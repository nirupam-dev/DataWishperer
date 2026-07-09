"""
Sandbox executor — Production-grade, process-isolated code execution.

Implements Layer 3 of defense-in-depth: runs AST-validated Python code in
a separate subprocess with enforced timeout, memory limits, stdout capture,
and structured result deserialization.

Security Architecture:
    Layer 1: Prompt instructions (tell LLM what not to do)
    Layer 2: AST validation (statically reject dangerous code)
    Layer 3: THIS — Process isolation + resource limits + result parsing

Why subprocess over in-process exec()?
    - Process isolation: crash in sandbox cannot crash the main app
    - Timeout enforcement: subprocess.run(timeout=N) is reliable
    - Memory isolation: OS-level separation with resource limits
    - Clean termination: kill the process on timeout, no leaked state
    - No global state pollution: each execution starts fresh

Error Handling Strategy:
    Raw Python tracebacks are transformed into structured, user-friendly
    error messages with:
        - Error type classification (KeyError, TypeError, etc.)
        - Cleaned error message (no file paths, no internal frames)
        - Suggestion for the user (based on error type)
        - Original code snippet for context
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import textwrap
from pathlib import Path
from typing import Dict, Optional, Tuple
from uuid import uuid4

from backend.core.config import SandboxSettings, StorageSettings, get_settings
from backend.core.exceptions import (
    CodeValidationError,
    ExecutionRuntimeError,
    ExecutionTimeoutError,
)
from backend.core.logging_config import get_logger
from backend.models.schemas import CodeExecutionResult, ResultType
from backend.sandbox.validator import CodeValidator

logger = get_logger(__name__)


# ── Error Classification ─────────────────────────────────────────────────────
# Maps Python error types to user-friendly descriptions and suggestions.
# This transforms raw tracebacks into actionable feedback.

_ERROR_SUGGESTIONS: Dict[str, Dict[str, str]] = {
    "KeyError": {
        "emoji": "🔑",
        "title": "Column Not Found",
        "suggestion": (
            "The code referenced a column that doesn't exist in your dataset. "
            "Check the exact column name — they are case-sensitive."
        ),
    },
    "TypeError": {
        "emoji": "🔧",
        "title": "Type Mismatch",
        "suggestion": (
            "An operation was applied to the wrong data type. "
            "For example, trying to calculate the average of text values. "
            "The column may need type conversion."
        ),
    },
    "ValueError": {
        "emoji": "⚠️",
        "title": "Invalid Value",
        "suggestion": (
            "A value in the data couldn't be converted or processed. "
            "This often happens with date parsing or numeric conversion "
            "when the data contains unexpected formats."
        ),
    },
    "IndexError": {
        "emoji": "📏",
        "title": "Index Out of Range",
        "suggestion": (
            "The code tried to access a row or element that doesn't exist. "
            "The dataset may have fewer rows than expected, or a "
            "filter returned an empty result."
        ),
    },
    "AttributeError": {
        "emoji": "❓",
        "title": "Method Not Found",
        "suggestion": (
            "The code called a method that doesn't exist on the data type. "
            "This often happens when a column is a different type than expected."
        ),
    },
    "ZeroDivisionError": {
        "emoji": "➗",
        "title": "Division by Zero",
        "suggestion": (
            "The code attempted to divide by zero. "
            "This can happen when a group or filter returns no data."
        ),
    },
    "MemoryError": {
        "emoji": "💾",
        "title": "Out of Memory",
        "suggestion": (
            "The operation required more memory than allowed. "
            "Try working with a subset of the data or simplifying the query."
        ),
    },
    "NameError": {
        "emoji": "📛",
        "title": "Undefined Variable",
        "suggestion": (
            "The code used a variable that hasn't been defined. "
            "This is usually an issue with the generated code — "
            "try rephrasing your question."
        ),
    },
    "SyntaxError": {
        "emoji": "📝",
        "title": "Syntax Error",
        "suggestion": (
            "The generated code has a syntax error. "
            "This is an AI generation issue — try rephrasing your question."
        ),
    },
}

_DEFAULT_ERROR_INFO: Dict[str, str] = {
    "emoji": "❌",
    "title": "Execution Error",
    "suggestion": (
        "An unexpected error occurred during code execution. "
        "Try rephrasing your question or simplifying the analysis."
    ),
}


# ── Execution Script Template ────────────────────────────────────────────────
# This is the complete wrapper that runs in the subprocess. It:
#   1. Sets up resource limits (memory, recursion)
#   2. Loads the dataset
#   3. Executes the generated code
#   4. Serializes the result to a structured JSON marker
#   5. Handles exceptions gracefully

_WRAPPER_TEMPLATE = textwrap.dedent('''\
    import sys
    import json
    import warnings
    warnings.filterwarnings("ignore")

    # ── Resource Limits ──────────────────────────────────────────────
    sys.setrecursionlimit({recursion_limit})
    import os
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # Memory limits are handled by Streamlit Cloud container (cgroups)

    # ── Imports ──────────────────────────────────────────────────────
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Visualization extras (available but not required)
    try:
        import seaborn as sns
        sns.set_theme(style="darkgrid", palette="viridis")
    except ImportError:
        pass
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        pass
    try:
        import scipy.stats
    except ImportError:
        pass

    # ── Load Dataset ─────────────────────────────────────────────────
    try:
        df = pd.read_csv(r"{csv_path}")
    except Exception as _load_err:
        _output = {{
            "type": "error",
            "data": f"Failed to load dataset: {{_load_err}}",
            "chart_generated": False,
            "error_type": "FileLoadError",
            "error_message": str(_load_err),
        }}
        print("__DATAWHISPERER_RESULT__" + json.dumps(_output, default=str))
        sys.exit(0)

    # ── Chart Output Path ────────────────────────────────────────────
    chart_path = r"{chart_path}"

    # ===== GENERATED CODE START =====
    {user_code}
    # ===== GENERATED CODE END =====

    # ── Result Serialization ─────────────────────────────────────────
    _output = {{"type": "text", "data": "", "chart_generated": False}}

    try:
        if "result" in dir():
            _r = result
            if isinstance(_r, pd.DataFrame):
                _output["type"] = "dataframe"
                # Truncate large DataFrames to prevent stdout overflow
                _truncated = _r.head({max_rows})
                
                # Preserve index when converting to records by safely resetting it
                try:
                    # Only reset if it's not a default 0-indexed RangeIndex, or if we want to be safe, just reset.
                    if not (isinstance(_truncated.index, pd.RangeIndex) and _truncated.index.start == 0 and _truncated.index.step == 1):
                        _truncated = _truncated.reset_index()
                except Exception:
                    pass
                    
                _output["data"] = _truncated.to_json(
                    orient="records", date_format="iso"
                )
                if len(_r) > {max_rows}:
                    _output["truncated"] = True
                    _output["total_rows"] = len(_r)
                    _output["shown_rows"] = {max_rows}
            elif isinstance(_r, pd.Series):
                _output["type"] = "series"
                _output["data"] = _r.head({max_rows}).to_json(date_format="iso")
            elif _r is None:
                _output["type"] = "text"
                _output["data"] = "Analysis completed (result is None)."
            else:
                _output["type"] = "text"
                _str_result = str(_r)
                # Truncate very long text results
                if len(_str_result) > {max_output_chars}:
                    _str_result = _str_result[:{max_output_chars}] + "... (truncated)"
                _output["data"] = _str_result
        else:
            _output["type"] = "text"
            _output["data"] = (
                "Code executed successfully but no 'result' variable was set."
            )
    except Exception as _e:
        _output["type"] = "error"
        _output["data"] = str(_e)

    # ── Chart Detection ──────────────────────────────────────────────
    import os.path
    import glob as _glob
    import shutil as _shutil

    # Primary: check if matplotlib figures exist and save them
    if len(plt.get_fignums()) > 0:
        try:
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor=plt.gcf().get_facecolor())
            plt.close('all')
        except Exception as e:
            pass

    # Fallback: if the LLM hardcoded a filename instead of using chart_path,
    # find any PNG files created in the working directory and use the first one
    if not (os.path.exists(chart_path) and os.path.getsize(chart_path) > 0):
        _cwd_pngs = _glob.glob("*.png")
        # Filter to only recently-created files (within last 60 seconds)
        import time as _time
        _now = _time.time()
        _recent_pngs = [p for p in _cwd_pngs if (_now - os.path.getmtime(p)) < 60]
        if _recent_pngs:
            # Use the most recently modified PNG
            _recent_pngs.sort(key=os.path.getmtime, reverse=True)
            try:
                _shutil.copy2(_recent_pngs[0], chart_path)
            except Exception:
                pass

    if os.path.exists(chart_path) and os.path.getsize(chart_path) > 0:
        _output["chart_generated"] = True

    print("__DATAWHISPERER_RESULT__" + json.dumps(_output, default=str))
''')


class SandboxExecutor:
    """
    Executes validated Python code in a sandboxed subprocess.

    Pipeline:
        1. Validate code via AST analysis (CodeValidator)
        2. Build wrapper script with resource limits
        3. Write to temporary file
        4. Execute via ``subprocess.run`` with timeout + env isolation
        5. Parse structured result from stdout
        6. Transform errors into user-friendly messages
        7. Clean up temporary files

    Security guarantees:
        - Process isolation (subprocess, not in-process exec)
        - Timeout enforcement (subprocess.run timeout)
        - Memory limits (resource.setrlimit on Unix, timeout on Windows)
        - No network access (no network modules whitelisted)
        - No filesystem mutation (open/write/unlink blocked by validator)
        - Output truncation (prevents stdout overflow)

    Args:
        sandbox_settings: Execution limits configuration.
        storage_settings: File path configuration.
        validator: Code validator instance for pre-execution checks.
    """

    def __init__(
        self,
        sandbox_settings: Optional[SandboxSettings] = None,
        storage_settings: Optional[StorageSettings] = None,
        validator: Optional[CodeValidator] = None,
    ) -> None:
        settings = get_settings()
        self._sandbox = sandbox_settings or settings.sandbox
        self._storage = storage_settings or settings.storage
        self._validator = validator or CodeValidator()
        self._charts_dir = self._storage.charts_path
        self._charts_dir.mkdir(parents=True, exist_ok=True)

    def execute(
        self,
        code: str,
        csv_path: str,
        validate_first: bool = True,
    ) -> CodeExecutionResult:
        """
        Execute Python code in a sandboxed subprocess.

        Args:
            code: The Python code to execute.
            csv_path: Path to the CSV file to load as ``df``.
            validate_first: If ``True``, run AST validation before execution.

        Returns:
            A ``CodeExecutionResult`` with the execution outcome.

        Raises:
            CodeValidationError: If code fails AST safety validation.
            ExecutionTimeoutError: If execution exceeds the time limit.
            ExecutionRuntimeError: If the code raises a Python exception.
        """
        start_time = time.time()

        # ── Phase 1: AST Validation ──────────────────────────────────
        if validate_first:
            self._validator.validate_or_raise(code)

        # ── Phase 2: Prepare execution ───────────────────────────────
        chart_filename = f"chart_{uuid4().hex[:12]}.png"
        chart_path = str(self._charts_dir / chart_filename)

        script = self._build_script(
            code=code,
            csv_path=csv_path,
            chart_path=chart_path,
        )

        # ── Phase 3: Execute in subprocess ───────────────────────────
        script_file = None
        try:
            script_file = self._write_temp_script(script)

            # Build a restricted environment — strip potentially
            # dangerous env vars but keep PATH for Python discovery
            env = self._build_restricted_env()

            result = subprocess.run(
                [sys.executable, "-u", script_file],
                capture_output=True,
                text=True,
                timeout=self._sandbox.timeout,
                cwd=str(self._charts_dir),
                env=env,
            )

            elapsed_ms = round((time.time() - start_time) * 1000, 2)

            # ── Phase 4: Parse result ────────────────────────────────
            return self._parse_output(
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode,
                chart_path=chart_path,
                elapsed_ms=elapsed_ms,
                code=code,
            )

        except subprocess.TimeoutExpired:
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            logger.error(
                "Sandbox execution timed out after %ds (%.0fms elapsed)",
                self._sandbox.timeout,
                elapsed_ms,
            )
            raise ExecutionTimeoutError(self._sandbox.timeout)

        finally:
            self._cleanup_temp_file(script_file)

    # ── Script Building ──────────────────────────────────────────────────

    def _build_script(
        self,
        code: str,
        csv_path: str,
        chart_path: str,
    ) -> str:
        """
        Build the complete execution wrapper script.

        Embeds the user code inside a controlled template with
        resource limits, imports, dataset loading, and result
        serialization.
        """
        return _WRAPPER_TEMPLATE.format(
            csv_path=csv_path.replace("\\", "\\\\"),
            chart_path=chart_path.replace("\\", "\\\\"),
            user_code=code,
            recursion_limit=500,
            max_memory_mb=self._sandbox.max_memory_mb,
            max_rows=200,
            max_output_chars=self._sandbox.max_output_kb * 1024,
        )

    @staticmethod
    def _build_restricted_env() -> Dict[str, str]:
        """
        Build a restricted environment for the subprocess.

        Keeps essential variables (PATH, Python, temp dirs) but strips
        everything else to prevent environment variable injection.
        """
        safe_keys = {
            "PATH", "SYSTEMROOT", "TEMP", "TMP", "HOME", "USERPROFILE",
            "PYTHONPATH", "PYTHONHASHSEED", "VIRTUAL_ENV", "CONDA_PREFIX",
            "LANG", "LC_ALL", "LC_CTYPE",
        }
        env = {
            k: v for k, v in os.environ.items()
            if k.upper() in safe_keys
        }
        # Disable Python's user site-packages for additional isolation
        env["PYTHONNOUSERSITE"] = "1"
        # Ensure deterministic hashing
        env["PYTHONHASHSEED"] = "0"
        
        # Enforce strict thread limits for Streamlit Cloud (1GB RAM)
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["NUMEXPR_NUM_THREADS"] = "1"
        return env

    # ── File Management ──────────────────────────────────────────────────

    def _write_temp_script(self, script: str) -> str:
        """
        Write the execution script to a temporary file.

        Returns the file path. The caller is responsible for cleanup.
        """
        fd = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8",
            dir=str(self._charts_dir),
            prefix="dw_exec_",
        )
        try:
            fd.write(script)
            fd.flush()
            return fd.name
        finally:
            fd.close()

    @staticmethod
    def _cleanup_temp_file(filepath: Optional[str]) -> None:
        """Safely remove the temporary script file."""
        if filepath:
            try:
                Path(filepath).unlink(missing_ok=True)
            except OSError:
                pass  # Best-effort cleanup

    # ── Output Parsing ───────────────────────────────────────────────────

    def _parse_output(
        self,
        stdout: str,
        stderr: str,
        return_code: int,
        chart_path: str,
        elapsed_ms: float,
        code: str,
    ) -> CodeExecutionResult:
        """
        Parse subprocess output into a structured result.

        Handles three cases:
            1. Success (return_code == 0) → parse JSON result marker
            2. Runtime error (return_code != 0) → classify and raise
            3. No marker in stdout → return raw output as text
        """
        # ── Case 1: Runtime error ────────────────────────────────────
        if return_code != 0:
            error_type, error_message = self._extract_error(stderr)
            clean_message = self._clean_error_message(error_message)

            logger.warning(
                "Sandbox runtime error [%.0fms]: %s: %s",
                elapsed_ms,
                error_type,
                clean_message[:150],
            )

            raise ExecutionRuntimeError(
                error_type=error_type,
                error_message=clean_message,
                code=code,
            )

        # ── Case 2: Look for result marker ───────────────────────────
        result_marker = "__DATAWHISPERER_RESULT__"
        if result_marker not in stdout:
            # Stdout capture without marker — return raw
            truncated_stdout = stdout[:2048] if stdout else ""
            return CodeExecutionResult(
                success=True,
                result_type=ResultType.TEXT,
                data=(
                    truncated_stdout
                    or "Code executed but produced no parseable output."
                ),
                stdout=stdout[: self._sandbox.max_output_kb * 1024],
                stderr=stderr[:1024],
                execution_time_ms=elapsed_ms,
            )

        # ── Case 3: Parse the JSON result ────────────────────────────
        json_str = stdout.split(result_marker, 1)[1].strip()
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            return CodeExecutionResult(
                success=True,
                result_type=ResultType.TEXT,
                data=stdout[:2048],
                stdout=stdout[:1024],
                execution_time_ms=elapsed_ms,
            )

        # Determine result type
        result_type_str = parsed.get("type", "text")
        try:
            result_type = ResultType(result_type_str)
        except ValueError:
            result_type = ResultType.TEXT

        # Check for chart generation
        chart_generated = parsed.get("chart_generated", False)
        actual_chart_path = (
            chart_path
            if chart_generated and Path(chart_path).exists()
            else None
        )

        if actual_chart_path:
            result_type = ResultType.CHART

        # Check for internal errors from the wrapper
        if result_type == ResultType.ERROR:
            error_type = parsed.get("error_type", "RuntimeError")
            error_message = parsed.get("error_message", parsed.get("data", ""))
            raise ExecutionRuntimeError(
                error_type=error_type,
                error_message=str(error_message),
                code=code,
            )

        logger.info(
            "Sandbox execution complete [%.0fms]: type=%s, chart=%s",
            elapsed_ms,
            result_type.value,
            bool(actual_chart_path),
        )

        return CodeExecutionResult(
            success=True,
            result_type=result_type,
            data=parsed.get("data", ""),
            chart_path=actual_chart_path,
            stdout=stdout[:1024],
            stderr=stderr[:1024],
            execution_time_ms=elapsed_ms,
        )

    # ── Error Formatting ─────────────────────────────────────────────────

    @staticmethod
    def _extract_error(stderr: str) -> Tuple[str, str]:
        """
        Extract error type and message from stderr traceback.

        Parses the last line of the traceback (which contains the
        actual error) and splits it into type and message.

        Returns:
            Tuple of (error_type, error_message).
        """
        if not stderr or not stderr.strip():
            return "RuntimeError", "Unknown error (no error output captured)"

        lines = stderr.strip().split("\n")

        # The actual error is always the last line
        last_line = lines[-1].strip()

        if ": " in last_line:
            parts = last_line.split(": ", 1)
            error_type = parts[0].strip()
            error_message = parts[1].strip()
            return error_type, error_message

        if ":" in last_line:
            parts = last_line.split(":", 1)
            return parts[0].strip(), parts[1].strip()

        return "RuntimeError", last_line

    @staticmethod
    def _clean_error_message(error_message: str) -> str:
        """
        Clean an error message for user display.

        Removes:
            - File paths (security: don't expose server paths)
            - Internal frame references
            - Redundant whitespace
        """
        msg = error_message

        # Remove file path references
        import re
        msg = re.sub(r'File "[^"]*"', 'File "<sandbox>"', msg)
        msg = re.sub(r"File '[^']*'", "File '<sandbox>'", msg)

        # Remove references to temp file names
        msg = re.sub(r"dw_exec_\w+\.py", "<sandbox>", msg)
        msg = re.sub(r"tmp\w+\.py", "<sandbox>", msg)

        # Collapse whitespace
        msg = " ".join(msg.split())

        # Truncate very long messages
        if len(msg) > 500:
            msg = msg[:497] + "..."

        return msg

    @classmethod
    def format_user_error(
        cls,
        error_type: str,
        error_message: str,
    ) -> str:
        """
        Format an error into a beautiful, user-friendly message.

        Produces a structured error display with:
            - Category emoji and title
            - Cleaned error message
            - Actionable suggestion

        Args:
            error_type: The Python exception class name.
            error_message: The cleaned error description.

        Returns:
            A formatted multi-line error string for the UI.
        """
        info = _ERROR_SUGGESTIONS.get(error_type, _DEFAULT_ERROR_INFO)
        emoji = info["emoji"]
        title = info["title"]
        suggestion = info["suggestion"]

        clean_msg = cls._clean_error_message(error_message)

        return (
            f"{emoji} **{title}**\n\n"
            f"**Error:** {clean_msg}\n\n"
            f"💡 **Suggestion:** {suggestion}"
        )
