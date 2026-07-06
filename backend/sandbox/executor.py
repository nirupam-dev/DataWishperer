"""
Sandbox executor — Secure, process-isolated code execution.

Implements Layer 3 of defense-in-depth: runs validated Python code in a
separate subprocess with enforced timeout, captures stdout, and deserializes
structured results (text, DataFrame, chart paths).

Why subprocess over in-process exec()?
    - Process isolation: crash in sandbox cannot crash the main app
    - Timeout enforcement: subprocess.run(timeout=N) is reliable
    - Memory isolation: OS-level separation
    - Clean termination: kill the process on timeout, no leaked state
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import textwrap
from pathlib import Path
from typing import Optional
from uuid import uuid4

from backend.core.config import SandboxSettings, StorageSettings, get_settings
from backend.core.exceptions import (
    ExecutionRuntimeError,
    ExecutionTimeoutError,
)
from backend.core.logging_config import get_logger
from backend.models.schemas import CodeExecutionResult, ResultType
from backend.sandbox.validator import CodeValidator

logger = get_logger(__name__)


# ── Execution Script Template ────────────────────────────────────────────────

_WRAPPER_TEMPLATE = textwrap.dedent('''\
    import sys
    import json
    import warnings
    warnings.filterwarnings("ignore")

    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Load the dataset
    df = pd.read_csv(r"{csv_path}")

    # Chart output path
    chart_path = r"{chart_path}"

    # ===== GENERATED CODE START =====
    {user_code}
    # ===== GENERATED CODE END =====

    # Serialize the result
    import json

    _output = {{"type": "text", "data": "", "chart_generated": False}}

    try:
        if "result" in dir():
            _r = result
            if isinstance(_r, pd.DataFrame):
                _output["type"] = "dataframe"
                _output["data"] = _r.head(200).to_json(orient="records", date_format="iso")
            elif isinstance(_r, pd.Series):
                _output["type"] = "series"
                _output["data"] = _r.head(200).to_json(date_format="iso")
            else:
                _output["type"] = "text"
                _output["data"] = str(_r)
        else:
            _output["type"] = "text"
            _output["data"] = "Code executed successfully but no 'result' variable was set."
    except Exception as _e:
        _output["type"] = "error"
        _output["data"] = str(_e)

    # Check if a chart was saved
    import os.path
    if os.path.exists(chart_path) and os.path.getsize(chart_path) > 0:
        _output["chart_generated"] = True

    print("__DATAWHISPERER_RESULT__" + json.dumps(_output, default=str))
''')


class SandboxExecutor:
    """
    Executes validated Python code in a sandboxed subprocess.

    The executor:
        1. Wraps user code in a controlled script template
        2. Writes the script to a temporary file
        3. Runs it via ``subprocess.run`` with a timeout
        4. Parses the structured result from stdout
        5. Cleans up temporary files

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
        """
        start_time = time.time()

        # Pre-execution validation
        if validate_first:
            self._validator.validate_or_raise(code)

        # Prepare chart output path
        chart_filename = f"chart_{uuid4().hex[:12]}.png"
        chart_path = str(self._charts_dir / chart_filename)

        # Build the execution script
        script = _WRAPPER_TEMPLATE.format(
            csv_path=csv_path.replace("\\", "\\\\"),
            chart_path=chart_path.replace("\\", "\\\\"),
            user_code=textwrap.indent(code, "    " * 0),
        )

        # Write to temporary file
        script_file = None
        try:
            script_file = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False,
                encoding="utf-8",
                dir=str(self._charts_dir),
            )
            script_file.write(script)
            script_file.flush()
            script_file.close()

            # Execute in subprocess
            result = subprocess.run(
                [sys.executable, script_file.name],
                capture_output=True,
                text=True,
                timeout=self._sandbox.timeout,
                cwd=str(self._charts_dir),
            )

            elapsed_ms = round((time.time() - start_time) * 1000, 2)

            # Parse result
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
            logger.error("Sandbox execution timed out after %ds", self._sandbox.timeout)
            raise ExecutionTimeoutError(self._sandbox.timeout)

        finally:
            # Clean up temp script file
            if script_file:
                try:
                    Path(script_file.name).unlink(missing_ok=True)
                except OSError:
                    pass

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

        Args:
            stdout: Captured standard output.
            stderr: Captured standard error.
            return_code: Process exit code.
            chart_path: Expected chart file path.
            elapsed_ms: Execution time in milliseconds.
            code: Original code (for error context).

        Returns:
            A ``CodeExecutionResult``.
        """
        # Check for runtime errors
        if return_code != 0:
            error_msg = stderr.strip() if stderr else "Unknown error"
            # Extract the last line which usually contains the actual error
            error_lines = error_msg.strip().split("\n")
            last_line = error_lines[-1] if error_lines else error_msg
            error_type = "RuntimeError"
            error_message = last_line

            if ":" in last_line:
                parts = last_line.split(":", 1)
                error_type = parts[0].strip()
                error_message = parts[1].strip()

            logger.warning(
                "Sandbox runtime error: %s: %s", error_type, error_message
            )
            raise ExecutionRuntimeError(
                error_type=error_type,
                error_message=error_message,
                code=code,
            )

        # Look for our result marker in stdout
        result_marker = "__DATAWHISPERER_RESULT__"
        if result_marker not in stdout:
            return CodeExecutionResult(
                success=True,
                result_type=ResultType.TEXT,
                data="Code executed but produced no parseable output.",
                stdout=stdout[:1024],
                stderr=stderr[:1024],
                execution_time_ms=elapsed_ms,
            )

        # Parse the JSON result
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

        result_type_str = parsed.get("type", "text")
        result_type = ResultType(result_type_str) if result_type_str in ResultType.__members__.values() else ResultType.TEXT

        # Check if chart was generated
        chart_generated = parsed.get("chart_generated", False)
        actual_chart_path = chart_path if chart_generated and Path(chart_path).exists() else None

        if actual_chart_path:
            result_type = ResultType.CHART

        return CodeExecutionResult(
            success=True,
            result_type=result_type,
            data=parsed.get("data", ""),
            chart_path=actual_chart_path,
            stdout=stdout[:1024],
            stderr=stderr[:1024],
            execution_time_ms=elapsed_ms,
        )
