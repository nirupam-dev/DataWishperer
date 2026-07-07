"""
Unit tests for the sandbox executor.

Tests the execution wrapper, output parsing, error formatting,
script building, and environment isolation — all without running
actual subprocesses (those are tested in integration tests).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.core.exceptions import (
    CodeValidationError,
    ExecutionRuntimeError,
    ExecutionTimeoutError,
)
from backend.models.schemas import CodeExecutionResult, ResultType
from backend.sandbox.executor import (
    SandboxExecutor,
    _DEFAULT_ERROR_INFO,
    _ERROR_SUGGESTIONS,
)


# ── Error Classification Tests ──────────────────────────────────────────────


class TestErrorClassification:
    """Test the error suggestion mapping."""

    @pytest.mark.parametrize("error_type,expected_title", [
        ("KeyError", "Column Not Found"),
        ("TypeError", "Type Mismatch"),
        ("ValueError", "Invalid Value"),
        ("IndexError", "Index Out of Range"),
        ("AttributeError", "Method Not Found"),
        ("ZeroDivisionError", "Division by Zero"),
        ("MemoryError", "Out of Memory"),
        ("NameError", "Undefined Variable"),
        ("SyntaxError", "Syntax Error"),
    ])
    def test_known_error_types_have_suggestions(self, error_type, expected_title):
        assert error_type in _ERROR_SUGGESTIONS
        assert _ERROR_SUGGESTIONS[error_type]["title"] == expected_title
        assert len(_ERROR_SUGGESTIONS[error_type]["suggestion"]) > 10
        assert len(_ERROR_SUGGESTIONS[error_type]["emoji"]) > 0

    def test_default_error_info(self):
        assert _DEFAULT_ERROR_INFO["title"] == "Execution Error"
        assert len(_DEFAULT_ERROR_INFO["suggestion"]) > 10


# ── Error Formatting Tests ──────────────────────────────────────────────────


class TestErrorFormatting:
    """Test format_user_error output quality."""

    def test_known_error_formatted(self):
        result = SandboxExecutor.format_user_error("KeyError", "'revenue'")
        assert "🔑" in result
        assert "Column Not Found" in result
        assert "revenue" in result
        assert "Suggestion" in result

    def test_unknown_error_uses_default(self):
        result = SandboxExecutor.format_user_error("CustomWeirdError", "something")
        assert "Execution Error" in result
        assert "❌" in result

    def test_error_message_cleaned(self):
        """File paths should be replaced with <sandbox> in error messages."""
        raw = 'File "/tmp/dw_exec_abc123.py", line 5, in <module>'
        cleaned = SandboxExecutor._clean_error_message(raw)
        # File paths are replaced with <sandbox>, not removed
        assert "dw_exec_abc123" not in cleaned
        assert "<sandbox>" in cleaned

    def test_long_error_message_truncated(self):
        long_msg = "x" * 1000
        cleaned = SandboxExecutor._clean_error_message(long_msg)
        assert len(cleaned) <= 500
        assert cleaned.endswith("...")


# ── Error Extraction Tests ──────────────────────────────────────────────────


class TestErrorExtraction:
    """Test _extract_error parsing of stderr."""

    def test_standard_traceback(self):
        stderr = (
            "Traceback (most recent call last):\n"
            '  File "/tmp/script.py", line 5, in <module>\n'
            "    df['nonexistent']\n"
            "KeyError: 'nonexistent'"
        )
        error_type, msg = SandboxExecutor._extract_error(stderr)
        assert error_type == "KeyError"
        assert "nonexistent" in msg

    def test_empty_stderr(self):
        error_type, msg = SandboxExecutor._extract_error("")
        assert error_type == "RuntimeError"
        assert "Unknown" in msg

    def test_whitespace_only_stderr(self):
        error_type, msg = SandboxExecutor._extract_error("   \n  \n  ")
        assert error_type == "RuntimeError"

    def test_no_colon_in_error(self):
        stderr = "SomeStrangeError"
        error_type, msg = SandboxExecutor._extract_error(stderr)
        assert error_type == "RuntimeError"

    def test_colon_without_space(self):
        stderr = "ValueError:invalid literal"
        error_type, msg = SandboxExecutor._extract_error(stderr)
        assert error_type == "ValueError"


# ── Script Building Tests ───────────────────────────────────────────────────


class TestScriptBuilding:
    """Test _build_script template rendering."""

    @patch("backend.sandbox.executor.get_settings")
    def test_script_contains_user_code(self, mock_settings):
        mock_settings.return_value = MagicMock()
        mock_settings.return_value.sandbox = MagicMock(
            timeout=30, max_memory_mb=512, max_output_kb=256
        )
        mock_settings.return_value.storage = MagicMock(
            charts_path=Path("/tmp/charts")
        )

        executor = SandboxExecutor.__new__(SandboxExecutor)
        executor._sandbox = mock_settings.return_value.sandbox
        executor._storage = mock_settings.return_value.storage
        executor._validator = MagicMock()
        executor._charts_dir = Path("/tmp/charts")

        script = executor._build_script(
            code="result = df.head()",
            csv_path="/tmp/test.csv",
            chart_path="/tmp/charts/chart.png",
        )
        assert "result = df.head()" in script
        assert "GENERATED CODE START" in script
        assert "GENERATED CODE END" in script
        assert "__DATAWHISPERER_RESULT__" in script

    @patch("backend.sandbox.executor.get_settings")
    def test_script_has_resource_limits(self, mock_settings):
        mock_settings.return_value = MagicMock()
        mock_settings.return_value.sandbox = MagicMock(
            timeout=30, max_memory_mb=512, max_output_kb=256
        )
        mock_settings.return_value.storage = MagicMock(
            charts_path=Path("/tmp/charts")
        )

        executor = SandboxExecutor.__new__(SandboxExecutor)
        executor._sandbox = mock_settings.return_value.sandbox
        executor._storage = mock_settings.return_value.storage
        executor._validator = MagicMock()
        executor._charts_dir = Path("/tmp/charts")

        script = executor._build_script(
            code="result = 42",
            csv_path="/tmp/test.csv",
            chart_path="/tmp/charts/chart.png",
        )
        assert "setrecursionlimit" in script
        assert "RLIMIT_AS" in script or "resource" in script


# ── Environment Isolation Tests ─────────────────────────────────────────────


class TestEnvironmentIsolation:
    """Test _build_restricted_env security."""

    def test_restricted_env_strips_dangerous_vars(self):
        import os
        original_env = os.environ.copy()
        os.environ["API_KEY"] = "secret_value"
        os.environ["DATABASE_URL"] = "postgres://evil"

        try:
            env = SandboxExecutor._build_restricted_env()
            assert "API_KEY" not in env
            assert "DATABASE_URL" not in env
        finally:
            os.environ.update(original_env)
            os.environ.pop("API_KEY", None)
            os.environ.pop("DATABASE_URL", None)

    def test_restricted_env_keeps_path(self):
        env = SandboxExecutor._build_restricted_env()
        assert "PATH" in env or "Path" in env

    def test_restricted_env_disables_user_site(self):
        env = SandboxExecutor._build_restricted_env()
        assert env.get("PYTHONNOUSERSITE") == "1"

    def test_restricted_env_deterministic_hash(self):
        env = SandboxExecutor._build_restricted_env()
        assert env.get("PYTHONHASHSEED") == "0"


# ── Output Parsing Tests ───────────────────────────────────────────────────


class TestOutputParsing:
    """Test _parse_output logic for various result types."""

    @patch("backend.sandbox.executor.get_settings")
    def _make_executor(self, mock_settings):
        mock_settings.return_value = MagicMock()
        mock_settings.return_value.sandbox = MagicMock(
            timeout=30, max_memory_mb=512, max_output_kb=256
        )
        mock_settings.return_value.storage = MagicMock(
            charts_path=Path("/tmp/charts")
        )
        executor = SandboxExecutor.__new__(SandboxExecutor)
        executor._sandbox = mock_settings.return_value.sandbox
        executor._storage = mock_settings.return_value.storage
        executor._validator = MagicMock()
        executor._charts_dir = Path("/tmp/charts")
        return executor

    def test_parse_text_result(self):
        executor = self._make_executor()
        result_json = json.dumps({
            "type": "text",
            "data": "Average revenue is $5,000",
            "chart_generated": False,
        })
        stdout = f"__DATAWHISPERER_RESULT__{result_json}"
        result = executor._parse_output(
            stdout=stdout, stderr="", return_code=0,
            chart_path="/tmp/charts/chart.png",
            elapsed_ms=100.0, code="result = 42",
        )
        assert result.success
        assert result.result_type == ResultType.TEXT
        assert "5,000" in result.data

    def test_parse_dataframe_result(self):
        executor = self._make_executor()
        result_json = json.dumps({
            "type": "dataframe",
            "data": '[{"a": 1, "b": 2}]',
            "chart_generated": False,
        })
        stdout = f"__DATAWHISPERER_RESULT__{result_json}"
        result = executor._parse_output(
            stdout=stdout, stderr="", return_code=0,
            chart_path="/tmp/charts/chart.png",
            elapsed_ms=50.0, code="result = df.head()",
        )
        assert result.success
        assert result.result_type == ResultType.DATAFRAME

    def test_parse_runtime_error_raises(self):
        executor = self._make_executor()
        stderr = "Traceback:\nKeyError: 'nonexistent'"
        with pytest.raises(ExecutionRuntimeError):
            executor._parse_output(
                stdout="", stderr=stderr, return_code=1,
                chart_path="/tmp/charts/chart.png",
                elapsed_ms=50.0, code="result = df['nonexistent']",
            )

    def test_parse_no_marker_returns_raw(self):
        executor = self._make_executor()
        result = executor._parse_output(
            stdout="some raw output", stderr="", return_code=0,
            chart_path="/tmp/charts/chart.png",
            elapsed_ms=50.0, code="result = 42",
        )
        assert result.success
        assert result.result_type == ResultType.TEXT
        assert "some raw output" in result.data

    def test_parse_empty_stdout_no_marker(self):
        executor = self._make_executor()
        result = executor._parse_output(
            stdout="", stderr="", return_code=0,
            chart_path="/tmp/charts/chart.png",
            elapsed_ms=50.0, code="result = 42",
        )
        assert result.success
        assert "no parseable output" in result.data.lower()

    def test_parse_invalid_json_returns_raw(self):
        executor = self._make_executor()
        stdout = "__DATAWHISPERER_RESULT__{invalid json"
        result = executor._parse_output(
            stdout=stdout, stderr="", return_code=0,
            chart_path="/tmp/charts/chart.png",
            elapsed_ms=50.0, code="result = 42",
        )
        assert result.success
        assert result.result_type == ResultType.TEXT

    def test_parse_error_type_raises(self):
        executor = self._make_executor()
        result_json = json.dumps({
            "type": "error",
            "data": "Failed to load dataset",
            "chart_generated": False,
            "error_type": "FileLoadError",
            "error_message": "file not found",
        })
        stdout = f"__DATAWHISPERER_RESULT__{result_json}"
        with pytest.raises(ExecutionRuntimeError):
            executor._parse_output(
                stdout=stdout, stderr="", return_code=0,
                chart_path="/tmp/charts/chart.png",
                elapsed_ms=50.0, code="result = 42",
            )
