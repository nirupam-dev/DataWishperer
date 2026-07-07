"""
Error handling tests.

Verify error propagation, classification, and user-facing messages
across all layers of the application.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.core.exceptions import (
    CodeValidationError,
    DataWhispererError,
    ExecutionRuntimeError,
    ExecutionTimeoutError,
    FileTooLargeError,
    GenerationError,
    GenerationTimeoutError,
    InvalidFileError,
    ModelNotFoundError,
    OllamaConnectionError,
    TooManyColumnsError,
)
from backend.sandbox.executor import SandboxExecutor
from backend.sandbox.validator import CodeValidator


# ── Exception Hierarchy Tests ───────────────────────────────────────────────


class TestExceptionHierarchy:
    """Verify all custom exceptions inherit from DataWhispererError."""

    @pytest.mark.parametrize("exc_class", [
        InvalidFileError,
        FileTooLargeError,
        TooManyColumnsError,
        CodeValidationError,
        ExecutionRuntimeError,
        ExecutionTimeoutError,
        GenerationError,
        GenerationTimeoutError,
        OllamaConnectionError,
        ModelNotFoundError,
    ])
    def test_inherits_from_base(self, exc_class):
        assert issubclass(exc_class, DataWhispererError)

    def test_base_exception_message(self):
        e = DataWhispererError("test message")
        assert str(e) == "test message"


# ── Exception Creation Tests ────────────────────────────────────────────────


class TestExceptionCreation:
    """Verify custom exceptions store correct metadata."""

    def test_invalid_file_error(self):
        e = InvalidFileError("Bad file format", filename="test.xyz")
        assert "Bad file format" in str(e)
        assert hasattr(e, 'error_code')

    def test_file_too_large_error(self):
        e = FileTooLargeError(file_size_mb=150.5, max_size_mb=100)
        assert "150" in str(e)
        assert e.context["file_size_mb"] == 150.5
        assert e.context["max_size_mb"] == 100

    def test_too_many_columns_error(self):
        e = TooManyColumnsError(col_count=200, max_columns=100)
        assert "200" in str(e)
        assert e.context["col_count"] == 200
        assert e.context["max_columns"] == 100

    def test_code_validation_error(self):
        e = CodeValidationError(["Blocked import: os"])
        assert "os" in str(e)

    def test_execution_runtime_error(self):
        e = ExecutionRuntimeError(error_type="KeyError", error_message="'revenue'")
        assert "revenue" in str(e)

    def test_execution_timeout_error(self):
        e = ExecutionTimeoutError(timeout_seconds=30)
        assert e.context["timeout_seconds"] == 30
        assert "30" in str(e)

    def test_generation_error(self):
        e = GenerationError("Could not extract code")
        assert "code" in str(e)

    def test_generation_timeout_error(self):
        e = GenerationTimeoutError(timeout_seconds=60)
        assert e.context["timeout_seconds"] == 60

    def test_ollama_connection_error(self):
        e = OllamaConnectionError(
            base_url="http://localhost:11434",
            original_error="Connection refused",
        )
        assert "localhost" in str(e)
        assert e.context["base_url"] == "http://localhost:11434"

    def test_model_not_found_error(self):
        e = ModelNotFoundError("qwen2.5:7b")
        assert "qwen2.5" in str(e)


# ── Validator Error Handling Tests ──────────────────────────────────────────


class TestValidatorErrorHandling:
    """Test validator error paths."""

    def setup_method(self):
        self.validator = CodeValidator()

    def test_validate_or_raise_raises_correctly(self):
        with pytest.raises(CodeValidationError) as exc_info:
            self.validator.validate_or_raise("import os\nresult = 42")
        assert "os" in str(exc_info.value)

    def test_validation_error_contains_violations(self):
        violations = self.validator.validate("import os\nimport subprocess\nresult = 42")
        critical = [v for v in violations if v.description]
        assert len(critical) >= 2

    def test_syntax_error_produces_clear_message(self):
        violations = self.validator.validate("def foo(:\n  pass")
        syntax_v = [v for v in violations if v.category == "syntax"]
        assert len(syntax_v) > 0
        assert len(syntax_v[0].description) > 10


# ── Executor Error Formatting Tests ─────────────────────────────────────────


class TestExecutorErrorFormatting:
    """Test user-facing error message quality."""

    @pytest.mark.parametrize("error_type,expected_keyword", [
        ("KeyError", "Column Not Found"),
        ("TypeError", "Type Mismatch"),
        ("ValueError", "Invalid Value"),
        ("IndexError", "Index Out of Range"),
        ("ZeroDivisionError", "Division by Zero"),
        ("MemoryError", "Out of Memory"),
        ("NameError", "Undefined Variable"),
        ("AttributeError", "Method Not Found"),
        ("SyntaxError", "Syntax Error"),
    ])
    def test_error_type_produces_title(self, error_type, expected_keyword):
        result = SandboxExecutor.format_user_error(error_type, "test message")
        assert expected_keyword in result

    def test_error_includes_suggestion(self):
        result = SandboxExecutor.format_user_error("KeyError", "'revenue'")
        assert "Suggestion" in result
        assert "case-sensitive" in result

    def test_error_includes_emoji(self):
        result = SandboxExecutor.format_user_error("KeyError", "'revenue'")
        assert "🔑" in result

    def test_error_includes_original_message(self):
        result = SandboxExecutor.format_user_error("ValueError", "could not convert")
        assert "could not convert" in result

    def test_unknown_error_type_handled(self):
        result = SandboxExecutor.format_user_error("CustomError", "weird thing")
        assert "Execution Error" in result
        assert "weird thing" in result

    def test_error_message_path_cleaning(self):
        raw = 'File "/tmp/dw_exec_abc123.py", line 5: error'
        cleaned = SandboxExecutor._clean_error_message(raw)
        assert "dw_exec_abc123" not in cleaned
        assert "<sandbox>" in cleaned

    def test_error_message_truncation(self):
        raw = "x" * 1000
        cleaned = SandboxExecutor._clean_error_message(raw)
        assert len(cleaned) <= 500


# ── Error Propagation Integration Tests ─────────────────────────────────────


class TestErrorPropagation:
    """Test that errors propagate correctly through the stack."""

    def test_upload_validation_chain(self):
        """Invalid extension should raise before reaching analysis."""
        from backend.core.security import validate_upload
        with pytest.raises(InvalidFileError):
            validate_upload(
                filename="data.txt",
                file_content=b"some content",
            )

    @patch("backend.core.security.get_settings")
    def test_size_validation_chain(self, mock_settings):
        """Oversized file should raise before reaching analysis."""
        mock_settings.return_value.storage = MagicMock(
            max_file_size_mb=1, max_columns=100
        )
        from backend.core.security import validate_upload
        with pytest.raises(FileTooLargeError):
            validate_upload(
                filename="huge.csv",
                file_content=b"x" * (2 * 1024 * 1024),
                settings=mock_settings.return_value.storage,
            )

    def test_quality_analyzer_handles_empty_df(self):
        import pandas as pd
        from backend.analytics.data_quality import DataQualityAnalyzer
        report = DataQualityAnalyzer().analyze(pd.DataFrame())
        assert report.overall_quality_score == 0
        assert "empty" in report.summary.lower()
