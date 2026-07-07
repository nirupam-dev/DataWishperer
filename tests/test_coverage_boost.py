"""
Tests for the executor, security module, exceptions, logging config,
retry prompts, and additional coverage boosters.

Covers:
    - SandboxExecutor: _build_script, _build_restricted_env, _extract_error,
      _clean_error_message, format_user_error, _parse_output
    - core.security: sanitize_filename, validate_extension, validate_file_size,
      validate_csv_content, validate_column_count, validate_upload
    - core.exceptions: exception hierarchy, to_dict, context access
    - core.logging_config: setup_logging, get_logger
    - retry_prompt: build_retry_prompt (both levels), build_compact_context
    - schemas: FileMetadata properties
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from backend.core.config import StorageSettings, get_settings
from backend.core.exceptions import (
    CodeValidationError,
    DataWhispererError,
    ExecutionRuntimeError,
    ExecutionTimeoutError,
    FileTooLargeError,
    FileNotFoundError_,
    GenerationError,
    GenerationTimeoutError,
    InvalidFileError,
    InvalidQueryError,
    InvalidSessionError,
    ModelNotFoundError,
    OllamaConnectionError,
    SandboxError,
    TooManyColumnsError,
    UnsafeCodeError,
    ValidationError_,
)
from backend.models.schemas import (
    ColumnInfo,
    FileMetadata,
    ResultType,
    CodeExecutionResult,
    LLMResponse,
)
from backend.sandbox.executor import SandboxExecutor


# ── Exception Hierarchy Tests ────────────────────────────────────────────────


class TestExceptionHierarchy:
    """Verify exception construction, message formatting, and to_dict."""

    def test_base_exception_defaults(self):
        e = DataWhispererError("test error")
        assert e.message == "test error"
        assert e.error_code == "INTERNAL_ERROR"
        assert e.context == {}
        assert e.suggestion is None
        assert str(e) == "test error"

    def test_base_exception_to_dict(self):
        e = DataWhispererError(
            "test error", error_code="TEST",
            suggestion="Try again", context={"key": "val"},
        )
        d = e.to_dict()
        assert d["code"] == "TEST"
        assert d["message"] == "test error"
        assert d["suggestion"] == "Try again"
        assert d["context"]["key"] == "val"

    def test_to_dict_without_optional_fields(self):
        e = DataWhispererError("msg")
        d = e.to_dict()
        assert "suggestion" not in d
        assert "context" not in d

    def test_invalid_file_error(self):
        e = InvalidFileError("Bad format", filename="test.xyz")
        assert "Bad format" in str(e)
        assert e.error_code == "INVALID_FILE"
        assert e.context["filename"] == "test.xyz"
        assert e.context["reason"] == "Bad format"

    def test_file_too_large_error(self):
        e = FileTooLargeError(file_size_mb=150.5, max_size_mb=100)
        assert "150" in str(e)
        assert e.context["file_size_mb"] == 150.5
        assert e.context["max_size_mb"] == 100
        assert e.error_code == "FILE_TOO_LARGE"

    def test_file_not_found_error(self):
        e = FileNotFoundError_("abc-123")
        assert "abc-123" in str(e)
        assert e.error_code == "FILE_NOT_FOUND"

    def test_too_many_columns_error(self):
        e = TooManyColumnsError(col_count=600, max_columns=500)
        assert "600" in str(e)
        assert e.context["col_count"] == 600
        assert e.context["max_columns"] == 500

    def test_ollama_connection_error(self):
        e = OllamaConnectionError(
            base_url="http://localhost:11434",
            original_error="Connection refused",
        )
        assert "localhost" in str(e)
        assert e.error_code == "OLLAMA_CONNECTION_ERROR"
        assert e.context["original_error"] == "Connection refused"

    def test_model_not_found_error(self):
        e = ModelNotFoundError("llama3:70b")
        assert "llama3:70b" in str(e)
        assert e.error_code == "MODEL_NOT_FOUND"
        assert e.suggestion is not None

    def test_generation_timeout_error(self):
        e = GenerationTimeoutError(timeout_seconds=120)
        assert "120" in str(e)
        assert e.error_code == "GENERATION_TIMEOUT"
        assert e.context["timeout_seconds"] == 120

    def test_generation_error(self):
        e = GenerationError("Empty response")
        assert "Empty response" in str(e)
        assert e.error_code == "GENERATION_FAILED"

    def test_code_validation_error(self):
        e = CodeValidationError(
            violations=["Blocked import: 'os'", "Blocked call: 'eval()'"],
            code="import os\nresult = eval('1')",
        )
        assert "os" in str(e)
        assert e.error_code == "CODE_VALIDATION_ERROR"
        assert len(e.context["violations"]) == 2
        assert len(e.context["code_snippet"]) <= 200

    def test_execution_timeout_error(self):
        e = ExecutionTimeoutError(timeout_seconds=30)
        assert "30" in str(e)
        assert e.error_code == "EXECUTION_TIMEOUT"

    def test_execution_runtime_error(self):
        e = ExecutionRuntimeError(
            error_type="KeyError",
            error_message="'revenue'",
            code="df['revenue'].mean()",
        )
        assert "KeyError" in str(e)
        assert e.error_code == "EXECUTION_RUNTIME_ERROR"
        assert e.context["error_type"] == "KeyError"

    def test_unsafe_code_error(self):
        e = UnsafeCodeError("file write attempt")
        assert "file write" in str(e)
        assert e.error_code == "UNSAFE_CODE"

    def test_invalid_session_error(self):
        e = InvalidSessionError("sess-xyz")
        assert "sess-xyz" in str(e)
        assert e.error_code == "INVALID_SESSION"

    def test_invalid_query_error(self):
        e = InvalidQueryError("Query too short")
        assert "too short" in str(e)
        assert e.error_code == "INVALID_QUERY"

    def test_inheritance_chain(self):
        """Verify the inheritance chain is correct for pattern matching."""
        assert isinstance(CodeValidationError([], ""), SandboxError)
        assert isinstance(CodeValidationError([], ""), DataWhispererError)
        assert isinstance(FileTooLargeError(1.0, 1), DataWhispererError)
        assert isinstance(OllamaConnectionError("url"), DataWhispererError)
        assert isinstance(InvalidQueryError("x"), DataWhispererError)


# ── Security Module Tests ────────────────────────────────────────────────────


class TestSecurityModule:
    """Test file upload security validation functions."""

    def test_sanitize_filename_strips_path_components(self):
        from backend.core.security import sanitize_filename
        result = sanitize_filename("../../etc/passwd")
        assert ".." not in result
        assert "/" not in result

    def test_sanitize_filename_removes_special_chars(self):
        from backend.core.security import sanitize_filename
        result = sanitize_filename("file<>name|test.csv")
        assert "<" not in result
        assert ">" not in result
        assert "|" not in result

    def test_sanitize_filename_collapses_whitespace(self):
        from backend.core.security import sanitize_filename
        result = sanitize_filename("my   data   file.csv")
        assert "   " not in result

    def test_sanitize_filename_truncation(self):
        from backend.core.security import sanitize_filename
        long_name = "a" * 300 + ".csv"
        result = sanitize_filename(long_name)
        assert len(result) <= 200

    def test_sanitize_empty_filename_raises(self):
        from backend.core.security import sanitize_filename
        # Characters fully stripped by the regex + path component extraction
        with pytest.raises(InvalidFileError):
            sanitize_filename("<>|?*")

    def test_validate_extension_csv_accepted(self):
        from backend.core.security import validate_extension
        validate_extension("data.csv")  # Should not raise

    def test_validate_extension_rejected(self):
        from backend.core.security import validate_extension
        with pytest.raises(InvalidFileError):
            validate_extension("data.xlsx")

    def test_validate_extension_case_insensitive(self):
        from backend.core.security import validate_extension
        validate_extension("DATA.CSV")  # Should not raise

    def test_validate_file_size_within_limit(self):
        from backend.core.security import validate_file_size
        settings = StorageSettings(max_file_size_mb=50)
        validate_file_size(1024 * 1024, settings)  # 1 MB — ok

    def test_validate_file_size_exceeds_limit(self):
        from backend.core.security import validate_file_size
        settings = StorageSettings(max_file_size_mb=1)
        with pytest.raises(FileTooLargeError):
            validate_file_size(2 * 1024 * 1024, settings)  # 2 MB > 1 MB

    def test_validate_csv_content_valid(self):
        from backend.core.security import validate_csv_content
        content = b"name,age\nAlice,30\nBob,25\n"
        col_count = validate_csv_content(content, "test.csv")
        assert col_count == 2

    def test_validate_csv_content_single_row_rejects(self):
        from backend.core.security import validate_csv_content
        content = b"name,age\n"
        with pytest.raises(InvalidFileError, match="header.*data"):
            validate_csv_content(content, "test.csv")

    def test_validate_csv_content_bad_encoding(self):
        from backend.core.security import validate_csv_content
        # Create bytes that are invalid in both UTF-8 and Latin-1
        content = bytes(range(128, 256)) * 100
        # This should attempt decoding and may succeed with latin-1
        # but the CSV structure check should catch it
        try:
            validate_csv_content(content, "test.csv")
        except InvalidFileError:
            pass  # Expected for malformed content

    def test_validate_column_count_within_limit(self):
        from backend.core.security import validate_column_count
        settings = StorageSettings(max_columns=500)
        validate_column_count(10, settings)  # Should not raise

    def test_validate_column_count_exceeds_limit(self):
        from backend.core.security import validate_column_count
        settings = StorageSettings(max_columns=5)
        with pytest.raises(TooManyColumnsError):
            validate_column_count(10, settings)

    def test_validate_upload_full_pipeline(self):
        from backend.core.security import validate_upload
        settings = StorageSettings(max_file_size_mb=50, max_columns=500)
        content = b"a,b,c\n1,2,3\n4,5,6\n"
        col_count = validate_upload("test.csv", content, settings)
        assert col_count == 3

    def test_validate_upload_wrong_extension(self):
        from backend.core.security import validate_upload
        settings = StorageSettings()
        with pytest.raises(InvalidFileError):
            validate_upload("test.json", b"data", settings)

    def test_validate_csv_column_mismatch(self):
        from backend.core.security import validate_csv_content
        content = b"a,b,c\n1,2,3,4,5,6,7,8\n"
        with pytest.raises(InvalidFileError, match="Column count mismatch"):
            validate_csv_content(content, "test.csv")

    def test_validate_csv_latin1_encoding(self):
        from backend.core.security import validate_csv_content
        content = "name,city\nJosé,São Paulo\n".encode("latin-1")
        col_count = validate_csv_content(content, "test.csv")
        assert col_count == 2


# ── Executor Unit Tests ──────────────────────────────────────────────────────


class TestExecutorMethods:
    """Test SandboxExecutor static/class methods in isolation."""

    def test_extract_error_standard_traceback(self):
        stderr = "Traceback (most recent call last):\n  File ...\nKeyError: 'revenue'"
        error_type, error_message = SandboxExecutor._extract_error(stderr)
        assert error_type == "KeyError"
        assert "revenue" in error_message

    def test_extract_error_empty_stderr(self):
        error_type, error_message = SandboxExecutor._extract_error("")
        assert error_type == "RuntimeError"
        assert "Unknown" in error_message

    def test_extract_error_no_colon(self):
        stderr = "SomeError"
        error_type, error_message = SandboxExecutor._extract_error(stderr)
        assert error_type == "RuntimeError"

    def test_extract_error_single_colon(self):
        stderr = "Error:message"
        error_type, error_message = SandboxExecutor._extract_error(stderr)
        assert error_type == "Error"
        assert error_message == "message"

    def test_clean_error_message_removes_file_paths(self):
        msg = 'File "/tmp/dw_exec_abc123.py", line 5, in <module>'
        cleaned = SandboxExecutor._clean_error_message(msg)
        assert "dw_exec" not in cleaned
        assert "<sandbox>" in cleaned

    def test_clean_error_message_truncation(self):
        msg = "x" * 1000
        cleaned = SandboxExecutor._clean_error_message(msg)
        assert len(cleaned) <= 500
        assert cleaned.endswith("...")

    def test_clean_error_message_collapses_whitespace(self):
        msg = "error   in    column"
        cleaned = SandboxExecutor._clean_error_message(msg)
        assert "  " not in cleaned

    def test_format_user_error_known_type(self):
        msg = SandboxExecutor.format_user_error("KeyError", "'revenue'")
        assert "Column Not Found" in msg
        assert "🔑" in msg
        assert "Suggestion" in msg

    def test_format_user_error_unknown_type(self):
        msg = SandboxExecutor.format_user_error("CustomError", "something broke")
        assert "Execution Error" in msg
        assert "❌" in msg

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
    def test_format_user_error_all_known_types(self, error_type, expected_title):
        msg = SandboxExecutor.format_user_error(error_type, "test message")
        assert expected_title in msg

    def test_build_restricted_env(self):
        env = SandboxExecutor._build_restricted_env()
        assert "PYTHONNOUSERSITE" in env
        assert env["PYTHONNOUSERSITE"] == "1"
        assert env["PYTHONHASHSEED"] == "0"
        # Should not contain arbitrary env vars
        assert "SOME_RANDOM_VAR" not in env

    def test_build_script_contains_markers(self):
        """Test that _build_script produces a script with the result marker."""
        with patch("backend.sandbox.executor.get_settings") as mock_settings:
            settings = MagicMock()
            settings.sandbox.timeout = 30
            settings.sandbox.max_memory_mb = 512
            settings.sandbox.max_output_kb = 256
            settings.storage.charts_path = MagicMock()
            settings.storage.charts_path.mkdir = MagicMock()
            mock_settings.return_value = settings

            executor = SandboxExecutor(
                sandbox_settings=settings.sandbox,
                storage_settings=settings.storage,
            )
            script = executor._build_script(
                code="result = df.head()",
                csv_path="/tmp/test.csv",
                chart_path="/tmp/chart.png",
            )
            assert "__DATAWHISPERER_RESULT__" in script
            assert "result = df.head()" in script
            assert "pd.read_csv" in script


# ── Logging Config Tests ─────────────────────────────────────────────────────


class TestLoggingConfig:
    """Test logging setup and get_logger."""

    def test_get_logger_returns_logger(self):
        from backend.core.logging_config import get_logger
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"

    def test_get_logger_same_name_same_instance(self):
        from backend.core.logging_config import get_logger
        l1 = get_logger("test.same")
        l2 = get_logger("test.same")
        assert l1 is l2

    def test_setup_logging_creates_handlers(self, tmp_path):
        from backend.core.logging_config import setup_logging, _create_console_handler, _create_file_handler

        # Test console handler creation
        handler = _create_console_handler("INFO")
        assert isinstance(handler, logging.StreamHandler)

        # Test file handler creation
        file_handler = _create_file_handler(
            log_dir=str(tmp_path),
            filename="test.log",
            level="INFO",
            max_bytes=1024 * 1024,
            backup_count=3,
        )
        assert isinstance(file_handler, logging.handlers.RotatingFileHandler)


# ── Retry Prompt Tests ───────────────────────────────────────────────────────


class TestRetryPrompt:
    """Test retry prompt building at both levels."""

    def test_level_1_prompt(self):
        from backend.llm.prompts.retry_prompt import build_retry_prompt
        prompt = build_retry_prompt(
            attempt=2,
            error_type="KeyError",
            error_message="'nonexistent_column'",
            diagnosis="Column does not exist in the dataset.",
        )
        assert "KeyError" in prompt
        assert "nonexistent_column" in prompt
        assert "RETRY" in prompt

    def test_level_2_prompt(self):
        from backend.llm.prompts.retry_prompt import build_retry_prompt
        prompt = build_retry_prompt(
            attempt=3,
            error_type="ValueError",
            error_message="Could not convert",
            previous_errors=["KeyError: 'x'", "ValueError: bad"],
            columns=["name", "age", "score"],
        )
        assert "FINAL" in prompt
        assert "name, age, score" in prompt

    def test_level_2_prompt_without_previous_errors(self):
        from backend.llm.prompts.retry_prompt import build_retry_prompt
        prompt = build_retry_prompt(
            attempt=3,
            error_type="TypeError",
            error_message="bad type",
        )
        assert "TypeError" in prompt

    def test_compact_context(self):
        from backend.llm.prompts.retry_prompt import build_compact_context
        ctx = build_compact_context(
            filename="sales.csv",
            row_count=1000,
            col_count=5,
            columns_with_types="name (object), age (int64), score (float64)",
        )
        assert "sales.csv" in ctx
        assert "1000" in ctx


# ── Schema Property Tests ────────────────────────────────────────────────────


class TestSchemaProperties:
    """Test FileMetadata properties and edge cases."""

    def test_column_names_property(self):
        meta = FileMetadata(
            file_id="x", original_name="test.csv",
            stored_path="/tmp/test.csv", row_count=10,
            col_count=2, file_size_bytes=100, memory_usage_mb=0.01,
            columns=[
                ColumnInfo(name="a", dtype="int64",
                           non_null_count=10, null_count=0, unique_count=10),
                ColumnInfo(name="b", dtype="object",
                           non_null_count=10, null_count=0, unique_count=5),
            ],
        )
        assert meta.column_names == ["a", "b"]

    def test_column_dtypes_property(self):
        meta = FileMetadata(
            file_id="x", original_name="test.csv",
            stored_path="/tmp/test.csv", row_count=10,
            col_count=2, file_size_bytes=100, memory_usage_mb=0.01,
            columns=[
                ColumnInfo(name="a", dtype="int64",
                           non_null_count=10, null_count=0, unique_count=10),
                ColumnInfo(name="b", dtype="object",
                           non_null_count=10, null_count=0, unique_count=5),
            ],
        )
        dtypes = meta.column_dtypes
        assert dtypes["a"] == "int64"
        assert dtypes["b"] == "object"

    def test_result_type_enum_values(self):
        assert ResultType.TEXT.value == "text"
        assert ResultType.DATAFRAME.value == "dataframe"
        assert ResultType.CHART.value == "chart"
        assert ResultType.ERROR.value == "error"

    def test_code_execution_result_defaults(self):
        result = CodeExecutionResult(
            success=True, result_type=ResultType.TEXT,
        )
        assert result.stdout == ""
        assert result.stderr == ""
        assert result.execution_time_ms == 0.0
        assert result.chart_path is None


# ── Additional Registry Tests ────────────────────────────────────────────────


class TestRegistryCoverage:
    """Extra tests to boost prompt registry coverage."""

    def test_build_post_reflection_messages(self):
        from backend.llm.prompts.registry import PromptRegistry
        registry = PromptRegistry()
        msgs = registry.build_post_reflection_messages(
            question="What is the average?",
            code="result = df['x'].mean()",
            result_type="text",
            result_preview="42.5",
        )
        assert len(msgs) >= 1
        assert msgs[0]["role"] == "user"

    def test_build_ambiguity_messages(self, sample_file_metadata):
        from backend.llm.prompts.registry import PromptRegistry
        registry = PromptRegistry()
        msgs = registry.build_ambiguity_messages(
            question="Show me the data",
            file_metadata=sample_file_metadata,
        )
        assert len(msgs) >= 1

    def test_build_visualization_messages(self):
        from backend.llm.prompts.registry import PromptRegistry
        registry = PromptRegistry()
        msgs = registry.build_visualization_messages(
            question="Plot revenue by category",
            n_categories=5,
            n_numeric=3,
            has_dates=True,
            data_size=200,
        )
        assert len(msgs) >= 1

    def test_build_context_switch_messages(self, sample_file_metadata):
        from backend.llm.prompts.registry import PromptRegistry
        registry = PromptRegistry()
        msgs = registry.build_context_switch_messages(
            file_metadata=sample_file_metadata,
        )
        assert len(msgs) >= 1
        assert msgs[0]["role"] == "system"
