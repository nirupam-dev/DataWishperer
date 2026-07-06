"""
Custom exception hierarchy for DataWhisperer.

Every exception carries a machine-readable ``error_code``, a human-friendly
``message``, and an optional ``context`` dict for structured logging.
The hierarchy mirrors the application layers so that error handlers in the
API / UI layer can pattern-match on category.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class DataWhispererError(Exception):
    """
    Base exception for all DataWhisperer errors.

    Attributes:
        message: Human-readable error description.
        error_code: Machine-readable error identifier (e.g. ``SANDBOX_TIMEOUT``).
        context: Additional structured data for logging / debugging.
        suggestion: Actionable suggestion shown to the end user.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "INTERNAL_ERROR",
        context: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None,
    ) -> None:
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.suggestion = suggestion
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the exception for JSON responses."""
        payload: Dict[str, Any] = {
            "code": self.error_code,
            "message": self.message,
        }
        if self.suggestion:
            payload["suggestion"] = self.suggestion
        if self.context:
            payload["context"] = self.context
        return payload


# ── File Errors ──────────────────────────────────────────────────────────────


class FileError(DataWhispererError):
    """Base class for file-related errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, error_code="FILE_ERROR", **kwargs)


class FileTooLargeError(FileError):
    """Raised when an uploaded file exceeds the size limit."""

    def __init__(self, file_size_mb: float, max_size_mb: int) -> None:
        super().__init__(
            message=f"File size ({file_size_mb:.1f} MB) exceeds the {max_size_mb} MB limit.",
            context={"file_size_mb": file_size_mb, "max_size_mb": max_size_mb},
            suggestion="Try uploading a smaller file or reduce the number of rows.",
        )
        self.error_code = "FILE_TOO_LARGE"


class InvalidFileError(FileError):
    """Raised when a file fails validation (not a valid CSV, wrong extension, etc.)."""

    def __init__(self, reason: str, filename: Optional[str] = None) -> None:
        super().__init__(
            message=f"Invalid file: {reason}",
            context={"filename": filename, "reason": reason},
            suggestion="Please upload a valid .csv file.",
        )
        self.error_code = "INVALID_FILE"


class FileNotFoundError_(FileError):
    """Raised when a referenced file does not exist on disk."""

    def __init__(self, file_id: str) -> None:
        super().__init__(
            message=f"File with ID '{file_id}' was not found.",
            context={"file_id": file_id},
            suggestion="The file may have been deleted. Please upload it again.",
        )
        self.error_code = "FILE_NOT_FOUND"


class TooManyColumnsError(FileError):
    """Raised when a CSV contains more columns than allowed."""

    def __init__(self, col_count: int, max_columns: int) -> None:
        super().__init__(
            message=f"CSV has {col_count} columns, exceeding the {max_columns} limit.",
            context={"col_count": col_count, "max_columns": max_columns},
            suggestion="Reduce the number of columns before uploading.",
        )
        self.error_code = "TOO_MANY_COLUMNS"


# ── LLM Errors ───────────────────────────────────────────────────────────────


class LLMError(DataWhispererError):
    """Base class for LLM / Ollama errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, error_code="LLM_ERROR", **kwargs)


class OllamaConnectionError(LLMError):
    """Raised when the Ollama server is unreachable."""

    def __init__(self, base_url: str, original_error: Optional[str] = None) -> None:
        super().__init__(
            message=f"Cannot connect to Ollama at {base_url}.",
            context={"base_url": base_url, "original_error": original_error},
            suggestion=(
                "Make sure Ollama is running:\n"
                "  1. Open a terminal\n"
                "  2. Run: ollama serve\n"
                "  3. Run: ollama pull qwen2.5:7b"
            ),
        )
        self.error_code = "OLLAMA_CONNECTION_ERROR"


class ModelNotFoundError(LLMError):
    """Raised when the requested model is not available in Ollama."""

    def __init__(self, model: str) -> None:
        super().__init__(
            message=f"Model '{model}' is not available in Ollama.",
            context={"model": model},
            suggestion=f"Pull the model by running: ollama pull {model}",
        )
        self.error_code = "MODEL_NOT_FOUND"


class GenerationTimeoutError(LLMError):
    """Raised when the LLM fails to respond within the timeout."""

    def __init__(self, timeout_seconds: int) -> None:
        super().__init__(
            message=f"LLM generation timed out after {timeout_seconds} seconds.",
            context={"timeout_seconds": timeout_seconds},
            suggestion="Try a simpler question or increase the timeout in Settings.",
        )
        self.error_code = "GENERATION_TIMEOUT"


class GenerationError(LLMError):
    """Raised when the LLM produces an unparseable or empty response."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            message=f"LLM generation failed: {reason}",
            context={"reason": reason},
            suggestion="Try rephrasing your question.",
        )
        self.error_code = "GENERATION_FAILED"


# ── Sandbox Errors ───────────────────────────────────────────────────────────


class SandboxError(DataWhispererError):
    """Base class for code execution sandbox errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, error_code="SANDBOX_ERROR", **kwargs)


class CodeValidationError(SandboxError):
    """Raised when generated code fails AST safety validation."""

    def __init__(self, violations: list[str], code: Optional[str] = None) -> None:
        super().__init__(
            message=f"Code validation failed: {'; '.join(violations)}",
            context={"violations": violations, "code_snippet": (code or "")[:200]},
            suggestion="The AI generated unsafe code. Retrying with stricter instructions.",
        )
        self.error_code = "CODE_VALIDATION_ERROR"


class ExecutionTimeoutError(SandboxError):
    """Raised when sandbox code execution exceeds the time limit."""

    def __init__(self, timeout_seconds: int) -> None:
        super().__init__(
            message=f"Code execution timed out after {timeout_seconds} seconds.",
            context={"timeout_seconds": timeout_seconds},
            suggestion="Your query may be too complex. Try simplifying or filtering first.",
        )
        self.error_code = "EXECUTION_TIMEOUT"


class ExecutionRuntimeError(SandboxError):
    """Raised when sandboxed code raises a Python exception at runtime."""

    def __init__(self, error_type: str, error_message: str, code: Optional[str] = None) -> None:
        super().__init__(
            message=f"Runtime error: {error_type}: {error_message}",
            context={
                "error_type": error_type,
                "error_message": error_message,
                "code_snippet": (code or "")[:200],
            },
            suggestion="The AI will try a different approach.",
        )
        self.error_code = "EXECUTION_RUNTIME_ERROR"


class UnsafeCodeError(SandboxError):
    """Raised when code attempts a disallowed operation at runtime."""

    def __init__(self, operation: str) -> None:
        super().__init__(
            message=f"Blocked unsafe operation: {operation}",
            context={"operation": operation},
            suggestion="The generated code attempted a disallowed operation and was blocked.",
        )
        self.error_code = "UNSAFE_CODE"


# ── Validation Errors ────────────────────────────────────────────────────────


class ValidationError_(DataWhispererError):
    """Base class for input validation errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, error_code="VALIDATION_ERROR", **kwargs)


class InvalidSessionError(ValidationError_):
    """Raised when a session ID is invalid or does not exist."""

    def __init__(self, session_id: str) -> None:
        super().__init__(
            message=f"Session '{session_id}' does not exist.",
            context={"session_id": session_id},
            suggestion="Start a new session or select an existing one.",
        )
        self.error_code = "INVALID_SESSION"


class InvalidQueryError(ValidationError_):
    """Raised when a user query fails validation."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            message=f"Invalid query: {reason}",
            context={"reason": reason},
            suggestion="Please enter a valid question about your data.",
        )
        self.error_code = "INVALID_QUERY"
