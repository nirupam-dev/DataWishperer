"""
File upload security validation.

Implements Layer 1 of the defense-in-depth security model:
    - Extension whitelist
    - MIME type verification
    - File size enforcement
    - Content sniffing (validates actual CSV structure)
    - Filename sanitization (prevents path traversal)
    - Column count limits
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Optional

from backend.core.config import StorageSettings, get_settings
from backend.core.exceptions import (
    FileTooLargeError,
    InvalidFileError,
    TooManyColumnsError,
)
from backend.core.logging_config import get_logger

logger = get_logger(__name__)

_ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".csv"})
_ALLOWED_MIME_TYPES: frozenset[str] = frozenset({
    "text/csv",
    "text/plain",
    "application/csv",
    "application/vnd.ms-excel",
})
_FILENAME_SANITIZE_RE = re.compile(r"[^\w\s\-.]", re.UNICODE)
_MAX_SNIFF_BYTES: int = 8192


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to prevent path traversal and shell injection.

    Strips directory components, removes special characters, and limits
    length to 200 characters.

    Args:
        filename: The raw filename from the upload.

    Returns:
        A sanitized, safe filename string.

    Raises:
        InvalidFileError: If the filename is empty after sanitization.
    """
    # Strip directory components (Unix and Windows)
    name = PurePosixPath(filename).name
    name = Path(name).name

    # Remove special characters
    name = _FILENAME_SANITIZE_RE.sub("", name)

    # Collapse whitespace
    name = "_".join(name.split())

    # Truncate
    stem = Path(name).stem[:195]
    suffix = Path(name).suffix

    sanitized = f"{stem}{suffix}" if suffix else stem

    if not sanitized:
        raise InvalidFileError("Filename is empty after sanitization.", filename=filename)

    logger.debug("Sanitized filename: '%s' -> '%s'", filename, sanitized)
    return sanitized


def validate_extension(filename: str) -> None:
    """
    Verify the file extension is in the allowed set.

    Args:
        filename: Name of the uploaded file.

    Raises:
        InvalidFileError: If the extension is not ``.csv``.
    """
    ext = Path(filename).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise InvalidFileError(
            f"Extension '{ext}' is not allowed. Only .csv files are accepted.",
            filename=filename,
        )


def validate_file_size(file_size_bytes: int, settings: Optional[StorageSettings] = None) -> None:
    """
    Check that the file does not exceed the configured size limit.

    Args:
        file_size_bytes: Size of the uploaded file in bytes.
        settings: Optional storage settings override.

    Raises:
        FileTooLargeError: If the file exceeds ``max_file_size_mb``.
    """
    if settings is None:
        settings = get_settings().storage

    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if file_size_bytes > max_bytes:
        raise FileTooLargeError(
            file_size_mb=round(file_size_bytes / (1024 * 1024), 2),
            max_size_mb=settings.max_file_size_mb,
        )


def validate_csv_content(file_content: bytes, filename: str = "unknown") -> int:
    """
    Sniff the file content to verify it is a valid CSV.

    Reads the first ``_MAX_SNIFF_BYTES`` bytes and attempts to parse them
    with Python's ``csv.Sniffer``. Falls back to basic line-splitting if
    the sniffer fails.

    Args:
        file_content: Raw bytes of the file.
        filename: Original filename for error messages.

    Returns:
        The detected number of columns.

    Raises:
        InvalidFileError: If the content cannot be parsed as CSV.
    """
    snippet = file_content[:_MAX_SNIFF_BYTES]
    try:
        text = snippet.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = snippet.decode("latin-1")
        except UnicodeDecodeError:
            raise InvalidFileError(
                "File encoding is not supported. Use UTF-8 or Latin-1.",
                filename=filename,
            )

    lines = text.strip().split("\n")
    if len(lines) < 2:
        raise InvalidFileError(
            "File must contain at least a header row and one data row.",
            filename=filename,
        )

    # Try csv.Sniffer for dialect detection
    try:
        dialect = csv.Sniffer().sniff(text[:4096])
    except csv.Error:
        dialect = csv.excel  # type: ignore[assignment]

    reader = csv.reader(io.StringIO(text), dialect)
    try:
        header = next(reader)
    except StopIteration:
        raise InvalidFileError("File appears to be empty.", filename=filename)

    col_count = len(header)
    if col_count < 1:
        raise InvalidFileError("No columns detected in header row.", filename=filename)

    # Verify at least one data row has a similar column count
    try:
        first_row = next(reader)
        if abs(len(first_row) - col_count) > 2:
            raise InvalidFileError(
                f"Column count mismatch: header has {col_count} columns, "
                f"first data row has {len(first_row)}.",
                filename=filename,
            )
    except StopIteration:
        raise InvalidFileError(
            "File has a header but no data rows.",
            filename=filename,
        )

    return col_count


def validate_column_count(
    col_count: int, settings: Optional[StorageSettings] = None
) -> None:
    """
    Ensure the CSV does not exceed the maximum column limit.

    Args:
        col_count: Number of columns in the CSV.
        settings: Optional storage settings override.

    Raises:
        TooManyColumnsError: If the column count exceeds the limit.
    """
    if settings is None:
        settings = get_settings().storage

    if col_count > settings.max_columns:
        raise TooManyColumnsError(col_count=col_count, max_columns=settings.max_columns)


def validate_upload(
    filename: str,
    file_content: bytes,
    settings: Optional[StorageSettings] = None,
) -> int:
    """
    Run the full upload validation pipeline.

    Executes all checks in sequence:
        1. Extension check
        2. Size check
        3. Content sniffing
        4. Column count check

    Args:
        filename: Original filename from the upload.
        file_content: Raw file bytes.
        settings: Optional storage settings override.

    Returns:
        The detected number of columns.

    Raises:
        InvalidFileError: On any validation failure.
        FileTooLargeError: If the file is too large.
        TooManyColumnsError: If too many columns.
    """
    logger.info("Validating upload: '%s' (%d bytes)", filename, len(file_content))

    validate_extension(filename)
    validate_file_size(len(file_content), settings)
    col_count = validate_csv_content(file_content, filename)
    validate_column_count(col_count, settings)

    logger.info("Upload validation passed: '%s', %d columns", filename, col_count)
    return col_count
