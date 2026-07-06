"""
Structured logging configuration for DataWhisperer.

Sets up dual-sink logging:
  - **Console**: Human-readable coloured output for development.
  - **File**: JSON-lines format with automatic rotation for production.

Usage::

    from backend.core.logging_config import setup_logging, get_logger

    setup_logging()  # Call once at startup
    logger = get_logger(__name__)
    logger.info("Server started", port=8501)
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

from backend.core.config import get_settings


def _create_console_handler(level: str) -> logging.StreamHandler:
    """Create a coloured console handler with human-readable formatting."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level))
    formatter = logging.Formatter(
        fmt="%(asctime)s │ %(levelname)-8s │ %(name)-30s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    return handler


def _create_file_handler(
    log_dir: str,
    filename: str,
    level: str,
    max_bytes: int,
    backup_count: int,
) -> logging.handlers.RotatingFileHandler:
    """
    Create a rotating file handler.

    Args:
        log_dir: Directory to write log files into.
        filename: Log file name (e.g. ``app.log``).
        level: Minimum log level for this handler.
        max_bytes: Maximum file size before rotation.
        backup_count: Number of rotated files to keep.

    Returns:
        A configured ``RotatingFileHandler``.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    filepath = log_path / filename

    handler = logging.handlers.RotatingFileHandler(
        filename=str(filepath),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setLevel(getattr(logging, level))

    formatter = logging.Formatter(
        fmt=(
            '{"timestamp":"%(asctime)s",'
            '"level":"%(levelname)s",'
            '"logger":"%(name)s",'
            '"message":"%(message)s"}'
        ),
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)
    return handler


def setup_logging(settings: Optional[object] = None) -> None:
    """
    Configure the application-wide logging infrastructure.

    Should be called **once** during application startup.  Subsequent calls
    are safe but will add duplicate handlers — avoid them.

    Args:
        settings: Optional ``AppSettings`` instance.  Falls back to
            ``get_settings()`` if not provided.
    """
    if settings is None:
        settings = get_settings()

    log_cfg = settings.logging  # type: ignore[union-attr]
    root_logger = logging.getLogger()

    # Prevent duplicate handlers on repeated calls
    if root_logger.handlers:
        return

    root_logger.setLevel(getattr(logging, log_cfg.level))

    # Console handler (always human-readable)
    root_logger.addHandler(_create_console_handler(log_cfg.level))

    # Application log file (all levels)
    root_logger.addHandler(
        _create_file_handler(
            log_dir=log_cfg.dir,
            filename="app.log",
            level=log_cfg.level,
            max_bytes=log_cfg.max_file_size_mb * 1024 * 1024,
            backup_count=log_cfg.backup_count,
        )
    )

    # Error-only log file
    root_logger.addHandler(
        _create_file_handler(
            log_dir=log_cfg.dir,
            filename="error.log",
            level="ERROR",
            max_bytes=5 * 1024 * 1024,
            backup_count=10,
        )
    )

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "watchdog", "fsevents"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root_logger.info("Logging configured — level=%s, dir=%s", log_cfg.level, log_cfg.dir)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.

    This is a thin wrapper that ensures consistent naming and provides a
    single import for all modules.

    Args:
        name: Logger name, typically ``__name__``.

    Returns:
        A ``logging.Logger`` instance.
    """
    return logging.getLogger(name)
