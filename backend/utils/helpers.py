"""
General-purpose utility functions.

All functions here are pure (no side effects) and stateless.
"""

from __future__ import annotations

import html
import re
from datetime import datetime, timedelta
from typing import Optional


def relative_time(dt: datetime) -> str:
    """
    Convert a datetime to a human-readable relative string.

    Args:
        dt: The datetime to convert (assumed UTC).

    Returns:
        A string like "2 minutes ago", "Yesterday", "3 days ago".
    """
    now = datetime.utcnow()
    diff = now - dt

    if diff < timedelta(seconds=60):
        return "Just now"
    elif diff < timedelta(minutes=60):
        mins = int(diff.total_seconds() / 60)
        return f"{mins} minute{'s' if mins > 1 else ''} ago"
    elif diff < timedelta(hours=24):
        hours = int(diff.total_seconds() / 3600)
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    elif diff < timedelta(days=2):
        return "Yesterday"
    elif diff < timedelta(days=30):
        days = diff.days
        return f"{days} day{'s' if days > 1 else ''} ago"
    else:
        return dt.strftime("%b %d, %Y")


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length with an ellipsis suffix.

    Args:
        text: The string to truncate.
        max_length: Maximum character count.
        suffix: String appended when truncation occurs.

    Returns:
        Truncated string.
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def sanitize_user_input(text: str) -> str:
    """
    Sanitize user input by removing control characters and escaping HTML.

    Args:
        text: Raw user input.

    Returns:
        Sanitized string safe for storage and display.
    """
    # Remove control characters except newlines and tabs
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Escape HTML entities
    cleaned = html.escape(cleaned)
    return cleaned.strip()


def format_file_size(size_bytes: int) -> str:
    """
    Format a file size in bytes to a human-readable string.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Formatted string like "2.3 MB" or "456 KB".
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def format_number(n: float, decimals: int = 2) -> str:
    """
    Format a number with thousands separators and decimal places.

    Args:
        n: The number to format.
        decimals: Number of decimal places.

    Returns:
        Formatted string like "1,234.56".
    """
    if n == int(n):
        return f"{int(n):,}"
    return f"{n:,.{decimals}f}"
