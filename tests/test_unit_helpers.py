"""
Unit tests for helper utilities.

Tests:
    - relative_time() formatting
    - truncate_text() behavior
    - sanitize_user_input() security
    - format_file_size() formatting
    - format_number() formatting
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from backend.utils.helpers import (
    format_file_size,
    format_number,
    relative_time,
    sanitize_user_input,
    truncate_text,
)


# ── relative_time ───────────────────────────────────────────────────────────


class TestRelativeTime:
    """Test human-readable relative time formatting."""

    def test_just_now(self):
        now = datetime.utcnow()
        result = relative_time(now)
        assert result == "Just now"

    def test_minutes_ago(self):
        dt = datetime.utcnow() - timedelta(minutes=5)
        result = relative_time(dt)
        assert "5 minutes ago" in result

    def test_minute_singular(self):
        dt = datetime.utcnow() - timedelta(minutes=1)
        result = relative_time(dt)
        assert "1 minute ago" in result

    def test_hours_ago(self):
        dt = datetime.utcnow() - timedelta(hours=3)
        result = relative_time(dt)
        assert "3 hours ago" in result

    def test_hour_singular(self):
        dt = datetime.utcnow() - timedelta(hours=1)
        result = relative_time(dt)
        assert "1 hour ago" in result

    def test_yesterday(self):
        dt = datetime.utcnow() - timedelta(days=1)
        result = relative_time(dt)
        assert result == "Yesterday"

    def test_days_ago(self):
        dt = datetime.utcnow() - timedelta(days=10)
        result = relative_time(dt)
        assert "10 days ago" in result

    def test_old_date_formatted(self):
        dt = datetime.utcnow() - timedelta(days=60)
        result = relative_time(dt)
        assert "," in result or "20" in result  # Date format like "May 06, 2024"


# ── truncate_text ───────────────────────────────────────────────────────────


class TestTruncateText:
    """Test text truncation."""

    def test_short_text_unchanged(self):
        assert truncate_text("hello", max_length=100) == "hello"

    def test_long_text_truncated(self):
        result = truncate_text("x" * 200, max_length=50)
        assert len(result) == 50
        assert result.endswith("...")

    def test_custom_suffix(self):
        result = truncate_text("x" * 200, max_length=50, suffix="…")
        assert result.endswith("…")
        assert len(result) == 50

    def test_exact_length_unchanged(self):
        text = "x" * 100
        assert truncate_text(text, max_length=100) == text

    def test_empty_string(self):
        assert truncate_text("", max_length=10) == ""


# ── sanitize_user_input ─────────────────────────────────────────────────────


class TestSanitizeUserInput:
    """Test user input sanitization for security."""

    def test_normal_text_preserved(self):
        assert sanitize_user_input("Hello world") == "Hello world"

    def test_html_escaped(self):
        result = sanitize_user_input("<script>alert('xss')</script>")
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_control_characters_removed(self):
        result = sanitize_user_input("hello\x00world\x07")
        assert "\x00" not in result
        assert "\x07" not in result

    def test_newlines_preserved(self):
        result = sanitize_user_input("line1\nline2")
        assert "\n" in result

    def test_tabs_preserved(self):
        result = sanitize_user_input("col1\tcol2")
        assert "\t" in result

    def test_leading_trailing_whitespace_stripped(self):
        result = sanitize_user_input("  hello  ")
        assert result == "hello"

    def test_empty_string(self):
        assert sanitize_user_input("") == ""

    def test_ampersand_escaped(self):
        result = sanitize_user_input("a & b")
        assert "&amp;" in result

    def test_quotes_escaped(self):
        result = sanitize_user_input('say "hello"')
        assert "&quot;" in result


# ── format_file_size ────────────────────────────────────────────────────────


class TestFormatFileSize:
    """Test human-readable file size formatting."""

    def test_bytes(self):
        assert format_file_size(500) == "500 B"

    def test_kilobytes(self):
        assert format_file_size(2048) == "2.0 KB"

    def test_megabytes(self):
        result = format_file_size(5 * 1024 * 1024)
        assert "5.0 MB" in result

    def test_gigabytes(self):
        result = format_file_size(2 * 1024 * 1024 * 1024)
        assert "2.0 GB" in result

    def test_zero_bytes(self):
        assert format_file_size(0) == "0 B"


# ── format_number ───────────────────────────────────────────────────────────


class TestFormatNumber:
    """Test number formatting with commas."""

    def test_integer(self):
        assert format_number(1000) == "1,000"

    def test_large_integer(self):
        assert format_number(1234567) == "1,234,567"

    def test_float(self):
        assert format_number(1234.56) == "1,234.56"

    def test_custom_decimals(self):
        assert format_number(3.14159, decimals=3) == "3.142"

    def test_zero(self):
        assert format_number(0) == "0"
