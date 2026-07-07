"""
Unit tests for CSV edge cases.

Tests:
    - Encoding detection (UTF-8, Latin-1, CP1252)
    - Unicode headers and data
    - File upload security validation
    - Column profiling (numeric, categorical, mixed)
    - Data quality report generation
    - Filename sanitization
    - Large/wide/empty CSVs
    - Malformed CSV structures
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from backend.utils.csv_analyzer import CSVAnalyzer
from backend.core.security import (
    sanitize_filename,
    validate_csv_content,
    validate_column_count,
    validate_extension,
    validate_file_size,
    validate_upload,
)
from backend.core.exceptions import (
    FileTooLargeError,
    InvalidFileError,
    TooManyColumnsError,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def analyzer():
    return CSVAnalyzer()


# ── CSV Encoding Tests ──────────────────────────────────────────────────────


class TestCSVEncoding:
    """Test CSV loading with various encodings."""

    def test_utf8_csv_loads(self, analyzer, sample_csv):
        df = analyzer.load_dataframe(sample_csv)
        assert len(df) > 0
        assert df.shape[1] > 0

    def test_latin1_csv_loads(self, analyzer, latin1_csv):
        df = analyzer.load_dataframe(latin1_csv)
        assert len(df) == 2
        assert "José" in df["name"].values

    def test_unicode_csv_loads(self, analyzer, unicode_csv):
        df = analyzer.load_dataframe(unicode_csv)
        assert len(df) == 3
        # Header names should be preserved
        assert "名前" in df.columns

    def test_nrows_limit_respected(self, analyzer, sample_csv):
        df = analyzer.load_dataframe(sample_csv, nrows=10)
        assert len(df) == 10

    def test_nonexistent_file_raises(self, analyzer, tmp_path):
        with pytest.raises(Exception):
            analyzer.load_dataframe(tmp_path / "nonexistent.csv")


# ── Column Profiling Tests ──────────────────────────────────────────────────


class TestColumnProfiling:
    """Test per-column metadata extraction."""

    def test_numeric_column_statistics(self, analyzer, sample_csv):
        metadata = analyzer.analyze(sample_csv, file_id="test-001")
        revenue_col = next(c for c in metadata.columns if c.name == "revenue")
        assert revenue_col.dtype == "float64"
        assert revenue_col.mean is not None
        assert revenue_col.std is not None
        assert revenue_col.min_val is not None
        assert revenue_col.max_val is not None
        assert revenue_col.non_null_count == 200

    def test_categorical_column_profiling(self, analyzer, sample_csv):
        metadata = analyzer.analyze(sample_csv, file_id="test-001")
        cat_col = next(c for c in metadata.columns if c.name == "category")
        assert cat_col.dtype == "object"
        assert cat_col.unique_count == 5
        assert len(cat_col.sample_values) <= 5

    def test_null_count_accuracy(self, analyzer, messy_csv):
        metadata = analyzer.analyze(messy_csv, file_id="test-002")
        value_col = next(c for c in metadata.columns if c.name == "value")
        assert value_col.null_count > 0

    def test_sample_values_truncated(self, analyzer, tmp_path):
        """Sample values should be truncated to 50 chars."""
        csv_path = tmp_path / "long_values.csv"
        df = pd.DataFrame({"text": ["x" * 100 for _ in range(10)]})
        df.to_csv(csv_path, index=False)
        metadata = analyzer.analyze(csv_path, file_id="test-003")
        text_col = next(c for c in metadata.columns if c.name == "text")
        for sv in text_col.sample_values:
            assert len(sv) <= 50


# ── CSV Analysis Tests ──────────────────────────────────────────────────────


class TestCSVAnalysis:
    """Test full CSV analysis pipeline."""

    def test_full_analysis_returns_metadata(self, analyzer, sample_csv):
        metadata = analyzer.analyze(sample_csv, file_id="test-001", original_name="test.csv")
        assert metadata.file_id == "test-001"
        assert metadata.original_name == "test.csv"
        assert metadata.row_count == 200
        assert metadata.col_count == 8
        assert metadata.memory_usage_mb > 0
        assert metadata.file_size_bytes > 0
        assert len(metadata.columns) == 8

    def test_analysis_with_defaults(self, analyzer, sample_csv):
        """file_id defaults to empty, original_name defaults to filename."""
        metadata = analyzer.analyze(sample_csv)
        assert metadata.file_id == ""
        assert metadata.original_name == "sample.csv"

    def test_preview_rows(self, analyzer, sample_csv):
        rows = analyzer.get_preview_rows(sample_csv, max_rows=5)
        assert len(rows) == 5
        assert isinstance(rows[0], dict)
        # All values should be strings
        for row in rows:
            for v in row.values():
                assert isinstance(v, str)

    def test_data_quality_report(self, analyzer, messy_csv):
        report = analyzer.get_data_quality_report(messy_csv)
        assert "completeness_pct" in report
        assert "total_cells" in report
        assert "warnings" in report
        assert report["completeness_pct"] < 100  # messy_df has nulls
        assert report["duplicate_rows"] > 0


# ── Empty and Edge Case CSVs ────────────────────────────────────────────────


class TestCSVEdgeCases:
    """Edge cases: empty files, single column, single row, all-null."""

    def test_single_column_csv(self, analyzer, tmp_path):
        csv_path = tmp_path / "single_col.csv"
        pd.DataFrame({"only_col": range(50)}).to_csv(csv_path, index=False)
        metadata = analyzer.analyze(csv_path, file_id="sc-001")
        assert metadata.col_count == 1
        assert metadata.row_count == 50

    def test_single_row_csv(self, analyzer, tmp_path):
        csv_path = tmp_path / "single_row.csv"
        pd.DataFrame({"a": [1], "b": [2], "c": [3]}).to_csv(csv_path, index=False)
        metadata = analyzer.analyze(csv_path, file_id="sr-001")
        assert metadata.row_count == 1
        assert metadata.col_count == 3

    def test_all_null_csv(self, analyzer, tmp_path):
        csv_path = tmp_path / "all_nulls.csv"
        df = pd.DataFrame({"a": [np.nan] * 20, "b": [None] * 20})
        df.to_csv(csv_path, index=False)
        metadata = analyzer.analyze(csv_path, file_id="an-001")
        for col in metadata.columns:
            assert col.null_count > 0

    def test_whitespace_column_names(self, analyzer, tmp_path):
        csv_path = tmp_path / "whitespace.csv"
        csv_path.write_text(" a , b , c \n1,2,3\n4,5,6\n")
        metadata = analyzer.analyze(csv_path, file_id="ws-001")
        assert metadata.col_count == 3

    def test_mixed_types_column(self, analyzer, tmp_path):
        csv_path = tmp_path / "mixed.csv"
        csv_path.write_text("val\n1\nhello\n3.14\nTrue\n")
        metadata = analyzer.analyze(csv_path, file_id="mix-001")
        assert metadata.row_count == 4

    def test_very_wide_csv(self, analyzer, tmp_path):
        csv_path = tmp_path / "wide.csv"
        df = pd.DataFrame(
            np.random.randn(5, 100),
            columns=[f"col_{i}" for i in range(100)],
        )
        df.to_csv(csv_path, index=False)
        metadata = analyzer.analyze(csv_path, file_id="wide-001")
        assert metadata.col_count == 100

    def test_empty_string_values(self, analyzer, tmp_path):
        csv_path = tmp_path / "empties.csv"
        csv_path.write_text('a,b,c\n"",1,""\n"x",2,"y"\n')
        metadata = analyzer.analyze(csv_path, file_id="e-001")
        assert metadata.row_count == 2


# ── Upload Security Validation ──────────────────────────────────────────────


class TestUploadSecurity:
    """Test file upload validation pipeline."""

    def test_valid_csv_extension(self):
        validate_extension("data.csv")

    def test_invalid_extension_rejected(self):
        with pytest.raises(InvalidFileError):
            validate_extension("data.xlsx")

    def test_exe_extension_rejected(self):
        with pytest.raises(InvalidFileError):
            validate_extension("evil.exe")

    def test_no_extension_rejected(self):
        with pytest.raises(InvalidFileError):
            validate_extension("noext")

    @patch("backend.core.security.get_settings")
    def test_file_size_within_limit(self, mock_settings):
        mock_settings.return_value.storage = MagicMock(max_file_size_mb=100)
        validate_file_size(50 * 1024 * 1024, mock_settings.return_value.storage)

    @patch("backend.core.security.get_settings")
    def test_file_size_exceeds_limit(self, mock_settings):
        mock_settings.return_value.storage = MagicMock(max_file_size_mb=10)
        with pytest.raises(FileTooLargeError):
            validate_file_size(
                50 * 1024 * 1024, mock_settings.return_value.storage
            )

    def test_valid_csv_content_parsed(self):
        content = b"name,age,city\nAlice,30,NYC\nBob,25,LA\n"
        col_count = validate_csv_content(content, "test.csv")
        assert col_count == 3

    def test_single_row_csv_rejected(self):
        """Must have at least a header + 1 data row."""
        content = b"name,age,city\n"
        with pytest.raises(InvalidFileError):
            validate_csv_content(content, "test.csv")

    def test_empty_content_rejected(self):
        with pytest.raises(InvalidFileError):
            validate_csv_content(b"", "test.csv")

    def test_column_count_mismatch_rejected(self):
        content = b"a,b,c\n1,2,3,4,5,6,7,8\n"
        with pytest.raises(InvalidFileError):
            validate_csv_content(content, "test.csv")

    @patch("backend.core.security.get_settings")
    def test_too_many_columns_rejected(self, mock_settings):
        mock_settings.return_value.storage = MagicMock(max_columns=10)
        with pytest.raises(TooManyColumnsError):
            validate_column_count(50, mock_settings.return_value.storage)


# ── Filename Sanitization Tests ─────────────────────────────────────────────


class TestFilenameSanitization:
    """Test filename cleaning for path traversal prevention."""

    def test_normal_filename_preserved(self):
        assert sanitize_filename("data.csv") == "data.csv"

    def test_path_traversal_stripped(self):
        result = sanitize_filename("../../etc/passwd.csv")
        assert ".." not in result
        assert "etc" not in result

    def test_windows_path_stripped(self):
        result = sanitize_filename("C:\\Users\\evil\\data.csv")
        assert "C:" not in result
        assert "\\" not in result

    def test_special_characters_removed(self):
        result = sanitize_filename("data!@#$%^&*.csv")
        assert "!" not in result
        assert "@" not in result

    def test_whitespace_collapsed(self):
        result = sanitize_filename("my  data   file.csv")
        assert "  " not in result

    def test_long_filename_truncated(self):
        long_name = "a" * 300 + ".csv"
        result = sanitize_filename(long_name)
        assert len(result) <= 200

    def test_empty_after_sanitization_raises(self):
        with pytest.raises(InvalidFileError):
            sanitize_filename("!@#$%^&*()")

    def test_unix_path_traversal(self):
        result = sanitize_filename("/tmp/../etc/shadow.csv")
        assert "/" not in result or result == "shadow.csv"
