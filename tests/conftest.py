"""
Shared test fixtures and configuration for the DataWhisperer test suite.

Provides:
    - Deterministic sample DataFrames of varying complexity
    - Temporary CSV files (auto-cleanup)
    - Mock FileMetadata objects
    - Settings cache reset
    - Shared constants for test data
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from backend.core.config import get_settings
from backend.models.schemas import ColumnInfo, FileMetadata


# ── Deterministic seed ───────────────────────────────────────────────────────
# All fixtures use np.random.seed(42) for reproducibility.


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    """Reset the LRU-cached settings singleton between tests."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ── DataFrame Fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Clean, well-structured dataset with multiple column types."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "category": np.random.choice(["A", "B", "C", "D", "E"], n),
        "region": np.random.choice(["North", "South", "East", "West"], n),
        "revenue": np.random.lognormal(8, 1, n).round(2),
        "quantity": np.random.randint(1, 100, n),
        "price": np.random.normal(50, 15, n).round(2),
        "discount": np.random.uniform(0, 0.5, n).round(3),
        "satisfaction": np.random.choice([1, 2, 3, 4, 5], n),
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
    })


@pytest.fixture
def messy_df() -> pd.DataFrame:
    """Dataset with intentional quality issues: nulls, dupes, outliers."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "id": range(n),
        "name": [f"item_{i}" for i in range(n)],
        "value": np.random.normal(100, 30, n).round(2),
        "category": np.random.choice(["A", "B", "C"], n),
    })
    # Inject missing values
    df.loc[10:20, "value"] = np.nan
    df.loc[50:55, "category"] = np.nan
    # Inject duplicates
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    # Inject outliers
    df.loc[98, "value"] = 9999.99
    df.loc[99, "value"] = -999.99
    return df


@pytest.fixture
def time_series_df() -> pd.DataFrame:
    """Dataset with clear temporal trends for predictive tests."""
    np.random.seed(42)
    n = 100
    x = np.arange(n, dtype=float)
    return pd.DataFrame({
        "day": pd.date_range("2024-01-01", periods=n, freq="D"),
        "sales": (x * 2.5 + 100 + np.random.normal(0, 10, n)).round(2),
        "cost": (x * 1.2 + 50 + np.random.normal(0, 5, n)).round(2),
        "random_noise": np.random.normal(0, 1, n).round(4),
    })


@pytest.fixture
def empty_df() -> pd.DataFrame:
    """Completely empty DataFrame."""
    return pd.DataFrame()


@pytest.fixture
def single_column_df() -> pd.DataFrame:
    """Edge case: DataFrame with a single column."""
    return pd.DataFrame({"x": range(100)})


@pytest.fixture
def single_row_df() -> pd.DataFrame:
    """Edge case: DataFrame with a single row."""
    return pd.DataFrame({"a": [1], "b": ["hello"], "c": [3.14]})


@pytest.fixture
def all_nulls_df() -> pd.DataFrame:
    """Edge case: DataFrame where every cell is null."""
    return pd.DataFrame({
        "a": [np.nan] * 50,
        "b": [None] * 50,
        "c": [np.nan] * 50,
    })


@pytest.fixture
def wide_df() -> pd.DataFrame:
    """Wide DataFrame: many columns, few rows."""
    np.random.seed(42)
    return pd.DataFrame(
        np.random.randn(20, 50),
        columns=[f"col_{i}" for i in range(50)],
    )


@pytest.fixture
def large_df() -> pd.DataFrame:
    """Performance test: moderately large DataFrame."""
    np.random.seed(42)
    n = 50_000
    return pd.DataFrame({
        "id": range(n),
        "value": np.random.normal(0, 1, n),
        "category": np.random.choice(["A", "B", "C", "D"], n),
        "amount": np.random.lognormal(5, 2, n).round(2),
    })


# ── CSV File Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def sample_csv(sample_df, tmp_path) -> Path:
    """Write sample_df to a temporary CSV and return the path."""
    csv_path = tmp_path / "sample.csv"
    sample_df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def messy_csv(messy_df, tmp_path) -> Path:
    """Write messy_df to a temporary CSV and return the path."""
    csv_path = tmp_path / "messy.csv"
    messy_df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def empty_csv(tmp_path) -> Path:
    """Write an empty CSV with headers only."""
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("a,b,c\n")
    return csv_path


@pytest.fixture
def unicode_csv(tmp_path) -> Path:
    """CSV with unicode characters in data and headers."""
    csv_path = tmp_path / "unicode.csv"
    csv_path.write_text(
        "名前,価格,カテゴリ\n"
        "りんご,100,果物\n"
        "バナナ,200,果物\n"
        "みかん,150,果物\n",
        encoding="utf-8",
    )
    return csv_path


@pytest.fixture
def latin1_csv(tmp_path) -> Path:
    """CSV encoded in Latin-1 (ISO 8859-1)."""
    csv_path = tmp_path / "latin1.csv"
    csv_path.write_bytes(
        "name,city,price\nJosé,São Paulo,100\nRené,Zürich,200\n".encode("latin-1")
    )
    return csv_path


@pytest.fixture
def large_csv(large_df, tmp_path) -> Path:
    """Write large_df to a temporary CSV."""
    csv_path = tmp_path / "large.csv"
    large_df.to_csv(csv_path, index=False)
    return csv_path


# ── Metadata Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def sample_file_metadata() -> FileMetadata:
    """FileMetadata matching sample_df structure."""
    return FileMetadata(
        file_id="test-file-001",
        original_name="sample.csv",
        stored_path="/tmp/test/sample.csv",
        row_count=200,
        col_count=8,
        file_size_bytes=15000,
        memory_usage_mb=0.12,
        columns=[
            ColumnInfo(
                name="category", dtype="object",
                non_null_count=200, null_count=0,
                unique_count=5, sample_values=["A", "B", "C"],
            ),
            ColumnInfo(
                name="region", dtype="object",
                non_null_count=200, null_count=0,
                unique_count=4, sample_values=["North", "South"],
            ),
            ColumnInfo(
                name="revenue", dtype="float64",
                non_null_count=200, null_count=0,
                unique_count=200, sample_values=["3246.78", "1543.21"],
                mean=5000.0, std=3000.0, min_val=100.0, max_val=25000.0,
            ),
            ColumnInfo(
                name="quantity", dtype="int64",
                non_null_count=200, null_count=0,
                unique_count=99, sample_values=["42", "17"],
                mean=50.0, std=29.0, min_val=1.0, max_val=99.0,
            ),
            ColumnInfo(
                name="price", dtype="float64",
                non_null_count=200, null_count=0,
                unique_count=200, sample_values=["50.25", "35.10"],
                mean=50.0, std=15.0, min_val=10.0, max_val=90.0,
            ),
            ColumnInfo(
                name="discount", dtype="float64",
                non_null_count=200, null_count=0,
                unique_count=200, sample_values=["0.15", "0.32"],
                mean=0.25, std=0.14, min_val=0.0, max_val=0.5,
            ),
            ColumnInfo(
                name="satisfaction", dtype="int64",
                non_null_count=200, null_count=0,
                unique_count=5, sample_values=["1", "3", "5"],
                mean=3.0, std=1.4, min_val=1.0, max_val=5.0,
            ),
            ColumnInfo(
                name="date", dtype="datetime64[ns]",
                non_null_count=200, null_count=0,
                unique_count=200, sample_values=["2024-01-01"],
            ),
        ],
    )
