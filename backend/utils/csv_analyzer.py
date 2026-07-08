"""
CSV Analyzer — Profiles a CSV file and extracts rich metadata.

This module is the "Data Inspector" of the architecture. It reads a CSV,
extracts column-level statistics, detects types, samples values, and
produces a ``FileMetadata`` schema that feeds into the LLM prompt context.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from backend.core.logging_config import get_logger
from backend.models.schemas import ColumnInfo, FileMetadata

logger = get_logger(__name__)

_SAMPLE_ROWS: int = 5
_MAX_PREVIEW_ROWS: int = 100


class CSVAnalyzer:
    """
    Analyzes a CSV file and produces structured metadata.

    This class handles:
        - Loading CSVs with encoding detection
        - Column-level profiling (dtype, nulls, uniques, stats)
        - Generating sample values for each column
        - Producing preview rows for the UI

    The analyzer is stateless and reusable — call ``analyze()`` for each file.
    """

    @staticmethod
    def load_dataframe(
        csv_path: Union[str, Path],
        nrows: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load a CSV file into a DataFrame with robust encoding handling.

        Attempts UTF-8 first, falls back to Latin-1 if decoding fails.

        Args:
            csv_path: Path to the CSV file.
            nrows: Optional row limit for partial loading.

        Returns:
            The loaded DataFrame.

        Raises:
            ValueError: If the file cannot be parsed as CSV.
        """
        path = Path(csv_path)
        for encoding in ("utf-8", "latin-1", "cp1252"):
            try:
                df = pd.read_csv(path, nrows=nrows, encoding=encoding)
                logger.debug("Loaded CSV with encoding '%s': %s", encoding, path.name)
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                raise ValueError(f"Failed to parse CSV '{path.name}': {e}") from e

        raise ValueError(f"Could not decode CSV '{path.name}' with any supported encoding.")

    @staticmethod
    def _profile_column(series: pd.Series) -> ColumnInfo:
        """
        Profile a single DataFrame column.

        Args:
            series: The pandas Series to profile.

        Returns:
            A ``ColumnInfo`` schema with all statistics populated.
        """
        dtype_str = str(series.dtype)
        non_null = int(series.count())
        null_count = int(series.isna().sum())
        unique_count = int(series.nunique())

        # Sample up to 5 non-null values, converted to strings
        non_null_values = series.dropna()
        sample_values = [
            str(v)[:50] for v in non_null_values.head(_SAMPLE_ROWS).tolist()
        ]

        # Numeric statistics
        mean = std = min_val = max_val = None
        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            mean = round(float(desc.get("mean", 0)), 4) if "mean" in desc else None
            std = round(float(desc.get("std", 0)), 4) if "std" in desc else None
            min_val = round(float(desc.get("min", 0)), 4) if "min" in desc else None
            max_val = round(float(desc.get("max", 0)), 4) if "max" in desc else None

        return ColumnInfo(
            name=series.name,
            dtype=dtype_str,
            non_null_count=non_null,
            null_count=null_count,
            unique_count=unique_count,
            sample_values=sample_values,
            mean=mean,
            std=std,
            min_val=min_val,
            max_val=max_val,
        )

    def analyze(
        self,
        csv_path: Union[str, Path],
        file_id: str = "",
        original_name: str = "",
    ) -> FileMetadata:
        """
        Perform full analysis of a CSV file.

        Loads the file, profiles every column, and returns a ``FileMetadata``
        schema ready for database persistence and prompt context building.

        Args:
            csv_path: Path to the CSV file on disk.
            file_id: UUID assigned to this file.
            original_name: Original filename from the upload.

        Returns:
            A complete ``FileMetadata`` schema.
        """
        path = Path(csv_path)
        logger.info("Analyzing CSV: '%s'", path.name)

        df = self.load_dataframe(path)

        columns: List[ColumnInfo] = [
            self._profile_column(df[col]) for col in df.columns
        ]

        memory_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)

        metadata = FileMetadata(
            file_id=file_id,
            original_name=original_name or path.name,
            stored_path=str(path),
            row_count=len(df),
            col_count=len(df.columns),
            file_size_bytes=path.stat().st_size,
            memory_usage_mb=memory_mb,
            columns=columns,
            sample_rows=df.head(_SAMPLE_ROWS).astype(str).to_dict(orient="records"),
        )

        logger.info(
            "Analysis complete: %d rows × %d cols, %.2f MB in memory",
            metadata.row_count, metadata.col_count, memory_mb,
        )
        return metadata

    def get_preview_rows(
        self,
        csv_path: Union[str, Path],
        max_rows: int = _MAX_PREVIEW_ROWS,
    ) -> List[dict]:
        """
        Load the first N rows as a list of dictionaries for UI display.

        Args:
            csv_path: Path to the CSV file.
            max_rows: Maximum rows to include.

        Returns:
            List of row dictionaries.
        """
        df = self.load_dataframe(csv_path, nrows=max_rows)
        # Convert to strings to ensure JSON serialization
        return df.head(max_rows).astype(str).to_dict(orient="records")

    def get_data_quality_report(self, csv_path: Union[str, Path]) -> dict:
        """
        Generate a quick data quality summary.

        Returns:
            Dictionary with completeness, type distribution, and warnings.
        """
        df = self.load_dataframe(csv_path)
        total_cells = df.shape[0] * df.shape[1]
        null_cells = int(df.isna().sum().sum())
        completeness = round((1 - null_cells / total_cells) * 100, 2) if total_cells > 0 else 0

        # Type distribution
        type_counts = df.dtypes.value_counts().to_dict()
        type_dist = {str(k): int(v) for k, v in type_counts.items()}

        # Warnings
        warnings: List[str] = []
        for col in df.columns:
            null_pct = df[col].isna().mean() * 100
            if null_pct > 50:
                warnings.append(f"Column '{col}' is {null_pct:.0f}% null")
            if df[col].nunique() == 1:
                warnings.append(f"Column '{col}' has only one unique value")
            if df[col].nunique() == len(df) and df[col].dtype == "object":
                warnings.append(f"Column '{col}' may be an ID column (all unique)")

        return {
            "completeness_pct": completeness,
            "total_cells": total_cells,
            "null_cells": null_cells,
            "type_distribution": type_dist,
            "warnings": warnings,
            "duplicate_rows": int(df.duplicated().sum()),
        }
