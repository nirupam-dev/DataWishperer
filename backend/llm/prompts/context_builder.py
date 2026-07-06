"""
Context builder — Transforms CSV metadata into structured prompt context.

Converts a ``FileMetadata`` schema into a compact, LLM-optimized text
representation that gives the model enough information to write accurate
pandas code without seeing the entire dataset.
"""

from __future__ import annotations

from typing import List

from backend.models.schemas import ColumnInfo, FileMetadata


class ContextBuilder:
    """
    Builds structured dataset context strings for LLM prompts.

    The context includes:
        - File overview (name, shape, memory)
        - Column table (name, type, nulls, uniques, samples)
        - Numeric summary (mean, std, min, max)
        - Data quality notes (null warnings, type hints)
    """

    def build(self, metadata: FileMetadata) -> str:
        """
        Build the full context string from file metadata.

        Args:
            metadata: The ``FileMetadata`` schema to convert.

        Returns:
            A multi-line string optimized for LLM context injection.
        """
        sections = [
            self._build_overview(metadata),
            self._build_column_table(metadata.columns),
            self._build_numeric_summary(metadata.columns),
            self._build_notes(metadata.columns),
        ]
        return "\n\n".join(filter(None, sections))

    def build_compact(self, metadata: FileMetadata) -> str:
        """
        Build a compact context for retry prompts (saves tokens).

        Args:
            metadata: The ``FileMetadata`` schema.

        Returns:
            A shorter context with only column names and types.
        """
        lines = [
            f"Dataset: {metadata.original_name} ({metadata.row_count} rows × {metadata.col_count} cols)",
            "Columns: " + ", ".join(
                f"{c.name} ({c.dtype})" for c in metadata.columns
            ),
        ]
        return "\n".join(lines)

    # ── Private builders ─────────────────────────────────────────────────

    @staticmethod
    def _build_overview(metadata: FileMetadata) -> str:
        """Build the dataset overview section."""
        return (
            f"DATASET INFORMATION:\n"
            f"- Filename: {metadata.original_name}\n"
            f"- Shape: {metadata.row_count:,} rows × {metadata.col_count} columns\n"
            f"- Memory: {metadata.memory_usage_mb} MB\n"
            f"- File size: {metadata.file_size_bytes:,} bytes"
        )

    @staticmethod
    def _build_column_table(columns: List[ColumnInfo]) -> str:
        """Build a tabular representation of column metadata."""
        lines = ["COLUMNS:"]
        lines.append(f"{'Column':<25} {'Type':<12} {'Non-Null':<10} {'Unique':<8} Sample Values")
        lines.append("-" * 90)

        for col in columns:
            samples = ", ".join(col.sample_values[:3]) if col.sample_values else "N/A"
            samples = samples[:40] + "..." if len(samples) > 40 else samples
            lines.append(
                f"{col.name:<25} {col.dtype:<12} {col.non_null_count:<10} "
                f"{col.unique_count:<8} {samples}"
            )

        return "\n".join(lines)

    @staticmethod
    def _build_numeric_summary(columns: List[ColumnInfo]) -> str:
        """Build a statistical summary for numeric columns."""
        numeric_cols = [c for c in columns if c.mean is not None]
        if not numeric_cols:
            return ""

        lines = ["NUMERIC SUMMARY:"]
        lines.append(f"{'Column':<25} {'Mean':<15} {'Std':<15} {'Min':<15} {'Max':<15}")
        lines.append("-" * 85)

        for col in numeric_cols:
            lines.append(
                f"{col.name:<25} {col.mean:<15} {col.std:<15} "
                f"{col.min_val:<15} {col.max_val:<15}"
            )

        return "\n".join(lines)

    @staticmethod
    def _build_notes(columns: List[ColumnInfo]) -> str:
        """Build data quality notes and type hints."""
        notes: List[str] = []

        for col in columns:
            if col.null_count > 0:
                pct = round(col.null_count / (col.non_null_count + col.null_count) * 100, 1)
                if pct > 5:
                    notes.append(f"- '{col.name}' has {col.null_count} nulls ({pct}%)")

            # Detect potential date columns
            if col.dtype == "object" and col.sample_values:
                sample = col.sample_values[0]
                if any(sep in sample for sep in ["-", "/", ":"]) and len(sample) >= 8:
                    notes.append(
                        f"- '{col.name}' appears to be a date/time column "
                        f"(consider pd.to_datetime)"
                    )

        if not notes:
            return ""

        return "NOTES:\n" + "\n".join(notes)
