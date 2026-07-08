"""
Dataset Explorer page — Preview, schema, statistics, data quality.

Renders the "Explore" tab content by reading from session_state.
Analytics are computed once and cached in session_state.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from backend.core.logging_config import get_logger
from backend.models.schemas import FileMetadata, ColumnInfo

from frontend.state import (
    get_analytics,
    get_file_service,
    has_dataset,
    DATAFRAME,
    FILE_METADATA,
    FILE_PATH,
    ANALYTICS_REPORT,
    UPLOAD_RESPONSE,
)

logger = get_logger(__name__)


def render_explorer() -> None:
    """Render the full dataset explorer page."""
    if not has_dataset():
        _render_empty_state()
        return

    metadata: FileMetadata = st.session_state[FILE_METADATA]
    df: pd.DataFrame = st.session_state[DATAFRAME]

    # ── Summary Metrics ──────────────────────────────────────────
    _render_summary_metrics(metadata)

    # ── Sub-tabs ─────────────────────────────────────────────────
    preview_tab, schema_tab, stats_tab, quality_tab = st.tabs(
        ["📋 Preview", "🏗️ Schema", "📈 Statistics", "🧹 Data Quality"]
    )

    with preview_tab:
        _render_preview(df)

    with schema_tab:
        _render_schema(metadata)

    with stats_tab:
        _render_statistics(df, metadata)

    with quality_tab:
        _render_quality(df)


# ── Private Renderers ───────────────────────────────────────────────────────


def _render_empty_state() -> None:
    st.markdown(
        '<div class="empty-state animate-in">'
        '<div class="empty-icon">📁</div>'
        "<h4>No Dataset Loaded</h4>"
        "<p>Upload a CSV file from the sidebar to start exploring.</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_summary_metrics(metadata: FileMetadata) -> None:
    """Render the row of metric cards at the top."""
    cols = st.columns(4)

    with cols[0]:
        st.markdown(
            '<div class="metric-card animate-in">'
            f'<p class="metric-value">{metadata.row_count:,}</p>'
            '<p class="metric-label">Rows</p>'
            "</div>",
            unsafe_allow_html=True,
        )

    with cols[1]:
        st.markdown(
            '<div class="metric-card animate-in">'
            f'<p class="metric-value">{metadata.col_count}</p>'
            '<p class="metric-label">Columns</p>'
            "</div>",
            unsafe_allow_html=True,
        )

    with cols[2]:
        total_nulls = sum(c.null_count for c in metadata.columns)
        st.markdown(
            '<div class="metric-card animate-in">'
            f'<p class="metric-value">{total_nulls:,}</p>'
            '<p class="metric-label">Missing Values</p>'
            "</div>",
            unsafe_allow_html=True,
        )

    with cols[3]:
        st.markdown(
            '<div class="metric-card animate-in">'
            f'<p class="metric-value">{metadata.memory_usage_mb:.1f} MB</p>'
            '<p class="metric-label">Memory Usage</p>'
            "</div>",
            unsafe_allow_html=True,
        )


def _render_preview(df: pd.DataFrame) -> None:
    """Show the first N rows of the dataset."""
    st.markdown(
        '<div class="section-header"><h3>Data Preview</h3></div>',
        unsafe_allow_html=True,
    )

    max_rows = st.slider(
        "Rows to display", 5, min(100, len(df)), 20, key="preview_rows_slider"
    )
    st.dataframe(
        df.head(max_rows),
        use_container_width=True,
        height=min(400, 35 * max_rows + 38),
    )
    st.caption(f"Showing {min(max_rows, len(df)):,} of {len(df):,} rows")


def _render_schema(metadata: FileMetadata) -> None:
    """Show column-level schema information."""
    st.markdown(
        '<div class="section-header"><h3>Schema &amp; Data Types</h3></div>',
        unsafe_allow_html=True,
    )

    schema_data = []
    for col in metadata.columns:
        null_pct = (
            f"{(col.null_count / (col.non_null_count + col.null_count) * 100):.1f}%"
            if (col.non_null_count + col.null_count) > 0
            else "N/A"
        )
        schema_data.append({
            "Column": col.name,
            "Type": col.dtype,
            "Non-Null": f"{col.non_null_count:,}",
            "Null": f"{col.null_count:,}",
            "Null %": null_pct,
            "Unique": f"{col.unique_count:,}",
            "Sample": ", ".join(col.sample_values[:3]) if col.sample_values else "—",
        })

    st.dataframe(
        pd.DataFrame(schema_data),
        use_container_width=True,
        hide_index=True,
    )


def _render_statistics(df: pd.DataFrame, metadata: FileMetadata) -> None:
    """Show descriptive statistics for numeric and categorical columns."""
    st.markdown(
        '<div class="section-header"><h3>Dataset Statistics</h3></div>',
        unsafe_allow_html=True,
    )

    numeric_cols = [c for c in metadata.columns if c.mean is not None]
    categorical_cols = [c for c in metadata.columns if c.mean is None]

    if numeric_cols:
        st.markdown("#### 🔢 Numeric Columns")
        num_data = []
        for col in numeric_cols:
            num_data.append({
                "Column": col.name,
                "Mean": f"{col.mean:,.4f}" if col.mean is not None else "—",
                "Std": f"{col.std:,.4f}" if col.std is not None else "—",
                "Min": f"{col.min_val:,.4f}" if col.min_val is not None else "—",
                "Max": f"{col.max_val:,.4f}" if col.max_val is not None else "—",
                "Unique": f"{col.unique_count:,}",
            })
        st.dataframe(pd.DataFrame(num_data), use_container_width=True, hide_index=True)

    if categorical_cols:
        st.markdown("#### 🏷️ Categorical Columns")
        cat_data = []
        for col in categorical_cols:
            cat_data.append({
                "Column": col.name,
                "Type": col.dtype,
                "Unique": f"{col.unique_count:,}",
                "Non-Null": f"{col.non_null_count:,}",
                "Top Values": ", ".join(col.sample_values[:5]) if col.sample_values else "—",
            })
        st.dataframe(pd.DataFrame(cat_data), use_container_width=True, hide_index=True)

    # Full describe()
    with st.expander("📊 Full `df.describe()` output", expanded=False):
        st.dataframe(df.describe(include="all").T, use_container_width=True)


def _render_quality(df: pd.DataFrame) -> None:
    """Run and display the analytics quality report."""
    st.markdown(
        '<div class="section-header"><h3>Data Quality Report</h3></div>',
        unsafe_allow_html=True,
    )

    # Cache the analytics report in session_state
    report = st.session_state.get(ANALYTICS_REPORT)
    if report is None:
        with st.spinner("Running data quality analysis…"):
            try:
                orchestrator = get_analytics()
                report = orchestrator.run_quick_scan(df)
                st.session_state[ANALYTICS_REPORT] = report
            except Exception as exc:
                st.error(f"Analysis failed: {str(exc)[:200]}")
                logger.exception("Analytics quick scan failed")
                return

    # Render the report
    st.markdown(report.to_markdown())

    if st.button("🔄 Re-run Full Analysis", key="rerun_analysis"):
        with st.spinner("Running full analysis…"):
            try:
                orchestrator = get_analytics()
                full_report = orchestrator.run_full_analysis(df)
                st.session_state[ANALYTICS_REPORT] = full_report
                st.rerun()
            except Exception as exc:
                st.error(f"Full analysis failed: {str(exc)[:200]}")
