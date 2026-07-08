"""
Export page component — Download transcripts, data, and charts.

Integrates with ExportService for transcript exports and provides
direct DataFrame download.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from backend.core.logging_config import get_logger
from backend.models.schemas import ChatMessage, ExportFormat, MessageRole

from frontend.state import (
    get_chat_service,
    get_export_service,
    get_viz_service,
    has_dataset,
    CHAT_HISTORY,
    DATAFRAME,
    FILE_NAME,
    SESSION_ID,
)

logger = get_logger(__name__)


def render_export() -> None:
    """Render the export page."""
    if not has_dataset():
        _render_empty_state()
        return

    st.markdown(
        '<div class="section-header"><h3>📥 Export Results</h3></div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        _render_data_export()

    with col2:
        _render_transcript_export()

    st.divider()
    _render_chart_export()


def _render_empty_state() -> None:
    st.markdown(
        '<div class="empty-state animate-in">'
        '<div class="empty-icon">📥</div>'
        "<h4>Nothing to Export</h4>"
        "<p>Upload a dataset and start a conversation to export results.</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_data_export() -> None:
    """Render data download options."""
    st.markdown("#### 📊 Data Export")

    df: pd.DataFrame = st.session_state[DATAFRAME]
    file_name = st.session_state.get(FILE_NAME, "data")
    base_name = Path(file_name).stem if file_name else "data"

    # CSV download
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="⬇️ Download CSV",
        data=csv_buffer.getvalue(),
        file_name=f"{base_name}_export.csv",
        mime="text/csv",
        use_container_width=True,
        key="export_csv",
    )

    # Excel download
    try:
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False, engine="openpyxl")
        st.download_button(
            label="⬇️ Download Excel",
            data=excel_buffer.getvalue(),
            file_name=f"{base_name}_export.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="export_xlsx",
        )
    except ImportError:
        st.caption("Install `openpyxl` for Excel export.")

    st.caption(f"{len(df):,} rows × {len(df.columns)} columns")


def _render_transcript_export() -> None:
    """Render chat transcript export options."""
    st.markdown("#### 💬 Chat Transcript")

    history: List[Dict[str, Any]] = st.session_state.get(CHAT_HISTORY, [])

    if not history:
        st.caption("No chat history to export.")
        return

    export_service = get_export_service()
    file_name = st.session_state.get(FILE_NAME, "Session")
    session_title = Path(file_name).stem if file_name else "Session"

    # Build ChatMessage objects from history
    messages: List[ChatMessage] = []
    for entry in history:
        role_str = entry.get("role", "user")
        role = MessageRole.USER if role_str == "user" else MessageRole.ASSISTANT
        resp = entry.get("response")
        messages.append(ChatMessage(
            session_id=st.session_state.get(SESSION_ID, ""),
            role=role,
            content=entry.get("content", ""),
            generated_code=resp.generated_code if resp else None,
            execution_result=str(resp.result_data)[:2000] if resp and resp.result_data else None,
        ))

    # Markdown export
    include_code = st.checkbox("Include generated code", value=True, key="export_include_code")

    if st.button("📝 Export as Markdown", use_container_width=True, key="export_md"):
        try:
            result = export_service.export_transcript(
                session_title=session_title,
                messages=messages,
                format=ExportFormat.MARKDOWN,
                include_code=include_code,
            )
            with open(result.filepath, "r", encoding="utf-8") as f:
                md_content = f.read()
            st.download_button(
                label="⬇️ Download Markdown",
                data=md_content,
                file_name=result.filename,
                mime="text/markdown",
                use_container_width=True,
                key="download_md",
            )
            st.success(f"Transcript exported: {result.filename}")
        except Exception as exc:
            st.error(f"Export failed: {str(exc)[:200]}")

    if st.button("📋 Export as JSON", use_container_width=True, key="export_json"):
        try:
            result = export_service.export_transcript(
                session_title=session_title,
                messages=messages,
                format=ExportFormat.JSON,
                include_code=include_code,
            )
            with open(result.filepath, "r", encoding="utf-8") as f:
                json_content = f.read()
            st.download_button(
                label="⬇️ Download JSON",
                data=json_content,
                file_name=result.filename,
                mime="application/json",
                use_container_width=True,
                key="download_json",
            )
            st.success(f"Transcript exported: {result.filename}")
        except Exception as exc:
            st.error(f"Export failed: {str(exc)[:200]}")

    st.caption(f"{len(messages)} messages in history")


def _render_chart_export() -> None:
    """Render chart gallery and export options."""
    st.markdown(
        '<div class="section-header"><h3>🎨 Generated Charts</h3></div>',
        unsafe_allow_html=True,
    )

    viz_service = get_viz_service()
    charts = viz_service.list_charts()

    if not charts:
        st.caption("No charts have been generated yet.")
        return

    cols = st.columns(3)
    for idx, chart_info in enumerate(charts[:9]):
        with cols[idx % 3]:
            chart_path = Path(chart_info["filepath"])
            if chart_path.exists():
                st.image(str(chart_path), use_container_width=True)
                with open(chart_path, "rb") as f:
                    st.download_button(
                        label=f"⬇️ {chart_path.name}",
                        data=f.read(),
                        file_name=chart_path.name,
                        mime="image/png",
                        key=f"dl_chart_{idx}",
                        use_container_width=True,
                    )

    st.caption(f"{len(charts)} chart(s) available")
