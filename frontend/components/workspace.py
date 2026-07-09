"""
Analytics Workspace — Full-featured data analysis interface.

Renders the sidebar (upload, dataset info, AI engine status) and
main content (hero, dataset cards, question panel, 2×2 result dashboard).
All data comes from actual session state — nothing is hardcoded.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from backend.core.config import get_settings
from backend.core.exceptions import DataWhispererError
from backend.core.logging_config import get_logger
from backend.models.schemas import (
    ChatResponse, FileMetadata, MessageRole, ResultType,
)
from backend.utils.helpers import format_file_size

from frontend.state import (
    get_agent, get_chat_service, get_file_service, get_session_service,
    get_viz_service, has_agent, has_dataset, clear_dataset,
    reinitialise_agent,
    FILE_ID, FILE_METADATA, FILE_PATH, FILE_NAME,
    DATAFRAME, SESSION_ID, UPLOAD_RESPONSE, ANALYTICS_REPORT, CHAT_HISTORY,
)

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def render_workspace() -> None:
    """Render the complete workspace: sidebar + main content."""
    _render_sidebar()
    _render_main_content()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def _render_sidebar() -> None:
    """Sidebar: brand, upload, dataset info, AI engine status, footer."""
    with st.sidebar:
        # Brand
        st.markdown("""
        <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.2rem;">
            <div style="width:32px; height:32px; border-radius:50%;
                         background:linear-gradient(135deg,#6366f1,#8b5cf6);
                         display:flex; align-items:center; justify-content:center;
                         font-size:0.85rem; color:white;">✦</div>
            <div>
                <div style="font-weight:700; font-size:0.8rem; color:#f1f5f9;
                             letter-spacing:1px;">DATAWHISPER</div>
                <div style="font-size:0.55rem; color:#64748b; text-transform:uppercase;
                             letter-spacing:0.8px;">TALK TO YOUR CSV WITH AI</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr style='margin:0.6rem 0; border-color:rgba(99,102,241,0.1);'>",
                    unsafe_allow_html=True)

        # Upload
        st.markdown("""
        <div style="font-size:0.7rem; font-weight:600; color:#94a3b8;
                     text-transform:uppercase; letter-spacing:1px; margin-bottom:0.4rem;">
            📁 UPLOAD DATASET</div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Choose a CSV file", type=["csv"],
            help="Upload a CSV file to start analysing.", key="csv_uploader",
        )
        if uploaded_file is not None:
            current_name = st.session_state.get(FILE_NAME)
            if current_name != uploaded_file.name or not has_dataset():
                _process_upload(uploaded_file)

        # Active Dataset
        if has_dataset():
            st.markdown("<hr style='margin:0.6rem 0; border-color:rgba(99,102,241,0.1);'>",
                        unsafe_allow_html=True)
            _render_sidebar_dataset_info()

        # AI Engine Status
        st.markdown("<hr style='margin:0.6rem 0; border-color:rgba(99,102,241,0.1);'>",
                    unsafe_allow_html=True)
        _render_ai_engine_status()

        # Footer
        st.markdown("""
        <div style="position:fixed; bottom:0.75rem; padding:0.5rem 0;">
            <div style="display:flex; align-items:center; gap:0.4rem;">
                <span style="font-size:0.7rem; color:#818cf8;">✦</span>
                <span style="font-size:0.65rem; font-weight:600; color:#64748b;
                              letter-spacing:0.5px;">DATAWHISPER AI</span>
            </div>
            <div style="font-size:0.5rem; color:#475569; margin-top:0.15rem;">
                BUILT FOR ANALYTICS. POWERED BY AI.</div>
        </div>
        """, unsafe_allow_html=True)


def _process_upload(uploaded_file) -> None:
    """Validate, save, and analyse an uploaded file (reuses existing backend)."""
    file_service = get_file_service()
    with st.spinner("Validating and analysing your dataset…"):
        try:
            content = uploaded_file.read()
            response = file_service.upload_file(
                filename=uploaded_file.name, content=content,
            )
            metadata = file_service.get_file_metadata(response.file_id)
            csv_path = file_service.get_file_path(response.file_id)
            df = pd.read_csv(csv_path)

            agent = get_agent()
            if agent is not None:
                agent.register_dataset(metadata)

            session_svc = get_session_service()
            session_id = session_svc.create_session(
                file_id=response.file_id,
                title=f"Analysis: {uploaded_file.name}",
            )

            st.session_state[FILE_ID] = response.file_id
            st.session_state[FILE_METADATA] = metadata
            st.session_state[FILE_PATH] = csv_path
            st.session_state[FILE_NAME] = uploaded_file.name
            st.session_state[DATAFRAME] = df
            st.session_state[SESSION_ID] = session_id
            st.session_state[UPLOAD_RESPONSE] = response
            st.session_state[CHAT_HISTORY] = []
            st.session_state[ANALYTICS_REPORT] = None
            st.session_state["_dw_upload_time"] = datetime.utcnow()

            logger.info("Dataset uploaded: %s (%d × %d)",
                        uploaded_file.name, response.row_count, response.col_count)
            st.rerun()

        except DataWhispererError as exc:
            st.error(f"❌ {exc.message}")
        except Exception as exc:
            st.error(f"❌ Upload failed: {str(exc)[:300]}")
            logger.exception("Upload failed")


def _render_sidebar_dataset_info() -> None:
    """Active dataset info in sidebar."""
    metadata: FileMetadata = st.session_state[FILE_METADATA]
    st.markdown("""
    <div style="font-size:0.7rem; font-weight:600; color:#94a3b8;
                 text-transform:uppercase; letter-spacing:1px; margin-bottom:0.3rem;">
        ACTIVE DATASET</div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:0.75rem; font-weight:600; color:#818cf8; margin-bottom:0.5rem;">
        {metadata.original_name.upper()}</div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    c1.markdown(f"""
    <div style="font-size:0.6rem; color:#64748b;">Rows</div>
    <div style="font-size:1.1rem; font-weight:700; color:#f1f5f9;">{metadata.row_count:,}</div>
    """, unsafe_allow_html=True)
    c2.markdown(f"""
    <div style="font-size:0.6rem; color:#64748b;">Columns</div>
    <div style="font-size:1.1rem; font-weight:700; color:#f1f5f9;">{metadata.col_count}</div>
    """, unsafe_allow_html=True)


def _render_ai_engine_status() -> None:
    """AI Engine status with Groq primary / Ollama fallback."""
    st.markdown("""
    <div style="font-size:0.7rem; font-weight:600; color:#94a3b8;
                 text-transform:uppercase; letter-spacing:1px; margin-bottom:0.4rem;">
        AI ENGINE STATUS</div>
    """, unsafe_allow_html=True)

    agent = get_agent()
    if agent is None:
        st.markdown("""
        <div style="margin-bottom:0.3rem;">
            <span class="status-dot status-offline"></span>
            <span style="font-size:0.75rem; font-weight:600; color:#f1f5f9;">Groq (Primary)</span>
            <div style="font-size:0.65rem; color:#64748b; margin-left:14px;">Offline</div>
        </div>
        <div>
            <span class="status-dot status-offline"></span>
            <span style="font-size:0.75rem; font-weight:600; color:#f1f5f9;">Ollama (Fallback)</span>
            <div style="font-size:0.65rem; color:#64748b; margin-left:14px;">Offline</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("RECONNECT", use_container_width=True, key="reconnect_btn"):
            if reinitialise_agent():
                st.success("Connected!")
                st.rerun()
            else:
                st.error("Could not connect.")
        return

    try:
        health = agent.health_check()
        local_only = bool(health.get("local_only_mode", False))
        pr = health.get("provider_router")

        if pr:
            primary = health.get("primary", {})
            fallback = health.get("fallback", {})
            p_conn = bool(primary.get("connected", False))
            f_conn = bool(fallback.get("connected", False))

            p_status = "disabled" if local_only else ("Online" if p_conn else "Offline")
            p_dot = "status-standby" if local_only else ("status-online" if p_conn else "status-offline")
            f_status = "Online" if f_conn else "Standby"
            f_dot = "status-online" if f_conn else "status-standby"
        else:
            ollama_info = health.get("ollama", {})
            p_status = "Offline"
            p_dot = "status-offline"
            f_conn = bool(ollama_info.get("connected", False))
            f_status = "Online" if f_conn else "Standby"
            f_dot = "status-online" if f_conn else "status-standby"

        st.markdown(f"""
        <div style="margin-bottom:0.4rem;">
            <span class="status-dot {p_dot}"></span>
            <span style="font-size:0.75rem; font-weight:600; color:#f1f5f9;">Groq (Primary)</span>
            <div style="font-size:0.65rem; color:#64748b; margin-left:14px;">{p_status}</div>
        </div>
        <div>
            <span class="status-dot {f_dot}"></span>
            <span style="font-size:0.75rem; font-weight:600; color:#f1f5f9;">Ollama (Fallback)</span>
            <div style="font-size:0.65rem; color:#64748b; margin-left:14px;">{f_status}</div>
        </div>
        """, unsafe_allow_html=True)

    except Exception:
        st.markdown('<span style="color:#ef4444; font-size:0.75rem;">● Status check failed</span>',
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════

def _render_main_content() -> None:
    """Main content area: hero → cards → question → results."""
    _render_workspace_hero()

    if has_dataset():
        _render_dataset_summary_cards()
        _render_question_panel()
        _render_results_dashboard()
    else:
        st.markdown("""
        <div style="text-align:center; padding:3rem 0; color:#64748b;">
            <p style="font-size:0.85rem;">👈 Upload a CSV file from the sidebar to get started.</p>
        </div>
        """, unsafe_allow_html=True)


def _render_workspace_hero() -> None:
    """Centered workspace hero: icon + heading + subtitle."""
    st.markdown("""
    <div style="text-align:center; padding:1rem 0 0.75rem 0;" class="animate-in">
        <div style="font-size:1.5rem; color:#818cf8; margin-bottom:0.4rem;">✦</div>
        <div style="font-size:1rem; font-weight:700; color:#f1f5f9;
                     letter-spacing:1.5px; text-transform:uppercase;">READY TO ANALYZE</div>
        <div style="font-size:0.8rem; color:#94a3b8; margin-top:0.25rem;">
            Ask any question about your data below.</div>
    </div>
    """, unsafe_allow_html=True)


def _render_dataset_summary_cards() -> None:
    """Five horizontal metric cards: Rows, Columns, File Size, Dataset, Last Updated."""
    metadata: FileMetadata = st.session_state[FILE_METADATA]
    file_size = format_file_size(metadata.file_size_bytes)
    upload_time = st.session_state.get("_dw_upload_time")
    if upload_time is None:
        upload_time = getattr(metadata, 'uploaded_at', datetime.utcnow())
    last_updated_date = upload_time.strftime("%d %b %Y")
    last_updated_time = upload_time.strftime("%I:%M %p")

    cards_data = [
        ("📋", str(f"{metadata.row_count:,}"), "Total Rows", "ROWS"),
        ("📊", str(metadata.col_count), "Total Columns", "COLUMNS"),
        ("💾", file_size, "On Disk", "FILE SIZE"),
        ("📁", "CSV", "File Type", "DATASET"),
        ("🕐", last_updated_date, last_updated_time, "LAST UPDATED"),
    ]

    cols = st.columns(5)
    for idx, (icon, value, sub, label) in enumerate(cards_data):
        with cols[idx]:
            st.markdown(f"""
            <div class="metric-card animate-in">
                <div class="metric-icon">{icon}</div>
                <div style="font-size:0.6rem; font-weight:600; color:#94a3b8;
                             text-transform:uppercase; letter-spacing:1px; margin-bottom:0.2rem;">
                    {label}</div>
                <div class="metric-value">{value}</div>
                <div style="font-size:0.6rem; color:#64748b; margin-top:0.15rem;">{sub}</div>
            </div>
            """, unsafe_allow_html=True)


def _render_question_panel() -> None:
    """Question input area with suggested questions."""
    st.markdown("")  # spacing

    # Suggested questions
    metadata: FileMetadata = st.session_state[FILE_METADATA]
    suggestions = _generate_suggestions(metadata)

    if suggestions:
        st.markdown("""
        <div style="font-size:0.8rem; font-weight:600; color:#f1f5f9; margin-bottom:0.5rem;">
            💡 TRY ASKING:</div>
        """, unsafe_allow_html=True)

        # 2x2 grid
        row1_cols = st.columns(2)
        for idx, q in enumerate(suggestions[:2]):
            with row1_cols[idx]:
                st.markdown('<div class="suggest-btn">', unsafe_allow_html=True)
                if st.button(q.upper(), key=f"suggest_{idx}", use_container_width=True):
                    _submit_question(q)
                st.markdown('</div>', unsafe_allow_html=True)

        if len(suggestions) > 2:
            row2_cols = st.columns(2)
            for idx, q in enumerate(suggestions[2:4]):
                with row2_cols[idx]:
                    st.markdown('<div class="suggest-btn">', unsafe_allow_html=True)
                    if st.button(q.upper(), key=f"suggest_{idx+2}", use_container_width=True):
                        _submit_question(q)
                    st.markdown('</div>', unsafe_allow_html=True)

    # Question input
    question = st.chat_input("Ask a question about your data...", key="workspace_chat_input")
    if question:
        _submit_question(question)

    # Loading state
    if st.session_state.get("_dw_processing", False):
        st.markdown("""
        <div style="text-align:center; padding:0.5rem; color:#94a3b8; font-size:0.8rem;">
            ⏳ Analyzing your question...</div>
        """, unsafe_allow_html=True)


def _generate_suggestions(metadata: FileMetadata) -> List[str]:
    """Generate suggested questions from schema (same logic as original chat.py)."""
    suggestions: List[str] = []
    col_names = [c.name for c in metadata.columns]
    numeric_cols = [c.name for c in metadata.columns if c.mean is not None]
    categorical_cols = [c.name for c in metadata.columns if c.mean is None and c.dtype == "object"]

    if numeric_cols:
        suggestions.append(f"What is the average {numeric_cols[0]}?")
    if len(numeric_cols) >= 2:
        suggestions.append(
            f"Show me the correlation between {numeric_cols[0]} and {numeric_cols[1]}")
    if categorical_cols and numeric_cols:
        suggestions.append(
            f"Compare {numeric_cols[0]} across different {categorical_cols[0]} values")
    if len(col_names) >= 2:
        target = numeric_cols[0] if numeric_cols else col_names[0]
        suggestions.append(f"Show me the top 10 rows by {target}")
    return suggestions


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS DASHBOARD (2×2 grid)
# ══════════════════════════════════════════════════════════════════════════════

def _render_results_dashboard() -> None:
    """Render the 2×2 result panels from the latest assistant response."""
    history: List[Dict[str, Any]] = st.session_state.get(CHAT_HISTORY, [])
    if not history:
        return

    # Find latest assistant response
    latest_response: Optional[ChatResponse] = None
    latest_question: str = ""
    for entry in reversed(history):
        if entry.get("role") == "assistant" and entry.get("response"):
            latest_response = entry["response"]
            break
        if entry.get("role") == "user":
            latest_question = entry.get("content", "")

    if latest_response is None:
        return

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    # Top row: Result Preview | Generated SQL
    top_left, top_right = st.columns(2)

    with top_left:
        _render_result_preview(latest_response)

    with top_right:
        _render_generated_sql(latest_response)

    # Bottom row: AI Answer | Visualization
    bot_left, bot_right = st.columns(2)

    with bot_left:
        _render_ai_answer(latest_response)

    with bot_right:
        _render_visualization(latest_response)


def _render_result_preview(response: ChatResponse) -> None:
    """Top-left panel: Result Preview table."""
    st.markdown("""
    <div class="result-panel">
        <div class="panel-header">
            <div class="panel-title">📋 RESULT PREVIEW</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if response.result_type == ResultType.ERROR:
        st.error(response.content[:300])
        return

    data = response.result_data
    if data is None:
        st.caption("No tabular result for this query.")
        return

    try:
        if isinstance(data, str):
            try:
                parsed = json.loads(data)
                if isinstance(parsed, list):
                    df = pd.DataFrame(parsed)
                elif isinstance(parsed, dict):
                    if all(not isinstance(v, (dict, list)) for v in parsed.values()):
                        df = pd.DataFrame([parsed]).T.reset_index()
                        df.columns = ["Index", "Value"]
                    else:
                        df = pd.DataFrame(parsed)
                else:
                    st.markdown(f"```\n{data[:500]}\n```")
                    return
            except json.JSONDecodeError:
                st.markdown(f"```\n{data[:500]}\n```")
                return
        elif isinstance(data, (dict, list)):
            df = pd.DataFrame(data)
        else:
            st.markdown(f"```\n{str(data)[:500]}\n```")
            return

        preview_rows = min(5, len(df))
        st.dataframe(df.head(preview_rows), use_container_width=True, hide_index=True,
                      height=min(220, 38 * preview_rows + 40))
        st.caption(f"Showing top {preview_rows} rows · {len(df)} rows returned")

    except Exception:
        st.markdown(f"```\n{str(data)[:500]}\n```")


def _render_generated_sql(response: ChatResponse) -> None:
    """Top-right panel: Generated SQL/code with copy button."""
    code = response.generated_code or ""

    st.markdown("""
    <div class="result-panel">
        <div class="panel-header">
            <div class="panel-title">⚡ GENERATED SQL</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if code:
        st.code(code, language="python")

        # Copy button
        if st.button("📋 Copy SQL", key="copy_sql_btn"):
            st.toast("Code copied to clipboard!")
    else:
        st.caption("No code generated for this query.")


def _render_ai_answer(response: ChatResponse) -> None:
    """Bottom-left panel: AI explanation/answer."""
    st.markdown("""
    <div class="result-panel">
        <div class="panel-header">
            <div class="panel-title">🤖 AI ANSWER</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    explanation = response.explanation
    if explanation:
        st.markdown(f"""
        <div style="font-size:0.82rem; color:#e2e8f0; line-height:1.7; padding:0.25rem 0;">
            {explanation}
        </div>
        """, unsafe_allow_html=True)

        if st.button("📋 Copy Answer", key="copy_answer_btn"):
            st.toast("Answer copied!")
    else:
        st.markdown(response.content[:500] if response.content else "No answer available.")

    # Metadata footer
    parts = []
    if response.provider_used:
        label = response.provider_used
        if response.model_used:
            label += f" · {response.model_used}"
        if response.fallback_used:
            label += " (fallback)"
        parts.append(f"Model: {label}")
    if response.latency_ms > 0:
        parts.append(f"Execution Time: {response.latency_ms/1000:.2f}s")

    # Count rows returned
    row_count = _count_result_rows(response)
    if row_count:
        parts.append(f"Rows Returned: {row_count}")

    if parts:
        st.caption(" • ".join(parts))


def _render_visualization(response: ChatResponse) -> None:
    """Bottom-right panel: Chart visualization."""
    st.markdown("""
    <div class="result-panel">
        <div class="panel-header">
            <div class="panel-title">📊 RESULT VISUALIZATION</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    chart_path = response.chart_path
    if chart_path and Path(chart_path).exists():
        st.image(chart_path, use_container_width=True)
        # Download button
        with open(chart_path, "rb") as f:
            st.download_button("⬇ Download", data=f.read(),
                               file_name=Path(chart_path).name,
                               mime="image/png", key="dl_chart_btn")
    elif response.result_type == ResultType.CHART:
        st.caption("Chart was generated but file not found.")
    else:
        st.caption("No visualization for this query result.")


def _count_result_rows(response: ChatResponse) -> Optional[int]:
    """Count rows in result data."""
    data = response.result_data
    if data is None:
        return None
    try:
        if isinstance(data, str):
            parsed = json.loads(data)
            if isinstance(parsed, list):
                return len(parsed)
            elif isinstance(parsed, dict):
                first_val = next(iter(parsed.values()), None)
                if isinstance(first_val, (dict, list)):
                    return len(first_val)
                return len(parsed)
        elif isinstance(data, list):
            return len(data)
        elif isinstance(data, dict):
            return len(data)
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# QUESTION SUBMISSION (reuses existing backend pipeline)
# ══════════════════════════════════════════════════════════════════════════════

def _submit_question(question: str) -> None:
    """Process a user question through the existing ChatService pipeline."""
    chat_service = get_chat_service()
    if chat_service is None:
        st.error("AI agent is unavailable. Please check the sidebar status.")
        return

    session_id = st.session_state.get(SESSION_ID)
    file_id = st.session_state.get(FILE_ID)
    file_metadata = st.session_state.get(FILE_METADATA)
    csv_path = st.session_state.get(FILE_PATH)

    if not all([session_id, file_id, file_metadata, csv_path]):
        st.error("Please upload a dataset first.")
        return

    history: List[Dict[str, Any]] = st.session_state.get(CHAT_HISTORY, [])
    history.append({"role": "user", "content": question})
    st.session_state[CHAT_HISTORY] = history

    with st.spinner("🔮 Analyzing your question…"):
        try:
            response: ChatResponse = chat_service.process_question(
                session_id=session_id, file_id=file_id,
                question=question, file_metadata=file_metadata,
                csv_path=csv_path,
            )
            history.append({
                "role": "assistant", "content": response.content, "response": response,
            })
            st.session_state[CHAT_HISTORY] = history

        except DataWhispererError as exc:
            history.append({"role": "assistant", "content": f"❌ {exc.message}", "response": None})
            st.session_state[CHAT_HISTORY] = history

        except Exception as exc:
            history.append({"role": "assistant",
                            "content": f"❌ Error: {str(exc)[:300]}", "response": None})
            st.session_state[CHAT_HISTORY] = history
            logger.exception("Chat processing error")

    st.rerun()
