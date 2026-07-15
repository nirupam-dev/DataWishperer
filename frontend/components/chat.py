"""
Chat interface component — Natural language querying with code-interpreter display.

Renders the chat history and input box, delegates question processing
to the existing ChatService, and displays answers with code, explanations,
tables, and charts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.io as pio
import streamlit as st

from backend.core.exceptions import DataWhispererError
from backend.core.logging_config import get_logger
from backend.models.schemas import (
    ChatResponse,
    FileMetadata,
    MessageRole,
    ResultType,
)

from frontend.state import (
    get_agent,
    get_chat_service,
    get_viz_service,
    has_agent,
    has_dataset,
    CHAT_HISTORY,
    DATAFRAME,
    FILE_ID,
    FILE_METADATA,
    FILE_PATH,
    SESSION_ID,
)

logger = get_logger(__name__)


def render_chat() -> None:
    """Render the chat interface."""
    if not has_dataset():
        _render_empty_state()
        return

    if not has_agent():
        _render_agent_unavailable()
        return

    # ── Chat History ──────────────────────────────────────────────
    _render_chat_history()

    # ── Input ─────────────────────────────────────────────────────
    _render_chat_input()


# ── Private Renderers ───────────────────────────────────────────────────────


def _render_empty_state() -> None:
    st.markdown(
        '<div class="empty-state animate-in">'
        '<div class="empty-icon">💬</div>'
        "<h4>Start a Conversation</h4>"
        "<p>Upload a CSV file first, then ask questions about your data.</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_agent_unavailable() -> None:
    st.warning(
        "⚠️ **AI Agent Unavailable**\n\n"
        "The Ollama LLM server is not reachable. Please:\n\n"
        "1. Open a terminal and run: `ollama serve`\n"
        "2. Pull the model: `ollama pull qwen2.5:7b`\n"
        "3. Click **Reconnect** in the sidebar.\n\n"
        "You can still explore your dataset in the **Explore** tab."
    )


def _render_chat_history() -> None:
    """Render all messages in the chat history."""
    history: List[Dict[str, Any]] = st.session_state.get(CHAT_HISTORY, [])

    if not history:
        st.markdown(
            '<div class="empty-state animate-in">'
            '<div class="empty-icon">🤖</div>'
            "<h4>Ready to Analyze</h4>"
            "<p>Ask any question about your data below.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        # Suggested questions
        _render_suggested_questions()
        return

    for entry in history:
        role = entry.get("role", "user")
        if role == "user":
            _render_user_message(entry["content"])
        else:
            _render_assistant_message(entry)


def _render_suggested_questions() -> None:
    """Show auto-generated suggested questions."""
    metadata: Optional[FileMetadata] = st.session_state.get(FILE_METADATA)
    agent = get_agent()

    if metadata is None or agent is None:
        return

    # Generate suggested questions (use a simple heuristic based on columns)
    suggestions = _generate_quick_suggestions(metadata)

    if suggestions:
        st.markdown("#### 💡 Try asking:")
        cols = st.columns(2)
        for idx, q in enumerate(suggestions[:4]):
            with cols[idx % 2]:
                if st.button(q, key=f"suggest_{idx}", use_container_width=True):
                    _submit_question(q)


def _generate_quick_suggestions(metadata: FileMetadata) -> List[str]:
    """Generate simple suggested questions based on column metadata."""
    suggestions: List[str] = []
    col_names = [c.name for c in metadata.columns]
    numeric_cols = [c.name for c in metadata.columns if c.mean is not None]
    categorical_cols = [c.name for c in metadata.columns if c.mean is None and c.dtype == "object"]

    if numeric_cols:
        suggestions.append(f"What is the average {numeric_cols[0]}?")
    if len(numeric_cols) >= 2:
        suggestions.append(f"Show me the correlation between {numeric_cols[0]} and {numeric_cols[1]}")
    if categorical_cols and numeric_cols:
        suggestions.append(f"Compare {numeric_cols[0]} across different {categorical_cols[0]} values")
    if len(col_names) >= 2:
        suggestions.append(f"Show me the top 10 rows by {numeric_cols[0] if numeric_cols else col_names[0]}")

    return suggestions


def _render_user_message(content: str) -> None:
    """Render a user chat message."""
    st.markdown(
        f'<div class="chat-user animate-in">'
        f"<strong>🧑 You</strong><br/>{content}"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_assistant_message(entry: Dict[str, Any]) -> None:
    """Render an assistant response with code, output, explanation, and chart."""
    st.markdown(
        '<div class="chat-assistant animate-in">',
        unsafe_allow_html=True,
    )
    st.markdown("**🤖 DataWhisperer**")

    response: Optional[ChatResponse] = entry.get("response")

    if response is None:
        # Fallback: plain text content
        st.markdown(entry.get("content", ""))
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # ── Generated Code ───────────────────────────────────────────
    if response.generated_code:
        with st.expander("📝 Generated Code", expanded=False):
            st.code(response.generated_code, language="python")

    # ── Output / Result ──────────────────────────────────────────
    _render_result(response)

    # ── Explanation ───────────────────────────────────────────────
    if response.explanation:
        st.markdown(
            f'<div class="explanation-box">'
            f"💡 **Explanation:** {response.explanation}"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Chart Reasoning ──────────────────────────────────────────
    if response.chart_explanation:
        st.markdown(
            f'<div class="explanation-box">'
            f"🎨 **Chart Reasoning:** {response.chart_explanation}"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Auto-Debug Note ──────────────────────────────────────────
    if response.auto_debug_applied:
        st.info(
            "🔧 The original code had an error and was automatically "
            "debugged and re-executed."
        )

    # ── Provider Metadata ────────────────────────────────────────
    provider_label = response.provider_used or "unknown"
    model_label = response.model_used or "unknown"
    provider_caption = f"🧠 Model: {provider_label} · {model_label}"
    if response.fallback_used:
        provider_caption += " · fallback applied"
        if response.fallback_reason:
            provider_caption += f" ({response.fallback_reason[:140]})"
    st.caption(provider_caption)

    # ── Performance ──────────────────────────────────────────────
    perf_parts = []
    if response.latency_ms > 0:
        perf_parts.append(f"{response.latency_ms:.0f}ms")
    if response.tokens_used > 0:
        perf_parts.append(f"{response.tokens_used} tokens")
    if response.retry_count > 0:
        perf_parts.append(f"{response.retry_count} retries")
    if perf_parts:
        st.caption(" · ".join(perf_parts))

    st.markdown("</div>", unsafe_allow_html=True)


def _render_result(response: ChatResponse) -> None:
    """Render the execution result based on its type."""
    if response.result_type == ResultType.ERROR:
        st.error(response.content)
        return

    if response.result_type == ResultType.CHART:
        _render_chart(response)
    elif response.result_type in (ResultType.DATAFRAME, ResultType.SERIES):
        _render_table_result(response)
    elif response.result_data is not None:
        st.markdown(f"**Result:**\n\n{response.result_data}")
    else:
        # Text-only response
        st.markdown(response.content)


def _render_chart(response: ChatResponse) -> None:
    """Render a chart from the response."""
    chart_path = response.chart_path

    if chart_path and Path(chart_path).exists():
        st.markdown('<div style="max-width:680px; margin:0.5rem auto;">', unsafe_allow_html=True)
        st.image(chart_path, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    elif response.result_data:
        st.markdown(str(response.result_data))


def _render_table_result(response: ChatResponse) -> None:
    """Render a DataFrame/Series result."""
    data = response.result_data
    if data is None:
        st.markdown(response.content)
        return

    try:
        if isinstance(data, str):
            try:
                # The executor returns DataFrames/Series as JSON strings
                parsed = json.loads(data)
                if isinstance(parsed, list):
                    df = pd.DataFrame(parsed)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    return
                elif isinstance(parsed, dict):
                    # For Series (dict with scalar values), orient index correctly
                    if all(not isinstance(v, (dict, list)) for v in parsed.values()):
                        df = pd.DataFrame([parsed]).T.reset_index()
                        df.columns = ["Index", "Value"]
                    else:
                        df = pd.DataFrame(parsed)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    return
            except json.JSONDecodeError:
                pass
            
            # Fallback to display as-is
            st.markdown(data)
        elif isinstance(data, (dict, list)):
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.markdown(f"```\n{data}\n```")
    except Exception:
        st.markdown(f"```\n{data}\n```")


# ── Chat Input & Submission ─────────────────────────────────────────────────


def _render_chat_input() -> None:
    """Render the chat input box."""
    question = st.chat_input(
        "Ask a question about your data…",
        key="chat_input",
    )
    if question:
        _submit_question(question)


def _submit_question(question: str) -> None:
    """Process a user question through the chat service."""
    chat_service = get_chat_service()
    if chat_service is None:
        st.error("AI agent is unavailable. Please check Ollama status.")
        return

    session_id = st.session_state.get(SESSION_ID)
    file_id = st.session_state.get(FILE_ID)
    file_metadata = st.session_state.get(FILE_METADATA)
    csv_path = st.session_state.get(FILE_PATH)

    if not all([session_id, file_id, file_metadata, csv_path]):
        st.error("Please upload a dataset first.")
        return

    # Append user message to local history immediately
    history: List[Dict[str, Any]] = st.session_state.get(CHAT_HISTORY, [])
    history.append({"role": "user", "content": question})
    st.session_state[CHAT_HISTORY] = history

    # Process via ChatService
    with st.spinner("🔮 Analyzing your question…"):
        try:
            response: ChatResponse = chat_service.process_question(
                session_id=session_id,
                file_id=file_id,
                question=question,
                file_metadata=file_metadata,
                csv_path=csv_path,
            )

            # Append assistant response
            history.append({
                "role": "assistant",
                "content": response.content,
                "response": response,
            })
            st.session_state[CHAT_HISTORY] = history

        except DataWhispererError as exc:
            history.append({
                "role": "assistant",
                "content": f"❌ {exc.message}",
                "response": None,
            })
            st.session_state[CHAT_HISTORY] = history
            if exc.suggestion:
                st.info(f"💡 {exc.suggestion}")

        except Exception as exc:
            error_msg = f"❌ An error occurred: {str(exc)[:300]}"
            history.append({
                "role": "assistant",
                "content": error_msg,
                "response": None,
            })
            st.session_state[CHAT_HISTORY] = history
            logger.exception("Chat processing error")

    st.rerun()
