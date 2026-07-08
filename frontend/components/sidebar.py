"""
Sidebar component — File upload, dataset info, session history, settings.

This module renders the left sidebar and handles file upload logic
by delegating to the existing FileService.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
from pathlib import Path

from backend.core.config import get_settings
from backend.core.exceptions import DataWhispererError
from backend.core.logging_config import get_logger
from backend.utils.helpers import format_file_size, relative_time

from frontend.state import (
    get_file_service,
    get_session_service,
    get_agent,
    has_dataset,
    has_agent,
    clear_dataset,
    reinitialise_agent,
    FILE_ID,
    FILE_METADATA,
    FILE_PATH,
    FILE_NAME,
    DATAFRAME,
    SESSION_ID,
    UPLOAD_RESPONSE,
    ANALYTICS_REPORT,
    CHAT_HISTORY,
)

logger = get_logger(__name__)


def render_sidebar() -> None:
    """Render the complete sidebar."""
    with st.sidebar:
        # ── Brand Header ─────────────────────────────────────────
        st.markdown("# 🔮 DataWhisperer")
        st.caption("Talk to your CSV with AI")
        st.divider()

        # ── File Upload ──────────────────────────────────────────
        _render_upload_section()

        # ── Dataset Info (if loaded) ─────────────────────────────
        if has_dataset():
            st.divider()
            _render_dataset_info()

        # ── Health Status ────────────────────────────────────────
        st.divider()
        _render_health_status()

        # ── Settings ─────────────────────────────────────────────
        st.divider()
        _render_settings_section()


# ── Private Renderers ───────────────────────────────────────────────────────


def _render_upload_section() -> None:
    """Render the CSV upload widget."""
    st.markdown("### 📁 Upload Dataset")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload a CSV file to start analysing your data.",
        key="csv_uploader",
    )

    if uploaded_file is not None:
        # Avoid re-processing the same file on Streamlit reruns
        current_name = st.session_state.get(FILE_NAME)
        if current_name == uploaded_file.name and has_dataset():
            return

        _process_upload(uploaded_file)


def _process_upload(uploaded_file) -> None:
    """Validate, save, and analyse an uploaded file."""
    file_service = get_file_service()

    with st.spinner("Validating and analysing your dataset…"):
        try:
            content = uploaded_file.read()
            response = file_service.upload_file(
                filename=uploaded_file.name,
                content=content,
            )

            # Store the metadata and path
            metadata = file_service.get_file_metadata(response.file_id)
            csv_path = file_service.get_file_path(response.file_id)

            # Load the DataFrame into memory (once)
            df = pd.read_csv(csv_path)

            # Register with the agent if available
            agent = get_agent()
            if agent is not None:
                agent.register_dataset(metadata)

            # Create a new session
            session_svc = get_session_service()
            session_id = session_svc.create_session(
                file_id=response.file_id,
                title=f"Analysis: {uploaded_file.name}",
            )

            # Persist to session_state
            st.session_state[FILE_ID] = response.file_id
            st.session_state[FILE_METADATA] = metadata
            st.session_state[FILE_PATH] = csv_path
            st.session_state[FILE_NAME] = uploaded_file.name
            st.session_state[DATAFRAME] = df
            st.session_state[SESSION_ID] = session_id
            st.session_state[UPLOAD_RESPONSE] = response
            st.session_state[CHAT_HISTORY] = []
            st.session_state[ANALYTICS_REPORT] = None

            st.success(
                f"✅ **{uploaded_file.name}** loaded — "
                f"{response.row_count:,} rows × {response.col_count} columns"
            )
            logger.info(
                "Dataset uploaded: %s (%d × %d)",
                uploaded_file.name, response.row_count, response.col_count,
            )
            st.rerun()

        except DataWhispererError as exc:
            st.error(f"❌ {exc.message}")
            if exc.suggestion:
                st.info(f"💡 {exc.suggestion}")
        except Exception as exc:
            st.error(f"❌ Upload failed: {str(exc)[:300]}")
            logger.exception("Upload failed")


def _render_dataset_info() -> None:
    """Show active dataset summary in the sidebar."""
    metadata = st.session_state.get(FILE_METADATA)
    if metadata is None:
        return

    st.markdown("### 📊 Active Dataset")
    st.markdown(f"**{metadata.original_name}**")

    col1, col2 = st.columns(2)
    col1.metric("Rows", f"{metadata.row_count:,}")
    col2.metric("Columns", f"{metadata.col_count}")

    st.caption(f"Memory: {metadata.memory_usage_mb:.1f} MB")
    st.caption(f"Size: {format_file_size(metadata.file_size_bytes)}")

    if st.button("🗑️ Clear Dataset", use_container_width=True):
        clear_dataset()
        st.rerun()


def _render_health_status() -> None:
    """Show Ollama/model connection status."""
    agent = get_agent()

    if agent is None:
        st.markdown(
            '<span class="badge badge-error">● Offline</span> '
            "AI agent unavailable",
            unsafe_allow_html=True,
        )
        st.caption(
            "Ollama may not be running. Start it with:\n"
            "```\nollama serve\n```"
        )
        if st.button("🔄 Reconnect", use_container_width=True):
            if reinitialise_agent():
                st.success("Connected!")
                st.rerun()
            else:
                st.error("Could not connect to Ollama.")
        return

    try:
        health = agent.health_check()
        if health.get("agent_ready"):
            st.markdown(
                '<span class="badge badge-success">● Online</span> '
                f"Model: `{health.get('model', 'unknown')}`",
                unsafe_allow_html=True,
            )
        else:
            ollama_info = health.get("ollama", {})
            if ollama_info.get("connected"):
                st.markdown(
                    '<span class="badge badge-warning">● Model Missing</span>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"Pull the model with:\n"
                    f"```\nollama pull {health.get('model', 'qwen2.5:7b')}\n```"
                )
            else:
                st.markdown(
                    '<span class="badge badge-error">● Disconnected</span>',
                    unsafe_allow_html=True,
                )
                st.caption("Start Ollama: `ollama serve`")

            if st.button("🔄 Reconnect", use_container_width=True):
                if reinitialise_agent():
                    st.success("Connected!")
                    st.rerun()
                else:
                    st.error("Reconnection failed.")
    except Exception:
        st.markdown(
            '<span class="badge badge-error">● Error</span>',
            unsafe_allow_html=True,
        )


def _render_settings_section() -> None:
    """Render basic model settings."""
    with st.expander("⚙️ Settings", expanded=False):
        settings = get_settings()

        st.text_input(
            "Ollama URL",
            value=settings.ollama.base_url,
            key="settings_ollama_url",
            disabled=True,
            help="Set via OLLAMA_BASE_URL environment variable.",
        )
        st.text_input(
            "Model",
            value=settings.ollama.model,
            key="settings_ollama_model",
            disabled=True,
            help="Set via OLLAMA_MODEL environment variable.",
        )
        st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=settings.ollama.temperature,
            step=0.1,
            disabled=True,
            help="Controls response randomness.",
        )
        st.caption(
            "Settings are configured via environment variables or `.env` file."
        )
