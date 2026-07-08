"""
Session state management for the Streamlit frontend.

Centralises all session_state initialisation and access so that
every page module reads from a single source of truth. Prevents
unnecessary reloads of the agent/services on Streamlit reruns.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st
import pandas as pd

from backend.core.config import get_settings
from backend.core.logging_config import get_logger, setup_logging
from backend.llm.factory import create_agent, create_chat_service
from backend.llm.agent import DataWhispererAgent
from backend.services.chat_service import ChatService
from backend.services.file_service import FileService
from backend.services.session_service import SessionService
from backend.services.export_service import ExportService
from backend.services.visualization_service import VisualizationService
from backend.analytics.orchestrator import AnalyticsOrchestrator
from backend.models.schemas import FileMetadata

logger = get_logger(__name__)


# ── State keys ──────────────────────────────────────────────────────────────

_INITIALISED = "_dw_initialised"
_AGENT = "_dw_agent"
_CHAT_SERVICE = "_dw_chat_service"
_FILE_SERVICE = "_dw_file_service"
_SESSION_SERVICE = "_dw_session_service"
_EXPORT_SERVICE = "_dw_export_service"
_VIZ_SERVICE = "_dw_viz_service"
_ANALYTICS = "_dw_analytics"

FILE_ID = "file_id"
FILE_METADATA = "file_metadata"
FILE_PATH = "file_path"
FILE_NAME = "file_name"
DATAFRAME = "dataframe"
SESSION_ID = "session_id"
CHAT_HISTORY = "chat_history"
ANALYTICS_REPORT = "analytics_report"
UPLOAD_RESPONSE = "upload_response"
ACTIVE_TAB = "active_tab"
SETTINGS_OLLAMA_URL = "settings_ollama_url"
SETTINGS_OLLAMA_MODEL = "settings_ollama_model"


def init_state() -> None:
    """
    Initialise session_state once per browser session.

    Creates backend services as singletons stored in session_state so that
    Streamlit reruns do not re-instantiate the model or database connections.
    """
    if st.session_state.get(_INITIALISED):
        return

    # Setup logging once
    setup_logging()

    settings = get_settings()

    # Create the agent and all services — expensive, done once.
    try:
        agent = create_agent()
        st.session_state[_AGENT] = agent
        st.session_state[_CHAT_SERVICE] = create_chat_service(agent)
    except Exception as exc:
        logger.warning("Agent creation failed (Ollama may be down): %s", exc)
        st.session_state[_AGENT] = None
        st.session_state[_CHAT_SERVICE] = None

    st.session_state[_FILE_SERVICE] = FileService()
    st.session_state[_SESSION_SERVICE] = SessionService()
    st.session_state[_EXPORT_SERVICE] = ExportService()
    st.session_state[_VIZ_SERVICE] = VisualizationService()
    st.session_state[_ANALYTICS] = AnalyticsOrchestrator()

    # Data state
    st.session_state.setdefault(FILE_ID, None)
    st.session_state.setdefault(FILE_METADATA, None)
    st.session_state.setdefault(FILE_PATH, None)
    st.session_state.setdefault(FILE_NAME, None)
    st.session_state.setdefault(DATAFRAME, None)
    st.session_state.setdefault(SESSION_ID, None)
    st.session_state.setdefault(CHAT_HISTORY, [])
    st.session_state.setdefault(ANALYTICS_REPORT, None)
    st.session_state.setdefault(UPLOAD_RESPONSE, None)
    st.session_state.setdefault(ACTIVE_TAB, "chat")

    st.session_state[_INITIALISED] = True
    logger.info("Session state initialised")


# ── Typed Accessors ─────────────────────────────────────────────────────────


def get_agent() -> Optional[DataWhispererAgent]:
    return st.session_state.get(_AGENT)


def get_chat_service() -> Optional[ChatService]:
    return st.session_state.get(_CHAT_SERVICE)


def get_file_service() -> FileService:
    return st.session_state[_FILE_SERVICE]


def get_session_service() -> SessionService:
    return st.session_state[_SESSION_SERVICE]


def get_export_service() -> ExportService:
    return st.session_state[_EXPORT_SERVICE]


def get_viz_service() -> VisualizationService:
    return st.session_state[_VIZ_SERVICE]


def get_analytics() -> AnalyticsOrchestrator:
    return st.session_state[_ANALYTICS]


def has_dataset() -> bool:
    """Return True if a CSV has been loaded."""
    return st.session_state.get(DATAFRAME) is not None


def has_agent() -> bool:
    """Return True if the LLM agent was created successfully."""
    return st.session_state.get(_AGENT) is not None


def clear_dataset() -> None:
    """Reset all dataset-related state."""
    st.session_state[FILE_ID] = None
    st.session_state[FILE_METADATA] = None
    st.session_state[FILE_PATH] = None
    st.session_state[FILE_NAME] = None
    st.session_state[DATAFRAME] = None
    st.session_state[SESSION_ID] = None
    st.session_state[CHAT_HISTORY] = []
    st.session_state[ANALYTICS_REPORT] = None
    st.session_state[UPLOAD_RESPONSE] = None


def reinitialise_agent() -> bool:
    """
    Attempt to recreate the agent (e.g. after settings change).

    Returns True on success, False on failure.
    """
    try:
        agent = create_agent()
        st.session_state[_AGENT] = agent
        st.session_state[_CHAT_SERVICE] = create_chat_service(agent)
        logger.info("Agent re-initialised successfully")
        return True
    except Exception as exc:
        logger.error("Agent re-initialisation failed: %s", exc)
        st.session_state[_AGENT] = None
        st.session_state[_CHAT_SERVICE] = None
        return False
