"""
Frontend integration tests for DataWhisperer.

Tests the state management, component rendering logic, and service
integration without requiring a running Streamlit server or Ollama.
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

from backend.core.config import get_settings
from backend.models.schemas import (
    ChatMessage,
    ChatResponse,
    ColumnInfo,
    ExportFormat,
    ExportResult,
    FileMetadata,
    FileUploadResponse,
    MessageRole,
    ResultType,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_settings():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def sample_metadata() -> FileMetadata:
    return FileMetadata(
        file_id="test-001",
        original_name="test_data.csv",
        stored_path="/tmp/test_data.csv",
        row_count=100,
        col_count=3,
        file_size_bytes=5000,
        memory_usage_mb=0.05,
        columns=[
            ColumnInfo(
                name="name", dtype="object",
                non_null_count=100, null_count=0,
                unique_count=50, sample_values=["Alice", "Bob"],
            ),
            ColumnInfo(
                name="age", dtype="int64",
                non_null_count=100, null_count=0,
                unique_count=30, sample_values=["25", "30"],
                mean=35.0, std=10.0, min_val=18.0, max_val=65.0,
            ),
            ColumnInfo(
                name="score", dtype="float64",
                non_null_count=95, null_count=5,
                unique_count=80, sample_values=["85.5", "92.0"],
                mean=75.0, std=15.0, min_val=20.0, max_val=100.0,
            ),
        ],
    )


@pytest.fixture
def sample_df() -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame({
        "name": [f"user_{i}" for i in range(100)],
        "age": np.random.randint(18, 65, 100),
        "score": np.random.normal(75, 15, 100).round(2),
    })


@pytest.fixture
def sample_chat_response() -> ChatResponse:
    return ChatResponse(
        message_id="msg-001",
        content="The average age is 35.0",
        generated_code="print(df['age'].mean())",
        result_type=ResultType.TEXT,
        result_data="35.0",
        explanation="Computed the mean of the age column.",
        tokens_used=150,
        latency_ms=1234.5,
    )


# ── State Module Tests ───────────────────────────────────────────────────────


class TestStateModule:
    """Tests for frontend.state — session state management."""

    def test_import_state_module(self):
        """Verify the state module can be imported without errors."""
        from frontend import state
        assert hasattr(state, "init_state")
        assert hasattr(state, "has_dataset")
        assert hasattr(state, "has_agent")
        assert hasattr(state, "clear_dataset")

    def test_state_key_constants(self):
        """Verify all state keys are defined as strings."""
        from frontend.state import (
            FILE_ID, FILE_METADATA, FILE_PATH, FILE_NAME,
            DATAFRAME, SESSION_ID, CHAT_HISTORY, ANALYTICS_REPORT,
        )
        keys = [FILE_ID, FILE_METADATA, FILE_PATH, FILE_NAME,
                DATAFRAME, SESSION_ID, CHAT_HISTORY, ANALYTICS_REPORT]
        for k in keys:
            assert isinstance(k, str)
            assert len(k) > 0


# ── Theme Module Tests ───────────────────────────────────────────────────────


class TestThemeModule:
    """Tests for frontend.theme — CSS injection."""

    def test_import_theme_module(self):
        from frontend import theme
        assert hasattr(theme, "inject_custom_css")
        assert hasattr(theme, "PRIMARY")

    def test_css_string_is_valid(self):
        from frontend.theme import _CSS
        assert "<style>" in _CSS
        assert "</style>" in _CSS
        assert "font-family" in _CSS

    def test_colour_constants(self):
        from frontend.theme import PRIMARY, ACCENT, SUCCESS, WARNING, BG_DARK
        # Verify they are hex colours
        for colour in [PRIMARY, ACCENT, SUCCESS, WARNING, BG_DARK]:
            assert colour.startswith("#")
            assert len(colour) == 7


# ── Component Logic Tests ────────────────────────────────────────────────────


class TestExplorerLogic:
    """Tests for explorer component data preparation logic."""

    def test_schema_data_construction(self, sample_metadata):
        """Verify schema data can be built from FileMetadata."""
        schema_data = []
        for col in sample_metadata.columns:
            total = col.non_null_count + col.null_count
            null_pct = f"{(col.null_count / total * 100):.1f}%" if total > 0 else "N/A"
            schema_data.append({
                "Column": col.name,
                "Type": col.dtype,
                "Non-Null": f"{col.non_null_count:,}",
                "Null": f"{col.null_count:,}",
                "Null %": null_pct,
                "Unique": f"{col.unique_count:,}",
            })

        assert len(schema_data) == 3
        assert schema_data[0]["Column"] == "name"
        assert schema_data[2]["Null %"] == "5.0%"

    def test_numeric_categorical_split(self, sample_metadata):
        """Verify numeric/categorical column classification."""
        numeric_cols = [c for c in sample_metadata.columns if c.mean is not None]
        categorical_cols = [c for c in sample_metadata.columns if c.mean is None]
        assert len(numeric_cols) == 2
        assert len(categorical_cols) == 1

    def test_summary_metrics_calculation(self, sample_metadata):
        """Verify total null count computation."""
        total_nulls = sum(c.null_count for c in sample_metadata.columns)
        assert total_nulls == 5


class TestChatLogic:
    """Tests for chat component data preparation logic."""

    def test_suggested_questions_generation(self, sample_metadata):
        """Verify suggested questions are generated from metadata."""
        from frontend.components.chat import _generate_quick_suggestions
        suggestions = _generate_quick_suggestions(sample_metadata)
        assert len(suggestions) > 0
        assert all(isinstance(s, str) for s in suggestions)
        # Should reference actual column names
        col_names = [c.name for c in sample_metadata.columns]
        found_match = any(
            any(col in s for col in col_names)
            for s in suggestions
        )
        assert found_match

    def test_performance_caption_formatting(self, sample_chat_response):
        """Verify performance info is formatted correctly."""
        resp = sample_chat_response
        perf_parts = []
        if resp.latency_ms > 0:
            perf_parts.append(f"{resp.latency_ms:.0f}ms")
        if resp.tokens_used > 0:
            perf_parts.append(f"{resp.tokens_used} tokens")
        if resp.retry_count > 0:
            perf_parts.append(f"{resp.retry_count} retries")

        caption = " · ".join(perf_parts)
        assert "1234ms" in caption or "1235ms" in caption
        assert "150 tokens" in caption
        assert "retries" not in caption  # retry_count is 0


class TestExportLogic:
    """Tests for export component data preparation logic."""

    def test_chat_message_construction_from_history(self, sample_chat_response):
        """Verify ChatMessage objects can be built from history entries."""
        history = [
            {"role": "user", "content": "What is the average age?"},
            {
                "role": "assistant",
                "content": sample_chat_response.content,
                "response": sample_chat_response,
            },
        ]

        messages = []
        for entry in history:
            role_str = entry.get("role", "user")
            role = MessageRole.USER if role_str == "user" else MessageRole.ASSISTANT
            resp = entry.get("response")
            messages.append(ChatMessage(
                session_id="test-session",
                role=role,
                content=entry.get("content", ""),
                generated_code=resp.generated_code if resp else None,
            ))

        assert len(messages) == 2
        assert messages[0].role == MessageRole.USER
        assert messages[1].role == MessageRole.ASSISTANT
        assert messages[1].generated_code == "print(df['age'].mean())"


# ── Integration Sanity Tests ────────────────────────────────────────────────


class TestImportIntegrity:
    """Verify all frontend modules import without errors."""

    def test_import_app(self):
        """The main app module should import cleanly."""
        # We can't call main() without Streamlit, but import should work
        import app  # noqa: F401

    def test_import_frontend_package(self):
        import frontend  # noqa: F401

    def test_import_state(self):
        from frontend import state  # noqa: F401

    def test_import_theme(self):
        from frontend import theme  # noqa: F401

    def test_import_sidebar(self):
        from frontend.components import sidebar  # noqa: F401

    def test_import_chat(self):
        from frontend.components import chat  # noqa: F401

    def test_import_explorer(self):
        from frontend.components import explorer  # noqa: F401

    def test_import_export(self):
        from frontend.components import export  # noqa: F401


class TestServiceIntegration:
    """Verify services can be instantiated (without Ollama)."""

    def test_file_service_creation(self):
        from backend.services.file_service import FileService
        svc = FileService()
        assert svc is not None

    def test_session_service_creation(self):
        from backend.services.session_service import SessionService
        svc = SessionService()
        assert svc is not None

    def test_export_service_creation(self):
        from backend.services.export_service import ExportService
        svc = ExportService()
        assert svc is not None

    def test_visualization_service_creation(self):
        from backend.services.visualization_service import VisualizationService
        svc = VisualizationService()
        assert svc is not None

    def test_analytics_orchestrator_creation(self):
        from backend.analytics.orchestrator import AnalyticsOrchestrator
        orch = AnalyticsOrchestrator()
        assert orch is not None
