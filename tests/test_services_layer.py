"""
Tests for the service layer: ChatService, FileService, SessionService,
ExportService, VisualizationService.

Covers all 0%-covered service modules and boosts ChatService from 40%.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, PropertyMock, patch, create_autospec
from uuid import uuid4

import pandas as pd
import pytest

from backend.core.config import ChatSettings, StorageSettings, get_settings
from backend.core.exceptions import InvalidQueryError
from backend.llm.agent import AgentResult, DataWhispererAgent
from backend.llm.memory import ConversationMemory
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
    SessionInfo,
)
from backend.services.chat_service import ChatService
from backend.services.export_service import ExportService
from backend.services.file_service import FileService
from backend.services.session_service import SessionService
from backend.services.visualization_service import VisualizationService


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_agent():
    agent = MagicMock(spec=DataWhispererAgent)
    agent.memory = MagicMock(spec=ConversationMemory)
    agent.memory.get_active_dataset.return_value = "file-1"
    agent.process_question.return_value = AgentResult(
        success=True, content="Answer", code="result = df.mean()",
        result_type=ResultType.TEXT, result_data="42",
        tokens_used=100, latency_ms=500.0, attempts=1,
    )
    return agent


@pytest.fixture
def mock_session_repo():
    repo = MagicMock()
    repo.touch.return_value = None
    return repo


@pytest.fixture
def mock_message_repo():
    repo = MagicMock()
    repo.save.return_value = MagicMock()
    repo.get_recent_messages.return_value = []
    repo.count_all.return_value = 5
    repo.count_by_session.return_value = 2
    repo.get_last_user_message.return_value = None
    return repo


@pytest.fixture
def chat_service(mock_agent, mock_session_repo, mock_message_repo):
    return ChatService(
        agent=mock_agent,
        session_repo=mock_session_repo,
        message_repo=mock_message_repo,
    )


@pytest.fixture
def sample_metadata():
    return FileMetadata(
        file_id="file-1", original_name="data.csv",
        stored_path="/tmp/data.csv", row_count=100, col_count=3,
        file_size_bytes=5000, memory_usage_mb=0.05,
        columns=[
            ColumnInfo(name="a", dtype="int64", non_null_count=100,
                       null_count=0, unique_count=100, sample_values=["1"]),
        ],
    )


# ── ChatService Tests ────────────────────────────────────────────────────────


class TestChatService:
    """Tests for ChatService orchestration."""

    def test_process_question_success(self, chat_service, sample_metadata):
        resp = chat_service.process_question(
            session_id="s1", file_id="file-1",
            question="What is the mean?",
            file_metadata=sample_metadata, csv_path="/tmp/data.csv",
        )
        assert isinstance(resp, ChatResponse)
        assert resp.content == "Answer"

    def test_process_question_calls_agent(self, chat_service, mock_agent, sample_metadata):
        chat_service.process_question(
            "s1", "file-1", "avg?", sample_metadata, "/tmp/data.csv",
        )
        mock_agent.process_question.assert_called_once()

    def test_process_question_saves_messages(self, chat_service, mock_message_repo, sample_metadata):
        chat_service.process_question(
            "s1", "file-1", "avg?", sample_metadata, "/tmp/data.csv",
        )
        assert mock_message_repo.save.call_count == 2  # user + assistant

    def test_validate_empty_question(self, chat_service, sample_metadata):
        with pytest.raises(InvalidQueryError):
            chat_service.process_question(
                "s1", "file-1", "   ", sample_metadata, "/tmp/data.csv",
            )

    def test_validate_long_question(self, chat_service, sample_metadata):
        with pytest.raises(InvalidQueryError):
            chat_service.process_question(
                "s1", "file-1", "x" * 10000, sample_metadata, "/tmp/data.csv",
            )

    def test_agent_property(self, chat_service, mock_agent):
        assert chat_service.agent is mock_agent

    def test_total_questions_count(self, chat_service, mock_message_repo):
        assert chat_service.get_total_questions_count() == 5

    def test_ensure_memory_loaded_skips_if_active(self, chat_service, mock_agent):
        mock_agent.memory.get_active_dataset.return_value = "file-1"
        chat_service._ensure_memory_loaded("s1")
        mock_agent.load_session_memory.assert_not_called()

    def test_ensure_memory_loaded_loads_from_db(self, chat_service, mock_agent, mock_message_repo):
        mock_agent.memory.get_active_dataset.return_value = None
        m = MagicMock()
        m.role = "user"
        m.content = "hello"
        mock_message_repo.get_recent_messages.return_value = [m]
        chat_service._ensure_memory_loaded("s1")
        mock_agent.load_session_memory.assert_called_once()

    def test_get_chat_history(self, chat_service, mock_message_repo):
        m = MagicMock()
        m.id = "m1"
        m.session_id = "s1"
        m.file_id = "f1"
        m.role = "user"
        m.content = "hi"
        m.generated_code = None
        m.execution_result = None
        m.result_type = "text"
        m.chart_path = None
        m.tokens_used = 0
        m.latency_ms = 0.0
        m.retry_count = 0
        m.created_at = datetime.utcnow()
        mock_message_repo.get_session_messages.return_value = [m]
        history = chat_service.get_chat_history("s1")
        assert len(history) == 1


# ── ExportService Tests ──────────────────────────────────────────────────────


class TestExportService:
    """Tests for ExportService transcript export."""

    @pytest.fixture
    def export_service(self, tmp_path):
        settings = StorageSettings(export_dir=str(tmp_path / "exports"))
        return ExportService(storage_settings=settings)

    @pytest.fixture
    def sample_messages(self):
        return [
            ChatMessage(
                session_id="s1", role=MessageRole.USER,
                content="What is the average?",
            ),
            ChatMessage(
                session_id="s1", role=MessageRole.ASSISTANT,
                content="The average is 42.",
                generated_code="result = df.mean()",
            ),
        ]

    def test_export_markdown(self, export_service, sample_messages):
        result = export_service.export_transcript(
            "Test Session", sample_messages, ExportFormat.MARKDOWN,
        )
        assert isinstance(result, ExportResult)
        assert result.format == ExportFormat.MARKDOWN
        content = Path(result.filepath).read_text(encoding="utf-8")
        assert "Test Session" in content
        assert "average" in content

    def test_export_markdown_includes_code(self, export_service, sample_messages):
        result = export_service.export_transcript(
            "Test", sample_messages, ExportFormat.MARKDOWN, include_code=True,
        )
        content = Path(result.filepath).read_text(encoding="utf-8")
        assert "df.mean()" in content

    def test_export_markdown_excludes_code(self, export_service, sample_messages):
        result = export_service.export_transcript(
            "Test", sample_messages, ExportFormat.MARKDOWN, include_code=False,
        )
        content = Path(result.filepath).read_text(encoding="utf-8")
        assert "df.mean()" not in content

    def test_export_json(self, export_service, sample_messages):
        result = export_service.export_transcript(
            "Test Session", sample_messages, ExportFormat.JSON,
        )
        assert result.format == ExportFormat.JSON
        data = json.loads(Path(result.filepath).read_text(encoding="utf-8"))
        assert data["message_count"] == 2
        assert len(data["messages"]) == 2

    def test_export_json_includes_code(self, export_service, sample_messages):
        result = export_service.export_transcript(
            "Test", sample_messages, ExportFormat.JSON, include_code=True,
        )
        data = json.loads(Path(result.filepath).read_text(encoding="utf-8"))
        assert any("generated_code" in m for m in data["messages"])

    def test_export_default_format(self, export_service, sample_messages):
        result = export_service.export_transcript(
            "Test", sample_messages, ExportFormat.CSV,
        )
        assert result.format == ExportFormat.MARKDOWN  # Falls back

    def test_list_exports(self, export_service, sample_messages):
        export_service.export_transcript("A", sample_messages)
        export_service.export_transcript("B", sample_messages)
        exports = export_service.list_exports()
        assert len(exports) == 2

    def test_get_export_path_exists(self, export_service, sample_messages):
        result = export_service.export_transcript("Test", sample_messages)
        path = export_service.get_export_path(result.filename)
        assert path is not None

    def test_get_export_path_not_found(self, export_service):
        assert export_service.get_export_path("nonexistent.md") is None


# ── FileService Tests ────────────────────────────────────────────────────────


class TestFileService:
    """Tests for FileService upload and management."""

    @pytest.fixture
    def file_service(self, tmp_path):
        mock_fm = MagicMock()
        mock_fm.save_file.return_value = ("fid", tmp_path / "stored.csv")
        mock_fr = MagicMock()
        mock_ca = MagicMock()
        mock_ca.analyze.return_value = FileMetadata(
            file_id="fid", original_name="test.csv",
            stored_path=str(tmp_path / "stored.csv"),
            row_count=50, col_count=2, file_size_bytes=1000,
            memory_usage_mb=0.01, columns=[],
        )
        mock_ca.get_preview_rows.return_value = [{"a": "1", "b": "2"}]
        mock_ca.get_data_quality_report.return_value = {"completeness_pct": 100}
        return FileService(
            file_manager=mock_fm, file_repo=mock_fr,
            csv_analyzer=mock_ca,
        )

    @patch("backend.services.file_service.validate_upload")
    def test_upload_file(self, mock_validate, file_service):
        resp = file_service.upload_file("test.csv", b"a,b\n1,2\n")
        assert isinstance(resp, FileUploadResponse)
        assert resp.file_id == "fid"
        assert resp.row_count == 50

    @patch("backend.services.file_service.validate_upload")
    def test_upload_cleans_up_on_failure(self, mock_validate, file_service):
        file_service._csv_analyzer.analyze.side_effect = ValueError("bad csv")
        with pytest.raises(ValueError):
            file_service.upload_file("test.csv", b"bad")
        file_service._file_manager.delete_file.assert_called_once()

    def test_get_file_metadata(self, file_service):
        file_service._file_repo.get_metadata.return_value = MagicMock()
        file_service.get_file_metadata("fid")
        file_service._file_repo.get_metadata.assert_called_with("fid")

    def test_get_file_path(self, file_service):
        meta = MagicMock()
        meta.stored_path = "/tmp/x.csv"
        file_service._file_repo.get_metadata.return_value = meta
        assert file_service.get_file_path("fid") == "/tmp/x.csv"

    def test_get_preview_rows(self, file_service):
        meta = MagicMock()
        meta.stored_path = "/tmp/x.csv"
        file_service._file_repo.get_metadata.return_value = meta
        rows = file_service.get_preview_rows("fid")
        assert len(rows) == 1

    def test_get_data_quality_report(self, file_service):
        meta = MagicMock()
        meta.stored_path = "/tmp/x.csv"
        file_service._file_repo.get_metadata.return_value = meta
        report = file_service.get_data_quality_report("fid")
        assert report["completeness_pct"] == 100

    def test_list_files(self, file_service):
        m1 = MagicMock()
        m1.id = "f1"
        m1.original_name = "a.csv"
        m1.row_count = 10
        m1.col_count = 2
        m1.file_size_bytes = 100
        m1.uploaded_at = datetime.utcnow()
        file_service._file_repo.list_all.return_value = [m1]
        result = file_service.list_files()
        assert len(result) == 1

    def test_delete_file(self, file_service):
        db_file = MagicMock()
        db_file.stored_path = "/tmp/x.csv"
        db_file.original_name = "x.csv"
        file_service._file_repo.get_by_id_or_raise.return_value = db_file
        file_service.delete_file("fid")
        file_service._file_manager.delete_file.assert_called()

    def test_get_disk_usage(self, file_service):
        file_service._file_manager.get_disk_usage_mb.return_value = 1.5
        assert file_service.get_disk_usage_mb() == 1.5

    def test_get_file_count(self, file_service):
        file_service._file_repo.count_all.return_value = 3
        assert file_service.get_file_count() == 3


# ── SessionService Tests ─────────────────────────────────────────────────────


class TestSessionService:
    """Tests for SessionService lifecycle."""

    @pytest.fixture
    def session_service(self):
        mock_sr = MagicMock()
        mock_mr = MagicMock()
        mock_mr.count_by_session.return_value = 5
        mock_mr.get_last_user_message.return_value = None
        return SessionService(session_repo=mock_sr, message_repo=mock_mr)

    def test_create_session(self, session_service):
        sid = session_service.create_session("file-1", "My Session")
        assert isinstance(sid, str)
        session_service._session_repo.create.assert_called_once()

    def test_get_or_create_existing(self, session_service):
        existing = MagicMock()
        existing.id = "s1"
        session_service._session_repo.get_by_id.return_value = existing
        result = session_service.get_or_create_session("s1", "f1")
        assert result == "s1"

    def test_get_or_create_new(self, session_service):
        session_service._session_repo.get_by_id.return_value = None
        result = session_service.get_or_create_session("bad-id", "f1")
        assert isinstance(result, str)

    def test_get_or_create_none_id(self, session_service):
        result = session_service.get_or_create_session(None, "f1")
        assert isinstance(result, str)

    def test_get_session_info(self, session_service):
        s = MagicMock()
        s.id = "s1"
        s.title = "Test"
        s.file_id = "f1"
        s.file = MagicMock()
        s.file.original_name = "data.csv"
        s.created_at = datetime.utcnow()
        s.updated_at = datetime.utcnow()
        s.is_active = True
        session_service._session_repo.get_by_id_or_raise.return_value = s
        info = session_service.get_session_info("s1")
        assert isinstance(info, SessionInfo)
        assert info.file_name == "data.csv"

    def test_get_session_info_no_file(self, session_service):
        s = MagicMock()
        s.id = "s1"
        s.title = "Test"
        s.file_id = "f1"
        s.file = None
        s.created_at = datetime.utcnow()
        s.updated_at = datetime.utcnow()
        s.is_active = True
        session_service._session_repo.get_by_id_or_raise.return_value = s
        info = session_service.get_session_info("s1")
        assert info.file_name == "Unknown"

    def test_list_sessions(self, session_service):
        s = MagicMock()
        s.id = "s1"
        s.title = "T"
        s.file_id = "f1"
        s.file = MagicMock()
        s.file.original_name = "x.csv"
        s.created_at = datetime.utcnow()
        s.updated_at = datetime.utcnow()
        s.is_active = True
        session_service._session_repo.list_active.return_value = [s]
        sessions = session_service.list_sessions()
        assert len(sessions) == 1

    def test_update_title(self, session_service):
        session_service.update_title("s1", "New")
        session_service._session_repo.update_title.assert_called_with("s1", "New")

    def test_delete_session(self, session_service):
        session_service.delete_session("s1")
        session_service._session_repo.soft_delete.assert_called_with("s1")

    def test_search_sessions(self, session_service):
        s = MagicMock()
        s.id = "s1"
        s.title = "Revenue"
        s.file_id = "f1"
        s.file = MagicMock()
        s.file.original_name = "x.csv"
        s.created_at = datetime.utcnow()
        s.updated_at = datetime.utcnow()
        s.is_active = True
        session_service._session_repo.search.return_value = [s]
        results = session_service.search_sessions("Revenue")
        assert len(results) == 1

    def test_get_session_count(self, session_service):
        session_service._session_repo.count_active.return_value = 3
        assert session_service.get_session_count() == 3


# ── VisualizationService Tests ───────────────────────────────────────────────


class TestVisualizationService:
    """Tests for VisualizationService chart management."""

    @pytest.fixture
    def viz_service(self, tmp_path):
        settings = StorageSettings(
            charts_dir=str(tmp_path / "charts"),
            export_dir=str(tmp_path / "exports"),
        )
        return VisualizationService(storage_settings=settings)

    def test_get_chart_path_nonexistent(self, viz_service):
        assert viz_service.get_chart_path("nope.png") is None

    def test_get_chart_path_exists(self, viz_service, tmp_path):
        chart = tmp_path / "charts" / "test.png"
        chart.write_bytes(b"\x89PNG")
        result = viz_service.get_chart_path("test.png")
        assert result is not None

    def test_list_charts_empty(self, viz_service):
        assert viz_service.list_charts() == []

    def test_list_charts(self, viz_service, tmp_path):
        (tmp_path / "charts" / "a.png").write_bytes(b"\x89PNG")
        (tmp_path / "charts" / "b.png").write_bytes(b"\x89PNG")
        charts = viz_service.list_charts()
        assert len(charts) == 2

    def test_cleanup_old_charts(self, viz_service, tmp_path):
        import time
        for i in range(5):
            (tmp_path / "charts" / f"c{i}.png").write_bytes(b"\x89PNG")
            time.sleep(0.01)
        deleted = viz_service.cleanup_old_charts(keep_count=2)
        assert deleted == 3

    def test_get_chart_count(self, viz_service, tmp_path):
        (tmp_path / "charts" / "a.png").write_bytes(b"\x89PNG")
        assert viz_service.get_chart_count() == 1

    def test_get_plotly_theme(self, viz_service):
        theme = viz_service.get_plotly_theme()
        assert isinstance(theme, dict)

    def test_export_chart_png(self, viz_service, tmp_path):
        chart = tmp_path / "charts" / "test.png"
        chart.write_bytes(b"\x89PNG")
        result = viz_service.export_chart(str(chart), format="png")
        assert result is not None

    def test_export_chart_html(self, viz_service):
        plotly_data = {"data": [{"x": [1, 2], "y": [3, 4], "type": "scatter"}], "layout": {}}
        result = viz_service.export_chart("dummy.png", format="html", plotly_json=plotly_data)
        assert result is not None

    def test_export_chart_unsupported(self, viz_service):
        result = viz_service.export_chart("dummy.png", format="svg")
        assert result is None

    def test_selector_property(self, viz_service):
        assert viz_service.selector is not None

    def test_generator_property(self, viz_service):
        assert viz_service.generator is not None

    def test_theme_manager_property(self, viz_service):
        assert viz_service.theme_manager is not None
