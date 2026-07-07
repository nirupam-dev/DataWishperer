"""
Tests for the storage layer: database, file_manager, repositories.

Covers:
    - backend.models.database (ORM models, engine factory, pragmas)
    - backend.storage.file_manager (FileManager disk ops)
    - backend.storage.repositories.file_repo (FileRepository CRUD)
    - backend.storage.repositories.session_repo (SessionRepository CRUD)
    - backend.storage.repositories.message_repo (MessageRepository CRUD)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session as SASession, sessionmaker

from backend.core.config import get_settings
from backend.core.exceptions import FileNotFoundError_, InvalidSessionError
from backend.models.database import (
    Base,
    FileModel,
    MessageModel,
    SessionModel,
    get_engine,
    get_session_factory,
    get_db_session,
)
from backend.models.schemas import (
    ChatMessage,
    ColumnInfo,
    FileMetadata,
    MessageRole,
    ResultType,
)
from backend.storage.file_manager import FileManager
from backend.storage.repositories.file_repo import FileRepository
from backend.storage.repositories.message_repo import MessageRepository
from backend.storage.repositories.session_repo import SessionRepository


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def db_session(tmp_path) -> Generator[SASession, None, None]:
    """Create an in-memory SQLite session for isolated DB tests."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = Session()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture
def file_repo(db_session) -> FileRepository:
    return FileRepository(db=db_session)


@pytest.fixture
def session_repo(db_session) -> SessionRepository:
    return SessionRepository(db=db_session)


@pytest.fixture
def message_repo(db_session) -> MessageRepository:
    return MessageRepository(db=db_session)


@pytest.fixture
def file_manager(tmp_path) -> FileManager:
    from backend.core.config import StorageSettings
    settings = StorageSettings(upload_dir=str(tmp_path / "uploads"))
    return FileManager(settings=settings)


@pytest.fixture
def sample_metadata() -> FileMetadata:
    return FileMetadata(
        file_id="file-001",
        original_name="test.csv",
        stored_path="/tmp/test.csv",
        row_count=100,
        col_count=3,
        file_size_bytes=5000,
        memory_usage_mb=0.05,
        columns=[
            ColumnInfo(
                name="id", dtype="int64",
                non_null_count=100, null_count=0,
                unique_count=100, sample_values=["1", "2"],
                mean=50.0, std=29.0, min_val=1.0, max_val=100.0,
            ),
            ColumnInfo(
                name="name", dtype="object",
                non_null_count=100, null_count=0,
                unique_count=80, sample_values=["Alice", "Bob"],
            ),
            ColumnInfo(
                name="value", dtype="float64",
                non_null_count=95, null_count=5,
                unique_count=90, sample_values=["3.14"],
                mean=50.0, std=10.0, min_val=0.0, max_val=100.0,
            ),
        ],
    )


def _create_file_in_db(db_session, file_id="file-001"):
    """Helper to create a FileModel directly in the DB."""
    f = FileModel(
        id=file_id, original_name="test.csv",
        stored_path="/tmp/test.csv", row_count=100, col_count=3,
        file_size_bytes=5000,
        column_metadata=json.dumps([
            {"name": "id", "dtype": "int64", "non_null_count": 100,
             "null_count": 0, "unique_count": 100, "sample_values": ["1"]},
        ]),
    )
    db_session.add(f)
    db_session.commit()
    return f


def _create_session_in_db(db_session, session_id="sess-001", file_id=None):
    """Helper to create a SessionModel directly in the DB."""
    s = SessionModel(id=session_id, title="Test Session", file_id=file_id)
    db_session.add(s)
    db_session.commit()
    return s


# ── ORM Model Tests ──────────────────────────────────────────────────────────


class TestOrmModels:
    """Tests for SQLAlchemy ORM models."""

    def test_session_model_repr(self, db_session):
        s = _create_session_in_db(db_session)
        assert "Session" in repr(s)
        assert s.title == "Test Session"

    def test_file_model_repr(self, db_session):
        f = _create_file_in_db(db_session)
        assert "File" in repr(f)
        assert f.original_name == "test.csv"

    def test_message_model_repr(self, db_session):
        _create_session_in_db(db_session, "s1")
        m = MessageModel(
            id="m1", session_id="s1", role="user",
            content="hello", result_type="text",
        )
        db_session.add(m)
        db_session.commit()
        assert "Message" in repr(m)

    def test_session_file_relationship(self, db_session):
        f = _create_file_in_db(db_session, "f1")
        s = _create_session_in_db(db_session, "s1", file_id="f1")
        db_session.refresh(s)
        assert s.file is not None
        assert s.file.original_name == "test.csv"

    def test_session_messages_cascade(self, db_session):
        _create_session_in_db(db_session, "s1")
        m1 = MessageModel(id="m1", session_id="s1", role="user", content="q1", result_type="text")
        m2 = MessageModel(id="m2", session_id="s1", role="assistant", content="a1", result_type="text")
        db_session.add_all([m1, m2])
        db_session.commit()
        s = db_session.query(SessionModel).get("s1")
        assert len(s.messages) == 2

    def test_message_defaults(self, db_session):
        _create_session_in_db(db_session, "s1")
        m = MessageModel(id="m1", session_id="s1", role="user", content="test")
        db_session.add(m)
        db_session.commit()
        db_session.refresh(m)
        assert m.tokens_used == 0
        assert m.latency_ms == 0.0
        assert m.retry_count == 0
        assert m.result_type == "text"


class TestDatabaseFactory:
    """Tests for get_engine, get_session_factory, get_db_session."""

    def test_get_engine_returns_engine(self):
        import backend.models.database as db_mod
        old_engine = db_mod._engine
        db_mod._engine = None
        try:
            engine = get_engine()
            assert engine is not None
        finally:
            db_mod._engine = old_engine

    def test_get_session_factory_returns_factory(self):
        import backend.models.database as db_mod
        old_factory = db_mod._SessionLocal
        db_mod._SessionLocal = None
        try:
            factory = get_session_factory()
            assert factory is not None
        finally:
            db_mod._SessionLocal = old_factory

    def test_get_db_session_returns_session(self):
        session = get_db_session()
        assert session is not None
        session.close()


# ── FileRepository Tests ─────────────────────────────────────────────────────


class TestFileRepository:
    """Tests for FileRepository CRUD operations."""

    def test_save_and_retrieve(self, file_repo, sample_metadata):
        file_repo.save(sample_metadata)
        result = file_repo.get_by_id("file-001")
        assert result is not None
        assert result.original_name == "test.csv"
        assert result.row_count == 100

    def test_get_metadata_with_columns(self, file_repo, sample_metadata):
        file_repo.save(sample_metadata)
        meta = file_repo.get_metadata("file-001")
        assert isinstance(meta, FileMetadata)
        assert len(meta.columns) == 3
        assert meta.columns[0].name == "id"

    def test_get_by_id_returns_none(self, file_repo):
        assert file_repo.get_by_id("nonexistent") is None

    def test_get_by_id_or_raise_raises(self, file_repo):
        with pytest.raises(FileNotFoundError_):
            file_repo.get_by_id_or_raise("nonexistent")

    def test_get_metadata_not_found(self, file_repo):
        with pytest.raises(FileNotFoundError_):
            file_repo.get_metadata("nonexistent")

    def test_list_all(self, file_repo, sample_metadata):
        file_repo.save(sample_metadata)
        meta2 = sample_metadata.model_copy(update={"file_id": "file-002", "original_name": "b.csv"})
        file_repo.save(meta2)
        files = file_repo.list_all()
        assert len(files) == 2

    def test_count_all(self, file_repo, sample_metadata):
        assert file_repo.count_all() == 0
        file_repo.save(sample_metadata)
        assert file_repo.count_all() == 1

    def test_delete(self, file_repo, sample_metadata):
        file_repo.save(sample_metadata)
        file_repo.delete("file-001")
        assert file_repo.get_by_id("file-001") is None

    def test_delete_nonexistent_raises(self, file_repo):
        with pytest.raises(FileNotFoundError_):
            file_repo.delete("nonexistent")

    def test_close(self, file_repo):
        file_repo.close()


# ── SessionRepository Tests ──────────────────────────────────────────────────


class TestSessionRepository:
    """Tests for SessionRepository CRUD operations."""

    def test_create_session(self, session_repo):
        s = session_repo.create("s1", title="My Session")
        assert s.id == "s1"
        assert s.title == "My Session"
        assert s.is_active is True

    def test_get_by_id(self, session_repo):
        session_repo.create("s1")
        result = session_repo.get_by_id("s1")
        assert result is not None

    def test_get_by_id_returns_none(self, session_repo):
        assert session_repo.get_by_id("none") is None

    def test_get_by_id_or_raise(self, session_repo):
        session_repo.create("s1")
        s = session_repo.get_by_id_or_raise("s1")
        assert s.id == "s1"

    def test_get_by_id_or_raise_error(self, session_repo):
        with pytest.raises(InvalidSessionError):
            session_repo.get_by_id_or_raise("none")

    def test_list_active(self, session_repo):
        session_repo.create("s1", title="A")
        session_repo.create("s2", title="B")
        active = session_repo.list_active()
        assert len(active) == 2

    def test_search(self, session_repo):
        session_repo.create("s1", title="Revenue Analysis")
        session_repo.create("s2", title="Cost Report")
        results = session_repo.search("Revenue")
        assert len(results) == 1
        assert results[0].title == "Revenue Analysis"

    def test_count_active(self, session_repo):
        assert session_repo.count_active() == 0
        session_repo.create("s1")
        assert session_repo.count_active() == 1

    def test_update_title(self, session_repo):
        session_repo.create("s1", title="Old")
        updated = session_repo.update_title("s1", "New Title")
        assert updated.title == "New Title"

    def test_touch(self, session_repo):
        session_repo.create("s1")
        session_repo.touch("s1")
        s = session_repo.get_by_id("s1")
        assert s is not None

    def test_touch_nonexistent(self, session_repo):
        session_repo.touch("none")  # Should not raise

    def test_soft_delete(self, session_repo):
        session_repo.create("s1")
        session_repo.soft_delete("s1")
        s = session_repo.get_by_id("s1")
        assert s.is_active is False

    def test_soft_delete_nonexistent(self, session_repo):
        with pytest.raises(InvalidSessionError):
            session_repo.soft_delete("none")

    def test_hard_delete(self, session_repo):
        session_repo.create("s1")
        session_repo.hard_delete("s1")
        assert session_repo.get_by_id("s1") is None

    def test_hard_delete_nonexistent(self, session_repo):
        with pytest.raises(InvalidSessionError):
            session_repo.hard_delete("none")

    def test_close(self, session_repo):
        session_repo.close()


# ── MessageRepository Tests ──────────────────────────────────────────────────


class TestMessageRepository:
    """Tests for MessageRepository CRUD operations."""

    def _make_msg(self, session_id="s1", role=MessageRole.USER, content="hi"):
        return ChatMessage(
            session_id=session_id, role=role, content=content,
        )

    def test_save_message(self, message_repo, db_session):
        _create_session_in_db(db_session, "s1")
        msg = self._make_msg()
        result = message_repo.save(msg)
        assert result.content == "hi"

    def test_get_session_messages(self, message_repo, db_session):
        _create_session_in_db(db_session, "s1")
        message_repo.save(self._make_msg(content="q1"))
        message_repo.save(self._make_msg(role=MessageRole.ASSISTANT, content="a1"))
        msgs = message_repo.get_session_messages("s1")
        assert len(msgs) == 2

    def test_get_session_messages_with_limit(self, message_repo, db_session):
        _create_session_in_db(db_session, "s1")
        for i in range(5):
            message_repo.save(self._make_msg(content=f"msg{i}"))
        msgs = message_repo.get_session_messages("s1", limit=3)
        assert len(msgs) == 3

    def test_get_recent_messages(self, message_repo, db_session):
        _create_session_in_db(db_session, "s1")
        for i in range(10):
            message_repo.save(self._make_msg(content=f"msg{i}"))
        recent = message_repo.get_recent_messages("s1", count=3)
        assert len(recent) == 3

    def test_get_last_user_message(self, message_repo, db_session):
        _create_session_in_db(db_session, "s1")
        message_repo.save(self._make_msg(content="first"))
        message_repo.save(self._make_msg(role=MessageRole.ASSISTANT, content="reply"))
        message_repo.save(self._make_msg(content="second"))
        last = message_repo.get_last_user_message("s1")
        assert last is not None
        assert last.content == "second"

    def test_get_last_user_message_none(self, message_repo):
        result = message_repo.get_last_user_message("empty")
        assert result is None

    def test_count_by_session(self, message_repo, db_session):
        _create_session_in_db(db_session, "s1")
        assert message_repo.count_by_session("s1") == 0
        message_repo.save(self._make_msg())
        assert message_repo.count_by_session("s1") == 1

    def test_count_all(self, message_repo, db_session):
        _create_session_in_db(db_session, "s1")
        _create_session_in_db(db_session, "s2")
        message_repo.save(self._make_msg("s1"))
        message_repo.save(self._make_msg("s2"))
        assert message_repo.count_all() == 2

    def test_delete_by_session(self, message_repo, db_session):
        _create_session_in_db(db_session, "s1")
        message_repo.save(self._make_msg())
        message_repo.save(self._make_msg())
        count = message_repo.delete_by_session("s1")
        assert count == 2
        assert message_repo.count_by_session("s1") == 0

    def test_close(self, message_repo):
        message_repo.close()


# ── FileManager Tests ────────────────────────────────────────────────────────


class TestFileManager:
    """Tests for FileManager disk operations."""

    def test_save_file(self, file_manager):
        fid, path = file_manager.save_file("test.csv", b"a,b\n1,2\n")
        assert fid is not None
        assert path.exists()
        assert path.read_bytes() == b"a,b\n1,2\n"

    def test_save_file_unique_names(self, file_manager):
        _, p1 = file_manager.save_file("data.csv", b"a\n1")
        _, p2 = file_manager.save_file("data.csv", b"a\n2")
        assert p1 != p2

    def test_get_file_path(self, file_manager):
        _, path = file_manager.save_file("test.csv", b"data")
        result = file_manager.get_file_path(str(path))
        assert result == path

    def test_get_file_path_not_found(self, file_manager):
        with pytest.raises(FileNotFoundError):
            file_manager.get_file_path("/nonexistent/file.csv")

    def test_delete_file(self, file_manager):
        _, path = file_manager.save_file("test.csv", b"data")
        assert file_manager.delete_file(str(path)) is True
        assert not path.exists()

    def test_delete_nonexistent_file(self, file_manager):
        assert file_manager.delete_file("/no/such/file") is False

    def test_disk_usage(self, file_manager):
        file_manager.save_file("a.csv", b"x" * 1000)
        file_manager.save_file("b.csv", b"y" * 2000)
        assert file_manager.get_disk_usage_bytes() >= 3000
        assert file_manager.get_disk_usage_mb() >= 0.0

    def test_list_files(self, file_manager):
        file_manager.save_file("a.csv", b"1")
        file_manager.save_file("b.csv", b"2")
        files = file_manager.list_files()
        assert len(files) == 2

    def test_cleanup_orphaned(self, file_manager):
        _, p1 = file_manager.save_file("a.csv", b"1")
        _, p2 = file_manager.save_file("b.csv", b"2")
        deleted = file_manager.cleanup_orphaned_files({str(p1)})
        assert deleted == 1
        assert p1.exists()
        assert not p2.exists()
