"""
SQLAlchemy ORM models for the DataWhisperer database.

Defines three core tables:
    - ``sessions``: Chat session metadata.
    - ``files``: Uploaded CSV file metadata.
    - ``messages``: Individual chat messages with code and results.

Uses SQLAlchemy 2.0 Mapped Column syntax for type safety.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    Session as SASession,
    sessionmaker,
)

from backend.core.config import get_settings


# ── Base ─────────────────────────────────────────────────────────────────────


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""

    pass


# ── Enable WAL mode and foreign keys for SQLite ─────────────────────────────


@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_conn: object, connection_record: object) -> None:
    """Configure SQLite pragmas on every new connection."""
    cursor = dbapi_conn.cursor()  # type: ignore[union-attr]
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA cache_size=-64000")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


# ── ORM Models ───────────────────────────────────────────────────────────────


class SessionModel(Base):
    """A chat session groups related messages about a CSV file."""

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    title: Mapped[str] = mapped_column(String(255), default="Untitled Session")
    file_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("files.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Relationships
    file: Mapped[Optional["FileModel"]] = relationship(
        "FileModel", back_populates="sessions"
    )
    messages: Mapped[List["MessageModel"]] = relationship(
        "MessageModel",
        back_populates="session",
        order_by="MessageModel.created_at",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Session(id={self.id!r}, title={self.title!r})>"


class FileModel(Base):
    """Metadata for an uploaded CSV file (the actual file lives on disk)."""

    __tablename__ = "files"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    original_name: Mapped[str] = mapped_column(String(255))
    stored_path: Mapped[str] = mapped_column(String(512))
    row_count: Mapped[int] = mapped_column(Integer, default=0)
    col_count: Mapped[int] = mapped_column(Integer, default=0)
    file_size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    column_metadata: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, doc="JSON-serialized column metadata"
    )
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )

    # Relationships
    sessions: Mapped[List["SessionModel"]] = relationship(
        "SessionModel", back_populates="file"
    )

    def __repr__(self) -> str:
        return f"<File(id={self.id!r}, name={self.original_name!r})>"


class MessageModel(Base):
    """A single chat message — either a user question or an AI response."""

    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("sessions.id", ondelete="CASCADE")
    )
    file_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("files.id", ondelete="SET NULL"), nullable=True
    )
    role: Mapped[str] = mapped_column(
        String(20), doc="user | assistant | system"
    )
    content: Mapped[str] = mapped_column(Text)
    generated_code: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    execution_result: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    result_type: Mapped[str] = mapped_column(String(20), default="text")
    chart_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )

    # Relationships
    session: Mapped["SessionModel"] = relationship(
        "SessionModel", back_populates="messages"
    )

    def __repr__(self) -> str:
        return f"<Message(id={self.id!r}, role={self.role!r})>"


# ── Database Engine Factory ──────────────────────────────────────────────────


_engine = None
_SessionLocal = None


def get_engine() -> Engine:
    """
    Return the SQLAlchemy engine singleton.

    Creates the engine and all tables on first call.

    Returns:
        The application database engine.
    """
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(
            settings.storage.database_url,
            echo=settings.debug,
            pool_pre_ping=True,
        )
        Base.metadata.create_all(_engine)
    return _engine


def get_session_factory() -> sessionmaker:
    """
    Return a configured session factory.

    Returns:
        A ``sessionmaker`` bound to the application engine.
    """
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(),
            autocommit=False,
            autoflush=False,
        )
    return _SessionLocal


def get_db_session() -> SASession:
    """
    Create and return a new database session.

    The caller is responsible for committing / closing.

    Returns:
        A new SQLAlchemy ``Session``.
    """
    factory = get_session_factory()
    return factory()
