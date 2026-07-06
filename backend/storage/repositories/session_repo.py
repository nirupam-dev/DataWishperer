"""
Session repository — CRUD operations for chat sessions.

Implements the Repository pattern, isolating all database access for the
``sessions`` table behind a clean interface. Business logic in
``SessionService`` never touches SQLAlchemy directly.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session as SASession

from backend.core.exceptions import InvalidSessionError
from backend.core.logging_config import get_logger
from backend.models.database import SessionModel, get_db_session

logger = get_logger(__name__)


class SessionRepository:
    """
    Data access object for chat sessions.

    Args:
        db: An active SQLAlchemy session. If ``None``, a new session is
            created automatically.
    """

    def __init__(self, db: Optional[SASession] = None) -> None:
        self._db = db or get_db_session()

    # ── Create ───────────────────────────────────────────────────────────

    def create(
        self,
        session_id: str,
        title: str = "Untitled Session",
        file_id: Optional[str] = None,
    ) -> SessionModel:
        """
        Create a new chat session.

        Args:
            session_id: Unique session identifier (UUID).
            title: Human-readable session title.
            file_id: Optional associated file ID.

        Returns:
            The newly created ``SessionModel``.
        """
        session = SessionModel(
            id=session_id,
            title=title,
            file_id=file_id,
        )
        self._db.add(session)
        self._db.commit()
        self._db.refresh(session)
        logger.info("Created session: id=%s, title='%s'", session_id, title)
        return session

    # ── Read ─────────────────────────────────────────────────────────────

    def get_by_id(self, session_id: str) -> Optional[SessionModel]:
        """
        Retrieve a session by its ID.

        Args:
            session_id: The session UUID.

        Returns:
            The ``SessionModel`` or ``None`` if not found.
        """
        return self._db.query(SessionModel).filter(SessionModel.id == session_id).first()

    def get_by_id_or_raise(self, session_id: str) -> SessionModel:
        """
        Retrieve a session by ID or raise ``InvalidSessionError``.

        Args:
            session_id: The session UUID.

        Returns:
            The ``SessionModel``.

        Raises:
            InvalidSessionError: If no session exists with the given ID.
        """
        session = self.get_by_id(session_id)
        if session is None:
            raise InvalidSessionError(session_id)
        return session

    def list_active(self, limit: int = 50) -> List[SessionModel]:
        """
        List all active sessions, most recent first.

        Args:
            limit: Maximum number of sessions to return.

        Returns:
            List of active ``SessionModel`` instances.
        """
        return (
            self._db.query(SessionModel)
            .filter(SessionModel.is_active == True)
            .order_by(SessionModel.updated_at.desc())
            .limit(limit)
            .all()
        )

    def search(self, query: str, limit: int = 20) -> List[SessionModel]:
        """
        Search sessions by title (case-insensitive partial match).

        Args:
            query: Search string.
            limit: Maximum results.

        Returns:
            Matching sessions ordered by recency.
        """
        return (
            self._db.query(SessionModel)
            .filter(
                SessionModel.is_active == True,
                SessionModel.title.ilike(f"%{query}%"),
            )
            .order_by(SessionModel.updated_at.desc())
            .limit(limit)
            .all()
        )

    def count_active(self) -> int:
        """Return the total number of active sessions."""
        return (
            self._db.query(SessionModel)
            .filter(SessionModel.is_active == True)
            .count()
        )

    # ── Update ───────────────────────────────────────────────────────────

    def update_title(self, session_id: str, title: str) -> SessionModel:
        """
        Update a session's title.

        Args:
            session_id: The session UUID.
            title: New title.

        Returns:
            The updated ``SessionModel``.
        """
        session = self.get_by_id_or_raise(session_id)
        session.title = title
        session.updated_at = datetime.utcnow()
        self._db.commit()
        self._db.refresh(session)
        return session

    def touch(self, session_id: str) -> None:
        """
        Update the ``updated_at`` timestamp to now.

        Args:
            session_id: The session UUID.
        """
        session = self.get_by_id(session_id)
        if session:
            session.updated_at = datetime.utcnow()
            self._db.commit()

    # ── Delete ───────────────────────────────────────────────────────────

    def soft_delete(self, session_id: str) -> None:
        """
        Mark a session as inactive (soft delete).

        Args:
            session_id: The session UUID.
        """
        session = self.get_by_id_or_raise(session_id)
        session.is_active = False
        session.updated_at = datetime.utcnow()
        self._db.commit()
        logger.info("Soft-deleted session: %s", session_id)

    def hard_delete(self, session_id: str) -> None:
        """
        Permanently delete a session and all its messages.

        Args:
            session_id: The session UUID.
        """
        session = self.get_by_id_or_raise(session_id)
        self._db.delete(session)
        self._db.commit()
        logger.info("Hard-deleted session: %s", session_id)

    def close(self) -> None:
        """Close the underlying database session."""
        self._db.close()
