"""
Session service — Manages chat session lifecycle.

Handles creation, retrieval, update, and deletion of chat sessions,
including auto-title generation and timestamp management.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

from backend.core.logging_config import get_logger
from backend.models.schemas import ChatMessage, MessageRole, SessionInfo
from backend.storage.repositories.message_repo import MessageRepository
from backend.storage.repositories.session_repo import SessionRepository

logger = get_logger(__name__)


class SessionService:
    """
    Business logic for chat session management.

    Args:
        session_repo: Session database repository.
        message_repo: Message database repository.
    """

    def __init__(
        self,
        session_repo: Optional[SessionRepository] = None,
        message_repo: Optional[MessageRepository] = None,
    ) -> None:
        self._session_repo = session_repo or SessionRepository()
        self._message_repo = message_repo or MessageRepository()

    def create_session(
        self,
        file_id: str,
        title: str = "Untitled Session",
    ) -> str:
        """
        Create a new chat session.

        Args:
            file_id: ID of the associated CSV file.
            title: Session title (can be auto-generated later).

        Returns:
            The new session ID (UUID string).
        """
        session_id = str(uuid4())
        self._session_repo.create(
            session_id=session_id,
            title=title,
            file_id=file_id,
        )
        logger.info("Created session: %s for file %s", session_id, file_id)
        return session_id

    def get_or_create_session(
        self,
        session_id: Optional[str],
        file_id: str,
    ) -> str:
        """
        Get an existing session or create a new one.

        Args:
            session_id: Optional existing session ID.
            file_id: Associated file ID (used for creation).

        Returns:
            The session ID (existing or newly created).
        """
        if session_id:
            session = self._session_repo.get_by_id(session_id)
            if session:
                return session.id

        return self.create_session(file_id=file_id)

    def get_session_info(self, session_id: str) -> SessionInfo:
        """
        Get complete session information.

        Args:
            session_id: The session UUID.

        Returns:
            A ``SessionInfo`` schema with message count and last question.

        Raises:
            InvalidSessionError: If the session doesn't exist.
        """
        session = self._session_repo.get_by_id_or_raise(session_id)
        msg_count = self._message_repo.count_by_session(session_id)
        last_msg = self._message_repo.get_last_user_message(session_id)

        file_name = session.file.original_name if session.file else "Unknown"

        return SessionInfo(
            id=session.id,
            title=session.title,
            file_id=session.file_id or "",
            file_name=file_name,
            message_count=msg_count,
            created_at=session.created_at,
            updated_at=session.updated_at,
            is_active=session.is_active,
            last_question=last_msg.content if last_msg else None,
        )

    def list_sessions(self, limit: int = 50) -> List[SessionInfo]:
        """
        List all active sessions with summary info.

        Args:
            limit: Maximum number of sessions.

        Returns:
            List of ``SessionInfo`` schemas, most recent first.
        """
        sessions = self._session_repo.list_active(limit=limit)
        result: List[SessionInfo] = []

        for session in sessions:
            msg_count = self._message_repo.count_by_session(session.id)
            last_msg = self._message_repo.get_last_user_message(session.id)
            file_name = session.file.original_name if session.file else "Unknown"

            result.append(SessionInfo(
                id=session.id,
                title=session.title,
                file_id=session.file_id or "",
                file_name=file_name,
                message_count=msg_count,
                created_at=session.created_at,
                updated_at=session.updated_at,
                is_active=session.is_active,
                last_question=last_msg.content if last_msg else None,
            ))

        return result

    def update_title(self, session_id: str, title: str) -> None:
        """
        Update a session's title.

        Args:
            session_id: The session UUID.
            title: New title string.
        """
        self._session_repo.update_title(session_id, title)
        logger.info("Updated session title: %s -> '%s'", session_id, title)

    def delete_session(self, session_id: str) -> None:
        """
        Soft-delete a session (mark as inactive).

        Args:
            session_id: The session UUID.
        """
        self._session_repo.soft_delete(session_id)

    def search_sessions(self, query: str) -> List[SessionInfo]:
        """
        Search sessions by title.

        Args:
            query: Search string.

        Returns:
            Matching sessions.
        """
        sessions = self._session_repo.search(query)
        return [
            SessionInfo(
                id=s.id,
                title=s.title,
                file_id=s.file_id or "",
                file_name=s.file.original_name if s.file else "Unknown",
                message_count=self._message_repo.count_by_session(s.id),
                created_at=s.created_at,
                updated_at=s.updated_at,
                is_active=s.is_active,
            )
            for s in sessions
        ]

    def get_session_count(self) -> int:
        """Return the number of active sessions."""
        return self._session_repo.count_active()
