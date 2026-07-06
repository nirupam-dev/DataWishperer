"""
Message repository — CRUD operations for chat messages.

Handles persistence and retrieval of user questions and AI responses,
including generated code, execution results, and chart paths.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session as SASession

from backend.core.logging_config import get_logger
from backend.models.database import MessageModel, get_db_session
from backend.models.schemas import ChatMessage, MessageRole

logger = get_logger(__name__)


class MessageRepository:
    """
    Data access object for chat messages.

    Args:
        db: An active SQLAlchemy session.
    """

    def __init__(self, db: Optional[SASession] = None) -> None:
        self._db = db or get_db_session()

    def save(self, message: ChatMessage) -> MessageModel:
        """
        Persist a chat message to the database.

        Args:
            message: The Pydantic ``ChatMessage`` to persist.

        Returns:
            The created ORM ``MessageModel``.
        """
        db_message = MessageModel(
            id=message.id,
            session_id=message.session_id,
            file_id=message.file_id,
            role=message.role.value,
            content=message.content,
            generated_code=message.generated_code,
            execution_result=message.execution_result,
            result_type=message.result_type.value,
            chart_path=message.chart_path,
            tokens_used=message.tokens_used,
            latency_ms=message.latency_ms,
            retry_count=message.retry_count,
            created_at=message.created_at,
        )
        self._db.add(db_message)
        self._db.commit()
        self._db.refresh(db_message)
        logger.debug(
            "Saved message: id=%s, session=%s, role=%s",
            message.id, message.session_id, message.role.value,
        )
        return db_message

    def get_session_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[MessageModel]:
        """
        Retrieve all messages for a session, ordered chronologically.

        Args:
            session_id: The parent session ID.
            limit: Optional maximum number of messages.

        Returns:
            List of ``MessageModel`` instances.
        """
        query = (
            self._db.query(MessageModel)
            .filter(MessageModel.session_id == session_id)
            .order_by(MessageModel.created_at.asc())
        )
        if limit:
            query = query.limit(limit)
        return query.all()

    def get_recent_messages(
        self,
        session_id: str,
        count: int = 6,
    ) -> List[MessageModel]:
        """
        Retrieve the most recent N messages for LLM context windowing.

        Returns messages in chronological order (oldest first among the
        most recent N).

        Args:
            session_id: The parent session ID.
            count: Number of recent messages to retrieve.

        Returns:
            List of ``MessageModel`` in chronological order.
        """
        # Get last N messages (newest first), then reverse for chronological order
        messages = (
            self._db.query(MessageModel)
            .filter(MessageModel.session_id == session_id)
            .order_by(MessageModel.created_at.desc())
            .limit(count)
            .all()
        )
        return list(reversed(messages))

    def get_last_user_message(self, session_id: str) -> Optional[MessageModel]:
        """
        Retrieve the last user message in a session.

        Args:
            session_id: The parent session ID.

        Returns:
            The last user ``MessageModel`` or ``None``.
        """
        return (
            self._db.query(MessageModel)
            .filter(
                MessageModel.session_id == session_id,
                MessageModel.role == MessageRole.USER.value,
            )
            .order_by(MessageModel.created_at.desc())
            .first()
        )

    def count_by_session(self, session_id: str) -> int:
        """
        Count total messages in a session.

        Args:
            session_id: The parent session ID.

        Returns:
            Message count.
        """
        return (
            self._db.query(MessageModel)
            .filter(MessageModel.session_id == session_id)
            .count()
        )

    def count_all(self) -> int:
        """Return the total number of messages across all sessions."""
        return self._db.query(MessageModel).count()

    def delete_by_session(self, session_id: str) -> int:
        """
        Delete all messages belonging to a session.

        Args:
            session_id: The parent session ID.

        Returns:
            Number of messages deleted.
        """
        count = (
            self._db.query(MessageModel)
            .filter(MessageModel.session_id == session_id)
            .delete()
        )
        self._db.commit()
        logger.info("Deleted %d messages from session %s", count, session_id)
        return count

    def close(self) -> None:
        """Close the underlying database session."""
        self._db.close()
