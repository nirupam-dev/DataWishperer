"""
Conversation memory manager — LangChain memory with persistence.

Manages conversation memory using LangChain's ``ChatMessageHistory``
backed by our existing SQLite repository. This bridges LangChain's
memory interface with DataWhisperer's persistence layer.

Architecture Decision:
    We use ``InMemoryChatMessageHistory`` per-session and sync to our
    own ``MessageRepository`` on save. This avoids double-storage while
    giving LangChain proper memory objects for prompt assembly.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from backend.core.config import ChatSettings, get_settings
from backend.core.logging_config import get_logger

logger = get_logger(__name__)


class ConversationMemory:
    """
    Manages LangChain conversation memory per session.

    Maintains an in-memory sliding window of chat history per session,
    formatted as LangChain message objects for direct injection into
    chains and prompts.

    Features:
        - Per-session isolation (supports multiple concurrent sessions)
        - Sliding window to limit token usage
        - Automatic format conversion between LangChain and dict formats
        - Dataset context tagging for multi-dataset support

    Args:
        chat_settings: Chat configuration with window size limits.
    """

    def __init__(
        self,
        chat_settings: Optional[ChatSettings] = None,
    ) -> None:
        self._settings = chat_settings or get_settings().chat
        self._sessions: Dict[str, InMemoryChatMessageHistory] = {}
        self._active_datasets: Dict[str, str] = {}  # session_id → file_id

    def get_or_create(self, session_id: str) -> InMemoryChatMessageHistory:
        """
        Get or create the chat history for a session.

        Args:
            session_id: Unique session identifier.

        Returns:
            The ``InMemoryChatMessageHistory`` for this session.
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = InMemoryChatMessageHistory()
            logger.debug("Created new memory for session %s", session_id)
        return self._sessions[session_id]

    def add_user_message(self, session_id: str, content: str) -> None:
        """
        Add a user message to the session's memory.

        Args:
            session_id: The session to add to.
            content: The user's message content.
        """
        history = self.get_or_create(session_id)
        history.add_user_message(content)
        self._trim_if_needed(session_id)

    def add_assistant_message(self, session_id: str, content: str) -> None:
        """
        Add an assistant message to the session's memory.

        Args:
            session_id: The session to add to.
            content: The assistant's response content.
        """
        history = self.get_or_create(session_id)
        history.add_ai_message(content)
        self._trim_if_needed(session_id)

    def get_langchain_messages(
        self,
        session_id: str,
        window_size: Optional[int] = None,
    ) -> List[BaseMessage]:
        """
        Get the recent messages as LangChain message objects.

        Applies a sliding window to limit context size for the LLM.

        Args:
            session_id: The session to retrieve from.
            window_size: Override the configured window size.

        Returns:
            List of LangChain message objects (most recent N).
        """
        history = self.get_or_create(session_id)
        messages = history.messages
        limit = window_size or self._settings.history_window_size
        return messages[-limit:] if len(messages) > limit else list(messages)

    def get_dict_messages(
        self,
        session_id: str,
        window_size: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """
        Get the recent messages as plain dicts.

        Useful for backward compatibility with non-LangChain components.

        Args:
            session_id: The session to retrieve from.
            window_size: Override the configured window size.

        Returns:
            List of ``{"role": "...", "content": "..."}`` dicts.
        """
        messages = self.get_langchain_messages(session_id, window_size)
        result = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                result.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                result.append({"role": "system", "content": msg.content})
        return result

    def load_from_db_messages(
        self,
        session_id: str,
        db_messages: List[Dict[str, str]],
    ) -> None:
        """
        Initialize session memory from database records.

        Called when resuming a session to restore context.

        Args:
            session_id: The session to populate.
            db_messages: List of ``{"role": "...", "content": "..."}`` dicts
                         from the message repository.
        """
        history = self.get_or_create(session_id)
        history.clear()

        for msg in db_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "user":
                history.add_user_message(content)
            elif role == "assistant":
                history.add_ai_message(content)

        logger.debug(
            "Loaded %d messages into memory for session %s",
            len(db_messages),
            session_id,
        )

    def set_active_dataset(self, session_id: str, file_id: str) -> Optional[str]:
        """
        Track the active dataset for a session (multi-dataset support).

        Args:
            session_id: The session identifier.
            file_id: The new active file ID.

        Returns:
            The previous file_id if a switch occurred, None if first set.
        """
        previous = self._active_datasets.get(session_id)
        self._active_datasets[session_id] = file_id

        if previous and previous != file_id:
            logger.info(
                "Dataset switch: session=%s, %s → %s",
                session_id,
                previous,
                file_id,
            )
            return previous
        return None

    def get_active_dataset(self, session_id: str) -> Optional[str]:
        """
        Get the currently active dataset for a session.

        Args:
            session_id: The session identifier.

        Returns:
            The active file_id, or None if not set.
        """
        return self._active_datasets.get(session_id)

    def clear_session(self, session_id: str) -> None:
        """
        Clear all memory for a session.

        Args:
            session_id: The session to clear.
        """
        if session_id in self._sessions:
            self._sessions[session_id].clear()
            del self._sessions[session_id]
        self._active_datasets.pop(session_id, None)
        logger.debug("Cleared memory for session %s", session_id)

    def get_session_count(self) -> int:
        """Return the number of active sessions in memory."""
        return len(self._sessions)

    # ── Private helpers ──────────────────────────────────────────────────

    def _trim_if_needed(self, session_id: str) -> None:
        """Trim the session's history if it exceeds the maximum."""
        history = self._sessions.get(session_id)
        if not history:
            return

        max_messages = self._settings.max_history_messages
        if len(history.messages) > max_messages:
            # Keep the most recent messages
            trimmed = history.messages[-max_messages:]
            history.clear()
            for msg in trimmed:
                history.add_message(msg)

            logger.debug(
                "Trimmed session %s memory to %d messages",
                session_id,
                max_messages,
            )
