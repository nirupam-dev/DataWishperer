"""
Chat service — Orchestrates the interpreter pipeline via the Agent.

This service bridges the UI/API layer with the DataWhispererAgent,
handling session management, message persistence, and response formatting.

The service layer is intentionally thin — the Agent contains all
intelligence. The service handles only:
    1. Input validation
    2. Session/message persistence (DB)
    3. Agent invocation
    4. Response mapping to ChatResponse schema
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from backend.core.config import ChatSettings, get_settings
from backend.core.exceptions import InvalidQueryError
from backend.core.logging_config import get_logger
from backend.llm.agent import DataWhispererAgent
from backend.models.schemas import (
    ChatMessage,
    ChatResponse,
    FileMetadata,
    MessageRole,
    ResultType,
)
from backend.storage.repositories.message_repo import MessageRepository
from backend.storage.repositories.session_repo import SessionRepository
from backend.utils.helpers import sanitize_user_input

logger = get_logger(__name__)


class ChatService:
    """
    Orchestrates the full question → answer pipeline via the Agent.

    This service is the primary entry point for processing user questions.
    It manages persistence and delegates intelligence to the Agent.

    Args:
        agent: The DataWhisperer AI agent.
        session_repo: Session persistence.
        message_repo: Message persistence.
        chat_settings: Chat configuration.
    """

    def __init__(
        self,
        agent: DataWhispererAgent,
        session_repo: Optional[SessionRepository] = None,
        message_repo: Optional[MessageRepository] = None,
        chat_settings: Optional[ChatSettings] = None,
    ) -> None:
        self._agent = agent
        self._session_repo = session_repo or SessionRepository()
        self._message_repo = message_repo or MessageRepository()

        settings = get_settings()
        self._chat_settings = chat_settings or settings.chat

    @property
    def agent(self) -> DataWhispererAgent:
        """Expose the underlying agent for direct access."""
        return self._agent

    def process_question(
        self,
        session_id: str,
        file_id: str,
        question: str,
        file_metadata: FileMetadata,
        csv_path: str,
    ) -> ChatResponse:
        """
        Process a user question through the interpreter pipeline.

        Pipeline:
            1. Validate and sanitize the question
            2. Save the user message to DB
            3. Load session memory if not already loaded
            4. Invoke the agent (8-stage interpreter pipeline)
            5. Save the assistant response to DB
            6. Touch the session timestamp
            7. Return structured ChatResponse with all interpreter outputs

        Args:
            session_id: The current chat session ID.
            file_id: The associated file ID.
            question: The user's natural language question.
            file_metadata: Metadata about the CSV file.
            csv_path: Disk path to the CSV file.

        Returns:
            A ``ChatResponse`` with all interpreter pipeline outputs:
                - generated_code: The pandas code that was generated
                - result_data: The execution output
                - explanation: Plain-English explanation
                - chart_explanation: Chart type reasoning (if applicable)
                - auto_debug_applied: Whether auto-debug was triggered

        Raises:
            InvalidQueryError: If the question fails validation.
        """
        # 1. Validate the question
        clean_question = self._validate_question(question)

        # 2. Save user message to DB
        user_message = ChatMessage(
            session_id=session_id,
            file_id=file_id,
            role=MessageRole.USER,
            content=clean_question,
        )
        self._message_repo.save(user_message)

        # 3. Ensure session memory is loaded
        self._ensure_memory_loaded(session_id)

        # 4. Invoke the agent (full 8-stage interpreter pipeline)
        result = self._agent.process_question(
            session_id=session_id,
            file_id=file_id,
            question=clean_question,
            csv_path=csv_path,
            file_metadata=file_metadata,
        )

        # 5. Save assistant message to DB
        assistant_message = ChatMessage(
            session_id=session_id,
            file_id=file_id,
            role=MessageRole.ASSISTANT,
            content=result.content,
            generated_code=result.code,
            execution_result=json.dumps(
                result.result_data, default=str
            )[:5000] if result.result_data else None,
            result_type=result.result_type,
            chart_path=result.chart_path,
            tokens_used=result.tokens_used,
            latency_ms=result.latency_ms,
            retry_count=result.attempts - 1,
        )
        self._message_repo.save(assistant_message)

        # 6. Touch session timestamp
        self._session_repo.touch(session_id)

        logger.info(
            "Question processed: session=%s, success=%s, attempts=%d, "
            "debug=%s, %.0fms",
            session_id,
            result.success,
            result.attempts,
            result.auto_debug_applied,
            result.latency_ms,
        )

        # 7. Return structured response with all interpreter outputs
        return ChatResponse(
            message_id=assistant_message.id,
            content=result.content,
            generated_code=result.code,
            result_type=result.result_type,
            result_data=result.result_data,
            chart_path=result.chart_path,
            explanation=result.explanation,
            chart_explanation=result.chart_explanation,
            tokens_used=result.tokens_used,
            latency_ms=result.latency_ms,
            retry_count=result.attempts - 1,
            auto_debug_applied=result.auto_debug_applied,
        )

    def get_chat_history(self, session_id: str) -> List[ChatMessage]:
        """
        Retrieve the full chat history for a session.

        Args:
            session_id: The session UUID.

        Returns:
            List of ``ChatMessage`` schemas in chronological order.
        """
        db_messages = self._message_repo.get_session_messages(session_id)
        return [
            ChatMessage(
                id=m.id,
                session_id=m.session_id,
                file_id=m.file_id,
                role=MessageRole(m.role),
                content=m.content,
                generated_code=m.generated_code,
                execution_result=m.execution_result,
                result_type=ResultType(m.result_type),
                chart_path=m.chart_path,
                tokens_used=m.tokens_used,
                latency_ms=m.latency_ms,
                retry_count=m.retry_count,
                created_at=m.created_at,
            )
            for m in db_messages
        ]

    def get_total_questions_count(self) -> int:
        """Return total questions asked across all sessions."""
        return self._message_repo.count_all()

    # ── Private helpers ──────────────────────────────────────────────────

    def _validate_question(self, question: str) -> str:
        """Validate and sanitize the user's question."""
        clean = sanitize_user_input(question)
        if not clean:
            raise InvalidQueryError("Question cannot be empty.")
        if len(clean) > self._chat_settings.max_query_length:
            raise InvalidQueryError(
                f"Question exceeds {self._chat_settings.max_query_length} "
                f"character limit."
            )
        return clean

    def _ensure_memory_loaded(self, session_id: str) -> None:
        """Load session history into agent memory if not already present."""
        if self._agent.memory.get_active_dataset(session_id) is not None:
            # Memory already initialized for this session
            return

        # Load from DB
        db_messages = self._message_repo.get_recent_messages(
            session_id,
            count=self._chat_settings.history_window_size,
        )
        if db_messages:
            dict_messages = [
                {"role": m.role, "content": m.content}
                for m in db_messages
            ]
            self._agent.load_session_memory(session_id, dict_messages)
            logger.debug(
                "Loaded %d messages into agent memory for session %s",
                len(dict_messages),
                session_id,
            )
