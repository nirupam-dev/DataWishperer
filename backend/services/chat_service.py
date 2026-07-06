"""
Chat service — The core orchestrator for the question-to-answer pipeline.

This is the most important service in the application. It coordinates:
    1. Session management
    2. Chat history retrieval
    3. LLM code generation
    4. Sandboxed code execution
    5. Result formatting
    6. Retry logic with error context
    7. Message persistence
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.core.config import ChatSettings, SandboxSettings, get_settings
from backend.core.exceptions import (
    CodeValidationError,
    DataWhispererError,
    ExecutionRuntimeError,
    ExecutionTimeoutError,
    InvalidQueryError,
)
from backend.core.logging_config import get_logger
from backend.llm.base_provider import BaseLLMProvider
from backend.llm.chains.query_chain import QueryChain
from backend.models.schemas import (
    ChatMessage,
    ChatResponse,
    CodeExecutionResult,
    FileMetadata,
    MessageRole,
    ResultType,
)
from backend.sandbox.executor import SandboxExecutor
from backend.storage.repositories.message_repo import MessageRepository
from backend.storage.repositories.session_repo import SessionRepository
from backend.utils.helpers import sanitize_user_input

logger = get_logger(__name__)


class ChatService:
    """
    Orchestrates the full question → answer pipeline.

    This service is the primary entry point for processing user questions.
    It manages the entire lifecycle of a query, including retries.

    Args:
        llm_provider: LLM backend for code generation.
        query_chain: Chain for prompt assembly and LLM interaction.
        sandbox_executor: Secure code execution engine.
        session_repo: Session persistence.
        message_repo: Message persistence.
        chat_settings: Chat configuration.
        sandbox_settings: Sandbox configuration.
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        query_chain: Optional[QueryChain] = None,
        sandbox_executor: Optional[SandboxExecutor] = None,
        session_repo: Optional[SessionRepository] = None,
        message_repo: Optional[MessageRepository] = None,
        chat_settings: Optional[ChatSettings] = None,
        sandbox_settings: Optional[SandboxSettings] = None,
    ) -> None:
        self._llm = llm_provider
        self._chain = query_chain or QueryChain(llm_provider)
        self._sandbox = sandbox_executor or SandboxExecutor()
        self._session_repo = session_repo or SessionRepository()
        self._message_repo = message_repo or MessageRepository()

        settings = get_settings()
        self._chat_settings = chat_settings or settings.chat
        self._sandbox_settings = sandbox_settings or settings.sandbox

    def process_question(
        self,
        session_id: str,
        file_id: str,
        question: str,
        file_metadata: FileMetadata,
        csv_path: str,
    ) -> ChatResponse:
        """
        Process a user question through the complete pipeline.

        Pipeline:
            1. Validate and sanitize the question
            2. Save the user message to history
            3. Retrieve recent chat history for context
            4. Generate code via LLM (with retry loop)
            5. Execute code in sandbox
            6. Format and save the assistant response

        Args:
            session_id: The current chat session ID.
            file_id: The associated file ID.
            question: The user's natural language question.
            file_metadata: Metadata about the CSV file.
            csv_path: Disk path to the CSV file.

        Returns:
            A ``ChatResponse`` with the analysis result.

        Raises:
            InvalidQueryError: If the question fails validation.
            DataWhispererError: On any pipeline failure after all retries.
        """
        start_time = time.time()

        # 1. Validate the question
        clean_question = self._validate_question(question)

        # 2. Save user message
        user_message = ChatMessage(
            session_id=session_id,
            file_id=file_id,
            role=MessageRole.USER,
            content=clean_question,
        )
        self._message_repo.save(user_message)

        # 3. Get chat history for context
        history = self._get_history_for_context(session_id)

        # 4-5. Generate and execute with retry loop
        max_retries = self._sandbox_settings.max_retries
        last_error: Optional[Exception] = None
        error_context: Optional[str] = None

        for attempt in range(1, max_retries + 2):  # +2 because range is exclusive and attempt is 1-indexed
            try:
                # Generate code
                code, llm_response = self._chain.generate_code(
                    question=clean_question,
                    file_metadata=file_metadata,
                    history=history,
                    error_context=error_context,
                    attempt=attempt,
                )

                # Execute in sandbox
                execution_result = self._sandbox.execute(
                    code=code,
                    csv_path=csv_path,
                )

                # Format the response content
                content = self._format_result(execution_result)

                elapsed_ms = round((time.time() - start_time) * 1000, 2)

                # Save assistant message
                assistant_message = ChatMessage(
                    session_id=session_id,
                    file_id=file_id,
                    role=MessageRole.ASSISTANT,
                    content=content,
                    generated_code=code,
                    execution_result=json.dumps(
                        execution_result.data, default=str
                    )[:5000] if execution_result.data else None,
                    result_type=execution_result.result_type,
                    chart_path=execution_result.chart_path,
                    tokens_used=llm_response.tokens_used,
                    latency_ms=elapsed_ms,
                    retry_count=attempt - 1,
                )
                self._message_repo.save(assistant_message)

                # Touch session timestamp
                self._session_repo.touch(session_id)

                logger.info(
                    "Question processed: session=%s, attempts=%d, %.0fms",
                    session_id, attempt, elapsed_ms,
                )

                return ChatResponse(
                    message_id=assistant_message.id,
                    content=content,
                    generated_code=code,
                    result_type=execution_result.result_type,
                    result_data=execution_result.data,
                    chart_path=execution_result.chart_path,
                    tokens_used=llm_response.tokens_used,
                    latency_ms=elapsed_ms,
                    retry_count=attempt - 1,
                )

            except (CodeValidationError, ExecutionRuntimeError) as e:
                last_error = e
                logger.warning(
                    "Attempt %d/%d failed: %s",
                    attempt, max_retries + 1, e.message,
                )
                if attempt <= max_retries:
                    error_context = self._chain.build_error_context(e, file_metadata)
                continue

            except ExecutionTimeoutError as e:
                last_error = e
                logger.warning("Execution timed out on attempt %d", attempt)
                if attempt <= max_retries:
                    error_context = self._chain.build_error_context(e, file_metadata)
                continue

        # All retries exhausted
        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        error_content = self._format_error(last_error)

        error_message = ChatMessage(
            session_id=session_id,
            file_id=file_id,
            role=MessageRole.ASSISTANT,
            content=error_content,
            result_type=ResultType.ERROR,
            latency_ms=elapsed_ms,
            retry_count=max_retries,
        )
        self._message_repo.save(error_message)

        return ChatResponse(
            message_id=error_message.id,
            content=error_content,
            result_type=ResultType.ERROR,
            latency_ms=elapsed_ms,
            retry_count=max_retries,
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
                f"Question exceeds {self._chat_settings.max_query_length} character limit."
            )
        return clean

    def _get_history_for_context(self, session_id: str) -> List[Dict[str, str]]:
        """Retrieve recent messages formatted for LLM context."""
        window = self._chat_settings.history_window_size
        messages = self._message_repo.get_recent_messages(session_id, count=window)
        return [
            {"role": m.role, "content": m.content}
            for m in messages
        ]

    @staticmethod
    def _format_result(result: CodeExecutionResult) -> str:
        """Format an execution result into a user-friendly response."""
        if result.result_type == ResultType.CHART:
            text_data = result.data if result.data else ""
            if text_data:
                return f"📊 Chart generated.\n\n{text_data}"
            return "📊 Chart generated successfully."

        if result.result_type == ResultType.DATAFRAME:
            try:
                df = pd.read_json(result.data)
                return f"📋 Results ({len(df)} rows):\n\n{df.to_string(index=False)}"
            except Exception:
                return f"📋 Results:\n\n{result.data}"

        if result.result_type == ResultType.SERIES:
            try:
                series = pd.read_json(result.data, typ="series")
                return f"📋 Results:\n\n{series.to_string()}"
            except Exception:
                return f"📋 Results:\n\n{result.data}"

        return str(result.data) if result.data else "Analysis complete."

    @staticmethod
    def _format_error(error: Optional[Exception]) -> str:
        """Format an error into a user-friendly message."""
        if error is None:
            return "❌ An unknown error occurred. Please try rephrasing your question."

        if isinstance(error, DataWhispererError):
            msg = f"❌ {error.message}"
            if error.suggestion:
                msg += f"\n\n💡 {error.suggestion}"
            return msg

        return f"❌ An unexpected error occurred: {str(error)[:200]}"
