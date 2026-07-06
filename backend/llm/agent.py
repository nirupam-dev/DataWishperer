"""
DataWhisperer AI Agent — The core agentic orchestrator.

This is the brain of the application. It coordinates all components
into a cohesive, intelligent pipeline that:

    1. Understands CSV schemas via metadata extraction
    2. Interprets user questions via LangChain prompt engineering
    3. Generates pandas code via Ollama/Qwen2.5:7B
    4. Reasons step-by-step internally (hidden from user)
    5. Validates generated code via AST analysis
    6. Retries on failure with error context injection
    7. Returns clean explanations (no code, no CoT)
    8. Maintains conversation memory per session
    9. Supports multiple datasets with context switching
    10. Tracks dataset state for accurate analysis

Architecture:
    The agent follows the ReAct (Reason + Act) pattern:
    - REASON: Internal chain-of-thought before code generation
    - ACT: Generate code, validate, execute
    - OBSERVE: Check execution result
    - RETRY: If failed, inject error context and re-reason

    Unlike a generic LangChain Agent with tools, we use a specialized
    pipeline because:
    - The task is narrow (CSV → pandas code → result)
    - We need deterministic security validation
    - We control the sandbox execution layer
    - Tool-based agents add unnecessary overhead for a focused task
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from backend.core.config import (
    ChatSettings,
    OllamaSettings,
    SandboxSettings,
    get_settings,
)
from backend.core.exceptions import (
    CodeValidationError,
    DataWhispererError,
    ExecutionRuntimeError,
    ExecutionTimeoutError,
    GenerationError,
    OllamaConnectionError,
)
from backend.core.logging_config import get_logger
from backend.llm.chains.output_parser import OutputParser
from backend.llm.chains.query_chain import QueryChain
from backend.llm.memory import ConversationMemory
from backend.llm.prompts.context_builder import ContextBuilder
from backend.llm.providers.ollama_provider import OllamaProvider
from backend.models.schemas import (
    ChatResponse,
    CodeExecutionResult,
    FileMetadata,
    LLMResponse,
    ResultType,
)
from backend.sandbox.executor import SandboxExecutor

logger = get_logger(__name__)


class AgentResult:
    """
    Encapsulates the complete result of an agent execution.

    Contains all information needed by the service layer to persist
    and display the result, without leaking internal details.

    Attributes:
        success: Whether the agent completed successfully.
        content: User-facing text response.
        code: The generated Python code (for display in UI).
        result_type: Type of result (text, dataframe, chart, error).
        result_data: The actual data (DataFrame JSON, text, etc.).
        chart_path: Path to generated chart image, if any.
        explanation: User-facing explanation of the results.
        tokens_used: Total tokens consumed across all attempts.
        latency_ms: Total wall-clock time in milliseconds.
        attempts: Number of generation attempts made.
        internal_reasoning: Internal CoT (NEVER show to user).
    """

    def __init__(
        self,
        success: bool,
        content: str,
        code: Optional[str] = None,
        result_type: ResultType = ResultType.TEXT,
        result_data: Optional[Any] = None,
        chart_path: Optional[str] = None,
        explanation: Optional[str] = None,
        tokens_used: int = 0,
        latency_ms: float = 0.0,
        attempts: int = 1,
        internal_reasoning: Optional[str] = None,
    ) -> None:
        self.success = success
        self.content = content
        self.code = code
        self.result_type = result_type
        self.result_data = result_data
        self.chart_path = chart_path
        self.explanation = explanation
        self.tokens_used = tokens_used
        self.latency_ms = latency_ms
        self.attempts = attempts
        self.internal_reasoning = internal_reasoning  # NEVER expose to user


class DataWhispererAgent:
    """
    The core AI agent that processes user questions about CSV data.

    Implements a specialized ReAct loop:
        REASON → GENERATE → VALIDATE → EXECUTE → OBSERVE → (RETRY)

    The agent:
        - Maintains per-session conversation memory
        - Supports multiple concurrent datasets
        - Tracks context switches between datasets
        - Generates explanations for non-technical users
        - Never exposes internal reasoning to the user

    Args:
        provider: The Ollama LLM provider.
        query_chain: The code generation chain (optional, auto-created).
        sandbox: The sandboxed code executor (optional, auto-created).
        memory: Conversation memory manager (optional, auto-created).
        chat_settings: Chat configuration.
        sandbox_settings: Sandbox configuration.
    """

    def __init__(
        self,
        provider: Optional[OllamaProvider] = None,
        query_chain: Optional[QueryChain] = None,
        sandbox: Optional[SandboxExecutor] = None,
        memory: Optional[ConversationMemory] = None,
        chat_settings: Optional[ChatSettings] = None,
        sandbox_settings: Optional[SandboxSettings] = None,
    ) -> None:
        settings = get_settings()

        # Initialize the LLM provider
        self._provider = provider or OllamaProvider()

        # Initialize shared memory
        self._memory = memory or ConversationMemory()

        # Initialize the query chain with shared memory
        self._chain = query_chain or QueryChain(
            provider=self._provider,
            memory=self._memory,
        )

        # Initialize the sandbox executor
        self._sandbox = sandbox or SandboxExecutor()

        # Configuration
        self._chat_settings = chat_settings or settings.chat
        self._sandbox_settings = sandbox_settings or settings.sandbox

        # Dataset registry: file_id → FileMetadata
        self._datasets: Dict[str, FileMetadata] = {}

        logger.info(
            "DataWhispererAgent initialized: model=%s, max_retries=%d",
            self._provider.get_model_name(),
            self._sandbox_settings.max_retries,
        )

    @property
    def memory(self) -> ConversationMemory:
        """Expose the conversation memory."""
        return self._memory

    @property
    def chain(self) -> QueryChain:
        """Expose the query chain."""
        return self._chain

    @property
    def provider(self) -> OllamaProvider:
        """Expose the LLM provider."""
        return self._provider

    # ── Dataset Management ───────────────────────────────────────────────

    def register_dataset(self, file_metadata: FileMetadata) -> None:
        """
        Register a dataset with the agent for analysis.

        Args:
            file_metadata: Complete metadata for the uploaded CSV.
        """
        self._datasets[file_metadata.file_id] = file_metadata
        logger.info(
            "Registered dataset: %s (%s, %d rows × %d cols)",
            file_metadata.file_id,
            file_metadata.original_name,
            file_metadata.row_count,
            file_metadata.col_count,
        )

    def unregister_dataset(self, file_id: str) -> None:
        """
        Remove a dataset from the agent's registry.

        Args:
            file_id: The file ID to unregister.
        """
        self._datasets.pop(file_id, None)
        logger.info("Unregistered dataset: %s", file_id)

    def get_dataset(self, file_id: str) -> Optional[FileMetadata]:
        """
        Get metadata for a registered dataset.

        Args:
            file_id: The file ID to look up.

        Returns:
            The ``FileMetadata``, or None if not registered.
        """
        return self._datasets.get(file_id)

    def get_all_datasets(self) -> Dict[str, FileMetadata]:
        """Return all registered datasets."""
        return dict(self._datasets)

    # ── Core Agent Execution ─────────────────────────────────────────────

    def process_question(
        self,
        session_id: str,
        file_id: str,
        question: str,
        csv_path: str,
        file_metadata: Optional[FileMetadata] = None,
    ) -> AgentResult:
        """
        Process a user question through the complete agent pipeline.

        Pipeline (ReAct loop):
            1. Resolve dataset metadata
            2. Detect context switch (multi-dataset support)
            3. REASON: Internal step-by-step reasoning
            4. ACT: Generate pandas code via LLM
            5. VALIDATE: AST security analysis
            6. EXECUTE: Run in sandboxed subprocess
            7. OBSERVE: Check result
            8. If failed → RETRY with error context (up to max_retries)
            9. Generate user-facing explanation
            10. Update conversation memory

        Args:
            session_id: The current chat session ID.
            file_id: The active file ID.
            question: The user's natural language question.
            csv_path: Disk path to the CSV file.
            file_metadata: Optional metadata override. If not provided,
                          uses the registered dataset.

        Returns:
            An ``AgentResult`` with the complete analysis outcome.
        """
        start_time = time.time()
        total_tokens = 0

        # 1. Resolve metadata
        metadata = file_metadata or self._datasets.get(file_id)
        if not metadata:
            return AgentResult(
                success=False,
                content="❌ Dataset not found. Please upload a CSV file first.",
                result_type=ResultType.ERROR,
                latency_ms=round((time.time() - start_time) * 1000, 2),
            )

        # Register if not already known
        if file_id not in self._datasets:
            self._datasets[file_id] = metadata

        # 2. Detect context switch
        previous_file_id = self._memory.set_active_dataset(session_id, file_id)
        if previous_file_id and previous_file_id != file_id:
            old_meta = self._datasets.get(previous_file_id)
            if old_meta:
                logger.info(
                    "Context switch detected: %s → %s",
                    old_meta.original_name,
                    metadata.original_name,
                )
                # Add context switch message to memory
                self._memory.add_assistant_message(
                    session_id,
                    f"[Switched to dataset: {metadata.original_name}]",
                )

        # 3-8. Generate → Validate → Execute → Retry loop
        max_retries = self._sandbox_settings.max_retries
        last_error: Optional[Exception] = None
        error_context: Optional[str] = None
        last_reasoning: Optional[str] = None

        for attempt in range(1, max_retries + 2):
            try:
                # REASON + ACT: Generate code
                code, llm_response, reasoning = self._chain.generate_code(
                    question=question,
                    file_metadata=metadata,
                    session_id=session_id,
                    all_datasets=self._datasets if len(self._datasets) > 1 else None,
                    error_context=error_context,
                    attempt=attempt,
                )

                total_tokens += llm_response.tokens_used
                if reasoning:
                    last_reasoning = reasoning

                # VALIDATE + EXECUTE: Run in sandbox
                execution_result = self._sandbox.execute(
                    code=code,
                    csv_path=csv_path,
                )

                # OBSERVE: Success!
                elapsed_ms = round((time.time() - start_time) * 1000, 2)

                # 9. Generate explanation
                result_summary = self._summarize_result(execution_result)
                explanation = self._chain.generate_explanation(
                    code=code,
                    result_summary=result_summary,
                )
                total_tokens += 50  # Approximate for explanation generation

                # 10. Update memory
                self._memory.add_user_message(session_id, question)
                self._memory.add_assistant_message(session_id, result_summary)

                content = self._format_result(execution_result, explanation)

                logger.info(
                    "Agent success: session=%s, attempts=%d, tokens=%d, %.0fms",
                    session_id,
                    attempt,
                    total_tokens,
                    elapsed_ms,
                )

                return AgentResult(
                    success=True,
                    content=content,
                    code=code,
                    result_type=execution_result.result_type,
                    result_data=execution_result.data,
                    chart_path=execution_result.chart_path,
                    explanation=explanation,
                    tokens_used=total_tokens,
                    latency_ms=elapsed_ms,
                    attempts=attempt,
                    internal_reasoning=last_reasoning,
                )

            except (CodeValidationError, ExecutionRuntimeError) as e:
                last_error = e
                logger.warning(
                    "Attempt %d/%d failed: %s",
                    attempt,
                    max_retries + 1,
                    e.message,
                )
                if attempt <= max_retries:
                    error_context = self._chain.build_error_context(e, metadata)
                continue

            except ExecutionTimeoutError as e:
                last_error = e
                logger.warning(
                    "Execution timed out on attempt %d", attempt
                )
                if attempt <= max_retries:
                    error_context = self._chain.build_error_context(e, metadata)
                continue

            except (GenerationError, OllamaConnectionError) as e:
                # Fatal errors — don't retry
                last_error = e
                logger.error("Fatal agent error: %s", str(e))
                break

        # All retries exhausted
        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        error_content = self._format_error(last_error)

        # Still update memory with the failed attempt
        self._memory.add_user_message(session_id, question)
        self._memory.add_assistant_message(
            session_id, "I couldn't complete this analysis."
        )

        return AgentResult(
            success=False,
            content=error_content,
            result_type=ResultType.ERROR,
            tokens_used=total_tokens,
            latency_ms=elapsed_ms,
            attempts=max_retries + 1,
            internal_reasoning=last_reasoning,
        )

    # ── Auxiliary Agent Capabilities ─────────────────────────────────────

    def generate_session_title(self, question: str) -> str:
        """
        Generate a short title for a new session.

        Args:
            question: The user's first question.

        Returns:
            A 4-6 word title string.
        """
        return self._chain.generate_title(question)

    def generate_suggested_questions(
        self,
        file_metadata: FileMetadata,
        count: int = 4,
    ) -> List[str]:
        """
        Generate suggested analytical questions for a dataset.

        Args:
            file_metadata: The dataset metadata.
            count: Number of suggestions to generate.

        Returns:
            List of suggested question strings.
        """
        return self._chain.generate_suggested_questions(file_metadata, count)

    def health_check(self) -> Dict[str, Any]:
        """
        Check the agent's health status.

        Returns:
            Dict with connectivity, model, and memory status.
        """
        ollama_health = self._provider.health_check()
        return {
            "agent_ready": ollama_health.get("connected", False) and ollama_health.get("model_loaded", False),
            "ollama": ollama_health,
            "model": self._provider.get_model_name(),
            "active_sessions": self._memory.get_session_count(),
            "registered_datasets": len(self._datasets),
        }

    def load_session_memory(
        self,
        session_id: str,
        messages: List[Dict[str, str]],
    ) -> None:
        """
        Load saved messages into session memory (for session resume).

        Args:
            session_id: The session to populate.
            messages: List of ``{"role": "...", "content": "..."}`` dicts.
        """
        self._memory.load_from_db_messages(session_id, messages)

    def clear_session(self, session_id: str) -> None:
        """
        Clear all state for a session.

        Args:
            session_id: The session to clear.
        """
        self._memory.clear_session(session_id)

    def close(self) -> None:
        """Release all resources."""
        self._provider.close()
        logger.info("Agent resources released.")

    # ── Private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _summarize_result(result: CodeExecutionResult) -> str:
        """Create a brief summary of an execution result for memory."""
        if result.result_type == ResultType.CHART:
            return "Generated a chart visualization."
        elif result.result_type == ResultType.DATAFRAME:
            return f"Returned a data table with results."
        elif result.result_type == ResultType.SERIES:
            return f"Returned a data series with results."
        else:
            data = str(result.data) if result.data else "No data"
            return data[:200]

    @staticmethod
    def _format_result(
        result: CodeExecutionResult,
        explanation: str,
    ) -> str:
        """Format execution result with explanation for the user."""
        parts = []

        # Add explanation first
        if explanation and explanation != "Analysis complete.":
            parts.append(explanation)

        # Add result data
        if result.result_type == ResultType.CHART:
            parts.append("📊 Chart generated successfully.")
            if result.data:
                parts.append(str(result.data))
        elif result.result_type == ResultType.DATAFRAME:
            parts.append(f"📋 Results:\n\n{result.data}")
        elif result.result_type == ResultType.SERIES:
            parts.append(f"📋 Results:\n\n{result.data}")
        elif result.data:
            parts.append(str(result.data))

        return "\n\n".join(parts) if parts else "Analysis complete."

    @staticmethod
    def _format_error(error: Optional[Exception]) -> str:
        """Format an error into a user-friendly message."""
        if error is None:
            return (
                "❌ An unknown error occurred. "
                "Please try rephrasing your question."
            )

        if isinstance(error, OllamaConnectionError):
            return (
                "❌ Cannot connect to the AI model.\n\n"
                "💡 Make sure Ollama is running:\n"
                "  1. Open a terminal\n"
                "  2. Run: `ollama serve`\n"
                "  3. Run: `ollama pull qwen2.5:7b`"
            )

        if isinstance(error, DataWhispererError):
            msg = f"❌ {error.message}"
            if error.suggestion:
                msg += f"\n\n💡 {error.suggestion}"
            return msg

        return f"❌ An unexpected error occurred: {str(error)[:200]}"
