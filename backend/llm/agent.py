"""
DataWhisperer AI Agent — Lightweight Code Interpreter.

This is the brain of the application. It implements a complete
code-interpreter pipeline that, for every user question:

    1. UNDERSTAND: Parse user intent from the question
    2. GENERATE: Produce optimized Pandas code via LLM
    3. DISPLAY: Return the generated code for the user to see
    4. EXECUTE: Run the code securely in a sandboxed subprocess
    5. OUTPUT: Return the execution results (tables, values, charts)
    6. EXPLAIN: Generate a plain-English explanation of what the code does
    7. CHART REASONING: If a chart was generated, explain why that chart type
    8. AUTO-DEBUG: If execution fails, automatically debug and retry ONCE

Architecture:
    The agent follows a specialized ReAct pattern:
    - REASON: Internal chain-of-thought before code generation (hidden)
    - ACT: Generate code → validate → execute
    - OBSERVE: Check execution result
    - DEBUG: On first failure, auto-debug the code and retry once
    - REPORT: On second failure, return structured error to user

    The auto-debug step is distinct from generic retry: it sends the
    failed code + error + dataset info to a specialized debug prompt
    that performs root-cause analysis.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from backend.core.config import (
    ChatSettings,
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
from backend.llm.base_provider import BaseLLMProvider
from backend.llm.chains.query_chain import QueryChain
from backend.llm.memory import ConversationMemory
from backend.models.schemas import (
    CodeExecutionResult,
    FileMetadata,
    LLMResponse,
    ResultType,
)
from backend.sandbox.executor import SandboxExecutor

logger = get_logger(__name__)


class AgentResult:
    """
    Encapsulates the complete result of the code-interpreter pipeline.

    Each field maps to an interpreter stage, giving the UI layer
    everything it needs to render the response:

    Attributes:
        success: Whether the pipeline completed successfully.
        content: Pre-formatted combined output for simple consumers.
        code: Stage 3 — The generated Python code to display.
        result_type: Stage 5 — Type of output (text, dataframe, chart, error).
        result_data: Stage 5 — The actual execution output.
        chart_path: Stage 5 — Path to chart image (if generated).
        explanation: Stage 6 — Plain-English explanation of code + results.
        chart_explanation: Stage 7 — Why this chart type was chosen.
        auto_debug_applied: Stage 8 — Whether auto-debug was triggered.
        debug_summary: Stage 8 — What the debugger changed (internal).
        tokens_used: Total tokens consumed across all LLM calls.
        latency_ms: Total wall-clock time in milliseconds.
        attempts: Number of execution attempts (1 = first try, 2 = after debug).
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
        chart_explanation: Optional[str] = None,
        auto_debug_applied: bool = False,
        debug_summary: Optional[str] = None,
        tokens_used: int = 0,
        latency_ms: float = 0.0,
        attempts: int = 1,
        internal_reasoning: Optional[str] = None,
        provider_used: Optional[str] = None,
        model_used: Optional[str] = None,
        fallback_used: bool = False,
        fallback_reason: Optional[str] = None,
    ) -> None:
        self.success = success
        self.content = content
        self.code = code
        self.result_type = result_type
        self.result_data = result_data
        self.chart_path = chart_path
        self.explanation = explanation
        self.chart_explanation = chart_explanation
        self.auto_debug_applied = auto_debug_applied
        self.debug_summary = debug_summary
        self.tokens_used = tokens_used
        self.latency_ms = latency_ms
        self.attempts = attempts
        self.internal_reasoning = internal_reasoning  # NEVER expose to user
        self.provider_used = provider_used
        self.model_used = model_used
        self.fallback_used = fallback_used
        self.fallback_reason = fallback_reason


class DataWhispererAgent:
    """
    The core AI agent — a lightweight code interpreter for CSV analysis.

    For every question, executes the full 8-stage pipeline:
        1. Understand → 2. Generate → 3. Display → 4. Execute →
        5. Output → 6. Explain → 7. Chart Reasoning → 8. Auto-Debug

    Features:
        - Per-session conversation memory
        - Multiple concurrent datasets
        - Context switching between datasets
        - Internal reasoning (never exposed)
        - Automatic code debugging on failure
        - Chart type reasoning

    Args:
        provider: The Ollama LLM provider.
        query_chain: The code generation chain.
        sandbox: The sandboxed code executor.
        memory: Conversation memory manager.
        chat_settings: Chat configuration.
        sandbox_settings: Sandbox configuration.
    """

    def __init__(
        self,
        provider: Optional[BaseLLMProvider] = None,
        query_chain: Optional[QueryChain] = None,
        sandbox: Optional[SandboxExecutor] = None,
        memory: Optional[ConversationMemory] = None,
        chat_settings: Optional[ChatSettings] = None,
        sandbox_settings: Optional[SandboxSettings] = None,
    ) -> None:
        settings = get_settings()

        if provider is None:
            from backend.llm.providers import create_default_provider

            provider = create_default_provider()

        self._provider = provider
        self._memory = memory or ConversationMemory()
        self._chain = query_chain or QueryChain(
            provider=self._provider,
            memory=self._memory,
        )
        self._sandbox = sandbox or SandboxExecutor()
        self._chat_settings = chat_settings or settings.chat
        self._sandbox_settings = sandbox_settings or settings.sandbox

        # Dataset registry: file_id → FileMetadata
        self._datasets: Dict[str, FileMetadata] = {}

        logger.info(
            "DataWhispererAgent initialized: model=%s",
            self._provider.get_model_name(),
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
    def provider(self) -> BaseLLMProvider:
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
        """Remove a dataset from the agent's registry."""
        self._datasets.pop(file_id, None)

    def get_dataset(self, file_id: str) -> Optional[FileMetadata]:
        """Get metadata for a registered dataset."""
        return self._datasets.get(file_id)

    def get_all_datasets(self) -> Dict[str, FileMetadata]:
        """Return all registered datasets."""
        return dict(self._datasets)

    # ── Core Interpreter Pipeline ────────────────────────────────────────

    def process_question(
        self,
        session_id: str,
        file_id: str,
        question: str,
        csv_path: str,
        file_metadata: Optional[FileMetadata] = None,
    ) -> AgentResult:
        """
        Process a user question through the complete interpreter pipeline.

        8-Stage Pipeline:
            1. UNDERSTAND: Resolve dataset, detect context switch
            2. GENERATE: Produce pandas code via LLM with internal reasoning
            3. (DISPLAY): Code is returned in AgentResult.code
            4. EXECUTE: Run code in sandboxed subprocess
            5. (OUTPUT): Result is returned in AgentResult.result_data
            6. EXPLAIN: Generate plain-English explanation
            7. CHART REASONING: Explain chart type choice (if applicable)
            8. AUTO-DEBUG: On failure, debug the code and retry ONCE

        Args:
            session_id: The current chat session ID.
            file_id: The active file ID.
            question: The user's natural language question.
            csv_path: Disk path to the CSV file.
            file_metadata: Optional metadata override.

        Returns:
            An ``AgentResult`` with all 8 pipeline stages populated.
        """
        start_time = time.time()
        total_tokens = 0

        # ── Stage 1: UNDERSTAND ──────────────────────────────────────────
        metadata = file_metadata or self._datasets.get(file_id)
        if not metadata:
            return AgentResult(
                success=False,
                content="❌ Dataset not found. Please upload a CSV file first.",
                result_type=ResultType.ERROR,
                latency_ms=round((time.time() - start_time) * 1000, 2),
            )

        if file_id not in self._datasets:
            self._datasets[file_id] = metadata

        # Detect context switch (multi-dataset support)
        previous_file_id = self._memory.set_active_dataset(session_id, file_id)
        if previous_file_id and previous_file_id != file_id:
            old_meta = self._datasets.get(previous_file_id)
            if old_meta:
                logger.info(
                    "Context switch: %s → %s",
                    old_meta.original_name,
                    metadata.original_name,
                )
                self._memory.add_assistant_message(
                    session_id,
                    f"[Switched to dataset: {metadata.original_name}]",
                )

        # ── Stage 2: GENERATE ────────────────────────────────────────────
        try:
            code, llm_response, reasoning = self._chain.generate_code(
                question=question,
                file_metadata=metadata,
                session_id=session_id,
                all_datasets=(
                    self._datasets if len(self._datasets) > 1 else None
                ),
                error_context=None,
                attempt=1,
            )
            total_tokens += llm_response.tokens_used
        except (GenerationError, OllamaConnectionError) as e:
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            provider_meta = self._get_provider_metadata()
            return AgentResult(
                success=False,
                content=self._format_error(e),
                result_type=ResultType.ERROR,
                tokens_used=total_tokens,
                latency_ms=elapsed_ms,
                provider_used=provider_meta.get("provider"),
                model_used=provider_meta.get("model"),
                fallback_used=bool(provider_meta.get("fallback_used", False)),
                fallback_reason=provider_meta.get("fallback_reason"),
            )

        # ── Stage 3: DISPLAY (code is captured in AgentResult.code) ──────
        # Stage 4: EXECUTE ────────────────────────────────────────────────
        execution_result: Optional[CodeExecutionResult] = None
        auto_debug_applied = False
        debug_summary: Optional[str] = None
        final_code = code

        try:
            execution_result = self._sandbox.execute(
                code=code,
                csv_path=csv_path,
            )
        except (CodeValidationError, ExecutionRuntimeError, ExecutionTimeoutError) as first_error:
            # ── Stage 8: AUTO-DEBUG (retry ONCE) ─────────────────────────
            logger.warning(
                "First execution failed: %s — attempting auto-debug",
                str(first_error)[:150],
            )

            # Short-circuit: environmental errors that LLM cannot fix
            error_str = str(first_error).lower()
            is_env_error = (
                "openblas" in error_str
                or "memory allocation" in error_str
                or "cannot allocate memory" in error_str
            )
            if is_env_error:
                elapsed_ms = round((time.time() - start_time) * 1000, 2)
                provider_meta = self._get_provider_metadata()
                self._memory.add_user_message(session_id, question)
                self._memory.add_assistant_message(
                    session_id, "I couldn't complete this analysis."
                )
                return AgentResult(
                    success=False,
                    content=(
                        "❌ **Memory Limit Reached**\n\n"
                        "The server doesn't have enough memory for this operation. "
                        "This is a hosting environment limitation, not a code issue.\n\n"
                        "💡 **Try:**\n"
                        "- Asking a simpler question (e.g., a specific column average)\n"
                        "- Breaking the analysis into smaller steps\n"
                        "- Working with fewer columns"
                    ),
                    code=code,
                    result_type=ResultType.ERROR,
                    auto_debug_applied=False,
                    tokens_used=total_tokens,
                    latency_ms=elapsed_ms,
                    attempts=1,
                    internal_reasoning=reasoning,
                    provider_used=provider_meta.get("provider"),
                    model_used=provider_meta.get("model"),
                    fallback_used=bool(provider_meta.get("fallback_used", False)),
                    fallback_reason=provider_meta.get("fallback_reason"),
                )

            try:
                # Debug the failed code
                fixed_code, debug_response = self._chain.debug_code(
                    failed_code=code,
                    error=first_error,
                    file_metadata=metadata,
                    question=question,
                )
                total_tokens += debug_response.tokens_used

                # Execute the debugged code
                execution_result = self._sandbox.execute(
                    code=fixed_code,
                    csv_path=csv_path,
                )

                auto_debug_applied = True
                final_code = fixed_code
                debug_summary = (
                    f"Original code failed with {type(first_error).__name__}. "
                    f"Auto-debugger fixed the issue."
                )
                logger.info(
                    "Auto-debug SUCCESS: fixed %s",
                    type(first_error).__name__,
                )

            except Exception as second_error:
                # Both attempts failed — report error
                elapsed_ms = round((time.time() - start_time) * 1000, 2)
                error_content = self._format_debug_failure(
                    first_error, second_error, code
                )
                provider_meta = self._get_provider_metadata()

                self._memory.add_user_message(session_id, question)
                self._memory.add_assistant_message(
                    session_id, "I couldn't complete this analysis."
                )

                return AgentResult(
                    success=False,
                    content=error_content,
                    code=code,
                    result_type=ResultType.ERROR,
                    auto_debug_applied=True,
                    tokens_used=total_tokens,
                    latency_ms=elapsed_ms,
                    attempts=2,
                    internal_reasoning=reasoning,
                    provider_used=provider_meta.get("provider"),
                    model_used=provider_meta.get("model"),
                    fallback_used=bool(provider_meta.get("fallback_used", False)),
                    fallback_reason=provider_meta.get("fallback_reason"),
                )

        # ── Stage 5: OUTPUT (result is captured in AgentResult) ──────────
        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        # ── Stage 6: EXPLAIN ─────────────────────────────────────────────
        result_summary = self._summarize_result(execution_result)
        
        # Build explanation payload with actual data preview
        data_preview = str(execution_result.data)[:500] if execution_result.data else "No data"
        if execution_result.result_type == ResultType.DATAFRAME:
            explanation_payload = f"Returned a data table: {data_preview}"
        elif execution_result.result_type == ResultType.SERIES:
            explanation_payload = f"Returned a data series: {data_preview}"
        else:
            explanation_payload = data_preview

        explanation = self._chain.generate_explanation(
            code=final_code,
            result_summary=explanation_payload,
        )
        total_tokens += 50  # Approximate for explanation

        # ── Stage 7: CHART REASONING ─────────────────────────────────────
        chart_explanation: Optional[str] = None
        if execution_result.result_type == ResultType.CHART:
            chart_explanation = self._chain.generate_chart_explanation(
                code=final_code,
                question=question,
            )
            if chart_explanation:
                total_tokens += 30  # Approximate for chart explanation

        # ── Update memory ────────────────────────────────────────────────
        self._memory.add_user_message(session_id, question)
        self._memory.add_assistant_message(session_id, result_summary)

        provider_meta = self._get_provider_metadata()

        # ── Format the combined content ──────────────────────────────────
        content = self._format_interpreter_output(
            code=final_code,
            execution_result=execution_result,
            explanation=explanation,
            chart_explanation=chart_explanation,
            auto_debug_applied=auto_debug_applied,
            provider_used=provider_meta.get("provider"),
            model_used=provider_meta.get("model"),
            fallback_used=bool(provider_meta.get("fallback_used", False)),
            fallback_reason=provider_meta.get("fallback_reason"),
        )

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        logger.info(
            "Interpreter pipeline complete: session=%s, attempts=%d, "
            "debug=%s, tokens=%d, %.0fms",
            session_id,
            2 if auto_debug_applied else 1,
            auto_debug_applied,
            total_tokens,
            elapsed_ms,
        )

        return AgentResult(
            success=True,
            content=content,
            code=final_code,
            result_type=execution_result.result_type,
            result_data=execution_result.data,
            chart_path=execution_result.chart_path,
            explanation=explanation,
            chart_explanation=chart_explanation,
            auto_debug_applied=auto_debug_applied,
            debug_summary=debug_summary,
            tokens_used=total_tokens,
            latency_ms=elapsed_ms,
            attempts=2 if auto_debug_applied else 1,
            internal_reasoning=reasoning,
            provider_used=provider_meta.get("provider"),
            model_used=provider_meta.get("model"),
            fallback_used=bool(provider_meta.get("fallback_used", False)),
            fallback_reason=provider_meta.get("fallback_reason"),
        )

    # ── Auxiliary Capabilities ───────────────────────────────────────────

    def generate_session_title(self, question: str) -> str:
        """Generate a short title for a new session."""
        return self._chain.generate_title(question)

    def generate_suggested_questions(
        self,
        file_metadata: FileMetadata,
        count: int = 4,
    ) -> List[str]:
        """Generate suggested analytical questions for a dataset."""
        return self._chain.generate_suggested_questions(file_metadata, count)

    def health_check(self) -> Dict[str, Any]:
        """Check the agent's health status."""
        provider_health = self._provider.health_check()

        if "primary" in provider_health and "fallback" in provider_health:
            fallback_health = provider_health.get("fallback", {})
            return {
                "agent_ready": bool(provider_health.get("ready", False)),
                "provider_router": provider_health,
                "primary": provider_health.get("primary", {}),
                "fallback": fallback_health,
                "ollama": fallback_health,
                "model": self._provider.get_model_name(),
                "local_only_mode": bool(provider_health.get("local_only_mode", False)),
                "active_sessions": self._memory.get_session_count(),
                "registered_datasets": len(self._datasets),
            }

        return {
            "agent_ready": (
                provider_health.get("connected", False)
                and provider_health.get("model_loaded", False)
            ),
            "ollama": provider_health,
            "model": self._provider.get_model_name(),
            "local_only_mode": False,
            "active_sessions": self._memory.get_session_count(),
            "registered_datasets": len(self._datasets),
        }

    def load_session_memory(
        self,
        session_id: str,
        messages: List[Dict[str, str]],
    ) -> None:
        """Load saved messages into session memory (for session resume)."""
        self._memory.load_from_db_messages(session_id, messages)

    def clear_session(self, session_id: str) -> None:
        """Clear all state for a session."""
        self._memory.clear_session(session_id)

    def close(self) -> None:
        """Release all resources."""
        self._provider.close()
        logger.info("Agent resources released.")

    # ── Private Formatters ───────────────────────────────────────────────

    @staticmethod
    def _format_interpreter_output(
        code: str,
        execution_result: CodeExecutionResult,
        explanation: str,
        chart_explanation: Optional[str],
        auto_debug_applied: bool,
        provider_used: Optional[str] = None,
        model_used: Optional[str] = None,
        fallback_used: bool = False,
        fallback_reason: Optional[str] = None,
    ) -> str:
        """
        Format all pipeline outputs into structured interpreter display.

        Produces a clean, sectioned output:
            📝 Generated Code
            📊 Output / Results
            💡 Explanation
            🎨 Chart Reasoning (if applicable)
            🔧 Auto-Debug Note (if applicable)
        """
        sections: List[str] = []

        # Section 1: Generated Code
        sections.append(f"📝 **Generated Code:**\n```python\n{code}\n```")

        # Section 2: Output
        if execution_result.result_type == ResultType.CHART:
            output_text = "📊 **Output:** Chart generated successfully."
            if execution_result.data and str(execution_result.data).strip():
                output_text += f"\n\n{execution_result.data}"
            sections.append(output_text)
        elif execution_result.result_type == ResultType.DATAFRAME:
            sections.append(f"📋 **Output:**\n\n{execution_result.data}")
        elif execution_result.result_type == ResultType.SERIES:
            sections.append(f"📋 **Output:**\n\n{execution_result.data}")
        elif execution_result.data:
            sections.append(f"📋 **Output:**\n\n{execution_result.data}")
        else:
            sections.append("✅ **Output:** Code executed successfully.")

        # Section 3: Explanation
        if explanation and explanation != "Analysis complete.":
            sections.append(f"💡 **Explanation:** {explanation}")

        # Section 3.5: Provider metadata
        if provider_used or model_used:
            provider_line = (
                f"🧠 **Model Used:** {provider_used or 'unknown'}"
                f" · {model_used or 'unknown'}"
            )
            if fallback_used:
                provider_line += " (fallback applied)"
                if fallback_reason:
                    provider_line += f" — {fallback_reason[:180]}"
            sections.append(provider_line)

        # Section 4: Chart Reasoning (only for charts)
        if chart_explanation:
            sections.append(f"🎨 **Chart Reasoning:** {chart_explanation}")

        # Section 5: Auto-Debug Note
        if auto_debug_applied:
            sections.append(
                "🔧 **Auto-Debug:** The original code had an error. "
                "It was automatically debugged and re-executed successfully."
            )

        return "\n\n".join(sections)

    @staticmethod
    def _summarize_result(result: CodeExecutionResult) -> str:
        """Create a brief summary of an execution result for memory."""
        if result.result_type == ResultType.CHART:
            return "Generated a chart visualization."
        elif result.result_type == ResultType.DATAFRAME:
            return "Returned a data table with results."
        elif result.result_type == ResultType.SERIES:
            return "Returned a data series with results."
        else:
            data = str(result.data) if result.data else "No data"
            return data[:200]

    def _get_provider_metadata(self) -> Dict[str, Any]:
        """Best-effort provider metadata for UI/response transparency."""
        meta_getter = getattr(self._provider, "get_last_generation_metadata", None)
        if callable(meta_getter):
            try:
                metadata = meta_getter()
                if isinstance(metadata, dict):
                    return metadata
            except Exception:
                logger.debug("Provider metadata getter failed", exc_info=True)

        return {
            "provider": getattr(self._provider, "__class__", type(self._provider)).__name__.replace("Provider", "").lower(),
            "model": self._provider.get_model_name(),
            "fallback_used": False,
            "fallback_reason": None,
        }

    @staticmethod
    def _format_error(error: Exception) -> str:
        """Format a fatal error into a user-friendly message."""
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

    @staticmethod
    def _format_debug_failure(
        first_error: Exception,
        second_error: Exception,
        original_code: str,
    ) -> str:
        """Format a message when both original and debugged code fail."""
        first_msg = (
            first_error.message
            if isinstance(first_error, DataWhispererError)
            else str(first_error)[:150]
        )
        second_msg = (
            second_error.message
            if isinstance(second_error, DataWhispererError)
            else str(second_error)[:150]
        )

        return (
            f"❌ **Execution Failed**\n\n"
            f"The generated code encountered an error, and the automatic "
            f"debugger was unable to fix it.\n\n"
            f"**Original Error:** {first_msg}\n\n"
            f"**Debug Attempt Error:** {second_msg}\n\n"
            f"💡 Try rephrasing your question or breaking it into "
            f"simpler steps.\n\n"
            f"📝 **Original Code:**\n```python\n{original_code}\n```"
        )
