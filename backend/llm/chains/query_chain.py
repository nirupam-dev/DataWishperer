"""
Query chain — LangChain-based question-to-code pipeline.

Orchestrates the complete code generation flow using LangChain primitives:
    1. Builds prompt with system instructions, dataset context, memory
    2. Invokes the LLM via ChatOllama
    3. Parses output, stripping chain-of-thought
    4. Manages retry with error context injection

Interpreter Pipeline Stages (this chain handles stages 2, 6, 7, 8):
    Stage 2: Generate optimized Pandas code
    Stage 6: Explain the code in simple English
    Stage 7: Explain chart type selection
    Stage 8: Auto-debug failed code and retry

Refactored to use PromptRegistry for all prompt composition.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from backend.core.config import ChatSettings, get_settings
from backend.core.exceptions import (
    CodeValidationError,
    ExecutionRuntimeError,
    GenerationError,
)
from backend.core.logging_config import get_logger
from backend.llm.chains.output_parser import OutputParser
from backend.llm.memory import ConversationMemory
from backend.llm.prompts.registry import PromptRegistry
from backend.llm.providers.ollama_provider import OllamaProvider
from backend.models.schemas import FileMetadata, LLMResponse

logger = get_logger(__name__)

# Chart type detection patterns for chart explanation generation
_CHART_TYPE_PATTERNS: Dict[str, str] = {
    r"\.bar\(": "bar chart",
    r"\.barh\(": "horizontal bar chart",
    r"\.plot\.bar\(": "bar chart",
    r"\.hist\(": "histogram",
    r"\.scatter\(": "scatter plot",
    r"\.plot\.scatter\(": "scatter plot",
    r"\.pie\(": "pie chart",
    r"\.plot\.pie\(": "pie chart",
    r"\.plot\.line\(|\.plot\(\)": "line chart",
    r"\.boxplot\(|\.plot\.box\(": "box plot",
    r"\.heatmap\(|\.imshow\(": "heatmap",
    r"\.violinplot\(": "violin plot",
    r"\.area\(|\.plot\.area\(": "area chart",
    # Seaborn chart detection
    r"sns\.barplot\(": "bar chart",
    r"sns\.histplot\(|sns\.distplot\(": "histogram",
    r"sns\.scatterplot\(": "scatter plot",
    r"sns\.heatmap\(": "heatmap",
    r"sns\.boxplot\(": "box plot",
    r"sns\.violinplot\(": "violin plot",
    r"sns\.lineplot\(": "line chart",
    r"sns\.kdeplot\(": "density plot",
    r"sns\.pairplot\(": "pair plot",
    # Plotly chart detection
    r"px\.bar\(": "bar chart",
    r"px\.scatter\(": "scatter plot",
    r"px\.line\(": "line chart",
    r"px\.histogram\(": "histogram",
    r"px\.pie\(": "pie chart",
    r"px\.box\(": "box plot",
    r"px\.violin\(": "violin plot",
    r"px\.imshow\(|px\.density_heatmap\(": "heatmap",
    r"go\.Bar\(": "bar chart",
    r"go\.Scatter\(": "scatter plot",
    r"go\.Heatmap\(": "heatmap",
    r"go\.Box\(": "box plot",
    r"go\.Violin\(": "violin plot",
    r"go\.Pie\(": "pie chart",
    r"\.corr\(\)": "correlation matrix",
}


class QueryChain:
    """
    LangChain-based query chain for the code interpreter pipeline.

    Handles code generation, explanation, chart reasoning, and auto-debug.
    Uses PromptRegistry for all prompt composition.

    Interpreter Stages:
        - ``generate_code()``: Stage 2 — Generate optimized Pandas code
        - ``generate_explanation()``: Stage 6 — Plain English code explanation
        - ``generate_chart_explanation()``: Stage 7 — Chart type reasoning
        - ``debug_code()``: Stage 8 — Auto-debug failed code

    Args:
        provider: The Ollama LLM provider.
        output_parser: Parser for extracting code from LLM output.
        prompt_registry: Registry for modular prompt composition.
        memory: Conversation memory manager.
        chat_settings: Chat configuration.
    """

    def __init__(
        self,
        provider: OllamaProvider,
        output_parser: Optional[OutputParser] = None,
        prompt_registry: Optional[PromptRegistry] = None,
        memory: Optional[ConversationMemory] = None,
        chat_settings: Optional[ChatSettings] = None,
    ) -> None:
        self._provider = provider
        self._parser = output_parser or OutputParser()
        self._registry = prompt_registry or PromptRegistry()
        self._memory = memory or ConversationMemory()
        self._chat_settings = chat_settings or get_settings().chat

    @property
    def memory(self) -> ConversationMemory:
        """Expose the conversation memory for external use."""
        return self._memory

    @property
    def registry(self) -> PromptRegistry:
        """Expose the prompt registry for external use."""
        return self._registry

    # ── Stage 2: Code Generation ─────────────────────────────────────────

    def generate_code(
        self,
        question: str,
        file_metadata: FileMetadata,
        session_id: str,
        all_datasets: Optional[Dict[str, FileMetadata]] = None,
        error_context: Optional[str] = None,
        attempt: int = 1,
    ) -> Tuple[str, LLMResponse, Optional[str]]:
        """
        Generate Python code to answer a user's question about their data.

        Internally reasons step-by-step but strips the reasoning from
        the final output. Only executable code is returned.

        Args:
            question: The user's natural language question.
            file_metadata: Metadata about the active CSV file.
            session_id: The current session ID for memory.
            all_datasets: Optional dict of all loaded datasets for context.
            error_context: Optional error from a previous failed attempt.
            attempt: Current attempt number (1-indexed).

        Returns:
            A tuple of ``(extracted_code, llm_response, reasoning_or_none)``.
            The reasoning is for internal logging only — NEVER show to user.

        Raises:
            GenerationError: If code cannot be extracted from the response.
            OllamaConnectionError: If the LLM is unreachable.
        """
        # Get conversation history from memory
        session_history = self._memory.get_dict_messages(session_id)

        # Build messages via the PromptRegistry
        messages = self._registry.build_generation_messages(
            question=question,
            file_metadata=file_metadata,
            session_history=session_history,
            all_datasets=all_datasets,
            error_context=error_context,
            attempt=attempt,
        )

        logger.info(
            "Generating code: question='%s' (attempt %d, %d messages)",
            question[:80],
            attempt,
            len(messages),
        )

        # Call the LLM
        llm_response = self._provider.generate(messages)

        # Extract code and reasoning separately
        code, reasoning = self._parser.extract_code_and_reasoning(
            llm_response.content
        )

        if reasoning:
            logger.debug("Internal reasoning: %s", reasoning[:200])

        logger.info(
            "Code generated: %d chars, %d tokens, %.0fms",
            len(code),
            llm_response.tokens_used,
            llm_response.latency_ms,
        )

        return code, llm_response, reasoning

    # ── Stage 6: Code Explanation ────────────────────────────────────────

    def generate_explanation(
        self,
        code: str,
        result_summary: str,
    ) -> str:
        """
        Stage 6: Explain the code and results in simple English.

        Args:
            code: The executed Python code.
            result_summary: Brief summary of the execution result.

        Returns:
            A 2-4 sentence explanation string.
        """
        messages = self._registry.build_explanation_messages(
            code=code,
            result_summary=result_summary,
        )

        try:
            response = self._provider.generate(
                messages, temperature=0.5, max_tokens=250
            )
            explanation = self._parser.extract_text_response(response.content)
            return explanation or "Analysis complete."
        except Exception:
            logger.warning("Explanation generation failed, using default.")
            return "Analysis complete."

    # ── Stage 7: Chart Explanation ───────────────────────────────────────

    def generate_chart_explanation(
        self,
        code: str,
        question: str,
    ) -> Optional[str]:
        """
        Stage 7: Explain why a specific chart type was chosen.

        Args:
            code: The executed code that generated a chart.
            question: The user's original question.

        Returns:
            A 1-2 sentence chart type explanation, or None if detection fails.
        """
        chart_type = self._detect_chart_type(code)
        if not chart_type:
            return None

        messages = self._registry.build_chart_explanation_messages(
            code=code,
            question=question,
            chart_type=chart_type,
        )

        try:
            response = self._provider.generate(
                messages, temperature=0.5, max_tokens=150
            )
            explanation = self._parser.extract_text_response(response.content)
            return explanation or None
        except Exception:
            logger.warning("Chart explanation generation failed.")
            return f"A {chart_type} was used to visualize this data."

    # ── Stage 8: Auto-Debug ──────────────────────────────────────────────

    def debug_code(
        self,
        failed_code: str,
        error: Exception,
        file_metadata: FileMetadata,
        question: str,
    ) -> Tuple[str, LLMResponse]:
        """
        Stage 8: Automatically debug failed code and produce a fix.

        Uses a specialized debug prompt that's more diagnostic than the
        generic error recovery — it asks the LLM to identify the root
        cause and write defensive code.

        Args:
            failed_code: The code that failed execution.
            error: The exception from execution.
            file_metadata: Dataset metadata for column reference.
            question: The original user question (for context).

        Returns:
            A tuple of ``(fixed_code, llm_response)``.

        Raises:
            GenerationError: If the debugger cannot produce valid code.
        """
        error_type = type(error).__name__
        error_message = str(error)

        if isinstance(error, (CodeValidationError, ExecutionRuntimeError)):
            error_message = error.message

        # Build debug messages via the PromptRegistry
        messages = self._registry.build_debug_messages(
            failed_code=failed_code,
            error_type=error_type,
            error_message=error_message,
            file_metadata=file_metadata,
        )

        logger.info(
            "Auto-debugging code: error=%s: %s",
            error_type,
            error_message[:100],
        )

        llm_response = self._provider.generate(messages)
        fixed_code = self._parser.extract_code(llm_response.content)

        logger.info(
            "Debug fix generated: %d chars, %d tokens",
            len(fixed_code),
            llm_response.tokens_used,
        )

        return fixed_code, llm_response

    # ── Reflection Methods ───────────────────────────────────────────────

    def reflect_on_code(
        self,
        code: str,
        file_metadata: FileMetadata,
    ) -> Tuple[bool, Optional[str]]:
        """
        Pre-execution reflection: validate code before running it.

        Returns (is_valid, fixed_code_or_none). If invalid, returns
        the corrected code from the reflection.

        Args:
            code: The generated code to validate.
            file_metadata: Dataset metadata for column cross-reference.

        Returns:
            Tuple of (is_valid, corrected_code_or_none).
        """
        messages = self._registry.build_reflection_messages(
            code=code,
            file_metadata=file_metadata,
        )

        try:
            response = self._provider.generate(
                messages, temperature=0.1, max_tokens=800
            )
            content = response.content.strip()

            # Parse the reflection verdict
            if "VERDICT: PASS" in content.upper():
                return True, None

            # Extract fixed code if present
            if "VERDICT: FAIL" in content.upper():
                try:
                    fixed_code = self._parser.extract_code(content)
                    return False, fixed_code
                except GenerationError:
                    logger.warning("Reflection found issues but no fix code.")
                    return False, None

            # Ambiguous response — treat as pass to avoid blocking
            return True, None

        except Exception:
            logger.warning("Reflection failed, proceeding without validation.")
            return True, None

    # ── Auxiliary Generation Methods ─────────────────────────────────────

    def generate_title(self, question: str) -> str:
        """Generate a short session title from the first user question."""
        messages = self._registry.build_title_messages(question)

        try:
            response = self._provider.generate(
                messages, temperature=0.5, max_tokens=30
            )
            title = response.content.strip().strip('"').strip("'")
            if len(title) > 60:
                title = title[:57] + "..."
            return title or "Data Analysis"
        except Exception:
            logger.warning("Title generation failed, using default.")
            return "Data Analysis"

    def generate_suggested_questions(
        self,
        file_metadata: FileMetadata,
        count: int = 4,
    ) -> List[str]:
        """Generate suggested analytical questions for a dataset."""
        messages = self._registry.build_suggested_questions_messages(
            file_metadata, count
        )

        try:
            response = self._provider.generate(
                messages, temperature=0.7, max_tokens=300
            )
            content = response.content.strip()
            start = content.find("[")
            end = content.rfind("]") + 1
            if start != -1 and end > start:
                questions = json.loads(content[start:end])
                return [str(q) for q in questions[:count]]
            return []
        except Exception:
            logger.warning("Suggested questions generation failed.")
            return []

    def build_error_context(
        self,
        error: Exception,
        file_metadata: FileMetadata,
    ) -> str:
        """Build an error context string for retry prompts."""
        from backend.llm.prompts.correction_prompt import build_error_recovery_prompt

        error_type = type(error).__name__
        error_message = str(error)

        if isinstance(error, (CodeValidationError, ExecutionRuntimeError)):
            error_message = error.message

        columns = [c.name for c in file_metadata.columns]
        dtypes = [f"{c.name}: {c.dtype}" for c in file_metadata.columns]

        return build_error_recovery_prompt(
            error_type=error_type,
            error_message=error_message,
            columns=columns,
            dtypes=dtypes,
        )

    # ── Private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _detect_chart_type(code: str) -> Optional[str]:
        """Detect the chart type from generated matplotlib/plotly code."""
        for pattern, chart_name in _CHART_TYPE_PATTERNS.items():
            if re.search(pattern, code):
                return chart_name

        if "plt.savefig" in code or "chart_path" in code:
            return "chart"

        return None
