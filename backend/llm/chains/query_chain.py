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
from backend.llm.prompts.context_builder import ContextBuilder
from backend.llm.prompts.few_shot import format_few_shot_examples
from backend.llm.prompts.system import (
    CHART_EXPLANATION_PROMPT,
    DEBUG_PROMPT,
    ERROR_RECOVERY_PROMPT,
    EXPLANATION_PROMPT,
    REASONING_PROMPT,
    SUGGESTED_QUESTIONS_PROMPT,
    SYSTEM_PROMPT,
    TITLE_GENERATION_PROMPT,
)
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
}


class QueryChain:
    """
    LangChain-based query chain for the code interpreter pipeline.

    Handles code generation, explanation, chart reasoning, and auto-debug.

    Interpreter Stages:
        - ``generate_code()``: Stage 2 — Generate optimized Pandas code
        - ``generate_explanation()``: Stage 6 — Plain English code explanation
        - ``generate_chart_explanation()``: Stage 7 — Chart type reasoning
        - ``debug_code()``: Stage 8 — Auto-debug failed code

    Args:
        provider: The Ollama LLM provider.
        output_parser: Parser for extracting code from LLM output.
        context_builder: Builder for dataset context strings.
        memory: Conversation memory manager.
        chat_settings: Chat configuration.
    """

    def __init__(
        self,
        provider: OllamaProvider,
        output_parser: Optional[OutputParser] = None,
        context_builder: Optional[ContextBuilder] = None,
        memory: Optional[ConversationMemory] = None,
        chat_settings: Optional[ChatSettings] = None,
    ) -> None:
        self._provider = provider
        self._parser = output_parser or OutputParser()
        self._context_builder = context_builder or ContextBuilder()
        self._memory = memory or ConversationMemory()
        self._chat_settings = chat_settings or get_settings().chat

    @property
    def memory(self) -> ConversationMemory:
        """Expose the conversation memory for external use."""
        return self._memory

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
        messages = self._build_messages(
            question=question,
            file_metadata=file_metadata,
            session_id=session_id,
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

        Generates a user-facing explanation that avoids jargon, focuses
        on key findings, and references specific numbers from the results.

        Args:
            code: The executed Python code.
            result_summary: Brief summary of the execution result.

        Returns:
            A 2-4 sentence explanation string.
        """
        prompt = EXPLANATION_PROMPT.format(
            code=code[:500],
            result_summary=result_summary[:300],
        )
        messages = [{"role": "user", "content": prompt}]

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

        Only called when the execution result includes a chart. Detects
        the chart type from the code and generates reasoning for the choice.

        Args:
            code: The executed code that generated a chart.
            question: The user's original question.

        Returns:
            A 1-2 sentence chart type explanation, or None if detection fails.
        """
        chart_type = self._detect_chart_type(code)
        if not chart_type:
            return None

        prompt = CHART_EXPLANATION_PROMPT.format(
            code=code[:400],
            question=question[:200],
            chart_type=chart_type,
        )
        messages = [{"role": "user", "content": prompt}]

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

        columns = ", ".join(f"'{c.name}'" for c in file_metadata.columns)
        dtypes = ", ".join(
            f"{c.name}: {c.dtype}" for c in file_metadata.columns
        )

        debug_prompt = DEBUG_PROMPT.format(
            failed_code=failed_code[:800],
            error_type=error_type,
            error_message=error_message,
            columns=columns,
            dtypes=dtypes,
            row_count=file_metadata.row_count,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": debug_prompt},
        ]

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

    # ── Auxiliary Generation Methods ─────────────────────────────────────

    def generate_title(self, question: str) -> str:
        """
        Generate a short session title from the first user question.

        Args:
            question: The user's first question.

        Returns:
            A 4-6 word title string.
        """
        prompt = TITLE_GENERATION_PROMPT.format(question=question[:200])
        messages = [{"role": "user", "content": prompt}]

        try:
            response = self._provider.generate(
                messages, temperature=0.5, max_tokens=30
            )
            title = response.content.strip().strip('"').strip("'")
            # Ensure reasonable length
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
        """
        Generate suggested questions based on the dataset schema.

        Args:
            file_metadata: Metadata about the CSV file.
            count: Number of questions to generate.

        Returns:
            List of suggested question strings.
        """
        columns_info = ", ".join(
            f"{c.name} ({c.dtype})" for c in file_metadata.columns
        )
        prompt = SUGGESTED_QUESTIONS_PROMPT.format(
            columns_info=columns_info, count=count
        )
        messages = [{"role": "user", "content": prompt}]

        try:
            response = self._provider.generate(
                messages, temperature=0.7, max_tokens=300
            )
            content = response.content.strip()
            # Find the JSON array in the response
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
        """
        Build an error context string for retry prompts.

        Args:
            error: The exception from the previous attempt.
            file_metadata: File metadata for column reference.

        Returns:
            Formatted error recovery prompt string.
        """
        error_type = type(error).__name__
        error_message = str(error)

        if isinstance(error, (CodeValidationError, ExecutionRuntimeError)):
            error_message = error.message

        columns = ", ".join(f"'{c.name}'" for c in file_metadata.columns)
        dtypes = ", ".join(
            f"{c.name}: {c.dtype}" for c in file_metadata.columns
        )

        return ERROR_RECOVERY_PROMPT.format(
            error_type=error_type,
            error_message=error_message,
            columns=columns,
            dtypes=dtypes,
        )

    # ── Private helpers ──────────────────────────────────────────────────

    def _build_messages(
        self,
        question: str,
        file_metadata: FileMetadata,
        session_id: str,
        all_datasets: Optional[Dict[str, FileMetadata]],
        error_context: Optional[str],
        attempt: int,
    ) -> List[Dict[str, str]]:
        """
        Assemble the full message list for the LLM.

        Message order:
            1. System prompt (role + rules + output format)
            2. Dataset context (full on first attempt, compact on retry)
            3. Few-shot examples (attempt 1 only, saves tokens on retries)
            4. Conversation history from memory (sliding window)
            5. Reasoning instruction + user question
        """
        messages: List[Dict[str, str]] = []

        # 1. System prompt
        messages.append({"role": "system", "content": SYSTEM_PROMPT})

        # 2. Dataset context
        if attempt <= 1:
            if all_datasets and len(all_datasets) > 1:
                dataset_context = self._context_builder.build_multi_dataset_context(
                    active_metadata=file_metadata,
                    all_datasets=all_datasets,
                )
            else:
                dataset_context = self._context_builder.build(file_metadata)
        else:
            dataset_context = self._context_builder.build_compact(file_metadata)

        messages.append({
            "role": "system",
            "content": f"DATASET CONTEXT:\n{dataset_context}",
        })

        # 3. Few-shot examples (first attempt only to save tokens)
        if attempt <= 1:
            messages.append({
                "role": "system",
                "content": format_few_shot_examples(),
            })

        # 4. Conversation history from memory
        history_messages = self._memory.get_dict_messages(session_id)
        for msg in history_messages:
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })

        # 5. User question with reasoning instruction and error context
        user_parts = [REASONING_PROMPT, question]

        if error_context and attempt > 1:
            user_parts.append(error_context)

        messages.append({
            "role": "user",
            "content": "\n\n".join(user_parts),
        })

        return messages

    @staticmethod
    def _detect_chart_type(code: str) -> Optional[str]:
        """
        Detect the chart type from generated matplotlib/plotly code.

        Args:
            code: The Python code to analyze.

        Returns:
            A human-readable chart type string, or None if not a chart.
        """
        for pattern, chart_name in _CHART_TYPE_PATTERNS.items():
            if re.search(pattern, code):
                return chart_name

        # Fallback: check for generic plot commands
        if "plt.savefig" in code or "chart_path" in code:
            return "chart"

        return None
