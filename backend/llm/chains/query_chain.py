"""
Query chain — Orchestrates the full question-to-answer pipeline.

This is the core chain that:
    1. Builds the prompt (system + context + history + question)
    2. Calls the LLM provider
    3. Parses the output to extract code
    4. Returns structured results

Implements retry logic with error context injection on failures.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from backend.core.config import ChatSettings, get_settings
from backend.core.exceptions import (
    CodeValidationError,
    ExecutionRuntimeError,
    GenerationError,
)
from backend.core.logging_config import get_logger
from backend.llm.base_provider import BaseLLMProvider
from backend.llm.chains.output_parser import OutputParser
from backend.llm.prompts.context_builder import ContextBuilder
from backend.llm.prompts.few_shot import format_few_shot_examples
from backend.llm.prompts.system import (
    ERROR_RECOVERY_PROMPT,
    SUGGESTED_QUESTIONS_PROMPT,
    SYSTEM_PROMPT,
    TITLE_GENERATION_PROMPT,
)
from backend.models.schemas import FileMetadata, LLMResponse

logger = get_logger(__name__)


class QueryChain:
    """
    Orchestrates the question → code generation pipeline.

    Responsible for:
        - Assembling prompts with system instructions, dataset context,
          chat history, and the user's question
        - Calling the LLM via the injected provider
        - Parsing generated code from the response
        - Managing retry logic with error context injection

    Args:
        llm_provider: The LLM backend to use for generation.
        output_parser: Parser for extracting code from LLM output.
        context_builder: Builder for dataset context strings.
        chat_settings: Chat configuration.
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        output_parser: Optional[OutputParser] = None,
        context_builder: Optional[ContextBuilder] = None,
        chat_settings: Optional[ChatSettings] = None,
    ) -> None:
        self._llm = llm_provider
        self._parser = output_parser or OutputParser()
        self._context_builder = context_builder or ContextBuilder()
        self._chat_settings = chat_settings or get_settings().chat

    def generate_code(
        self,
        question: str,
        file_metadata: FileMetadata,
        history: Optional[List[Dict[str, str]]] = None,
        error_context: Optional[str] = None,
        attempt: int = 1,
    ) -> tuple[str, LLMResponse]:
        """
        Generate Python code to answer a user's question about their data.

        Args:
            question: The user's natural language question.
            file_metadata: Metadata about the uploaded CSV file.
            history: Optional list of previous chat messages for context.
            error_context: Optional error from a previous failed attempt.
            attempt: Current attempt number (1-indexed).

        Returns:
            A tuple of ``(extracted_code, llm_response)``.

        Raises:
            GenerationError: If code cannot be extracted from the response.
            OllamaConnectionError: If the LLM is unreachable.
        """
        messages = self._build_messages(
            question=question,
            file_metadata=file_metadata,
            history=history,
            error_context=error_context,
            attempt=attempt,
        )

        logger.info(
            "Generating code: question='%s' (attempt %d, %d messages)",
            question[:80], attempt, len(messages),
        )

        # Call the LLM
        llm_response = self._llm.generate(messages)

        # Extract code from the response
        code = self._parser.extract_code(llm_response.content)

        logger.info(
            "Code generated: %d chars, %d tokens, %.0fms",
            len(code), llm_response.tokens_used, llm_response.latency_ms,
        )

        return code, llm_response

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
            response = self._llm.generate(messages, temperature=0.5, max_tokens=30)
            title = response.content.strip().strip('"').strip("'")
            # Ensure it's reasonable length
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
        Generate suggested questions based on the dataset.

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
            response = self._llm.generate(messages, temperature=0.7, max_tokens=300)
            # Parse JSON array from response
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

    # ── Private helpers ──────────────────────────────────────────────────

    def _build_messages(
        self,
        question: str,
        file_metadata: FileMetadata,
        history: Optional[List[Dict[str, str]]],
        error_context: Optional[str],
        attempt: int,
    ) -> List[Dict[str, str]]:
        """
        Assemble the full message list for the LLM.

        Message order:
            1. System prompt (role + rules + output format)
            2. Dataset context
            3. Few-shot examples (attempt 1 only, to save tokens on retries)
            4. Chat history (sliding window)
            5. User question (with error context on retries)
        """
        messages: List[Dict[str, str]] = []

        # 1. System prompt
        messages.append({"role": "system", "content": SYSTEM_PROMPT})

        # 2. Dataset context
        if attempt <= 1:
            dataset_context = self._context_builder.build(file_metadata)
        else:
            dataset_context = self._context_builder.build_compact(file_metadata)

        messages.append({
            "role": "system",
            "content": f"DATASET CONTEXT:\n{dataset_context}",
        })

        # 3. Few-shot examples (first attempt only)
        if attempt <= 1:
            messages.append({
                "role": "system",
                "content": format_few_shot_examples(),
            })

        # 4. Chat history (sliding window)
        if history:
            window_size = self._chat_settings.history_window_size
            for msg in history[-window_size:]:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                })

        # 5. User question (with error context on retries)
        if error_context and attempt > 1:
            user_content = f"{question}\n\n{error_context}"
        else:
            user_content = question

        messages.append({"role": "user", "content": user_content})

        return messages

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
