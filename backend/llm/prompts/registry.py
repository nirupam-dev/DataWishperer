"""
Prompt Registry — Central orchestrator for the modular prompt architecture.

This is the single entry point for all prompt assembly in the application.
It composes prompts from the 8 specialized modules based on pipeline stage
and context:

    ┌─────────────────────────────────────────────────────────┐
    │                    PROMPT REGISTRY                       │
    │                                                         │
    │  Stage 2 (Generate):                                    │
    │    system_prompt + safety_prompt + developer_prompt      │
    │    + context_builder + few_shot + user_prompt            │
    │                                                         │
    │  Stage 6 (Explain):                                     │
    │    output_format_prompt.EXPLANATION_PROMPT               │
    │                                                         │
    │  Stage 7 (Chart Reason):                                │
    │    output_format_prompt.CHART_EXPLANATION_PROMPT         │
    │                                                         │
    │  Stage 8 (Debug):                                       │
    │    system_prompt + correction_prompt + safety_prompt     │
    │                                                         │
    │  Retry:                                                 │
    │    system_prompt_compact + retry_prompt + safety_prompt  │
    │                                                         │
    │  Reflection:                                            │
    │    reflection_prompt (pre/post execution)                │
    │                                                         │
    │  Ambiguity:                                             │
    │    output_format_prompt.AMBIGUITY_DETECTION_PROMPT       │
    └─────────────────────────────────────────────────────────┘

Design Philosophy:
    - Each prompt module is a pure function or constant — no state
    - The registry is the ONLY place where prompts are composed
    - Pipeline stages request prompts by semantic name, not by module
    - This makes it trivial to A/B test prompt variants
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.core.logging_config import get_logger
from backend.llm.prompts.system_prompt import SYSTEM_PROMPT, SYSTEM_PROMPT_COMPACT
from backend.llm.prompts.developer_prompt import build_developer_prompt
from backend.llm.prompts.safety_prompt import build_safety_prompt
from backend.llm.prompts.correction_prompt import (
    build_debug_prompt,
    build_error_recovery_prompt,
    build_column_mismatch_prompt,
)
from backend.llm.prompts.retry_prompt import build_retry_prompt, build_compact_context
from backend.llm.prompts.reflection_prompt import (
    build_reflection_prompt,
    build_post_execution_reflection,
)
from backend.llm.prompts.user_prompt import build_user_prompt
from backend.llm.prompts.output_format_prompt import (
    AMBIGUITY_DETECTION_PROMPT,
    CHART_EXPLANATION_PROMPT,
    CONTEXT_SWITCH_PROMPT,
    EXPLANATION_PROMPT,
    SUGGESTED_QUESTIONS_PROMPT,
    TITLE_GENERATION_PROMPT,
    VISUALIZATION_SELECTION_PROMPT,
)
from backend.llm.prompts.context_builder import ContextBuilder
from backend.llm.prompts.few_shot import format_few_shot_examples
from backend.models.schemas import FileMetadata

logger = get_logger(__name__)


class PromptRegistry:
    """
    Central prompt orchestrator for the DataWhisperer pipeline.

    Assembles prompt message lists from modular components based on
    the current pipeline stage and context. All prompt composition
    flows through this class.

    Usage:
        registry = PromptRegistry()

        # For code generation (Stage 2):
        messages = registry.build_generation_messages(
            question="What is the average revenue?",
            file_metadata=metadata,
            session_history=[...],
        )

        # For debugging (Stage 8):
        messages = registry.build_debug_messages(
            failed_code="...",
            error=some_exception,
            file_metadata=metadata,
        )
    """

    def __init__(self) -> None:
        self._context_builder = ContextBuilder()

    # ── Stage 2: Code Generation Messages ────────────────────────────────

    def build_generation_messages(
        self,
        question: str,
        file_metadata: FileMetadata,
        session_history: Optional[List[Dict[str, str]]] = None,
        all_datasets: Optional[Dict[str, FileMetadata]] = None,
        error_context: Optional[str] = None,
        attempt: int = 1,
    ) -> List[Dict[str, str]]:
        """
        Build the complete message list for code generation.

        Composes: system + safety + dataset context + few-shot +
                  history + developer reasoning + user question.

        Args:
            question: The user's question.
            file_metadata: Active dataset metadata.
            session_history: Previous conversation messages.
            all_datasets: All loaded datasets (for multi-dataset context).
            error_context: Error from a previous failed attempt.
            attempt: Current attempt number (1-indexed).

        Returns:
            List of message dicts ready for the LLM provider.
        """
        messages: List[Dict[str, str]] = []

        # 1. System prompt (full on first attempt, compact on retry)
        if attempt <= 1:
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
        else:
            messages.append({"role": "system", "content": SYSTEM_PROMPT_COMPACT})

        # 2. Safety guardrails with column validation
        column_names = [c.name for c in file_metadata.columns]
        safety = build_safety_prompt(
            column_names=column_names,
            include_code_safety=(attempt <= 1),
        )
        messages.append({"role": "system", "content": safety})

        # 3. Dataset context (full on first attempt, compact on retry)
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

        # 4. Few-shot examples (first attempt only to save tokens)
        if attempt <= 1:
            messages.append({
                "role": "system",
                "content": format_few_shot_examples(),
            })

        # 5. Conversation history
        if session_history:
            for msg in session_history:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                })

        # 6. Developer reasoning + user question
        developer_prompt = build_developer_prompt(
            question=question,
            include_inspection=(attempt <= 1),
        )
        user_content = build_user_prompt(
            question=question,
            reasoning_preamble=developer_prompt,
            error_context=error_context,
        )
        messages.append({"role": "user", "content": user_content})

        logger.debug(
            "Built generation messages: %d messages, attempt %d",
            len(messages),
            attempt,
        )

        return messages

    # ── Stage 6: Explanation Messages ────────────────────────────────────

    def build_explanation_messages(
        self,
        code: str,
        result_summary: str,
    ) -> List[Dict[str, str]]:
        """Build messages for the explanation stage."""
        prompt = EXPLANATION_PROMPT.format(
            code=code[:500],
            result_summary=result_summary[:1000],
        )
        return [{"role": "user", "content": prompt}]

    # ── Stage 7: Chart Explanation Messages ──────────────────────────────

    def build_chart_explanation_messages(
        self,
        code: str,
        question: str,
        chart_type: str,
    ) -> List[Dict[str, str]]:
        """Build messages for chart type reasoning."""
        prompt = CHART_EXPLANATION_PROMPT.format(
            code=code[:400],
            question=question[:200],
            chart_type=chart_type,
        )
        return [{"role": "user", "content": prompt}]

    # ── Stage 8: Debug Messages ──────────────────────────────────────────

    def build_debug_messages(
        self,
        failed_code: str,
        error_type: str,
        error_message: str,
        file_metadata: FileMetadata,
    ) -> List[Dict[str, str]]:
        """Build messages for the auto-debug stage."""
        columns = [c.name for c in file_metadata.columns]
        dtypes = [f"{c.name}: {c.dtype}" for c in file_metadata.columns]

        # Use column mismatch prompt for KeyError
        if "KeyError" in error_type:
            debug_prompt = build_column_mismatch_prompt(
                error_message=error_message,
                columns=columns,
            )
        else:
            debug_prompt = build_debug_prompt(
                failed_code=failed_code,
                error_type=error_type,
                error_message=error_message,
                columns=columns,
                dtypes=dtypes,
                row_count=file_metadata.row_count,
            )

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": debug_prompt},
        ]

    # ── Reflection Messages ──────────────────────────────────────────────

    def build_reflection_messages(
        self,
        code: str,
        file_metadata: FileMetadata,
    ) -> List[Dict[str, str]]:
        """Build messages for pre-execution code reflection."""
        column_names = [c.name for c in file_metadata.columns]
        prompt = build_reflection_prompt(code=code, column_names=column_names)
        return [{"role": "user", "content": prompt}]

    def build_post_reflection_messages(
        self,
        question: str,
        code: str,
        result_type: str,
        result_preview: str,
    ) -> List[Dict[str, str]]:
        """Build messages for post-execution result validation."""
        prompt = build_post_execution_reflection(
            question=question,
            code=code,
            result_type=result_type,
            result_preview=result_preview,
        )
        return [{"role": "user", "content": prompt}]

    # ── Ambiguity Detection Messages ─────────────────────────────────────

    def build_ambiguity_messages(
        self,
        question: str,
        file_metadata: FileMetadata,
    ) -> List[Dict[str, str]]:
        """Build messages for ambiguity detection."""
        columns = ", ".join(f"'{c.name}'" for c in file_metadata.columns)
        prompt = AMBIGUITY_DETECTION_PROMPT.format(
            question=question[:300],
            columns=columns,
        )
        return [{"role": "user", "content": prompt}]

    # ── Visualization Selection Messages ─────────────────────────────────

    def build_visualization_messages(
        self,
        question: str,
        n_categories: int,
        n_numeric: int,
        has_dates: bool,
        data_size: int,
    ) -> List[Dict[str, str]]:
        """Build messages for automatic chart type selection."""
        prompt = VISUALIZATION_SELECTION_PROMPT.format(
            question=question[:200],
            n_categories=n_categories,
            n_numeric=n_numeric,
            has_dates=str(has_dates),
            data_size=data_size,
        )
        return [{"role": "user", "content": prompt}]

    # ── Auxiliary Messages ───────────────────────────────────────────────

    def build_title_messages(self, question: str) -> List[Dict[str, str]]:
        """Build messages for session title generation."""
        prompt = TITLE_GENERATION_PROMPT.format(question=question[:200])
        return [{"role": "user", "content": prompt}]

    def build_suggested_questions_messages(
        self,
        file_metadata: FileMetadata,
        count: int = 4,
    ) -> List[Dict[str, str]]:
        """Build messages for suggested questions generation."""
        columns_info = ", ".join(
            f"{c.name} ({c.dtype})" for c in file_metadata.columns
        )
        prompt = SUGGESTED_QUESTIONS_PROMPT.format(
            columns_info=columns_info,
            count=count,
        )
        return [{"role": "user", "content": prompt}]

    def build_context_switch_messages(
        self,
        file_metadata: FileMetadata,
    ) -> List[Dict[str, str]]:
        """Build messages for dataset context switch notification."""
        columns = ", ".join(c.name for c in file_metadata.columns)
        prompt = CONTEXT_SWITCH_PROMPT.format(
            filename=file_metadata.original_name,
            row_count=file_metadata.row_count,
            col_count=file_metadata.col_count,
            columns=columns,
        )
        return [{"role": "system", "content": prompt}]
