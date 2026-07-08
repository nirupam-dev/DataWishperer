"""
User Prompt — Dynamic user message construction.

Transforms the raw user question into a structured prompt that includes:
    - The reasoning instruction preamble
    - The actual user question
    - Error context (on retries)
    - Ambiguity handling hints

This module is responsible for the FINAL message in the conversation
(the "user" role message that triggers code generation).
"""

from __future__ import annotations

from typing import Optional


def build_user_prompt(
    question: str,
    reasoning_preamble: str,
    error_context: Optional[str] = None,
) -> str:
    """
    Construct the user-role message for code generation.

    Args:
        question: The user's natural language question.
        reasoning_preamble: Developer reasoning instructions.
        error_context: Optional error from a previous failed attempt.

    Returns:
        The assembled user prompt string.
    """
    formatted_question = f'USER QUESTION:\n"{question}"'
    parts = [reasoning_preamble, formatted_question]

    if error_context:
        parts.append(error_context)

    return "\n\n".join(parts)
