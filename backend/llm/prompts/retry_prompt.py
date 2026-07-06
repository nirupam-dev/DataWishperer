"""
Retry Prompt — Structured retry with progressive context reduction.

Progressive retry levels:
    Level 1: Error context + compact schema
    Level 2: Pattern matching + minimal context + forced simplification
"""

from __future__ import annotations
from typing import Optional


RETRY_LEVEL_1_PROMPT: str = """\
Your previous code attempt FAILED. Here is the error:

ERROR: {error_type}: {error_message}

WHAT WENT WRONG: {diagnosis}

RETRY INSTRUCTIONS:
1. Do NOT repeat the same mistake
2. Address the specific error above
3. Use ONLY verified column names from the dataset context
4. Add defensive checks for the issue that caused the failure
5. Keep the same analytical goal

Write corrected code in ```python ... ``` fences.
Assign to `result`. Output ONLY the code block.\
"""


RETRY_LEVEL_2_PROMPT: str = """\
TWO previous attempts failed. This is your FINAL chance.

Previous errors:
1. {error_1}
2. {error_2}

SIMPLIFIED APPROACH — write the SIMPLEST possible code:
- Use basic pandas only: .groupby(), .mean(), .sum(), .value_counts()
- Add .dropna() before every numeric operation
- If you cannot answer, set: result = "Cannot answer: [reason]"

Dataset columns: {columns}

Write code in ```python ... ``` fences. Assign to `result`.\
"""


RETRY_COMPACT_CONTEXT: str = """\
Dataset: {filename} ({row_count} rows × {col_count} cols)
Columns: {columns_with_types}\
"""


def build_retry_prompt(
    attempt: int,
    error_type: str,
    error_message: str,
    diagnosis: Optional[str] = None,
    previous_errors: Optional[list] = None,
    columns: Optional[list] = None,
) -> str:
    """Build the retry prompt based on attempt number."""
    if attempt <= 2:
        return RETRY_LEVEL_1_PROMPT.format(
            error_type=error_type,
            error_message=error_message[:300],
            diagnosis=diagnosis or "Review column names and types.",
        )
    else:
        errors = previous_errors or [f"{error_type}: {error_message}"]
        error_1 = errors[0] if len(errors) > 0 else "Unknown"
        error_2 = errors[1] if len(errors) > 1 else errors[0]
        cols = ", ".join(columns) if columns else "See dataset context"
        return RETRY_LEVEL_2_PROMPT.format(
            error_1=str(error_1)[:150],
            error_2=str(error_2)[:150],
            columns=cols,
        )


def build_compact_context(
    filename: str,
    row_count: int,
    col_count: int,
    columns_with_types: str,
) -> str:
    """Build a token-efficient dataset context for retry attempts."""
    return RETRY_COMPACT_CONTEXT.format(
        filename=filename,
        row_count=row_count,
        col_count=col_count,
        columns_with_types=columns_with_types,
    )
