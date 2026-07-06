"""
Reflection Prompt — Self-validation loop for generated code.

Implements a "generate → reflect → correct" pattern where the LLM
reviews its own generated code BEFORE execution to catch errors that
would otherwise require a full retry cycle.

This saves significant latency: catching a column name typo in
reflection (~200ms) is much cheaper than executing → failing →
debugging → re-executing (~4000ms).

Qwen2.5 Optimization:
    - Checklist format (Qwen2.5 follows checklists reliably)
    - Binary yes/no output format (reduces ambiguity in parsing)
    - Explicit "output ONLY" instruction (prevents prose contamination)
"""

from __future__ import annotations

from typing import List


# ── Pre-Execution Reflection Prompt ──────────────────────────────────────────
# Asks the LLM to validate its own code before we send it to the sandbox

REFLECTION_PROMPT: str = """\
Review this Python code for correctness BEFORE execution.

CODE TO REVIEW:
```python
{code}
```

DATASET COLUMNS: {columns}

CHECKLIST — answer each with PASS or FAIL:
1. COLUMNS: Does every column reference match a name in DATASET COLUMNS exactly?
2. TYPES: Are numeric operations applied only to numeric columns?
3. RESULT: Is there a `result = ...` assignment?
4. SAFETY: No print(), eval(), exec(), os, subprocess, or network calls?
5. SYNTAX: Is the Python syntax valid?
6. EDGE CASES: Are NaN values and empty DataFrames handled?

OUTPUT FORMAT (use EXACTLY this):
VERDICT: PASS or FAIL
ISSUES: [list any issues found, or "None"]
FIX: [corrected code in ```python``` fences if FAIL, or "N/A" if PASS]\
"""


# ── Post-Execution Reflection Prompt ────────────────────────────────────────
# Validates the result AFTER execution to catch silent logical errors

POST_EXECUTION_REFLECTION: str = """\
The code executed successfully. Verify the result makes sense.

QUESTION: {question}
CODE: {code}
RESULT TYPE: {result_type}
RESULT PREVIEW: {result_preview}

SANITY CHECK:
1. Does the result actually answer the question asked?
2. Are the numbers in a reasonable range?
3. Is the result empty when it shouldn't be?

If the result looks wrong, explain why in 1 sentence.
Otherwise respond with: VERIFIED\
"""


def build_reflection_prompt(
    code: str,
    column_names: List[str],
) -> str:
    """
    Build the pre-execution reflection prompt.

    Args:
        code: The generated code to validate.
        column_names: Actual column names for cross-reference.

    Returns:
        The formatted reflection prompt.
    """
    columns_str = ", ".join(f"'{c}'" for c in column_names)
    return REFLECTION_PROMPT.format(
        code=code[:1000],
        columns=columns_str,
    )


def build_post_execution_reflection(
    question: str,
    code: str,
    result_type: str,
    result_preview: str,
) -> str:
    """
    Build the post-execution sanity check prompt.

    Args:
        question: The user's original question.
        code: The executed code.
        result_type: Type of result (text, dataframe, chart, etc.).
        result_preview: First 500 chars of the result.

    Returns:
        The formatted post-execution reflection prompt.
    """
    return POST_EXECUTION_REFLECTION.format(
        question=question[:200],
        code=code[:500],
        result_type=result_type,
        result_preview=result_preview[:500],
    )
