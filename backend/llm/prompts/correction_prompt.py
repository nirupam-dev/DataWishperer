"""
Correction Prompt — Auto-debug and error recovery instructions.

When generated code fails execution, these prompts guide the LLM to:
    1. Diagnose the root cause from the error message
    2. Cross-reference against actual dataset schema
    3. Write a corrected version with defensive checks
    4. Preserve the original analytical intent

This module provides TWO levels of correction:
    - DEBUG_PROMPT: Diagnostic, root-cause-focused (used on first failure)
    - ERROR_RECOVERY_PROMPT: Pattern-matching, quick fix (used on second failure)

Qwen2.5 Optimization:
    - Error type → fix pattern mapping (Qwen2.5 excels at pattern matching)
    - Concrete column list injection (eliminates guessing)
    - "Fix the bug, don't change the analysis" instruction (prevents drift)
"""

from __future__ import annotations

from typing import List, Optional


# ── Primary Debug Prompt ─────────────────────────────────────────────────────
# Used on FIRST execution failure — diagnostic and thorough

DEBUG_PROMPT: str = """\
The following Python pandas code FAILED during execution.

FAILED CODE:
```python
{failed_code}
```

ERROR:
{error_type}: {error_message}

DATASET SCHEMA:
- Columns: {columns}
- Types: {dtypes}
- Row count: {row_count}

DEBUG INSTRUCTIONS:
1. Read the error message carefully — identify the ROOT CAUSE
2. Cross-reference column names against the DATASET SCHEMA above
3. Check for type mismatches between operations and column dtypes
4. Write a FIXED version with defensive checks:
   - Add .dropna() before numeric operations
   - Add pd.to_numeric(errors='coerce') for type conversions
   - Add column existence checks where appropriate
5. PRESERVE the same analytical intent — fix the bug, NOT the analysis
6. If the original code was generating a chart:
   - YOU MUST STILL CALL plt.savefig(chart_path)
   - Assign the UNDERLYING DATA to `result` (e.g., the filtered DataFrame/Series)
   - NEVER assign the Matplotlib axes/figure to `result`.

OUTPUT: Write the corrected code in ```python ... ``` fences.
Assign the final data answer to `result`.\
"""


# ── Fallback Error Recovery Prompt ───────────────────────────────────────────
# Used when the first debug attempt also fails — simpler, pattern-based

ERROR_RECOVERY_PROMPT: str = """\
Your previous code FAILED with:
  {error_type}: {error_message}

Available columns: {columns}
Column types: {dtypes}

COMMON FIX PATTERNS:
- KeyError → wrong column name. Check exact spelling and case from the list above.
- TypeError → type mismatch. Use pd.to_numeric(df['col'], errors='coerce').
- ValueError → bad conversion. Use pd.to_datetime(df['col'], errors='coerce').
- AttributeError → wrong method. Check pandas API for the correct method name.
- ZeroDivisionError → add a check: if denominator != 0.
- IndexError → empty result. Add len() check before indexing.

Write corrected code in ```python ... ``` fences.
Assign the final answer to `result` (if chart, assign underlying data, NOT axes. Call plt.savefig).
Output ONLY the code block.\
"""


# ── Column Mismatch Recovery ────────────────────────────────────────────────
# Specialized prompt for the most common error: wrong column names

COLUMN_MISMATCH_PROMPT: str = """\
The code failed because it referenced a column that does not exist.

Error: {error_message}

THE ONLY VALID COLUMNS ARE:
{column_list}

Rewrite the code using ONLY columns from the list above.
If no suitable column exists, set:
result = "Cannot answer: the dataset does not contain a column for [what you need]"

Write corrected code in ```python ... ``` fences.\
"""


def build_debug_prompt(
    failed_code: str,
    error_type: str,
    error_message: str,
    columns: List[str],
    dtypes: List[str],
    row_count: int,
) -> str:
    """
    Build the primary debug prompt with full context.

    Args:
        failed_code: The code that failed.
        error_type: The exception class name.
        error_message: The error description.
        columns: List of column names.
        dtypes: List of "col: dtype" strings.
        row_count: Number of rows in the dataset.

    Returns:
        The formatted debug prompt.
    """
    return DEBUG_PROMPT.format(
        failed_code=failed_code[:800],
        error_type=error_type,
        error_message=error_message[:300],
        columns=", ".join(f"'{c}'" for c in columns),
        dtypes=", ".join(dtypes),
        row_count=row_count,
    )


def build_error_recovery_prompt(
    error_type: str,
    error_message: str,
    columns: List[str],
    dtypes: List[str],
) -> str:
    """
    Build the fallback error recovery prompt.

    Args:
        error_type: The exception class name.
        error_message: The error description.
        columns: List of column names.
        dtypes: List of "col: dtype" strings.

    Returns:
        The formatted recovery prompt.
    """
    return ERROR_RECOVERY_PROMPT.format(
        error_type=error_type,
        error_message=error_message[:200],
        columns=", ".join(f"'{c}'" for c in columns),
        dtypes=", ".join(dtypes),
    )


def build_column_mismatch_prompt(
    error_message: str,
    columns: List[str],
) -> str:
    """
    Build a specialized prompt for column name errors.

    Args:
        error_message: The KeyError or similar message.
        columns: List of valid column names.

    Returns:
        The formatted column mismatch recovery prompt.
    """
    column_list = "\n".join(f"  - '{c}'" for c in columns)
    return COLUMN_MISMATCH_PROMPT.format(
        error_message=error_message[:200],
        column_list=column_list,
    )
