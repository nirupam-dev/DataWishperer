"""
Safety Prompt — Anti-hallucination and data integrity guardrails.

This prompt prevents the most common LLM failures when generating
pandas code against real datasets:

    1. Column hallucination (inventing columns that don't exist)
    2. Data fabrication (making up numbers or statistics)
    3. Unsafe operations (file access, network, code injection)
    4. Type assumption errors (treating strings as numbers)
    5. Silent failures (code that runs but produces wrong results)

These guardrails are particularly important for Qwen2.5:7B because
smaller models are more prone to hallucination under ambiguity.

Qwen2.5 Optimization:
    - Negative rules with "NEVER" and "FORBIDDEN" — high compliance
    - Concrete examples of violations — reduces ambiguity
    - Short, scannable rules — fits well within limited context
"""

from __future__ import annotations

from typing import List


# ── Anti-Hallucination Guardrails ────────────────────────────────────────────
# Injected as a system message to prevent column/data fabrication

ANTI_HALLUCINATION_PROMPT: str = """\
ANTI-HALLUCINATION RULES (CRITICAL):
1. Use ONLY columns listed in the DATASET CONTEXT. No exceptions.
2. If you need a column that doesn't exist, say so — do NOT create it silently.
3. NEVER invent sample data, fake statistics, or placeholder values.
4. If the dataset cannot answer the question, set:
   result = "Cannot answer: [explain what data is missing]"
5. Prefer df.columns.tolist() checks over assumptions about column names.
6. NEVER rename columns unless the user explicitly asks for renaming.\
"""


# ── Code Safety Guardrails ───────────────────────────────────────────────────
# Prevents dangerous code patterns even if the LLM tries to generate them.
# Note: The sandbox validator is the REAL enforcement layer. These prompts
# are a defense-in-depth measure to prevent the model from even attempting
# unsafe operations (saving an LLM round-trip on rejection).

CODE_SAFETY_PROMPT: str = """\
FORBIDDEN OPERATIONS (code will be rejected if these appear):
- os, sys, subprocess, shutil — no filesystem or process access
- requests, urllib, httpx, socket — no network access
- eval(), exec(), compile() — no dynamic code execution
- __import__, importlib — no dynamic imports
- open(), read(), write() — no direct file I/O (except plt.savefig to chart_path)
- globals(), locals(), vars() — no scope inspection
- pickle, shelve, marshal — no serialization of arbitrary objects

ALLOWED IMPORTS ONLY: pandas, numpy, matplotlib, datetime, math, re, collections\
"""


# ── Column Validation Prompt ─────────────────────────────────────────────────
# Dynamic prompt that embeds the actual column list for cross-reference

COLUMN_VALIDATION_TEMPLATE: str = """\
VERIFIED COLUMNS IN THIS DATASET:
{column_list}

You MUST use column names EXACTLY as listed above.
Common mistakes to avoid:
- Wrong case: 'Name' vs 'name' vs 'NAME'
- Extra spaces: ' Name' vs 'Name'
- Abbreviations: 'qty' when the column is 'quantity'
- Plurals: 'sales' when the column is 'sale'\
"""


def build_safety_prompt(
    column_names: List[str],
    include_code_safety: bool = True,
) -> str:
    """
    Assemble the complete safety prompt with dataset-specific context.

    Args:
        column_names: List of actual column names from the dataset.
        include_code_safety: Whether to include code safety rules.

    Returns:
        The assembled safety prompt string.
    """
    parts = [ANTI_HALLUCINATION_PROMPT]

    if include_code_safety:
        parts.append(CODE_SAFETY_PROMPT)

    # Build column validation with actual columns
    if column_names:
        formatted_columns = "\n".join(
            f"  - '{col}'" for col in column_names
        )
        parts.append(
            COLUMN_VALIDATION_TEMPLATE.format(column_list=formatted_columns)
        )

    return "\n\n".join(parts)
