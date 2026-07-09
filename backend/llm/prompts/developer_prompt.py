"""
Developer Prompt — Internal chain-of-thought reasoning instructions.

This prompt guides the LLM's internal reasoning process. It forces
step-by-step analysis BEFORE code generation, ensuring the model:
    1. Inspects the dataframe schema before writing code
    2. Identifies the correct columns
    3. Plans transformations
    4. Selects the right pandas operations
    5. Validates its own approach

The reasoning output is NEVER shown to the user — it's extracted and
logged internally for debugging. The user sees only the final code.

Qwen2.5 Optimization:
    - Explicit numbered reasoning steps
    - "Do NOT show" instruction — Qwen2.5 respects negative directives
    - Separation of reasoning from output with clear markers
"""

from __future__ import annotations

from typing import Optional


# ── Reasoning Prompt (injected before the user question) ─────────────────────
# Forces internal step-by-step analysis before code generation

DEVELOPER_REASONING_PROMPT: str = """\
Before writing code, silently reason through these steps:
1. INSPECT: Which columns from the dataset are relevant to this question?
2. VALIDATE: Are those columns the correct dtype? Do they need conversion?
3. NULLS: Do any relevant columns have significant null values?
4. PLAN: What pandas operations will answer this question?
5. EDGE CASES: What could go wrong? (empty results, division by zero, mixed types)
6. VISUALIZATION: Does this question need a chart? If so, what type?

Then write ONLY the code block. Do NOT output your reasoning steps.\
"""


# ── DataFrame Inspection Prompt ──────────────────────────────────────────────
# Used when the model needs to inspect the dataframe before answering

DATAFRAME_INSPECTION_PROMPT: str = """\
CRITICAL — Before writing code, mentally verify that the required columns exist in the dataset context and have appropriate dtypes.
If a needed column is missing, your code should be:
result = "Cannot answer: column '[name]' not found in dataset"

If the user asks to "explain" or "describe" data (e.g., "explain the first 5 rows"), write the Pandas code to extract that data (e.g., `result = df.head(5)`). The system will automatically generate the text explanation in the next step.

Otherwise, proceed directly to writing the Pandas code that answers the user's question.\
"""


# ── Statistical Analysis Prompt ──────────────────────────────────────────────
# Injected when the question involves statistics or numerical analysis

STATISTICAL_ANALYSIS_PROMPT: str = """\
STATISTICAL RIGOR REQUIREMENTS:
- IMPORTANT: Do NOT call df.describe() on the entire DataFrame — it can exhaust memory on small servers.
  Instead, use targeted aggregations on specific columns: df['col'].mean(), df[['col1','col2']].describe(), etc.
- Report sample size (n) alongside statistics
- Use .median() instead of .mean() for skewed distributions
- For correlations, use .corr() with method='pearson' or 'spearman'
- For comparisons, show both absolute and percentage differences
- Round all statistics to appropriate significant figures
- Acknowledge limitations: small sample sizes, missing data impact\
"""


def build_developer_prompt(
    question: str,
    include_statistics: bool = False,
    include_inspection: bool = True,
) -> str:
    """
    Assemble the developer-level reasoning prompt.

    Composes the relevant developer instructions based on the
    question type and pipeline stage.

    Args:
        question: The user's question (used to detect statistical intent).
        include_statistics: Force include statistical rigor instructions.
        include_inspection: Include dataframe inspection reminder.

    Returns:
        The assembled developer prompt string.
    """
    parts = [DEVELOPER_REASONING_PROMPT]

    if include_inspection:
        parts.append(DATAFRAME_INSPECTION_PROMPT)

    # Auto-detect statistical questions
    stat_keywords = {
        "average", "mean", "median", "std", "deviation", "correlation",
        "percentile", "quartile", "distribution", "variance", "outlier",
        "significant", "regression", "trend", "p-value", "confidence",
        "hypothesis", "skew", "kurtosis", "normal",
    }
    question_lower = question.lower()
    needs_stats = include_statistics or any(
        kw in question_lower for kw in stat_keywords
    )

    if needs_stats:
        parts.append(STATISTICAL_ANALYSIS_PROMPT)

    return "\n\n".join(parts)
