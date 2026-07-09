"""
Output Formatting Prompt — Controls how the LLM presents results.

Separate prompts for each output type in the interpreter pipeline:
    - Code explanation (Stage 6)
    - Chart type reasoning (Stage 7)
    - Session title generation
    - Suggested questions generation

These are intentionally separate from the code generation system prompt
to avoid bloating the primary context window.

Qwen2.5 Optimization:
    - Explicit length limits ("2-4 sentences", "4-6 words")
    - "Do NOT include code" — prevents format contamination
    - Template variables for dynamic context injection
"""

from __future__ import annotations


# ── Code Explanation Prompt (Stage 6) ────────────────────────────────────────

EXPLANATION_PROMPT: str = """\
You are explaining a data analysis result to a non-technical user.

Code executed:
```python
{code}
```

Result: {result_summary}

Write a clear, brief explanation (2-4 sentences):
- What the code does in simple terms (no jargon)
- Key findings with specific numbers (ONLY if numbers are present in the Result)
- Any notable patterns

Rules:
- Do NOT include any code or technical terms
- Do NOT say "the code does X" — say "the analysis shows X"
- Be specific about numbers IF they are in the Result, but NEVER invent, guess, or hallucinate numbers. If the Result is just a message (e.g. "chart saved"), describe what the chart represents without making up data.\
"""


# ── Chart Explanation Prompt (Stage 7) ───────────────────────────────────────

CHART_EXPLANATION_PROMPT: str = """\
A chart was generated for: "{question}"
Chart type: {chart_type}

Explain in 1-2 sentences:
1. WHY this chart type was chosen for this data
2. What the chart reveals

Do NOT include any code. Be concise.\
"""


# ── Title Generation Prompt ──────────────────────────────────────────────────

TITLE_GENERATION_PROMPT: str = """\
Generate a 4-6 word title for this data analysis question:
"{question}"

Return ONLY the title. No quotes, no punctuation at the end.\
"""


# ── Suggested Questions Prompt ───────────────────────────────────────────────

SUGGESTED_QUESTIONS_PROMPT: str = """\
Dataset columns: {columns_info}

Generate {count} analytical questions about this data. Cover:
- Aggregations (averages, totals, counts)
- Comparisons (top/bottom N, rankings)
- Distributions (outliers, unique values, ranges)
- Trends (if date columns exist)

Return as a JSON array: ["question1", "question2", ...]
No other text.\
"""


# ── Context Switch Prompt ────────────────────────────────────────────────────

CONTEXT_SWITCH_PROMPT: str = """\
The user has switched to a NEW dataset: {filename}
Shape: {row_count} rows × {col_count} columns
Columns: {columns}

IMPORTANT: All previous analysis was on a DIFFERENT dataset.
Use ONLY the columns listed above for new queries.\
"""


# ── Ambiguity Detection Prompt ───────────────────────────────────────────────
# Used when the question is unclear and could map to multiple analyses

AMBIGUITY_DETECTION_PROMPT: str = """\
The user asked: "{question}"

Available columns: {columns}

Is this question AMBIGUOUS? A question is ambiguous if:
- It refers to a concept not clearly mapped to any column
- Multiple columns could be the intended target
- The aggregation method is unclear (mean vs sum vs count)
- The time period or grouping is unspecified

If AMBIGUOUS, respond with:
AMBIGUOUS: [explain what is unclear]
SUGGESTION: [your best interpretation]
ALTERNATIVES: [1-2 alternative interpretations]

If CLEAR, respond with:
CLEAR: [proceed with code generation]\
"""


# ── Visualization Selection Prompt ───────────────────────────────────────────
# Guides automatic chart type selection based on data characteristics

VISUALIZATION_SELECTION_PROMPT: str = """\
Choose the BEST chart type for this analysis:

Question: "{question}"
Data characteristics:
- Number of categories: {n_categories}
- Number of numeric values: {n_numeric}
- Has time series: {has_dates}
- Data size: {data_size} rows

SELECTION RULES:
- Bar chart: comparing categories (<15 categories)
- Horizontal bar: comparing categories (long labels or >10 categories)
- Line chart: trends over time
- Scatter plot: relationship between 2 numeric variables
- Histogram: distribution of a single numeric variable
- Pie chart: proportions (<7 categories, must sum to 100%)
- Box plot: distribution comparison across groups, outlier detection
- Violin plot: density-based distribution comparison
- Heatmap: 2D aggregation or intensity visualization
- Correlation matrix: pairwise correlation between all numeric columns

Respond with ONLY the chart type name.\
"""
