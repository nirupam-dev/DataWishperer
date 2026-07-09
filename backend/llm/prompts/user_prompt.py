"""
User Prompt — Dynamic user message construction.

Transforms the raw user question into a structured prompt that includes:
    - The reasoning instruction preamble
    - The actual user question
    - Error context (on retries)
    - Ambiguity handling hints
    - Visualization mandate (when chart/graph/plot is requested)

This module is responsible for the FINAL message in the conversation
(the "user" role message that triggers code generation).
"""

from __future__ import annotations

import re
from typing import Optional


# ── Chart Intent Detection ───────────────────────────────────────────────────
# Keywords and phrases that indicate the user wants a visual chart/graph.
# Checked case-insensitively against the user's question.

_CHART_KEYWORDS = {
    "graph", "chart", "plot", "visualize", "visualization", "visualise",
    "histogram", "bar chart", "scatter", "pie chart", "heatmap",
    "box plot", "boxplot", "violin", "line chart", "area chart",
    "trend", "distribution", "correlation matrix",
    "draw", "show me a graph", "show me a chart", "create a chart",
    "generate a graph", "generate a chart", "make a graph", "make a chart",
}

_CHART_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(kw) for kw in _CHART_KEYWORDS) + r")\b",
    re.IGNORECASE,
)

# ── Visualization Mandate ────────────────────────────────────────────────────
# Injected when chart intent is detected to FORCE the LLM to generate a chart.

_VISUALIZATION_MANDATE = """\
VISUALIZATION REQUIRED — The user is asking for a CHART/GRAPH.
You MUST generate a matplotlib visualization. Follow this pattern EXACTLY:

1. import matplotlib.pyplot as plt
2. fig, ax = plt.subplots(figsize=(12, 7))
3. fig.patch.set_facecolor('#1a1a2e')
4. ax.set_facecolor('#16213e')
5. Create the chart (bar, scatter, histogram, etc.)
6. Style with colors from ['#6C5CE7','#00CEC9','#FD79A8','#FDCB6E','#55EFC4','#A29BFE','#FF7675','#74B9FF']
7. Set title, labels, tick colors to '#F0F0F5'
8. plt.tight_layout()
9. plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor=plt.gcf().get_facecolor())
10. plt.close('all')
11. result = <the underlying data used for the chart>

Do NOT just return a DataFrame. You MUST call plt.savefig(chart_path, ...).
The variable `chart_path` is already defined — just use it directly.
FAILURE to generate a chart when the user asks for one is UNACCEPTABLE."""


def _detect_chart_intent(question: str) -> bool:
    """Return True if the user's question asks for a chart/graph/plot."""
    return bool(_CHART_PATTERN.search(question))


def build_user_prompt(
    question: str,
    reasoning_preamble: str,
    error_context: Optional[str] = None,
) -> str:
    """
    Construct the user-role message for code generation.

    If the question contains chart/graph/plot keywords, a strong
    visualization mandate is injected to force the LLM to generate
    matplotlib code rather than just returning a DataFrame.

    Args:
        question: The user's natural language question.
        reasoning_preamble: Developer reasoning instructions.
        error_context: Optional error from a previous failed attempt.

    Returns:
        The assembled user prompt string.
    """
    formatted_question = f'USER QUESTION:\n"{question}"'
    parts = [reasoning_preamble, formatted_question]

    # Inject visualization mandate when user asks for a chart
    if _detect_chart_intent(question):
        parts.append(_VISUALIZATION_MANDATE)

    if error_context:
        parts.append(error_context)

    return "\n\n".join(parts)
