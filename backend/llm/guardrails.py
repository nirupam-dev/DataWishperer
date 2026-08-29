"""
Off-Topic Query Guardrail — Prevents token waste on unrelated questions.

This module provides a fast, heuristic-based relevance check that runs
BEFORE any LLM call, saving tokens when users ask questions completely
unrelated to their uploaded CSV data (e.g., "write python code to add
2 numbers", "tell me a joke", "what is the capital of France").

Two-layer defence:
    Layer 1 (this module): Fast regex/keyword check — zero tokens used.
    Layer 2 (system prompt): Reinforcement in the LLM prompt itself.

The heuristic is intentionally conservative: it only blocks questions
that are CLEARLY off-topic. Ambiguous questions are allowed through
so the LLM can attempt an answer against the dataset.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from backend.core.logging_config import get_logger

logger = get_logger(__name__)


# ── Off-Topic Patterns ──────────────────────────────────────────────────────
# These patterns match questions that are CLEARLY unrelated to data analysis.
# Each tuple: (compiled regex, category label for logging).

_OFF_TOPIC_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # General programming / coding requests
    (re.compile(
        r"\b(?:write|create|make|build|code|implement|develop|program)\b"
        r".*\b(?:python|java|javascript|c\+\+|html|css|react|flask|django|"
        r"function|class|algorithm|program|script|app|application|website|"
        r"game|api|server|bot|calculator)\b",
        re.IGNORECASE,
    ), "programming_request"),

    # Math/coding problems unrelated to data
    (re.compile(
        r"\b(?:write|create|make)\b.*\b(?:code|program|function|script)\b"
        r".*\b(?:add|subtract|multiply|divide|sort|reverse|fibonacci|"
        r"factorial|prime|palindrome|binary|search|linked list|stack|queue|"
        r"tree|matrix|array)\b",
        re.IGNORECASE,
    ), "coding_problem"),

    # General knowledge / trivia
    (re.compile(
        r"\b(?:what is the|who is|who was|when was|where is|where was|"
        r"how old is|tell me about|explain|define|meaning of|"
        r"capital of|president of|history of|population of)\b",
        re.IGNORECASE,
    ), "general_knowledge"),

    # Creative writing / content generation
    (re.compile(
        r"\b(?:write|compose|draft|create)\b.*\b(?:essay|poem|story|letter|"
        r"email|blog|article|resume|cv|cover letter|speech|song|lyrics|"
        r"paragraph|report)\b",
        re.IGNORECASE,
    ), "creative_writing"),

    # Conversational / chit-chat
    (re.compile(
        r"^(?:hello|hi|hey|how are you|what's up|good morning|good evening|"
        r"good night|thank you|thanks|bye|goodbye|who are you|what can you do|"
        r"tell me a joke|sing|are you ai|are you human|what is your name)"
        r"[\s!?.]*$",
        re.IGNORECASE,
    ), "chitchat"),

    # Sports / entertainment / general trivia
    (re.compile(
        r"\b(?:who won|who will win|world cup|olympics|super bowl|champion|"
        r"best movie|best song|best book|favorite|weather|horoscope|"
        r"news today|current affairs)\b",
        re.IGNORECASE,
    ), "trivia"),

    # Homework / exam questions
    (re.compile(
        r"\b(?:solve|calculate|compute|evaluate|simplify|prove|derive)\b"
        r".*\b(?:equation|integral|derivative|limit|matrix|vector|"
        r"probability|theorem|formula|expression)\b",
        re.IGNORECASE,
    ), "homework"),

    # Translation requests
    (re.compile(
        r"\b(?:translate|convert|say)\b.*\b(?:to|in|into)\b.*"
        r"\b(?:hindi|french|spanish|german|chinese|japanese|arabic|"
        r"english|urdu|bengali|tamil|telugu|korean|russian)\b",
        re.IGNORECASE,
    ), "translation"),

    # Recipe / cooking
    (re.compile(
        r"\b(?:recipe|cook|bake|how to make|ingredients for)\b",
        re.IGNORECASE,
    ), "recipe"),

    # Medical / legal advice
    (re.compile(
        r"\b(?:symptoms of|cure for|treatment for|medicine for|"
        r"legal advice|law about|is it legal)\b",
        re.IGNORECASE,
    ), "advice"),
]

# ── Data-Related Whitelist ──────────────────────────────────────────────────
# If any of these keywords appear, the question is likely about the data
# and should NOT be blocked, even if an off-topic pattern also matches.

_DATA_KEYWORDS = re.compile(
    r"\b(?:data|dataset|csv|column|row|table|dataframe|df|"
    r"average|mean|median|mode|sum|total|count|min|max|std|"
    r"group by|groupby|filter|sort|top|bottom|highest|lowest|"
    r"correlation|distribution|outlier|missing|null|nan|"
    r"trend|compare|comparison|percentage|ratio|proportion|"
    r"revenue|sales|profit|price|cost|amount|quantity|"
    r"bar chart|pie chart|histogram|scatter|heatmap|plot|graph|"
    r"chart|visualize|visualization|show me|analyze|analysis|"
    r"aggregate|pivot|crosstab|merge|join|unique|distinct|"
    r"describe|summary|statistics|info|shape|head|tail|"
    r"upload|file|this data|the data|my data|our data|"
    r"first \d|last \d|top \d|bottom \d)\b",
    re.IGNORECASE,
)


def check_relevance(
    question: str,
    column_names: Optional[List[str]] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Check if a user question is relevant to CSV data analysis.

    This is a fast, zero-cost heuristic check that runs BEFORE any
    LLM call. It catches obviously off-topic questions to save tokens.

    Args:
        question: The user's natural language question.
        column_names: Optional list of column names from the active dataset.
                      If the question mentions any column name, it's relevant.

    Returns:
        A tuple of (is_relevant, rejection_message_or_none).
        If is_relevant is True, rejection_message is None.
        If is_relevant is False, rejection_message contains the user-facing reply.
    """
    q = question.strip()

    # Catch short chitchat / greetings before the length check
    _CHITCHAT_EXACT = {
        "hello", "hi", "hey", "thanks", "thank you", "bye", "goodbye",
        "good morning", "good evening", "good night", "ok", "okay",
    }
    if q.lower().rstrip("!?., ") in _CHITCHAT_EXACT:
        logger.info("Off-topic query blocked (category=chitchat): '%s'", q)
        return False, _build_rejection_message("chitchat")

    # Very short questions are likely data-related ("mean?", "top 5?")
    if len(q) < 10:
        return True, None

    # If the question mentions any column name, it's definitely relevant
    if column_names:
        q_lower = q.lower()
        for col in column_names:
            if col.lower() in q_lower:
                return True, None

    # If the question contains data-related keywords, it's relevant
    if _DATA_KEYWORDS.search(q):
        return True, None

    # Check against off-topic patterns
    for pattern, category in _OFF_TOPIC_PATTERNS:
        if pattern.search(q):
            logger.info(
                "Off-topic query blocked (category=%s): '%s'",
                category,
                q[:100],
            )
            return False, _build_rejection_message(category)

    # Default: allow through (conservative — let the LLM handle ambiguity)
    return True, None


def _build_rejection_message(category: str) -> str:
    """Build a friendly, helpful rejection message for off-topic queries."""
    return (
        "🚫 **Off-Topic Question Detected**\n\n"
        "I'm **DataWhisper AI** — I'm specifically designed to analyze "
        "your uploaded CSV data. I can't help with general programming, "
        "trivia, or other non-data questions.\n\n"
        "💡 **Try asking things like:**\n"
        "- *\"What is the average revenue by region?\"*\n"
        "- *\"Show me the top 10 customers by sales\"*\n"
        "- *\"Plot a bar chart of monthly expenses\"*\n"
        "- *\"How many missing values are in each column?\"*\n"
        "- *\"What is the correlation between price and quantity?\"*\n\n"
        "📊 Ask me anything about **your data** and I'll analyze it for you!"
    )
