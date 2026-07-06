"""
LLM prompts sub-package — Modular prompt architecture for DataWhisperer.

Architecture:
    ┌──────────────────────────────────────────────────────────────┐
    │                      PromptRegistry                          │
    │  (Central orchestrator — composes stage-specific messages)    │
    ├──────────────────────────────────────────────────────────────┤
    │                                                              │
    │  system_prompt.py      — Core identity & behavioral rules    │
    │  developer_prompt.py   — Internal reasoning & inspection     │
    │  user_prompt.py        — User message construction           │
    │  safety_prompt.py      — Anti-hallucination & code safety    │
    │  correction_prompt.py  — Debug & error recovery              │
    │  retry_prompt.py       — Progressive retry strategies        │
    │  reflection_prompt.py  — Self-validation loops               │
    │  output_format_prompt.py — Explanation, chart, title formats │
    │                                                              │
    │  context_builder.py    — CSV metadata → LLM context          │
    │  few_shot.py           — Concrete input→output examples      │
    │  system.py             — Legacy prompts (deprecated)         │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘

Usage:
    from backend.llm.prompts.registry import PromptRegistry

    registry = PromptRegistry()
    messages = registry.build_generation_messages(question=..., ...)
"""

from backend.llm.prompts.registry import PromptRegistry

__all__ = ["PromptRegistry"]
