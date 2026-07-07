"""
Unit tests for prompt architecture modules.

Tests:
    - Safety prompt assembly (anti-hallucination, code safety)
    - Context builder (full, compact, multi-dataset, switch)
    - Output parser (code extraction, reasoning separation)
    - Prompt registry message building
    - Few-shot example formatting
"""

from __future__ import annotations

import pytest

from backend.llm.prompts.safety_prompt import (
    ANTI_HALLUCINATION_PROMPT,
    CODE_SAFETY_PROMPT,
    COLUMN_VALIDATION_TEMPLATE,
    build_safety_prompt,
)
from backend.llm.prompts.context_builder import ContextBuilder
from backend.llm.chains.output_parser import OutputParser
from backend.llm.prompts.registry import PromptRegistry
from backend.models.schemas import ColumnInfo, FileMetadata


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def column_names():
    return ["category", "revenue", "quantity", "date"]


@pytest.fixture
def context_builder():
    return ContextBuilder()


@pytest.fixture
def output_parser():
    return OutputParser()


@pytest.fixture
def registry():
    return PromptRegistry()


# ── Safety Prompt Tests ─────────────────────────────────────────────────────


class TestSafetyPrompt:
    """Test safety prompt assembly and content."""

    def test_anti_hallucination_rules_present(self):
        assert "ANTI-HALLUCINATION" in ANTI_HALLUCINATION_PROMPT
        assert "ONLY columns" in ANTI_HALLUCINATION_PROMPT
        assert "NEVER invent" in ANTI_HALLUCINATION_PROMPT

    def test_code_safety_rules_present(self):
        assert "FORBIDDEN" in CODE_SAFETY_PROMPT
        assert "os" in CODE_SAFETY_PROMPT
        assert "eval()" in CODE_SAFETY_PROMPT
        assert "ALLOWED IMPORTS" in CODE_SAFETY_PROMPT

    def test_build_safety_prompt_includes_all_sections(self, column_names):
        prompt = build_safety_prompt(column_names, include_code_safety=True)
        assert "ANTI-HALLUCINATION" in prompt
        assert "FORBIDDEN" in prompt
        assert "VERIFIED COLUMNS" in prompt
        for col in column_names:
            assert col in prompt

    def test_build_safety_prompt_without_code_safety(self, column_names):
        prompt = build_safety_prompt(column_names, include_code_safety=False)
        assert "ANTI-HALLUCINATION" in prompt
        assert "FORBIDDEN" not in prompt

    def test_build_safety_prompt_empty_columns(self):
        prompt = build_safety_prompt([], include_code_safety=True)
        assert "ANTI-HALLUCINATION" in prompt
        assert "VERIFIED COLUMNS" not in prompt

    def test_column_validation_template_format(self):
        formatted = COLUMN_VALIDATION_TEMPLATE.format(
            column_list="  - 'revenue'\n  - 'quantity'"
        )
        assert "revenue" in formatted
        assert "EXACTLY" in formatted


# ── Context Builder Tests ───────────────────────────────────────────────────


class TestContextBuilder:
    """Test dataset context string generation."""

    def test_full_context_has_all_sections(
        self, context_builder, sample_file_metadata
    ):
        ctx = context_builder.build(sample_file_metadata)
        assert "DATASET INFORMATION" in ctx
        assert "COLUMNS:" in ctx
        assert "sample.csv" in ctx
        assert "200" in ctx

    def test_compact_context_is_shorter(
        self, context_builder, sample_file_metadata
    ):
        full = context_builder.build(sample_file_metadata)
        compact = context_builder.build_compact(sample_file_metadata)
        assert len(compact) < len(full)
        assert "sample.csv" in compact
        assert "200 rows" in compact

    def test_multi_dataset_context(self, context_builder, sample_file_metadata):
        other = FileMetadata(
            file_id="test-file-002",
            original_name="other.csv",
            stored_path="/tmp/other.csv",
            row_count=50, col_count=3,
            file_size_bytes=2000, memory_usage_mb=0.01,
            columns=[
                ColumnInfo(
                    name="x", dtype="float64",
                    non_null_count=50, null_count=0,
                    unique_count=50, sample_values=["1.0"],
                ),
            ],
        )
        ctx = context_builder.build_multi_dataset_context(
            active_metadata=sample_file_metadata,
            all_datasets={
                "test-file-001": sample_file_metadata,
                "test-file-002": other,
            },
        )
        assert "ACTIVE DATASET" in ctx
        assert "OTHER LOADED DATASETS" in ctx
        assert "other.csv" in ctx

    def test_switch_context(self, context_builder, sample_file_metadata):
        ctx = context_builder.build_switch_context(
            new_metadata=sample_file_metadata,
            old_name="old_data.csv",
        )
        assert "SWITCHED" in ctx
        assert "old_data.csv" in ctx
        assert "sample.csv" in ctx

    def test_numeric_summary_for_numeric_columns(
        self, context_builder, sample_file_metadata
    ):
        ctx = context_builder.build(sample_file_metadata)
        assert "NUMERIC SUMMARY" in ctx

    def test_notes_for_missing_values(self, context_builder):
        metadata = FileMetadata(
            file_id="test-003", original_name="nulls.csv",
            stored_path="/tmp/nulls.csv",
            row_count=100, col_count=2,
            file_size_bytes=1000, memory_usage_mb=0.01,
            columns=[
                ColumnInfo(
                    name="value", dtype="float64",
                    non_null_count=40, null_count=60,
                    unique_count=40, sample_values=["1.0"],
                    mean=5.0, std=2.0, min_val=0.0, max_val=10.0,
                ),
            ],
        )
        ctx = context_builder.build(metadata)
        assert "NOTES:" in ctx
        assert "nulls" in ctx.lower()


# ── Output Parser Tests ─────────────────────────────────────────────────────


class TestOutputParser:
    """Test code extraction from LLM responses."""

    def test_extract_fenced_code(self, output_parser):
        raw = "Here's the code:\n```python\nresult = df.head()\n```\nDone."
        code = output_parser.extract_code(raw)
        assert code == "result = df.head()"

    def test_extract_multiple_fenced_blocks(self, output_parser):
        raw = (
            "```python\nimport pandas as pd\n```\n"
            "Some text.\n"
            "```python\nresult = pd.DataFrame()\n```"
        )
        code = output_parser.extract_code(raw)
        assert "import pandas" in code
        assert "result =" in code

    def test_extract_by_result_assignment(self, output_parser):
        raw = (
            "Let me think about this.\n"
            "result = df.groupby('cat')['val'].mean()\n"
            "\n**Done**"
        )
        code = output_parser.extract_code(raw)
        assert "result =" in code

    def test_extract_raw_python(self, output_parser):
        raw = (
            "import numpy as np\n"
            "df['new'] = df['val'] * 2\n"
            "result = df.head()"
        )
        code = output_parser.extract_code(raw)
        assert "result =" in code

    def test_empty_response_raises(self, output_parser):
        from backend.core.exceptions import GenerationError
        with pytest.raises(GenerationError):
            output_parser.extract_code("")

    def test_no_code_raises(self, output_parser):
        from backend.core.exceptions import GenerationError
        with pytest.raises(GenerationError):
            output_parser.extract_code("I don't know the answer.")

    def test_extract_code_and_reasoning(self, output_parser):
        raw = (
            "Step 1: First, I need to group the data.\n"
            "Step 2: Then calculate the mean.\n"
            "```python\nresult = df.groupby('cat').mean()\n```"
        )
        code, reasoning = output_parser.extract_code_and_reasoning(raw)
        assert "result =" in code
        assert reasoning is not None
        assert "Step 1" in reasoning

    def test_reasoning_with_think_tags(self, output_parser):
        raw = (
            "<think>I should use groupby for this.</think>\n"
            "```python\nresult = df.groupby('cat').mean()\n```"
        )
        code, reasoning = output_parser.extract_code_and_reasoning(raw)
        assert "result =" in code
        assert reasoning is not None
        assert "groupby" in reasoning

    def test_extract_text_response_strips_code(self, output_parser):
        raw = (
            "This analysis shows the average.\n"
            "```python\nresult = 42\n```\n"
            "The result is 42."
        )
        text = output_parser.extract_text_response(raw)
        assert "result = 42" not in text
        assert "analysis" in text.lower()

    def test_no_reasoning_returns_none(self, output_parser):
        raw = "```python\nresult = df.head()\n```"
        code, reasoning = output_parser.extract_code_and_reasoning(raw)
        assert code == "result = df.head()"
        assert reasoning is None


# ── Prompt Registry Tests ───────────────────────────────────────────────────


class TestPromptRegistry:
    """Test prompt registry message assembly."""

    def test_generation_messages_structure(
        self, registry, sample_file_metadata
    ):
        msgs = registry.build_generation_messages(
            question="What is the average revenue?",
            file_metadata=sample_file_metadata,
        )
        assert len(msgs) >= 4  # system + safety + context + user
        roles = [m["role"] for m in msgs]
        assert roles[0] == "system"
        assert roles[-1] == "user"

    def test_generation_messages_retry_uses_compact(
        self, registry, sample_file_metadata
    ):
        msgs_first = registry.build_generation_messages(
            question="What is the average?",
            file_metadata=sample_file_metadata,
            attempt=1,
        )
        msgs_retry = registry.build_generation_messages(
            question="What is the average?",
            file_metadata=sample_file_metadata,
            attempt=2,
        )
        # Retry should have fewer/shorter messages
        first_total = sum(len(m["content"]) for m in msgs_first)
        retry_total = sum(len(m["content"]) for m in msgs_retry)
        assert retry_total < first_total

    def test_generation_includes_history(
        self, registry, sample_file_metadata
    ):
        history = [
            {"role": "user", "content": "Show me the top 5 rows"},
            {"role": "assistant", "content": "Here are the top 5 rows..."},
        ]
        msgs = registry.build_generation_messages(
            question="Now filter by category A",
            file_metadata=sample_file_metadata,
            session_history=history,
        )
        contents = [m["content"] for m in msgs]
        assert any("top 5 rows" in c for c in contents)

    def test_explanation_messages(self, registry):
        msgs = registry.build_explanation_messages(
            code="result = df.head()",
            result_summary="Returned 5 rows",
        )
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_chart_explanation_messages(self, registry):
        msgs = registry.build_chart_explanation_messages(
            code="plt.bar(x, y)\nresult = 'done'",
            question="Show revenue by category",
            chart_type="bar chart",
        )
        assert len(msgs) == 1
        assert "bar chart" in msgs[0]["content"]

    def test_debug_messages(self, registry, sample_file_metadata):
        msgs = registry.build_debug_messages(
            failed_code="result = df['nonexistent']",
            error_type="KeyError",
            error_message="'nonexistent'",
            file_metadata=sample_file_metadata,
        )
        assert len(msgs) >= 2

    def test_title_messages(self, registry):
        msgs = registry.build_title_messages("What is the average revenue?")
        assert len(msgs) == 1

    def test_suggested_questions_messages(
        self, registry, sample_file_metadata
    ):
        msgs = registry.build_suggested_questions_messages(
            sample_file_metadata, count=4
        )
        assert len(msgs) == 1
