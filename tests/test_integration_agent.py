"""
Integration tests for the agent pipeline.

Tests the 8-stage interpreter pipeline with mocked LLM:
    Stage 1: Question intake
    Stage 2: Code generation
    Stage 3: AST validation
    Stage 4: Sandbox execution
    Stage 5: Result packaging
    Stage 6: Explanation generation
    Stage 7: Chart explanation
    Stage 8: Auto-debug

All tests mock OllamaProvider to avoid requiring a running Ollama instance.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.llm.agent import DataWhispererAgent
from backend.llm.chains.output_parser import OutputParser
from backend.llm.chains.query_chain import QueryChain
from backend.llm.prompts.registry import PromptRegistry
from backend.models.schemas import (
    CodeExecutionResult,
    FileMetadata,
    LLMResponse,
    ResultType,
)
from backend.sandbox.executor import SandboxExecutor
from backend.sandbox.validator import CodeValidator


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_provider():
    """Mock OllamaProvider that returns controlled responses."""
    provider = MagicMock()
    provider.get_model_name.return_value = "qwen2.5:7b"
    return provider


@pytest.fixture
def mock_llm_response():
    """Factory for LLMResponse objects."""
    def _make(content: str = "```python\nresult = df.head()\n```", tokens: int = 100):
        return LLMResponse(
            content=content,
            model="qwen2.5:7b",
            tokens_used=tokens,
            latency_ms=500.0,
            finish_reason="stop",
        )
    return _make


@pytest.fixture
def query_chain(mock_provider):
    return QueryChain(
        provider=mock_provider,
        output_parser=OutputParser(),
        prompt_registry=PromptRegistry(),
    )


# ── QueryChain Code Generation Tests ───────────────────────────────────────


class TestQueryChainGeneration:
    """Test code generation via the query chain."""

    def test_generate_code_extracts_python(
        self, query_chain, mock_provider, mock_llm_response, sample_file_metadata
    ):
        mock_provider.generate.return_value = mock_llm_response(
            "```python\nresult = df.groupby('category')['revenue'].mean()\n```"
        )
        code, response, reasoning = query_chain.generate_code(
            question="Average revenue by category?",
            file_metadata=sample_file_metadata,
            session_id="test-session",
        )
        assert "result =" in code
        assert "groupby" in code
        assert response.tokens_used == 100

    def test_generate_code_strips_reasoning(
        self, query_chain, mock_provider, mock_llm_response, sample_file_metadata
    ):
        mock_provider.generate.return_value = mock_llm_response(
            "Step 1: I need to group the data.\n"
            "Step 2: Then calculate the mean.\n"
            "```python\nresult = df.groupby('cat').mean()\n```"
        )
        code, response, reasoning = query_chain.generate_code(
            question="Average?",
            file_metadata=sample_file_metadata,
            session_id="test-session",
        )
        assert "Step 1" not in code
        assert reasoning is not None

    def test_generate_code_with_error_context(
        self, query_chain, mock_provider, mock_llm_response, sample_file_metadata
    ):
        mock_provider.generate.return_value = mock_llm_response(
            "```python\nresult = df['revenue'].mean()\n```"
        )
        code, response, _ = query_chain.generate_code(
            question="Average revenue?",
            file_metadata=sample_file_metadata,
            session_id="test-session",
            error_context="Previous attempt failed with KeyError",
            attempt=2,
        )
        assert "result =" in code

    def test_empty_response_raises(
        self, query_chain, mock_provider, mock_llm_response, sample_file_metadata
    ):
        from backend.core.exceptions import GenerationError
        mock_provider.generate.return_value = mock_llm_response("")
        # Disable the auto-created fallback_on_malformed_output so the
        # chain re-raises GenerationError instead of attempting a fallback
        # with a non-LLMResponse mock.
        mock_provider.fallback_on_malformed_output = None
        with pytest.raises(GenerationError):
            query_chain.generate_code(
                question="Something",
                file_metadata=sample_file_metadata,
                session_id="test-session",
            )


# ── QueryChain Explanation Tests ────────────────────────────────────────────


class TestQueryChainExplanation:
    """Test explanation and chart reasoning generation."""

    def test_generate_explanation(self, query_chain, mock_provider, mock_llm_response):
        mock_provider.generate.return_value = mock_llm_response(
            "The analysis grouped revenue by category and calculated the mean."
        )
        explanation = query_chain.generate_explanation(
            code="result = df.groupby('cat').mean()",
            result_summary="DataFrame with 5 rows",
        )
        assert len(explanation) > 10

    def test_explanation_fallback_on_error(self, query_chain, mock_provider):
        mock_provider.generate.side_effect = Exception("LLM down")
        explanation = query_chain.generate_explanation(
            code="result = df.head()",
            result_summary="5 rows",
        )
        assert explanation == "Analysis complete."

    def test_chart_explanation(self, query_chain, mock_provider, mock_llm_response):
        mock_provider.generate.return_value = mock_llm_response(
            "A bar chart was chosen because it effectively compares categories."
        )
        explanation = query_chain.generate_chart_explanation(
            code="df['cat'].value_counts().plot.bar()\nplt.savefig(chart_path)\nresult = 'done'",
            question="Show distribution of categories",
        )
        assert explanation is not None

    def test_chart_explanation_no_chart(self, query_chain, mock_provider):
        """Code without chart patterns should return None."""
        explanation = query_chain.generate_chart_explanation(
            code="result = df.head()",
            question="Show top rows",
        )
        assert explanation is None


# ── QueryChain Debug Tests ──────────────────────────────────────────────────


class TestQueryChainDebug:
    """Test the auto-debug stage."""

    def test_debug_produces_fixed_code(
        self, query_chain, mock_provider, mock_llm_response, sample_file_metadata
    ):
        mock_provider.generate.return_value = mock_llm_response(
            "```python\nresult = df['revenue'].mean()\n```"
        )
        fixed_code, response = query_chain.debug_code(
            failed_code="result = df['revnue'].mean()",  # typo
            error=KeyError("revnue"),
            file_metadata=sample_file_metadata,
            question="Average revenue?",
        )
        assert "result =" in fixed_code
        assert "revenue" in fixed_code

    def test_debug_handles_different_error_types(
        self, query_chain, mock_provider, mock_llm_response, sample_file_metadata
    ):
        for error_cls, msg in [
            (TypeError, "unsupported operand type"),
            (ValueError, "could not convert string to float"),
            (IndexError, "list index out of range"),
        ]:
            mock_provider.generate.return_value = mock_llm_response(
                "```python\nresult = df.head()\n```"
            )
            fixed_code, _ = query_chain.debug_code(
                failed_code="result = df.head()",
                error=error_cls(msg),
                file_metadata=sample_file_metadata,
                question="test",
            )
            assert "result =" in fixed_code


# ── QueryChain Reflection Tests ─────────────────────────────────────────────


class TestQueryChainReflection:
    """Test pre-execution code reflection."""

    def test_reflection_pass_verdict(
        self, query_chain, mock_provider, mock_llm_response, sample_file_metadata
    ):
        mock_provider.generate.return_value = mock_llm_response(
            "VERDICT: PASS\nThe code correctly uses the available columns."
        )
        is_valid, fixed = query_chain.reflect_on_code(
            code="result = df['revenue'].mean()",
            file_metadata=sample_file_metadata,
        )
        assert is_valid
        assert fixed is None

    def test_reflection_fail_with_fix(
        self, query_chain, mock_provider, mock_llm_response, sample_file_metadata
    ):
        mock_provider.generate.return_value = mock_llm_response(
            "VERDICT: FAIL\nColumn 'rev' not found.\n"
            "```python\nresult = df['revenue'].mean()\n```"
        )
        is_valid, fixed = query_chain.reflect_on_code(
            code="result = df['rev'].mean()",
            file_metadata=sample_file_metadata,
        )
        assert not is_valid
        assert fixed is not None
        assert "revenue" in fixed

    def test_reflection_error_fallback(
        self, query_chain, mock_provider, sample_file_metadata
    ):
        mock_provider.generate.side_effect = Exception("LLM crashed")
        is_valid, fixed = query_chain.reflect_on_code(
            code="result = df.head()",
            file_metadata=sample_file_metadata,
        )
        # On error, default to PASS to avoid blocking
        assert is_valid
        assert fixed is None


# ── Chart Type Detection Tests ──────────────────────────────────────────────


class TestChartTypeDetection:
    """Test the _detect_chart_type utility."""

    @pytest.mark.parametrize("code,expected", [
        ("df.plot.bar()", "bar chart"),
        ("df.hist()", "histogram"),
        ("df.plot.scatter(x='a', y='b')", "scatter plot"),
        ("df.plot.pie()", "pie chart"),
        ("sns.boxplot(data=df)", "box plot"),
        ("sns.heatmap(corr)", "heatmap"),
        ("px.bar(df)", "bar chart"),
        ("px.scatter(df)", "scatter plot"),
        ("go.Bar(x=x, y=y)", "bar chart"),
        ("plt.savefig(chart_path)", "chart"),
    ])
    def test_chart_type_detection(self, code, expected):
        result = QueryChain._detect_chart_type(code)
        assert result == expected

    def test_no_chart_returns_none(self):
        result = QueryChain._detect_chart_type("result = df.head()")
        assert result is None


# ── Auxiliary Generation Tests ──────────────────────────────────────────────


class TestAuxiliaryGeneration:
    """Test title and suggested question generation."""

    def test_generate_title(self, query_chain, mock_provider, mock_llm_response):
        mock_provider.generate.return_value = mock_llm_response(
            "Revenue Analysis"
        )
        title = query_chain.generate_title("What is the average revenue?")
        assert len(title) > 0
        assert len(title) <= 60

    def test_generate_title_truncated(self, query_chain, mock_provider, mock_llm_response):
        mock_provider.generate.return_value = mock_llm_response(
            "x" * 100
        )
        title = query_chain.generate_title("Something")
        assert len(title) <= 60

    def test_generate_title_fallback(self, query_chain, mock_provider):
        mock_provider.generate.side_effect = Exception("fail")
        title = query_chain.generate_title("test")
        assert title == "Data Analysis"

    def test_suggested_questions_json_parsed(
        self, query_chain, mock_provider, mock_llm_response, sample_file_metadata
    ):
        mock_provider.generate.return_value = mock_llm_response(
            '["What is the average revenue?", "Show revenue by region"]'
        )
        questions = query_chain.generate_suggested_questions(
            sample_file_metadata, count=2
        )
        assert len(questions) == 2

    def test_suggested_questions_fallback(
        self, query_chain, mock_provider, sample_file_metadata
    ):
        mock_provider.generate.side_effect = Exception("fail")
        questions = query_chain.generate_suggested_questions(
            sample_file_metadata
        )
        assert questions == []
