"""
Tests for LLM layer coverage gaps: OllamaProvider, QueryChain, OutputParser.

Targets:
    - ollama_provider.py: 46% → 90%+
    - query_chain.py: 85% → 95%+
    - output_parser.py: 87% → 95%+
    - chart_export.py: 63% → 95%+
    - chart_explainer.py: 67% → 95%+
    - logging_config.py: 66% → 85%+
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, PropertyMock, patch, create_autospec

import numpy as np
import pandas as pd
import pytest

from backend.core.exceptions import (
    CodeValidationError,
    ExecutionRuntimeError,
    GenerationError,
    GenerationTimeoutError,
    ModelNotFoundError,
    OllamaConnectionError,
)
from backend.llm.chains.output_parser import OutputParser
from backend.llm.chains.query_chain import QueryChain, _CHART_TYPE_PATTERNS
from backend.llm.prompts.registry import PromptRegistry
from backend.models.schemas import (
    ColumnInfo,
    FileMetadata,
    LLMResponse,
    ResultType,
)
from backend.visualization.chart_export import ChartExporter
from backend.visualization.chart_explainer import ChartExplainer
from backend.visualization.chart_selector import ChartSpec, ChartType


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def parser():
    return OutputParser()


@pytest.fixture
def sample_metadata():
    return FileMetadata(
        file_id="f1", original_name="data.csv", stored_path="/tmp/data.csv",
        row_count=100, col_count=3, file_size_bytes=5000, memory_usage_mb=0.05,
        columns=[
            ColumnInfo(name="a", dtype="int64", non_null_count=100,
                       null_count=0, unique_count=100, sample_values=["1"]),
            ColumnInfo(name="b", dtype="float64", non_null_count=100,
                       null_count=0, unique_count=90, sample_values=["3.14"],
                       mean=50.0, std=10.0, min_val=0.0, max_val=100.0),
        ],
    )


@pytest.fixture
def mock_provider():
    p = MagicMock()
    p.generate.return_value = LLMResponse(
        content="```python\nresult = df.mean()\n```",
        model="test", tokens_used=50, latency_ms=100.0,
    )
    return p


@pytest.fixture
def query_chain(mock_provider):
    return QueryChain(provider=mock_provider)


# ── OutputParser Tests ───────────────────────────────────────────────────────


class TestOutputParserExtended:
    """Extended tests for OutputParser edge cases."""

    def test_extract_raw_with_indicators(self, parser):
        raw = "import pandas as pd\ndf.groupby('a').mean()\nresult = df.sum()"
        code = parser.extract_code(raw)
        assert "result" in code

    def test_extract_raw_not_enough_indicators(self, parser):
        with pytest.raises(GenerationError):
            parser.extract_code("This is just a sentence about data.")

    def test_result_assignment_extraction(self, parser):
        text = "Here's the code:\nimport pandas as pd\nresult = df['a'].mean()"
        code = parser.extract_code(text)
        assert "result" in code

    def test_result_assignment_stops_at_markdown(self, parser):
        text = "import pandas\nresult = 42\n## Explanation\nsome text"
        code = parser.extract_code(text)
        assert "Explanation" not in code

    def test_extract_reasoning_think_tags(self, parser):
        text = "<think>Let me analyze step by step</think>\n```python\nresult = 1\n```"
        code, reasoning = parser.extract_code_and_reasoning(text)
        assert reasoning is not None
        assert "step by step" in reasoning

    def test_extract_reasoning_pre_code(self, parser):
        text = "Step 1: First, I need to check columns.\n```python\nresult = 1\n```"
        _, reasoning = parser.extract_code_and_reasoning(text)
        assert reasoning is not None

    def test_extract_reasoning_no_reasoning(self, parser):
        text = "```python\nresult = 1\n```"
        _, reasoning = parser.extract_code_and_reasoning(text)
        assert reasoning is None

    def test_extract_text_response_strips_code(self, parser):
        text = "Here is the answer.\n```python\nresult = 1\n```\nDone."
        result = parser.extract_text_response(text)
        assert "result" not in result
        assert "answer" in result

    def test_extract_text_strips_cot(self, parser):
        text = "<think>internal reasoning</think>\nThe average is 42."
        result = parser.extract_text_response(text)
        assert "internal" not in result
        assert "42" in result

    def test_extract_reasoning_no_indicators(self, parser):
        text = "Hello world\n```python\nresult = 1\n```"
        _, reasoning = parser.extract_code_and_reasoning(text)
        assert reasoning is None  # "Hello world" has no reasoning indicators


# ── QueryChain Tests ─────────────────────────────────────────────────────────


class TestQueryChainExtended:
    """Extended tests for QueryChain methods."""

    def test_generate_code(self, query_chain, sample_metadata):
        code, response, reasoning = query_chain.generate_code(
            "What is the average?", sample_metadata, "s1",
        )
        assert "result" in code

    def test_generate_code_with_error_context(self, query_chain, sample_metadata):
        code, _, _ = query_chain.generate_code(
            "avg?", sample_metadata, "s1",
            error_context="KeyError: 'x'", attempt=2,
        )
        assert code is not None

    def test_generate_explanation(self, query_chain, mock_provider):
        mock_provider.generate.return_value = LLMResponse(
            content="The code calculates the mean.", model="t",
        )
        result = query_chain.generate_explanation("result=df.mean()", "42")
        assert isinstance(result, str)

    def test_generate_explanation_failure(self, query_chain, mock_provider):
        mock_provider.generate.side_effect = Exception("LLM down")
        result = query_chain.generate_explanation("code", "result")
        assert result == "Analysis complete."

    def test_generate_explanation_empty(self, query_chain, mock_provider):
        mock_provider.generate.return_value = LLMResponse(content="", model="t")
        result = query_chain.generate_explanation("code", "result")
        assert result == "Analysis complete."

    def test_generate_chart_explanation(self, query_chain, mock_provider):
        mock_provider.generate.return_value = LLMResponse(
            content="A bar chart was chosen because...", model="t",
        )
        result = query_chain.generate_chart_explanation(
            "df.plot.bar()", "show distribution",
        )
        assert result is not None

    def test_generate_chart_explanation_no_chart(self, query_chain):
        result = query_chain.generate_chart_explanation(
            "result = df.mean()", "avg?",
        )
        assert result is None

    def test_generate_chart_explanation_failure(self, query_chain, mock_provider):
        mock_provider.generate.side_effect = Exception("fail")
        result = query_chain.generate_chart_explanation(
            "df.plot.bar()", "show dist",
        )
        assert "bar chart" in result

    def test_debug_code(self, query_chain, sample_metadata, mock_provider):
        mock_provider.generate.return_value = LLMResponse(
            content="```python\nresult = df['a'].mean()\n```", model="t",
        )
        error = ExecutionRuntimeError(error_type="KeyError", error_message="'x'")
        code, resp = query_chain.debug_code(
            "result = df['x'].mean()", error, sample_metadata, "avg?",
        )
        assert "result" in code

    def test_debug_code_with_validation_error(self, query_chain, sample_metadata, mock_provider):
        mock_provider.generate.return_value = LLMResponse(
            content="```python\nresult = 1\n```", model="t",
        )
        error = CodeValidationError("blocked import")
        code, _ = query_chain.debug_code("import os", error, sample_metadata, "q")
        assert code is not None

    def test_reflect_on_code_pass(self, query_chain, sample_metadata, mock_provider):
        mock_provider.generate.return_value = LLMResponse(
            content="VERDICT: PASS\nCode looks correct.", model="t",
        )
        is_valid, fixed = query_chain.reflect_on_code("result=1", sample_metadata)
        assert is_valid is True
        assert fixed is None

    def test_reflect_on_code_fail_with_fix(self, query_chain, sample_metadata, mock_provider):
        mock_provider.generate.return_value = LLMResponse(
            content="VERDICT: FAIL\n```python\nresult = df['a'].mean()\n```",
            model="t",
        )
        is_valid, fixed = query_chain.reflect_on_code("result=df['x']", sample_metadata)
        assert is_valid is False
        assert fixed is not None

    def test_reflect_on_code_fail_no_fix(self, query_chain, sample_metadata, mock_provider):
        mock_provider.generate.return_value = LLMResponse(
            content="VERDICT: FAIL\nBut no code provided.", model="t",
        )
        is_valid, fixed = query_chain.reflect_on_code("bad", sample_metadata)
        assert is_valid is False
        assert fixed is None

    def test_reflect_on_code_ambiguous(self, query_chain, sample_metadata, mock_provider):
        mock_provider.generate.return_value = LLMResponse(
            content="Hmm, not sure.", model="t",
        )
        is_valid, _ = query_chain.reflect_on_code("code", sample_metadata)
        assert is_valid is True  # Ambiguous → pass

    def test_reflect_on_code_exception(self, query_chain, sample_metadata, mock_provider):
        mock_provider.generate.side_effect = Exception("fail")
        is_valid, _ = query_chain.reflect_on_code("code", sample_metadata)
        assert is_valid is True

    def test_generate_title(self, query_chain, mock_provider):
        mock_provider.generate.return_value = LLMResponse(
            content="Revenue Analysis", model="t",
        )
        title = query_chain.generate_title("What is the total revenue?")
        assert title == "Revenue Analysis"

    def test_generate_title_failure(self, query_chain, mock_provider):
        mock_provider.generate.side_effect = Exception("fail")
        title = query_chain.generate_title("q")
        assert title == "Data Analysis"

    def test_generate_title_long(self, query_chain, mock_provider):
        mock_provider.generate.return_value = LLMResponse(
            content="A" * 100, model="t",
        )
        title = query_chain.generate_title("q")
        assert len(title) <= 60

    def test_generate_title_empty(self, query_chain, mock_provider):
        mock_provider.generate.return_value = LLMResponse(content="", model="t")
        title = query_chain.generate_title("q")
        assert title == "Data Analysis"

    def test_generate_suggested_questions(self, query_chain, sample_metadata, mock_provider):
        mock_provider.generate.return_value = LLMResponse(
            content='["Q1?", "Q2?"]', model="t",
        )
        qs = query_chain.generate_suggested_questions(sample_metadata)
        assert len(qs) == 2

    def test_generate_suggested_questions_failure(self, query_chain, sample_metadata, mock_provider):
        mock_provider.generate.side_effect = Exception("fail")
        qs = query_chain.generate_suggested_questions(sample_metadata)
        assert qs == []

    def test_generate_suggested_questions_no_json(self, query_chain, sample_metadata, mock_provider):
        mock_provider.generate.return_value = LLMResponse(
            content="Here are some questions", model="t",
        )
        qs = query_chain.generate_suggested_questions(sample_metadata)
        assert qs == []

    def test_build_error_context(self, query_chain, sample_metadata):
        err = ExecutionRuntimeError(error_type="KeyError", error_message="'x'")
        ctx = query_chain.build_error_context(err, sample_metadata)
        assert isinstance(ctx, str)
        assert "KeyError" in ctx or "Error" in ctx

    def test_detect_chart_type_patterns(self):
        assert QueryChain._detect_chart_type("df.plot.bar()") == "bar chart"
        assert QueryChain._detect_chart_type("sns.heatmap(corr)") == "heatmap"
        assert QueryChain._detect_chart_type("px.scatter(df)") == "scatter plot"
        assert QueryChain._detect_chart_type("plt.savefig('x.png')") == "chart"
        assert QueryChain._detect_chart_type("result = df.mean()") is None

    def test_memory_property(self, query_chain):
        assert query_chain.memory is not None

    def test_registry_property(self, query_chain):
        assert query_chain.registry is not None


# ── OllamaProvider Tests ────────────────────────────────────────────────────


class TestOllamaProvider:
    """Tests for OllamaProvider (mocked HTTP/LLM)."""

    @pytest.fixture
    def provider(self):
        from backend.llm.providers.ollama_provider import OllamaProvider
        with patch("backend.llm.providers.ollama_provider.ChatOllama"):
            with patch("backend.llm.providers.ollama_provider.httpx.Client"):
                from backend.core.config import OllamaSettings
                settings = OllamaSettings(model="test-model", base_url="http://localhost:11434")
                return OllamaProvider(settings=settings)

    def test_get_model_name(self, provider):
        assert provider.get_model_name() == "test-model"

    def test_llm_property(self, provider):
        assert provider.llm is not None

    def test_create_llm(self, provider):
        with patch("backend.llm.providers.ollama_provider.ChatOllama") as mock:
            provider.create_llm(temperature=0.5, max_tokens=100)
            mock.assert_called_once()

    def test_generate_success(self, provider):
        mock_response = MagicMock()
        mock_response.content = "```python\nresult = 42\n```"
        mock_response.response_metadata = {"eval_count": 50, "done_reason": "stop"}
        provider._llm.invoke.return_value = mock_response
        result = provider.generate([{"role": "user", "content": "hi"}])
        assert result.content == "```python\nresult = 42\n```"
        assert result.tokens_used == 50

    def test_generate_empty_response(self, provider):
        mock_response = MagicMock()
        mock_response.content = "   "
        mock_response.response_metadata = {}
        provider._llm.invoke.return_value = mock_response
        with pytest.raises(GenerationError):
            provider.generate([{"role": "user", "content": "hi"}])

    def test_generate_connection_error(self, provider):
        provider._llm.invoke.side_effect = Exception("connection refused")
        with pytest.raises(OllamaConnectionError):
            provider.generate([{"role": "user", "content": "hi"}])

    def test_generate_model_not_found(self, provider):
        provider._llm.invoke.side_effect = Exception("model not found 404")
        with pytest.raises(ModelNotFoundError):
            provider.generate([{"role": "user", "content": "hi"}])

    def test_generate_timeout(self, provider):
        provider._llm.invoke.side_effect = Exception("timeout exceeded")
        with pytest.raises(GenerationTimeoutError):
            provider.generate([{"role": "user", "content": "hi"}])

    def test_generate_generic_error(self, provider):
        provider._llm.invoke.side_effect = Exception("something weird")
        with pytest.raises(GenerationError):
            provider.generate([{"role": "user", "content": "hi"}])

    def test_generate_with_overrides(self, provider):
        with patch("backend.llm.providers.ollama_provider.ChatOllama") as mock_cls:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "result = 1"
            mock_response.response_metadata = {}
            mock_llm.invoke.return_value = mock_response
            mock_cls.return_value = mock_llm
            result = provider.generate(
                [{"role": "user", "content": "hi"}],
                temperature=0.1, max_tokens=50,
            )
            assert result.content == "result = 1"

    def test_health_check_success(self, provider):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": [{"name": "test-model"}]}
        provider._http_client.get.return_value = mock_resp
        result = provider.health_check()
        assert result["connected"] is True
        assert result["model_loaded"] is True

    def test_health_check_model_not_loaded(self, provider):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": [{"name": "other-model"}]}
        provider._http_client.get.return_value = mock_resp
        result = provider.health_check()
        assert result["connected"] is True
        assert result["model_loaded"] is False

    def test_health_check_bad_status(self, provider):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        provider._http_client.get.return_value = mock_resp
        result = provider.health_check()
        assert result["connected"] is False

    def test_health_check_connect_error(self, provider):
        import httpx
        provider._http_client.get.side_effect = httpx.ConnectError("refused")
        result = provider.health_check()
        assert result["connected"] is False

    def test_health_check_generic_error(self, provider):
        provider._http_client.get.side_effect = Exception("unknown")
        result = provider.health_check()
        assert result["connected"] is False

    def test_list_models_success(self, provider):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": [{"name": "m1"}, {"name": "m2"}]}
        provider._http_client.get.return_value = mock_resp
        models = provider.list_models()
        assert len(models) == 2

    def test_list_models_empty(self, provider):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        provider._http_client.get.return_value = mock_resp
        assert provider.list_models() == []

    def test_list_models_connect_error(self, provider):
        import httpx
        provider._http_client.get.side_effect = httpx.ConnectError("refused")
        with pytest.raises(OllamaConnectionError):
            provider.list_models()

    def test_close(self, provider):
        provider.close()
        provider._http_client.close.assert_called_once()

    def test_dict_to_langchain_message(self):
        from backend.llm.providers.ollama_provider import _dict_to_langchain_message
        from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
        assert isinstance(_dict_to_langchain_message({"role": "system", "content": "x"}), SystemMessage)
        assert isinstance(_dict_to_langchain_message({"role": "assistant", "content": "x"}), AIMessage)
        assert isinstance(_dict_to_langchain_message({"role": "user", "content": "x"}), HumanMessage)
        assert isinstance(_dict_to_langchain_message({"content": "x"}), HumanMessage)


# ── ChartExporter Tests ─────────────────────────────────────────────────────


class TestChartExporter:
    """Tests for ChartExporter multi-format export."""

    @pytest.fixture
    def exporter(self, tmp_path):
        return ChartExporter(str(tmp_path / "exports"))

    def test_export_png(self, exporter, tmp_path):
        src = tmp_path / "chart.png"
        src.write_bytes(b"\x89PNG")
        result = exporter.export_png(str(src))
        assert result == str(src)

    def test_export_png_not_found(self, exporter):
        with pytest.raises(FileNotFoundError):
            exporter.export_png("/nonexistent.png")

    def test_export_svg_returns_none(self, exporter, tmp_path):
        src = tmp_path / "chart.png"
        src.write_bytes(b"\x89PNG")
        result = exporter.export_svg(str(src))
        assert result is None

    def test_export_plotly_html(self, exporter):
        plotly = {"data": [{"x": [1], "y": [2]}], "layout": {"title": "Test"}}
        result = exporter.export_plotly_html(plotly)
        assert result.endswith(".html")
        content = Path(result).read_text()
        assert "Plotly" in content

    def test_get_available_formats(self, exporter):
        assert exporter.get_available_formats() == ["png"]
        assert "html" in exporter.get_available_formats(has_plotly=True)


# ── ChartExplainer Tests ────────────────────────────────────────────────────


class TestChartExplainerExtended:
    """Extended tests for ChartExplainer rule-based explanations."""

    @pytest.fixture
    def explainer(self):
        return ChartExplainer()

    def _spec(self, chart_type, x=None, y=None, reasoning="test"):
        return ChartSpec(
            chart_type=chart_type, x_column=x, y_column=y,
            title="Test", reasoning=reasoning, confidence=0.9,
        )

    def test_explain_bar(self, explainer):
        df = pd.DataFrame({"cat": ["A", "B", "A"], "val": [10, 20, 30]})
        result = explainer.explain(df, self._spec(ChartType.BAR, "cat", "val"))
        assert "bar chart" in result.lower()

    def test_explain_bar_missing_col(self, explainer):
        df = pd.DataFrame({"x": [1]})
        result = explainer.explain(df, self._spec(ChartType.BAR, "missing", "val"))
        assert "bar chart" in result.lower()

    def test_explain_pie(self, explainer):
        df = pd.DataFrame({"cat": ["A", "B", "C"], "val": [10, 20, 30]})
        result = explainer.explain(df, self._spec(ChartType.PIE, "cat", "val"))
        assert "pie" in result.lower()

    def test_explain_pie_missing_col(self, explainer):
        df = pd.DataFrame({"x": [1]})
        result = explainer.explain(df, self._spec(ChartType.PIE, "z", "w"))
        assert "pie" in result.lower()

    def test_explain_histogram(self, explainer):
        df = pd.DataFrame({"val": np.random.normal(0, 1, 100)})
        result = explainer.explain(df, self._spec(ChartType.HISTOGRAM, x="val"))
        assert "histogram" in result.lower()

    def test_explain_histogram_missing_col(self, explainer):
        df = pd.DataFrame({"x": [1]})
        result = explainer.explain(df, self._spec(ChartType.HISTOGRAM, x="missing"))
        assert "histogram" in result.lower()

    def test_explain_scatter(self, explainer):
        df = pd.DataFrame({"x": range(50), "y": range(50)})
        result = explainer.explain(df, self._spec(ChartType.SCATTER, "x", "y"))
        assert "scatter" in result.lower()
        assert "correlation" in result.lower()

    def test_explain_scatter_missing(self, explainer):
        df = pd.DataFrame({"x": [1]})
        result = explainer.explain(df, self._spec(ChartType.SCATTER, "a", "b"))
        assert "scatter" in result.lower()

    def test_explain_heatmap(self, explainer):
        df = pd.DataFrame({"x": [1]})
        result = explainer.explain(df, self._spec(ChartType.HEATMAP))
        assert "heatmap" in result.lower()

    def test_explain_correlation(self, explainer):
        df = pd.DataFrame({"a": range(20), "b": range(20), "c": np.random.rand(20)})
        result = explainer.explain(df, self._spec(ChartType.CORRELATION_MATRIX))
        assert "correlation" in result.lower()

    def test_explain_correlation_few_cols(self, explainer):
        df = pd.DataFrame({"a": ["x", "y"]})
        result = explainer.explain(df, self._spec(ChartType.CORRELATION_MATRIX))
        assert "correlation" in result.lower()

    def test_explain_box(self, explainer):
        df = pd.DataFrame({"val": np.random.normal(0, 1, 100)})
        result = explainer.explain(df, self._spec(ChartType.BOX_PLOT, y="val"))
        assert "box" in result.lower()

    def test_explain_box_missing(self, explainer):
        df = pd.DataFrame({"x": [1]})
        result = explainer.explain(df, self._spec(ChartType.BOX_PLOT, y="missing"))
        assert "box" in result.lower()

    def test_explain_violin(self, explainer):
        df = pd.DataFrame({"val": np.random.normal(0, 1, 100)})
        result = explainer.explain(df, self._spec(ChartType.VIOLIN_PLOT, y="val"))
        assert "violin" in result.lower()

    def test_explain_violin_missing(self, explainer):
        df = pd.DataFrame({"x": [1]})
        result = explainer.explain(df, self._spec(ChartType.VIOLIN_PLOT, y="z"))
        assert "violin" in result.lower()

    def test_explain_line(self, explainer):
        df = pd.DataFrame({"val": [10, 20, 30, 40]})
        result = explainer.explain(df, self._spec(ChartType.LINE, y="val"))
        assert "line" in result.lower()

    def test_explain_line_missing(self, explainer):
        df = pd.DataFrame({"x": [1]})
        result = explainer.explain(df, self._spec(ChartType.LINE, y="z"))
        assert "line" in result.lower()

    def test_explain_area(self, explainer):
        df = pd.DataFrame({"val": [10, 20, 30]})
        result = explainer.explain(df, self._spec(ChartType.AREA, y="val"))
        assert "line" in result.lower() or "trend" in result.lower()

    def test_explain_default(self, explainer):
        df = pd.DataFrame({"x": [1]})
        result = explainer.explain(df, self._spec(ChartType.AREA, y="x"))
        assert isinstance(result, str) and len(result) > 0

    def test_explain_with_reasoning(self, explainer):
        df = pd.DataFrame({"x": [1]})
        result = explainer.explain(df, self._spec(ChartType.HEATMAP, reasoning="Best for matrix"))
        assert "Why this chart" in result

    def test_explain_exception_fallback(self, explainer):
        df = pd.DataFrame({"cat": ["A"], "val": ["not_numeric"]})
        spec = self._spec(ChartType.BAR, "cat", "val")
        result = explainer.explain(df, spec)
        assert isinstance(result, str)


# ── Logging Config Tests ────────────────────────────────────────────────────


class TestLoggingConfig:
    """Tests for logging configuration setup."""

    def test_get_logger(self):
        from backend.core.logging_config import get_logger
        logger = get_logger("test_module")
        assert logger is not None
        assert logger.name == "test_module"

    def test_setup_logging(self, tmp_path):
        import logging
        from unittest.mock import MagicMock, patch
        from backend.core.logging_config import setup_logging
        # Clear root handlers so setup_logging proceeds
        root = logging.getLogger()
        original_handlers = root.handlers[:]
        root.handlers = []
        try:
            mock_settings = MagicMock()
            mock_settings.logging.level = "DEBUG"
            mock_settings.logging.dir = str(tmp_path / "logs")
            mock_settings.logging.max_file_size_mb = 1
            mock_settings.logging.backup_count = 2
            setup_logging(settings=mock_settings)
            assert len(root.handlers) > 0
        finally:
            root.handlers = original_handlers
