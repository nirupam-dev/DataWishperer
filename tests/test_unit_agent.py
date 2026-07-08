"""
Unit tests for the OllamaProvider and DataWhispererAgent.

All LLM calls are mocked to avoid requiring a running Ollama instance.
Tests cover:
    - Provider: generate, health_check, error classification, model listing
    - Agent: dataset management, process_question pipeline, error handling,
             health check, session management, formatting methods
    - Factory: create_provider, create_agent, create_chat_service
"""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from backend.core.config import ChatSettings, OllamaSettings, SandboxSettings
from backend.core.exceptions import (
    CodeValidationError,
    ExecutionRuntimeError,
    ExecutionTimeoutError,
    GenerationError,
    OllamaConnectionError,
)
from backend.llm.agent import AgentResult, DataWhispererAgent
from backend.llm.chains.query_chain import QueryChain
from backend.llm.memory import ConversationMemory
from backend.llm.providers.ollama_provider import OllamaProvider, _dict_to_langchain_message
from backend.models.schemas import (
    CodeExecutionResult,
    ColumnInfo,
    FileMetadata,
    LLMResponse,
    ResultType,
)
from backend.sandbox.executor import SandboxExecutor


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_provider():
    """Create a mocked OllamaProvider."""
    provider = MagicMock(spec=OllamaProvider)
    provider.get_model_name.return_value = "qwen2.5:7b"
    provider.health_check.return_value = {
        "connected": True,
        "model_loaded": True,
        "models": ["qwen2.5:7b"],
        "error": None,
    }
    return provider


@pytest.fixture
def mock_chain():
    """Create a mocked QueryChain."""
    chain = MagicMock(spec=QueryChain)
    chain.generate_code.return_value = (
        "result = df.head()",
        LLMResponse(content="code", model="qwen2.5:7b", tokens_used=100, latency_ms=500),
        None,  # reasoning
    )
    chain.generate_explanation.return_value = "This shows the first 5 rows."
    chain.generate_chart_explanation.return_value = None
    chain.generate_title.return_value = "Data Preview"
    chain.generate_suggested_questions.return_value = ["Q1?", "Q2?"]
    return chain


@pytest.fixture
def mock_sandbox():
    """Create a mocked SandboxExecutor."""
    sandbox = MagicMock(spec=SandboxExecutor)
    sandbox.execute.return_value = CodeExecutionResult(
        success=True,
        result_type=ResultType.TEXT,
        data="Preview of data",
        execution_time_ms=150.0,
    )
    return sandbox


@pytest.fixture
def mock_memory():
    """Create a mocked ConversationMemory."""
    memory = MagicMock(spec=ConversationMemory)
    memory.set_active_dataset.return_value = None
    memory.get_session_count.return_value = 1
    return memory


@pytest.fixture
def agent(mock_provider, mock_chain, mock_sandbox, mock_memory):
    """Create an agent with all dependencies mocked."""
    with patch("backend.llm.agent.get_settings") as mock_settings:
        settings = MagicMock()
        settings.chat = ChatSettings()
        settings.sandbox = SandboxSettings()
        mock_settings.return_value = settings

        return DataWhispererAgent(
            provider=mock_provider,
            query_chain=mock_chain,
            sandbox=mock_sandbox,
            memory=mock_memory,
            chat_settings=settings.chat,
            sandbox_settings=settings.sandbox,
        )


@pytest.fixture
def file_metadata():
    return FileMetadata(
        file_id="test-001",
        original_name="sales.csv",
        stored_path="/tmp/test/sales.csv",
        row_count=100,
        col_count=3,
        file_size_bytes=5000,
        memory_usage_mb=0.05,
        columns=[
            ColumnInfo(name="product", dtype="object",
                       non_null_count=100, null_count=0,
                       unique_count=10, sample_values=["Widget"]),
            ColumnInfo(name="revenue", dtype="float64",
                       non_null_count=100, null_count=0,
                       unique_count=100, sample_values=["500.0"],
                       mean=500.0, std=200.0, min_val=10.0, max_val=2000.0),
            ColumnInfo(name="quantity", dtype="int64",
                       non_null_count=100, null_count=0,
                       unique_count=50, sample_values=["10"],
                       mean=25.0, std=15.0, min_val=1.0, max_val=99.0),
        ],
    )


# ── Message Conversion Tests ────────────────────────────────────────────────


class TestMessageConversion:
    """Test _dict_to_langchain_message helper."""

    def test_user_message(self):
        from langchain_core.messages import HumanMessage
        msg = _dict_to_langchain_message({"role": "user", "content": "Hello"})
        assert isinstance(msg, HumanMessage)
        assert msg.content == "Hello"

    def test_system_message(self):
        from langchain_core.messages import SystemMessage
        msg = _dict_to_langchain_message({"role": "system", "content": "System"})
        assert isinstance(msg, SystemMessage)

    def test_assistant_message(self):
        from langchain_core.messages import AIMessage
        msg = _dict_to_langchain_message({"role": "assistant", "content": "Resp"})
        assert isinstance(msg, AIMessage)

    def test_default_role_is_user(self):
        from langchain_core.messages import HumanMessage
        msg = _dict_to_langchain_message({"content": "No role"})
        assert isinstance(msg, HumanMessage)


# ── Agent Dataset Management Tests ──────────────────────────────────────────


class TestAgentDatasetManagement:
    """Test dataset registration, retrieval, and unregistration."""

    def test_register_dataset(self, agent, file_metadata):
        agent.register_dataset(file_metadata)
        assert agent.get_dataset("test-001") is not None

    def test_unregister_dataset(self, agent, file_metadata):
        agent.register_dataset(file_metadata)
        agent.unregister_dataset("test-001")
        assert agent.get_dataset("test-001") is None

    def test_get_all_datasets(self, agent, file_metadata):
        agent.register_dataset(file_metadata)
        datasets = agent.get_all_datasets()
        assert "test-001" in datasets

    def test_unregister_nonexistent_is_safe(self, agent):
        agent.unregister_dataset("nonexistent")  # Should not raise


# ── Agent Pipeline Tests ────────────────────────────────────────────────────


class TestAgentProcessQuestion:
    """Test the 8-stage interpreter pipeline with mocked components."""

    def test_successful_pipeline(self, agent, file_metadata, mock_chain, mock_sandbox):
        agent.register_dataset(file_metadata)

        result = agent.process_question(
            session_id="sess-1",
            file_id="test-001",
            question="Show first rows",
            csv_path="/tmp/test/sales.csv",
            file_metadata=file_metadata,
        )

        assert result.success is True
        assert result.code is not None
        assert result.explanation is not None
        assert result.attempts == 1

    def test_dataset_not_found(self, agent):
        result = agent.process_question(
            session_id="sess-1",
            file_id="nonexistent",
            question="Show data",
            csv_path="/tmp/test.csv",
        )
        assert result.success is False
        assert "not found" in result.content.lower()

    def test_generation_error_handled(self, agent, file_metadata, mock_chain):
        mock_chain.generate_code.side_effect = GenerationError("Empty response")
        agent.register_dataset(file_metadata)

        result = agent.process_question(
            session_id="sess-1",
            file_id="test-001",
            question="Bad question",
            csv_path="/tmp/test.csv",
            file_metadata=file_metadata,
        )

        assert result.success is False
        assert result.result_type == ResultType.ERROR

    def test_ollama_connection_error_handled(self, agent, file_metadata, mock_chain):
        mock_chain.generate_code.side_effect = OllamaConnectionError(
            base_url="http://localhost:11434"
        )
        agent.register_dataset(file_metadata)

        result = agent.process_question(
            session_id="sess-1",
            file_id="test-001",
            question="Test",
            csv_path="/tmp/test.csv",
            file_metadata=file_metadata,
        )

        assert result.success is False
        assert "connect" in result.content.lower()

    def test_auto_debug_on_execution_failure(self, agent, file_metadata, mock_chain, mock_sandbox):
        """When first execution fails but debug succeeds."""
        mock_sandbox.execute.side_effect = [
            ExecutionRuntimeError(
                error_type="KeyError", error_message="'nonexistent'", code="bad"
            ),
            CodeExecutionResult(
                success=True, result_type=ResultType.TEXT,
                data="fixed result", execution_time_ms=200.0,
            ),
        ]
        mock_chain.debug_code.return_value = (
            "result = 'fixed'",
            LLMResponse(content="fixed", model="qwen2.5:7b", tokens_used=50),
        )
        agent.register_dataset(file_metadata)

        result = agent.process_question(
            session_id="sess-1",
            file_id="test-001",
            question="Show data",
            csv_path="/tmp/test.csv",
            file_metadata=file_metadata,
        )

        assert result.success is True
        assert result.auto_debug_applied is True
        assert result.attempts == 2

    def test_double_failure_returns_error(self, agent, file_metadata, mock_chain, mock_sandbox):
        """When both original and debug execution fail."""
        mock_sandbox.execute.side_effect = ExecutionRuntimeError(
            error_type="TypeError", error_message="bad type", code="bad"
        )
        mock_chain.debug_code.side_effect = GenerationError("Cannot fix")
        agent.register_dataset(file_metadata)

        result = agent.process_question(
            session_id="sess-1",
            file_id="test-001",
            question="complex query",
            csv_path="/tmp/test.csv",
            file_metadata=file_metadata,
        )

        assert result.success is False
        assert result.auto_debug_applied is True
        assert result.attempts == 2

    def test_chart_result_includes_chart_explanation(self, agent, file_metadata, mock_chain, mock_sandbox):
        mock_sandbox.execute.return_value = CodeExecutionResult(
            success=True, result_type=ResultType.CHART,
            data="Chart generated", chart_path="/tmp/chart.png",
            execution_time_ms=300.0,
        )
        mock_chain.generate_chart_explanation.return_value = "A bar chart was chosen."
        agent.register_dataset(file_metadata)

        result = agent.process_question(
            session_id="sess-1",
            file_id="test-001",
            question="Plot revenue",
            csv_path="/tmp/test.csv",
            file_metadata=file_metadata,
        )

        assert result.success is True
        assert result.chart_explanation is not None


# ── Agent Auxiliary Tests ────────────────────────────────────────────────────


class TestAgentAuxiliary:
    """Test helper/auxiliary capabilities."""

    def test_generate_session_title(self, agent, mock_chain):
        title = agent.generate_session_title("Show average revenue")
        mock_chain.generate_title.assert_called_once_with("Show average revenue")
        assert title == "Data Preview"

    def test_generate_suggested_questions(self, agent, file_metadata, mock_chain):
        questions = agent.generate_suggested_questions(file_metadata, count=3)
        mock_chain.generate_suggested_questions.assert_called_once()

    def test_health_check(self, agent, mock_provider, mock_memory):
        health = agent.health_check()
        assert "agent_ready" in health
        assert "ollama" in health
        assert "model" in health

    def test_clear_session(self, agent, mock_memory):
        agent.clear_session("sess-1")
        mock_memory.clear_session.assert_called_once_with("sess-1")

    def test_close(self, agent, mock_provider):
        agent.close()
        mock_provider.close.assert_called_once()

    def test_load_session_memory(self, agent, mock_memory):
        msgs = [{"role": "user", "content": "hi"}]
        agent.load_session_memory("sess-1", msgs)
        mock_memory.load_from_db_messages.assert_called_once_with("sess-1", msgs)

    def test_properties_expose_dependencies(self, agent, mock_provider, mock_chain, mock_memory):
        assert agent.provider is mock_provider
        assert agent.chain is mock_chain
        assert agent.memory is mock_memory


# ── Agent Result Tests ───────────────────────────────────────────────────────


class TestAgentResult:
    """Test AgentResult data structure."""

    def test_default_values(self):
        r = AgentResult(success=True, content="ok")
        assert r.success is True
        assert r.content == "ok"
        assert r.code is None
        assert r.result_type == ResultType.TEXT
        assert r.auto_debug_applied is False
        assert r.tokens_used == 0
        assert r.attempts == 1

    def test_all_fields(self):
        r = AgentResult(
            success=True, content="ok", code="x = 1",
            result_type=ResultType.DATAFRAME, result_data="{}",
            chart_path="/tmp/chart.png", explanation="Explains it",
            chart_explanation="Bar chart", auto_debug_applied=True,
            debug_summary="Fixed KeyError", tokens_used=500,
            latency_ms=1234.56, attempts=2,
            internal_reasoning="Step 1...",
        )
        assert r.chart_path == "/tmp/chart.png"
        assert r.attempts == 2
        assert r.internal_reasoning == "Step 1..."


# ── Agent Formatter Tests ────────────────────────────────────────────────────


class TestAgentFormatters:
    """Test static formatting methods."""

    def test_format_interpreter_output_text(self):
        result = CodeExecutionResult(
            success=True, result_type=ResultType.TEXT,
            data="42", execution_time_ms=100.0,
        )
        output = DataWhispererAgent._format_interpreter_output(
            code="result = 42",
            execution_result=result,
            explanation="Returns 42",
            chart_explanation=None,
            auto_debug_applied=False,
        )
        assert "Generated Code" in output
        assert "42" in output
        assert "Auto-Debug" not in output

    def test_format_interpreter_output_chart(self):
        result = CodeExecutionResult(
            success=True, result_type=ResultType.CHART,
            data="", chart_path="/tmp/chart.png",
            execution_time_ms=200.0,
        )
        output = DataWhispererAgent._format_interpreter_output(
            code="plt.savefig(chart_path)",
            execution_result=result,
            explanation="Creates a chart",
            chart_explanation="Bar chart chosen for comparison",
            auto_debug_applied=True,
        )
        assert "Chart Reasoning" in output
        assert "Auto-Debug" in output

    def test_format_interpreter_output_no_data(self):
        result = CodeExecutionResult(
            success=True, result_type=ResultType.TEXT,
            data=None, execution_time_ms=100.0,
        )
        output = DataWhispererAgent._format_interpreter_output(
            code="x = 1",
            execution_result=result,
            explanation="Analysis complete.",
            chart_explanation=None,
            auto_debug_applied=False,
        )
        assert "executed successfully" in output

    def test_summarize_result_chart(self):
        result = CodeExecutionResult(
            success=True, result_type=ResultType.CHART, execution_time_ms=100.0,
        )
        assert "chart" in DataWhispererAgent._summarize_result(result).lower()

    def test_summarize_result_dataframe(self):
        result = CodeExecutionResult(
            success=True, result_type=ResultType.DATAFRAME,
            data="[{...}]", execution_time_ms=100.0,
        )
        assert "table" in DataWhispererAgent._summarize_result(result).lower()

    def test_summarize_result_series(self):
        result = CodeExecutionResult(
            success=True, result_type=ResultType.SERIES,
            data="{}", execution_time_ms=100.0,
        )
        assert "series" in DataWhispererAgent._summarize_result(result).lower()

    def test_format_error_ollama(self):
        err = OllamaConnectionError(base_url="http://localhost:11434")
        msg = DataWhispererAgent._format_error(err)
        assert "connect" in msg.lower()

    def test_format_error_generic(self):
        err = RuntimeError("Unexpected crash")
        msg = DataWhispererAgent._format_error(err)
        assert "unexpected" in msg.lower()

    def test_format_debug_failure(self):
        e1 = ExecutionRuntimeError(error_type="KeyError", error_message="'col'", code="x")
        e2 = GenerationError("Cannot fix")
        msg = DataWhispererAgent._format_debug_failure(e1, e2, "result = 42")
        assert "Failed" in msg
        assert "Original Error" in msg
        assert "Debug Attempt" in msg


# ── Factory Tests ────────────────────────────────────────────────────────────


class TestFactory:
    """Test the agent factory DI container."""

    @patch("backend.llm.factory.FailoverLLMProvider")
    @patch("backend.llm.factory.GrokProvider")
    @patch("backend.llm.factory.OllamaProvider")
    @patch("backend.llm.factory.get_settings")
    def test_create_provider(
        self,
        mock_settings,
        mock_ollama_cls,
        mock_grok_cls,
        mock_failover_cls,
    ):
        from backend.llm.factory import create_provider

        settings = MagicMock()
        settings.local_only_mode = False
        settings.grok.model = "grok-3-mini"
        settings.ollama.model = "qwen2.5:7b"
        mock_settings.return_value = settings
        mock_ollama = MagicMock()
        mock_grok = MagicMock()
        mock_router = MagicMock()
        mock_ollama_cls.return_value = mock_ollama
        mock_grok_cls.return_value = mock_grok
        mock_failover_cls.return_value = mock_router

        provider = create_provider()
        mock_ollama_cls.assert_called_once_with(settings=settings.ollama)
        mock_grok_cls.assert_called_once_with(settings=settings.grok)
        mock_failover_cls.assert_called_once()
        assert provider is mock_router

    @patch("backend.llm.factory.FailoverLLMProvider")
    @patch("backend.llm.factory.GrokProvider")
    @patch("backend.llm.factory.OllamaProvider")
    @patch("backend.llm.factory.get_settings")
    def test_create_provider_local_only_skips_grok(
        self,
        mock_settings,
        mock_ollama_cls,
        mock_grok_cls,
        mock_failover_cls,
    ):
        from backend.llm.factory import create_provider

        settings = MagicMock()
        settings.local_only_mode = True
        settings.grok.model = "grok-3-mini"
        settings.ollama.model = "qwen2.5:7b"
        mock_settings.return_value = settings
        mock_ollama = MagicMock()
        mock_router = MagicMock()
        mock_ollama_cls.return_value = mock_ollama
        mock_failover_cls.return_value = mock_router

        provider = create_provider()
        mock_ollama_cls.assert_called_once_with(settings=settings.ollama)
        mock_grok_cls.assert_not_called()
        mock_failover_cls.assert_called_once()
        assert provider is mock_router

    @patch("backend.llm.factory.OllamaProvider")
    @patch("backend.llm.factory.SandboxExecutor")
    @patch("backend.llm.factory.get_settings")
    def test_create_agent(self, mock_settings, mock_sandbox, mock_provider_cls):
        from backend.llm.factory import create_agent
        settings = MagicMock()
        settings.chat = ChatSettings()
        settings.sandbox = SandboxSettings()
        mock_settings.return_value = settings
        mock_provider = MagicMock()
        mock_provider.get_model_name.return_value = "test"
        mock_provider_cls.return_value = mock_provider

        agent = create_agent()
        assert agent is not None

    @patch("backend.llm.factory.ChatService")
    @patch("backend.llm.factory.SessionRepository")
    @patch("backend.llm.factory.MessageRepository")
    @patch("backend.llm.factory.get_settings")
    def test_create_chat_service_with_agent(
        self, mock_settings, mock_msg_repo, mock_sess_repo, mock_chat_svc
    ):
        from backend.llm.factory import create_chat_service
        settings = MagicMock()
        settings.chat = ChatSettings()
        mock_settings.return_value = settings
        mock_agent = MagicMock()

        create_chat_service(agent=mock_agent)
        mock_chat_svc.assert_called_once()
