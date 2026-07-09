"""Tests for Groq/Ollama failover routing and safety fallbacks."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend.core.exceptions import (
    CodeValidationError,
    GenerationError,
    GrokCredentialsError,
    ProviderFallbackError,
)
from backend.llm.agent import DataWhispererAgent
from backend.llm.chains.query_chain import QueryChain
from backend.llm.providers.failover_provider import FailoverLLMProvider
from backend.models.schemas import LLMResponse, ResultType


@pytest.fixture
def sample_messages():
    return [{"role": "user", "content": "show mean"}]


def test_grok_success_no_ollama_call(sample_messages):
    grok = MagicMock()
    ollama = MagicMock()
    grok.generate.return_value = LLMResponse(content="ok", model="llama-3.3-70b-versatile", provider="Groq")

    provider = FailoverLLMProvider(grok_provider=grok, ollama_provider=ollama, local_only_mode=False)
    response = provider.generate(sample_messages)

    assert response.provider == "Groq"
    grok.generate.assert_called_once()
    ollama.generate.assert_not_called()
    meta = provider.get_last_generation_metadata()
    assert meta["provider"] == "Groq"


def test_grok_failure_then_ollama_success(sample_messages):
    grok = MagicMock()
    ollama = MagicMock()
    grok.generate.side_effect = RuntimeError("grok unavailable")
    ollama.generate.return_value = LLMResponse(content="ok", model="qwen2.5:7b", provider="ollama")

    provider = FailoverLLMProvider(grok_provider=grok, ollama_provider=ollama, local_only_mode=False)
    response = provider.generate(sample_messages)

    assert response.provider == "ollama"
    grok.generate.assert_called_once()
    ollama.generate.assert_called_once()
    meta = provider.get_last_generation_metadata()
    assert meta["fallback_used"] is True
    assert meta["provider"] == "Ollama Fallback"


def test_both_providers_fail_raises_provider_fallback_error(sample_messages):
    grok = MagicMock()
    ollama = MagicMock()
    grok.generate.side_effect = RuntimeError("grok down")
    ollama.generate.side_effect = RuntimeError("ollama down")

    provider = FailoverLLMProvider(grok_provider=grok, ollama_provider=ollama, local_only_mode=False)

    with pytest.raises(ProviderFallbackError):
        provider.generate(sample_messages)


def test_missing_groq_api_key_falls_back_to_ollama(sample_messages):
    grok = MagicMock()
    ollama = MagicMock()
    grok.generate.side_effect = GrokCredentialsError("GROQ_API_KEY is missing")
    ollama.generate.return_value = LLMResponse(content="ok", model="qwen2.5:7b", provider="ollama")

    provider = FailoverLLMProvider(grok_provider=grok, ollama_provider=ollama, local_only_mode=False)
    response = provider.generate(sample_messages)

    assert response.provider == "ollama"
    meta = provider.get_last_generation_metadata()
    assert meta["fallback_used"] is True
    assert meta["provider"] == "Ollama Fallback"
    assert "groq_api_key" in str(meta["fallback_reason"]).lower()


def test_local_only_mode_skips_grok(sample_messages):
    grok = MagicMock()
    ollama = MagicMock()
    ollama.get_model_name.return_value = "qwen2.5:7b"
    ollama.generate.return_value = LLMResponse(content="ok", model="qwen2.5:7b", provider="ollama")

    provider = FailoverLLMProvider(grok_provider=grok, ollama_provider=ollama, local_only_mode=True)
    response = provider.generate(sample_messages)

    assert response.provider == "ollama"
    grok.generate.assert_not_called()
    ollama.generate.assert_called_once()


def test_dangerous_code_rejection_sandbox_validation_path(sample_file_metadata):
    provider = MagicMock()
    provider.get_model_name.return_value = "router-model"
    provider.get_last_generation_metadata.return_value = {
        "provider": "ollama",
        "model": "qwen2.5:7b",
        "fallback_used": True,
        "fallback_reason": "unsafe code fallback",
    }

    chain = MagicMock(spec=QueryChain)
    chain.generate_code.return_value = (
        "import os\nresult = 1",
        LLMResponse(content="code", model="llama-3.3-70b-versatile", provider="Groq", tokens_used=10),
        None,
    )
    chain.debug_code.side_effect = GenerationError("cannot safely repair")

    sandbox = MagicMock()
    sandbox.execute.side_effect = CodeValidationError(["Blocked import: os"], code="import os")

    memory = MagicMock()
    memory.set_active_dataset.return_value = None

    agent = DataWhispererAgent(
        provider=provider,
        query_chain=chain,
        sandbox=sandbox,
        memory=memory,
    )
    agent.register_dataset(sample_file_metadata)

    result = agent.process_question(
        session_id="s1",
        file_id=sample_file_metadata.file_id,
        question="run shell command",
        csv_path=sample_file_metadata.stored_path,
        file_metadata=sample_file_metadata,
    )

    assert result.success is False
    assert result.result_type == ResultType.ERROR
    assert result.auto_debug_applied is True
    assert result.fallback_used is True


def test_query_chain_malformed_output_without_fallback_helper_is_safe(sample_file_metadata):
    class _ProviderWithoutFallback:
        def generate(self, messages, temperature=None, max_tokens=None):
            return LLMResponse(
                content="no executable code here",
                model="grok-3-mini",
                provider="grok",
            )

        def health_check(self):
            return {"connected": True, "model_loaded": True}

        def get_model_name(self):
            return "grok-3-mini"

        def close(self):
            return None

    provider = _ProviderWithoutFallback()

    chain = QueryChain(provider=provider)

    with pytest.raises(GenerationError):
        chain.generate_code(
            question="mean?",
            file_metadata=sample_file_metadata,
            session_id="sess-malformed",
        )
