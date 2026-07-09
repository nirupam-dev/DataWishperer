"""LLM provider implementations and default provider builder."""

from __future__ import annotations

from backend.core.config import get_settings
from backend.llm.base_provider import BaseLLMProvider
from backend.llm.providers.failover_provider import FailoverLLMProvider
from backend.llm.providers.groq_provider import GrokProvider
from backend.llm.providers.ollama_provider import OllamaProvider


def create_default_provider() -> BaseLLMProvider:
	"""Create the default provider router (Grok primary, Ollama fallback)."""
	settings = get_settings()
	ollama_provider = OllamaProvider(settings=settings.ollama)

	grok_provider = None
	if not settings.local_only_mode:
		try:
			grok_provider = GrokProvider(settings=settings.grok)
		except Exception:
			# Keep startup resilient: fallback-only mode is valid.
			grok_provider = None

	return FailoverLLMProvider(
		grok_provider=grok_provider,
		ollama_provider=ollama_provider,
		local_only_mode=settings.local_only_mode,
	)


__all__ = [
	"OllamaProvider",
	"GrokProvider",
	"FailoverLLMProvider",
	"create_default_provider",
]
