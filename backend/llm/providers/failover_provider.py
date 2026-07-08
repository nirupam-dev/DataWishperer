"""Provider router: Grok primary, Ollama fallback."""

from __future__ import annotations

from typing import Dict, List, Optional

from backend.core.exceptions import ProviderFallbackError
from backend.core.logging_config import get_logger
from backend.llm.base_provider import BaseLLMProvider
from backend.llm.providers.grok_provider import GrokProvider
from backend.llm.providers.ollama_provider import OllamaProvider
from backend.models.schemas import LLMResponse

logger = get_logger(__name__)


class FailoverLLMProvider(BaseLLMProvider):
    """Routes LLM generation to Grok first, then Ollama on failure."""

    def __init__(
        self,
        grok_provider: Optional[GrokProvider],
        ollama_provider: OllamaProvider,
        local_only_mode: bool = False,
    ) -> None:
        self._grok = grok_provider
        self._ollama = ollama_provider
        self._local_only_mode = local_only_mode
        self._last_generation: Dict[str, object] = {
            "provider": "ollama" if local_only_mode else "unknown",
            "model": self._ollama.get_model_name() if local_only_mode else "unknown",
            "fallback_used": False,
            "fallback_reason": None,
        }

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate via Grok first (unless local-only), then fallback to Ollama."""
        if self._local_only_mode or self._grok is None:
            response = self._ollama.generate(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            self._last_generation = {
                "provider": "ollama",
                "model": response.model,
                "fallback_used": False,
                "fallback_reason": "Local Only Mode enabled" if self._local_only_mode else "Grok unavailable",
            }
            return response

        try:
            response = self._grok.generate(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            self._last_generation = {
                "provider": "grok",
                "model": response.model,
                "fallback_used": False,
                "fallback_reason": None,
            }
            return response
        except Exception as grok_error:
            grok_reason = str(grok_error)[:240]
            logger.warning("Grok failed, falling back to Ollama: %s", grok_reason)

            try:
                response = self._ollama.generate(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                self._last_generation = {
                    "provider": "ollama",
                    "model": response.model,
                    "fallback_used": True,
                    "fallback_reason": grok_reason,
                }
                return response
            except Exception as ollama_error:
                raise ProviderFallbackError(
                    primary_error=grok_reason,
                    fallback_error=str(ollama_error)[:240],
                )

    def fallback_on_malformed_output(
        self,
        messages: List[Dict[str, str]],
        reason: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Force fallback when Grok response is malformed for parser extraction."""
        response = self._ollama.generate(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._last_generation = {
            "provider": "ollama",
            "model": response.model,
            "fallback_used": True,
            "fallback_reason": f"Malformed Grok output: {reason[:180]}",
        }
        return response

    def get_last_generation_metadata(self) -> Dict[str, object]:
        """Expose provider routing metadata for UI/reporting."""
        return dict(self._last_generation)

    def health_check(self) -> Dict[str, object]:
        """Aggregated health for primary/fallback providers."""
        ollama_health = self._ollama.health_check()
        grok_health = (
            self._grok.health_check()
            if self._grok is not None
            else {
                "connected": False,
                "model_loaded": False,
                "credentials_present": False,
                "error": "Grok unavailable",
            }
        )

        ready = bool(
            ollama_health.get("connected")
            and ollama_health.get("model_loaded")
        ) or (
            not self._local_only_mode
            and bool(grok_health.get("connected"))
            and bool(grok_health.get("model_loaded"))
        )

        return {
            "ready": ready,
            "local_only_mode": self._local_only_mode,
            "primary": grok_health,
            "fallback": ollama_health,
            "last_provider": self._last_generation.get("provider"),
            "last_fallback": self._last_generation.get("fallback_used", False),
            "last_fallback_reason": self._last_generation.get("fallback_reason"),
        }

    def get_model_name(self) -> str:
        if self._local_only_mode or self._grok is None:
            return self._ollama.get_model_name()
        return self._grok.get_model_name()

    def close(self) -> None:
        self._ollama.close()
        if self._grok is not None:
            self._grok.close()
