"""
Ollama LLM provider — Concrete implementation using the Ollama REST API.

Connects to a locally running Ollama server and uses the Qwen2.5:7B model
(configurable) for code generation. Communicates via the ``/api/chat``
endpoint.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import httpx

from backend.core.config import OllamaSettings, get_settings
from backend.core.exceptions import (
    GenerationError,
    GenerationTimeoutError,
    ModelNotFoundError,
    OllamaConnectionError,
)
from backend.core.logging_config import get_logger
from backend.llm.base_provider import BaseLLMProvider
from backend.models.schemas import LLMResponse

logger = get_logger(__name__)


class OllamaProvider(BaseLLMProvider):
    """
    Ollama LLM provider using the local REST API.

    Implements the ``BaseLLMProvider`` interface for the Ollama inference
    server. Supports health checks, model listing, and synchronous
    generation.

    Args:
        settings: Ollama configuration. Defaults to global settings.
    """

    def __init__(self, settings: Optional[OllamaSettings] = None) -> None:
        self._settings = settings or get_settings().ollama
        self._base_url = self._settings.base_url.rstrip("/")
        self._client = httpx.Client(timeout=self._settings.timeout)

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Generate a response using the Ollama chat API.

        Args:
            messages: List of ``{"role": "...", "content": "..."}`` dicts.
            temperature: Override the configured temperature.
            max_tokens: Override the configured max tokens.

        Returns:
            An ``LLMResponse`` with the generated text and metadata.

        Raises:
            OllamaConnectionError: If Ollama is unreachable.
            ModelNotFoundError: If the model is not pulled.
            GenerationTimeoutError: If the request times out.
            GenerationError: If the response is empty or malformed.
        """
        url = f"{self._base_url}/api/chat"
        payload: Dict[str, Any] = {
            "model": self._settings.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self._settings.temperature,
                "num_predict": max_tokens or self._settings.num_predict,
                "num_ctx": self._settings.num_ctx,
                "top_p": self._settings.top_p,
                "repeat_penalty": self._settings.repeat_penalty,
            },
        }

        start_time = time.time()
        logger.info(
            "Sending request to Ollama: model=%s, messages=%d",
            self._settings.model, len(messages),
        )

        try:
            response = self._client.post(url, json=payload)
        except httpx.ConnectError as e:
            raise OllamaConnectionError(
                base_url=self._base_url, original_error=str(e)
            )
        except httpx.TimeoutException:
            raise GenerationTimeoutError(self._settings.timeout)

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        if response.status_code == 404:
            raise ModelNotFoundError(self._settings.model)

        if response.status_code != 200:
            raise GenerationError(
                f"Ollama returned HTTP {response.status_code}: {response.text[:200]}"
            )

        try:
            data = response.json()
        except Exception:
            raise GenerationError("Failed to parse Ollama response as JSON.")

        content = data.get("message", {}).get("content", "")
        if not content.strip():
            raise GenerationError("Ollama returned an empty response.")

        tokens_used = data.get("eval_count", 0)

        logger.info(
            "Ollama response received: %d chars, %d tokens, %.0fms",
            len(content), tokens_used, elapsed_ms,
        )

        return LLMResponse(
            content=content,
            model=self._settings.model,
            tokens_used=tokens_used,
            latency_ms=elapsed_ms,
            finish_reason=data.get("done_reason", "stop"),
        )

    def health_check(self) -> Dict[str, object]:
        """
        Check Ollama connectivity and model availability.

        Returns:
            Dict with ``connected``, ``model_loaded``, ``models``, and
            ``error`` keys.
        """
        result: Dict[str, object] = {
            "connected": False,
            "model_loaded": False,
            "models": [],
            "error": None,
        }

        try:
            response = self._client.get(f"{self._base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                result["connected"] = True
                data = response.json()
                model_names = [
                    m.get("name", "") for m in data.get("models", [])
                ]
                result["models"] = model_names

                # Check if our target model is available
                target = self._settings.model
                result["model_loaded"] = any(
                    target in name for name in model_names
                )

                if not result["model_loaded"]:
                    result["error"] = f"Model '{target}' not found. Available: {model_names}"
            else:
                result["error"] = f"Ollama returned HTTP {response.status_code}"
        except httpx.ConnectError:
            result["error"] = f"Cannot connect to Ollama at {self._base_url}"
        except Exception as e:
            result["error"] = str(e)

        return result

    def get_model_name(self) -> str:
        """Return the configured model identifier."""
        return self._settings.model

    def list_models(self) -> List[str]:
        """
        List all models available in the local Ollama instance.

        Returns:
            List of model name strings.

        Raises:
            OllamaConnectionError: If Ollama is unreachable.
        """
        try:
            response = self._client.get(f"{self._base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [m.get("name", "") for m in data.get("models", [])]
            return []
        except httpx.ConnectError as e:
            raise OllamaConnectionError(
                base_url=self._base_url, original_error=str(e)
            )

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()
