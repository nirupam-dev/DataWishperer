"""
Ollama LLM provider — LangChain ChatOllama integration.

Replaces raw httpx calls with LangChain's ChatOllama for proper
prompt handling, streaming support, and token accounting. Retains
the BaseLLMProvider interface for dependency injection.

Architecture Decision:
    Using ``langchain_ollama.ChatOllama`` instead of raw HTTP because:
    - Automatic message formatting for chat models
    - Built-in retry/timeout logic
    - Consistent interface with other LangChain providers
    - Proper token counting support
    - Streaming generator support for future UI integration
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import httpx
from langchain_ollama import ChatOllama
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

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


def _dict_to_langchain_message(msg: Dict[str, str]) -> BaseMessage:
    """
    Convert a plain dict message to a LangChain message object.

    Args:
        msg: Dict with ``role`` and ``content`` keys.

    Returns:
        The appropriate LangChain message subclass.
    """
    role = msg.get("role", "user")
    content = msg.get("content", "")

    if role == "system":
        return SystemMessage(content=content)
    elif role == "assistant":
        return AIMessage(content=content)
    else:
        return HumanMessage(content=content)


class OllamaProvider(BaseLLMProvider):
    """
    Ollama LLM provider using LangChain's ChatOllama.

    Implements the ``BaseLLMProvider`` interface with LangChain
    integration, providing health checks, model listing, and
    synchronous generation with proper error classification.

    Args:
        settings: Ollama configuration. Defaults to global settings.
    """

    def __init__(self, settings: Optional[OllamaSettings] = None) -> None:
        self._settings = settings or get_settings().ollama
        self._base_url = self._settings.base_url.rstrip("/")

        # Primary LangChain ChatOllama instance
        self._llm = ChatOllama(
            model=self._settings.model,
            base_url=self._base_url,
            temperature=self._settings.temperature,
            num_predict=self._settings.num_predict,
            num_ctx=self._settings.num_ctx,
            top_p=self._settings.top_p,
            repeat_penalty=self._settings.repeat_penalty,
        )

        # HTTP client for health checks and model listing
        self._http_client = httpx.Client(timeout=10)

        logger.info(
            "OllamaProvider initialized: model=%s, url=%s, ctx=%d",
            self._settings.model,
            self._base_url,
            self._settings.num_ctx,
        )

    @property
    def llm(self) -> ChatOllama:
        """Expose the underlying ChatOllama for direct LangChain use."""
        return self._llm

    def create_llm(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ChatOllama:
        """
        Create a new ChatOllama instance with overridden parameters.

        Useful for different chain stages that need different settings
        (e.g., title generation with higher temperature).

        Args:
            temperature: Override sampling temperature.
            max_tokens: Override max prediction tokens.

        Returns:
            A new ``ChatOllama`` instance.
        """
        return ChatOllama(
            model=self._settings.model,
            base_url=self._base_url,
            temperature=temperature if temperature is not None else self._settings.temperature,
            num_predict=max_tokens or self._settings.num_predict,
            num_ctx=self._settings.num_ctx,
            top_p=self._settings.top_p,
            repeat_penalty=self._settings.repeat_penalty,
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Generate a response using LangChain ChatOllama.

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
        # Select or create appropriate LLM instance
        if temperature is not None or max_tokens is not None:
            llm = self.create_llm(temperature=temperature, max_tokens=max_tokens)
        else:
            llm = self._llm

        # Convert to LangChain messages
        lc_messages = [_dict_to_langchain_message(m) for m in messages]

        start_time = time.time()
        logger.info(
            "Generating via ChatOllama: model=%s, messages=%d",
            self._settings.model,
            len(lc_messages),
        )

        try:
            response = llm.invoke(lc_messages)
        except Exception as e:
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            error_str = str(e).lower()

            if "connection" in error_str or "refused" in error_str:
                raise OllamaConnectionError(
                    base_url=self._base_url,
                    original_error=str(e),
                )
            elif "not found" in error_str or "404" in error_str:
                raise ModelNotFoundError(self._settings.model)
            elif "timeout" in error_str or elapsed_ms > (self._settings.timeout * 1000):
                raise GenerationTimeoutError(self._settings.timeout)
            else:
                raise GenerationError(f"LangChain generation failed: {str(e)[:300]}")

        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        content = response.content if isinstance(response.content, str) else str(response.content)

        if not content.strip():
            raise GenerationError("Ollama returned an empty response.")

        # Extract token usage from response metadata
        metadata = response.response_metadata or {}
        tokens_used = metadata.get("eval_count", 0)
        finish_reason = metadata.get("done_reason", "stop")

        logger.info(
            "ChatOllama response: %d chars, %d tokens, %.0fms",
            len(content),
            tokens_used,
            elapsed_ms,
        )

        return LLMResponse(
            content=content,
            model=self._settings.model,
            tokens_used=tokens_used,
            latency_ms=elapsed_ms,
            finish_reason=finish_reason,
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
            response = self._http_client.get(
                f"{self._base_url}/api/tags", timeout=5
            )
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
                    result["error"] = (
                        f"Model '{target}' not found. "
                        f"Available: {model_names}"
                    )
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
            response = self._http_client.get(
                f"{self._base_url}/api/tags", timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                return [m.get("name", "") for m in data.get("models", [])]
            return []
        except httpx.ConnectError as e:
            raise OllamaConnectionError(
                base_url=self._base_url, original_error=str(e)
            )

    def close(self) -> None:
        """Close HTTP client resources."""
        self._http_client.close()
