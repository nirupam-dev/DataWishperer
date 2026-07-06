"""
Abstract LLM provider interface.

Implements the Strategy pattern: all LLM backends (Ollama, future local
providers) implement this interface. Business logic depends on the
abstraction, never on a concrete provider.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from backend.models.schemas import LLMResponse


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All providers must implement ``generate()`` and ``health_check()``.
    Optionally implement ``stream_generate()`` for token streaming.
    """

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            messages: List of message dicts with ``role`` and ``content`` keys.
            temperature: Optional temperature override.
            max_tokens: Optional max token override.

        Returns:
            An ``LLMResponse`` containing the generated text and metadata.

        Raises:
            OllamaConnectionError: If the provider is unreachable.
            ModelNotFoundError: If the requested model is not available.
            GenerationTimeoutError: If generation exceeds the timeout.
        """
        ...

    @abstractmethod
    def health_check(self) -> Dict[str, object]:
        """
        Check if the LLM provider is healthy and the model is available.

        Returns:
            A dict with at minimum ``connected`` (bool) and
            ``model_loaded`` (bool) keys.
        """
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the name of the currently configured model."""
        ...
