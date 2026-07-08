"""
Grok LLM provider — LangChain ChatOpenAI integration against xAI API.

Uses xAI's official API endpoint (https://api.x.ai/v1) through LangChain's
ChatOpenAI wrapper so the project keeps a consistent LangChain pipeline.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from backend.core.config import GrokSettings, get_settings
from backend.core.exceptions import (
    GenerationError,
    GenerationTimeoutError,
    GrokAPIError,
    GrokCredentialsError,
    GrokRateLimitError,
)
from backend.core.logging_config import get_logger
from backend.llm.base_provider import BaseLLMProvider
from backend.models.schemas import LLMResponse

logger = get_logger(__name__)

_XAI_BASE_URL = "https://api.x.ai/v1"


def _dict_to_langchain_message(msg: Dict[str, str]) -> BaseMessage:
    """Convert plain message dicts to LangChain messages."""
    role = msg.get("role", "user")
    content = msg.get("content", "")

    if role == "system":
        return SystemMessage(content=content)
    if role == "assistant":
        return AIMessage(content=content)
    return HumanMessage(content=content)


class GrokProvider(BaseLLMProvider):
    """xAI Grok provider with retry and error classification."""

    def __init__(self, settings: Optional[GrokSettings] = None) -> None:
        self._settings = settings or get_settings().grok
        self._api_key = os.getenv("XAI_API_KEY", "").strip()
        self._llm = self._create_llm()

        logger.info(
            "GrokProvider initialized: model=%s, timeout=%ss, retries=%d, key_present=%s",
            self._settings.model,
            self._settings.timeout_seconds,
            self._settings.max_retries,
            bool(self._api_key),
        )

    def _create_llm(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ChatOpenAI:
        """Create a ChatOpenAI client configured for xAI API."""
        return ChatOpenAI(
            model=self._settings.model,
            api_key=self._api_key or "MISSING_KEY",
            base_url=self._settings.base_url,
            timeout=float(self._settings.timeout_seconds),
            max_retries=0,
            temperature=0.1 if temperature is None else temperature,
            max_tokens=max_tokens,
        )

    def _ensure_credentials(self) -> None:
        if not self._api_key:
            raise GrokCredentialsError("XAI_API_KEY is missing")

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate a response from Grok with bounded retries."""
        self._ensure_credentials()

        llm = self._llm
        if temperature is not None or max_tokens is not None:
            llm = self._create_llm(temperature=temperature, max_tokens=max_tokens)

        lc_messages = [_dict_to_langchain_message(m) for m in messages]

        attempts = self._settings.max_retries + 1
        last_error: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            start_time = time.time()
            try:
                response = llm.invoke(lc_messages)
                elapsed_ms = round((time.time() - start_time) * 1000, 2)

                content = response.content if isinstance(response.content, str) else str(response.content)
                if not content.strip():
                    raise GenerationError("Grok returned an empty response")

                metadata = response.response_metadata or {}
                usage = metadata.get("token_usage") or {}
                tokens_used = int(
                    usage.get("total_tokens")
                    or usage.get("completion_tokens")
                    or 0
                )
                finish_reason = metadata.get("finish_reason", "stop")

                return LLMResponse(
                    content=content,
                    model=metadata.get("model_name", self._settings.model),
                    tokens_used=tokens_used,
                    latency_ms=elapsed_ms,
                    finish_reason=finish_reason,
                    provider="grok",
                )
            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                if "401" in error_str or "unauthorized" in error_str or "invalid api key" in error_str:
                    raise GrokCredentialsError("Invalid XAI_API_KEY")
                if "429" in error_str or "rate" in error_str:
                    if attempt < attempts:
                        logger.warning("Grok rate-limited (attempt %d/%d), retrying", attempt, attempts)
                        continue
                    raise GrokRateLimitError()
                if "timeout" in error_str:
                    if attempt < attempts:
                        logger.warning("Grok timeout (attempt %d/%d), retrying", attempt, attempts)
                        continue
                    raise GenerationTimeoutError(self._settings.timeout_seconds)

                if attempt < attempts:
                    logger.warning("Grok call failed (attempt %d/%d): %s", attempt, attempts, str(e)[:180])
                    continue

        raise GrokAPIError(str(last_error)[:300] if last_error else "Unknown Grok API error")

    def health_check(self) -> Dict[str, object]:
        """Return configuration-level health for Grok."""
        key_present = bool(self._api_key)
        return {
            "connected": key_present,
            "model_loaded": key_present,
            "credentials_present": key_present,
            "model": self._settings.model,
            "error": None if key_present else "Missing XAI_API_KEY",
        }

    def get_model_name(self) -> str:
        return self._settings.model

    def close(self) -> None:
        """No-op for interface compatibility."""
        return
