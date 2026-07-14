"""
Groq LLM provider — LangChain ChatOpenAI integration against Groq API.

Uses Groq's official API endpoint (https://api.groq.com/openai/v1) through
LangChain's ChatOpenAI wrapper so the project keeps a consistent pipeline.
Groq is the PRIMARY LLM provider; Ollama is the fallback.

Supports multiple API keys (GROQ_API_KEY, GROQ_API_KEY_2, …) for automatic
rotation when a key hits rate-limits, quota exhaustion, or auth errors.
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

# Groq is the primary cloud provider; key accepted via GROQ_API_KEY or XAI_API_KEY


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
    """Groq cloud LLM provider with retry, error classification, and API key rotation."""

    def __init__(self, settings: Optional[GrokSettings] = None) -> None:
        self._settings = settings or get_settings().grok

        # Collect all available API keys for rotation.
        # Accepts GROQ_API_KEY, GROQ_API_KEY_2, …, and XAI_API_KEY (legacy).
        self._api_keys = self._collect_api_keys()
        self._active_key_index = 0
        self._api_key = self._api_keys[0] if self._api_keys else ""

        self._llm = self._create_llm()

        logger.info(
            "GrokProvider initialized: model=%s, timeout=%ss, retries=%d, "
            "api_keys_available=%d, active_key_index=%d",
            self._settings.model,
            self._settings.timeout_seconds,
            self._settings.max_retries,
            len(self._api_keys),
            self._active_key_index,
        )

    @staticmethod
    def _collect_api_keys() -> List[str]:
        """Gather all non-empty Groq API keys from the environment.

        Checks GROQ_API_KEY, GROQ_API_KEY_2, GROQ_API_KEY_3, … (up to 10),
        plus XAI_API_KEY as a legacy fallback.
        """
        keys: List[str] = []
        # Primary key
        primary = os.getenv("GROQ_API_KEY", "").strip()
        if primary:
            keys.append(primary)
        # Numbered backup keys: GROQ_API_KEY_2 … GROQ_API_KEY_10
        for i in range(2, 11):
            k = os.getenv(f"GROQ_API_KEY_{i}", "").strip()
            if k:
                keys.append(k)
        # Legacy alias
        legacy = os.getenv("XAI_API_KEY", "").strip()
        if legacy and legacy not in keys:
            keys.append(legacy)
        return keys

    def _rotate_api_key(self) -> bool:
        """Rotate to the next available API key.

        Returns True if a new key was activated, False if no more keys remain.
        """
        if len(self._api_keys) <= 1:
            return False

        next_index = (self._active_key_index + 1) % len(self._api_keys)
        if next_index == self._active_key_index:
            return False  # Full circle — no fresh key available

        self._active_key_index = next_index
        self._api_key = self._api_keys[next_index]
        # Rebuild the default LLM client with the new key
        self._llm = self._create_llm()

        logger.info(
            "Rotated Groq API key → index %d/%d",
            next_index + 1,
            len(self._api_keys),
        )
        return True

    def _create_llm(
        self,  # noqa: D102
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ChatOpenAI:
        """Create a ChatOpenAI client configured for Groq API."""
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
            raise GrokCredentialsError(
                "No GROQ_API_KEY found. "
                "Set GROQ_API_KEY (and optionally GROQ_API_KEY_2) in your .env file."
            )

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate a response from Groq with bounded retries and API key rotation.

        On rate-limit (429) or auth (401) errors, the provider automatically
        rotates to the next available API key before retrying.
        """
        self._ensure_credentials()

        llm = self._llm
        if temperature is not None or max_tokens is not None:
            llm = self._create_llm(temperature=temperature, max_tokens=max_tokens)

        lc_messages = [_dict_to_langchain_message(m) for m in messages]

        attempts = self._settings.max_retries + 1
        last_error: Optional[Exception] = None
        # Track which keys we've already tried to avoid infinite loops
        keys_tried: set[int] = {self._active_key_index}

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
                    provider="Groq",
                )
            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # ── Key-rotatable errors (auth / rate-limit / quota) ────
                is_auth_error = (
                    "401" in error_str
                    or "unauthorized" in error_str
                    or "invalid api key" in error_str
                )
                is_rate_limit = "429" in error_str or "rate" in error_str

                if is_auth_error or is_rate_limit:
                    # Try rotating to the next API key
                    rotated = self._rotate_api_key()
                    if rotated and self._active_key_index not in keys_tried:
                        keys_tried.add(self._active_key_index)
                        # Rebuild the LLM with the rotated key
                        llm = self._create_llm(
                            temperature=temperature, max_tokens=max_tokens
                        )
                        error_kind = "auth" if is_auth_error else "rate-limit"
                        logger.warning(
                            "Groq %s error — rotated to API key %d/%d, retrying",
                            error_kind,
                            self._active_key_index + 1,
                            len(self._api_keys),
                        )
                        continue  # retry immediately with new key

                    # No more keys to try
                    if is_auth_error:
                        raise GrokCredentialsError(
                            "All GROQ_API_KEY(s) are invalid or expired."
                        )
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
        """Return configuration-level health for Groq."""
        key_present = bool(self._api_key)
        return {
            "connected": key_present,
            "model_loaded": key_present,
            "credentials_present": key_present,
            "model": self._settings.model,
            "api_keys_available": len(self._api_keys),
            "active_key_index": self._active_key_index + 1,
            "error": None if key_present else "Missing GROQ_API_KEY",
        }

    def get_model_name(self) -> str:
        return self._settings.model

    def close(self) -> None:
        """No-op for interface compatibility."""
        return
