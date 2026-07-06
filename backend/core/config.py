"""
Configuration management using Pydantic Settings.

Provides a centralized, type-safe configuration system with support for
environment variables, .env files, and sensible defaults. Follows the
Singleton pattern to ensure a single configuration instance across the
application lifecycle.

Configuration Hierarchy (highest precedence first):
    1. Environment variables
    2. .env file values
    3. Default values in this module
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class _PathDefaults:
    """Resolve default paths relative to project root."""

    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

    @classmethod
    def data_dir(cls) -> str:
        return str(cls.PROJECT_ROOT / "data")

    @classmethod
    def upload_dir(cls) -> str:
        return str(cls.PROJECT_ROOT / "uploads")

    @classmethod
    def log_dir(cls) -> str:
        return str(cls.PROJECT_ROOT / "logs")

    @classmethod
    def export_dir(cls) -> str:
        return str(cls.PROJECT_ROOT / "exports")

    @classmethod
    def charts_dir(cls) -> str:
        return str(cls.PROJECT_ROOT / "charts")


class OllamaSettings(BaseSettings):
    """Ollama LLM server configuration."""

    model_config = SettingsConfigDict(env_prefix="OLLAMA_")

    base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server base URL.",
    )
    model: str = Field(
        default="qwen2.5:7b",
        description="Model identifier to use for inference.",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Low values produce deterministic code.",
    )
    num_predict: int = Field(
        default=2048,
        ge=128,
        le=8192,
        description="Maximum number of tokens to generate.",
    )
    num_ctx: int = Field(
        default=4096,
        ge=1024,
        le=32768,
        description="Context window size in tokens.",
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling probability threshold.",
    )
    repeat_penalty: float = Field(
        default=1.1,
        ge=1.0,
        le=2.0,
        description="Penalty for repeated tokens.",
    )
    timeout: int = Field(
        default=120,
        ge=10,
        le=600,
        description="HTTP timeout for Ollama requests in seconds.",
    )


class SandboxSettings(BaseSettings):
    """Sandboxed code execution configuration."""

    model_config = SettingsConfigDict(env_prefix="SANDBOX_")

    timeout: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Maximum execution time in seconds.",
    )
    max_memory_mb: int = Field(
        default=512,
        ge=64,
        le=2048,
        description="Maximum memory allocation in megabytes.",
    )
    max_output_kb: int = Field(
        default=256,
        ge=16,
        le=1024,
        description="Maximum stdout capture size in kilobytes.",
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Maximum LLM retry attempts on execution failure.",
    )


class StorageSettings(BaseSettings):
    """Database and file storage configuration."""

    model_config = SettingsConfigDict(env_prefix="STORAGE_")

    database_dir: str = Field(default_factory=_PathDefaults.data_dir)
    database_name: str = Field(default="datawhisperer.db")
    upload_dir: str = Field(default_factory=_PathDefaults.upload_dir)
    export_dir: str = Field(default_factory=_PathDefaults.export_dir)
    charts_dir: str = Field(default_factory=_PathDefaults.charts_dir)
    max_file_size_mb: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum upload file size in megabytes.",
    )
    max_columns: int = Field(
        default=500,
        ge=1,
        le=5000,
        description="Maximum number of CSV columns allowed.",
    )

    @property
    def database_url(self) -> str:
        """Construct SQLite connection URL."""
        return f"sqlite:///{Path(self.database_dir) / self.database_name}"

    @property
    def upload_path(self) -> Path:
        return Path(self.upload_dir)

    @property
    def export_path(self) -> Path:
        return Path(self.export_dir)

    @property
    def charts_path(self) -> Path:
        return Path(self.charts_dir)


class ChatSettings(BaseSettings):
    """Chat and conversation configuration."""

    model_config = SettingsConfigDict(env_prefix="CHAT_")

    max_history_messages: int = Field(
        default=20,
        ge=2,
        le=100,
        description="Maximum messages to store per session.",
    )
    history_window_size: int = Field(
        default=6,
        ge=2,
        le=20,
        description="Number of recent messages included in LLM context.",
    )
    max_query_length: int = Field(
        default=2000,
        ge=10,
        le=10000,
        description="Maximum user query length in characters.",
    )
    suggested_questions_count: int = Field(
        default=4,
        ge=0,
        le=8,
        description="Number of auto-generated suggested questions.",
    )


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    model_config = SettingsConfigDict(env_prefix="LOG_")

    level: str = Field(default="INFO", description="Root log level.")
    dir: str = Field(default_factory=_PathDefaults.log_dir)
    max_file_size_mb: int = Field(default=10, ge=1, le=100)
    backup_count: int = Field(default=5, ge=1, le=20)
    format_json: bool = Field(
        default=True,
        description="If True, logs are JSON lines. If False, human-readable.",
    )

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"Log level must be one of {allowed}, got '{v}'")
        return upper


class AppSettings(BaseSettings):
    """
    Root application settings.

    Aggregates all sub-configurations into a single entry point.
    Access via ``get_settings()`` for cached singleton behavior.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="APP_",
        extra="ignore",
    )

    name: str = Field(default="DataWhisperer")
    version: str = Field(default="1.0.0")
    debug: bool = Field(default=False)
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8501)

    # Sub-configurations
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    sandbox: SandboxSettings = Field(default_factory=SandboxSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    chat: ChatSettings = Field(default_factory=ChatSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    def ensure_directories(self) -> None:
        """Create all required directories if they do not exist."""
        dirs = [
            self.storage.upload_path,
            self.storage.export_path,
            self.storage.charts_path,
            Path(self.storage.database_dir),
            Path(self.logging.dir),
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """
    Return the cached application settings singleton.

    Uses ``lru_cache`` to ensure only one instance exists.
    Call ``get_settings.cache_clear()`` in tests to reset.

    Returns:
        AppSettings: The application configuration instance.
    """
    settings = AppSettings()
    settings.ensure_directories()
    return settings
