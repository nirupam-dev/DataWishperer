"""
Pydantic schemas for request / response validation and internal data transfer.

These schemas enforce strict typing at every boundary:
    - API requests and responses
    - Internal service-to-service data transfer
    - Serialization of LLM results and sandbox outputs
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# ── Enumerations ─────────────────────────────────────────────────────────────


class MessageRole(str, Enum):
    """Chat message author role."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ResultType(str, Enum):
    """Type of data returned by the sandbox executor."""

    TEXT = "text"
    DATAFRAME = "dataframe"
    SERIES = "series"
    CHART = "chart"
    ERROR = "error"


class ExportFormat(str, Enum):
    """Supported export formats."""

    CSV = "csv"
    EXCEL = "xlsx"
    MARKDOWN = "md"
    JSON = "json"
    PNG = "png"
    SVG = "svg"


# ── File Schemas ─────────────────────────────────────────────────────────────


class ColumnInfo(BaseModel):
    """Metadata for a single CSV column."""

    name: str
    dtype: str
    non_null_count: int
    null_count: int
    unique_count: int
    sample_values: List[str] = Field(default_factory=list, max_length=5)

    # Numeric-only fields (None for non-numeric columns)
    mean: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None


class FileMetadata(BaseModel):
    """Complete metadata extracted from a CSV file."""

    file_id: str = Field(default_factory=lambda: str(uuid4()))
    original_name: str
    stored_path: str
    row_count: int
    col_count: int
    file_size_bytes: int
    memory_usage_mb: float
    columns: List[ColumnInfo]
    sample_rows: List[Dict[str, Any]] = Field(default_factory=list, max_length=5)
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def column_names(self) -> List[str]:
        """Return a flat list of column names."""
        return [c.name for c in self.columns]

    @property
    def column_dtypes(self) -> Dict[str, str]:
        """Return a mapping of column name → dtype."""
        return {c.name: c.dtype for c in self.columns}


class FileUploadResponse(BaseModel):
    """Response returned after a successful file upload."""

    file_id: str
    filename: str
    row_count: int
    col_count: int
    file_size_mb: float
    columns: List[ColumnInfo]
    preview_rows: List[Dict[str, Any]] = Field(default_factory=list)


# ── Chat Schemas ─────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    """Incoming chat request from the user."""

    session_id: str
    file_id: str
    question: str = Field(..., min_length=1, max_length=2000)


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    file_id: Optional[str] = None
    role: MessageRole
    content: str
    generated_code: Optional[str] = None
    execution_result: Optional[str] = None
    result_type: ResultType = ResultType.TEXT
    chart_path: Optional[str] = None
    tokens_used: int = 0
    latency_ms: float = 0.0
    retry_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ChatResponse(BaseModel):
    """
    Response returned to the user after processing a question.

    Structured for code-interpreter-style display:
        - ``generated_code``: The pandas code that was generated
        - ``result_data``: The execution output (table, text, chart path)
        - ``explanation``: Plain-English explanation of the code & results
        - ``chart_explanation``: Why this chart type was chosen (charts only)
        - ``content``: Pre-formatted combined output for simple consumers
    """

    message_id: str
    content: str
    generated_code: Optional[str] = None
    result_type: ResultType
    result_data: Optional[Any] = None
    chart_path: Optional[str] = None
    explanation: Optional[str] = None
    chart_explanation: Optional[str] = None
    tokens_used: int = 0
    latency_ms: float = 0.0
    retry_count: int = 0
    auto_debug_applied: bool = False
    provider_used: Optional[str] = None
    model_used: Optional[str] = None
    fallback_used: bool = False
    fallback_reason: Optional[str] = None


# ── Session Schemas ──────────────────────────────────────────────────────────


class SessionCreate(BaseModel):
    """Data required to create a new chat session."""

    file_id: str
    title: Optional[str] = None


class SessionInfo(BaseModel):
    """Summary information about a chat session."""

    id: str
    title: str
    file_id: str
    file_name: str
    message_count: int
    created_at: datetime
    updated_at: datetime
    is_active: bool = True
    last_question: Optional[str] = None


class SessionDetail(SessionInfo):
    """Full session including message history."""

    messages: List[ChatMessage] = Field(default_factory=list)


# ── Execution Schemas ────────────────────────────────────────────────────────


class CodeExecutionRequest(BaseModel):
    """Request to execute generated Python code in the sandbox."""

    code: str
    csv_path: str
    chart_dir: str


class CodeExecutionResult(BaseModel):
    """Result from sandboxed code execution."""

    success: bool
    result_type: ResultType
    data: Optional[Any] = None
    chart_path: Optional[str] = None
    stdout: str = ""
    stderr: str = ""
    execution_time_ms: float = 0.0
    error_type: Optional[str] = None
    error_message: Optional[str] = None


# ── LLM Schemas ──────────────────────────────────────────────────────────────


class LLMRequest(BaseModel):
    """Internal request to the LLM provider."""

    messages: List[Dict[str, str]]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class LLMResponse(BaseModel):
    """Internal response from the LLM provider."""

    content: str
    model: str
    tokens_used: int = 0
    latency_ms: float = 0.0
    finish_reason: str = "stop"
    provider: str = "unknown"


# ── Export Schemas ───────────────────────────────────────────────────────────


class ExportRequest(BaseModel):
    """Request to export analysis results."""

    session_id: str
    format: ExportFormat
    include_code: bool = True
    include_charts: bool = True


class ExportResult(BaseModel):
    """Result of an export operation."""

    filename: str
    filepath: str
    format: ExportFormat
    size_bytes: int
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ── Health Schemas ───────────────────────────────────────────────────────────


class HealthStatus(BaseModel):
    """Application health check response."""

    status: str = "healthy"
    version: str
    ollama_connected: bool
    ollama_model_loaded: bool
    database_ok: bool
    uptime_seconds: float
    details: Dict[str, Any] = Field(default_factory=dict)
