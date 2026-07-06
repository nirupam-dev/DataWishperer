"""
Export service — Generates downloadable files from analysis results.

Supports exporting:
    - Chat transcripts (Markdown, JSON)
    - Data results (CSV, Excel)
    - Charts (PNG, SVG)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.core.config import StorageSettings, get_settings
from backend.core.logging_config import get_logger
from backend.models.schemas import (
    ChatMessage,
    ExportFormat,
    ExportResult,
    MessageRole,
)

logger = get_logger(__name__)


class ExportService:
    """
    Generates export files from chat sessions and analysis results.

    Args:
        storage_settings: Storage configuration for export directory.
    """

    def __init__(self, storage_settings: Optional[StorageSettings] = None) -> None:
        settings = get_settings()
        self._storage = storage_settings or settings.storage
        self._export_dir = self._storage.export_path
        self._export_dir.mkdir(parents=True, exist_ok=True)

    def export_transcript(
        self,
        session_title: str,
        messages: List[ChatMessage],
        format: ExportFormat = ExportFormat.MARKDOWN,
        include_code: bool = True,
    ) -> ExportResult:
        """
        Export a chat session transcript.

        Args:
            session_title: The session title for the document header.
            messages: List of chat messages to export.
            format: Output format (Markdown or JSON).
            include_code: Whether to include generated Python code.

        Returns:
            An ``ExportResult`` with the file path and metadata.
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in session_title)

        if format == ExportFormat.MARKDOWN:
            return self._export_markdown(safe_title, timestamp, messages, include_code)
        elif format == ExportFormat.JSON:
            return self._export_json(safe_title, timestamp, messages, include_code)
        else:
            return self._export_markdown(safe_title, timestamp, messages, include_code)

    def _export_markdown(
        self,
        title: str,
        timestamp: str,
        messages: List[ChatMessage],
        include_code: bool,
    ) -> ExportResult:
        """Generate a Markdown transcript file."""
        filename = f"{title}_{timestamp}.md"
        filepath = self._export_dir / filename

        lines: List[str] = [
            f"# DataWhisperer — {title}",
            f"*Exported on {datetime.utcnow().strftime('%B %d, %Y at %H:%M UTC')}*",
            "",
            "---",
            "",
        ]

        for msg in messages:
            role_label = "🧑 You" if msg.role == MessageRole.USER else "🤖 DataWhisperer"
            lines.append(f"### {role_label}")
            lines.append("")
            lines.append(msg.content)
            lines.append("")

            if include_code and msg.generated_code:
                lines.append("<details>")
                lines.append("<summary>📝 Generated Code</summary>")
                lines.append("")
                lines.append("```python")
                lines.append(msg.generated_code)
                lines.append("```")
                lines.append("</details>")
                lines.append("")

            lines.append("---")
            lines.append("")

        content = "\n".join(lines)
        filepath.write_text(content, encoding="utf-8")

        logger.info("Exported markdown transcript: '%s'", filepath)

        return ExportResult(
            filename=filename,
            filepath=str(filepath),
            format=ExportFormat.MARKDOWN,
            size_bytes=filepath.stat().st_size,
        )

    def _export_json(
        self,
        title: str,
        timestamp: str,
        messages: List[ChatMessage],
        include_code: bool,
    ) -> ExportResult:
        """Generate a JSON transcript file."""
        filename = f"{title}_{timestamp}.json"
        filepath = self._export_dir / filename

        data: Dict[str, Any] = {
            "title": title,
            "exported_at": datetime.utcnow().isoformat(),
            "message_count": len(messages),
            "messages": [],
        }

        for msg in messages:
            entry: Dict[str, Any] = {
                "role": msg.role.value,
                "content": msg.content,
                "created_at": msg.created_at.isoformat(),
            }
            if include_code and msg.generated_code:
                entry["generated_code"] = msg.generated_code
            if msg.execution_result:
                entry["execution_result"] = msg.execution_result
            data["messages"].append(entry)

        content = json.dumps(data, indent=2, default=str)
        filepath.write_text(content, encoding="utf-8")

        logger.info("Exported JSON transcript: '%s'", filepath)

        return ExportResult(
            filename=filename,
            filepath=str(filepath),
            format=ExportFormat.JSON,
            size_bytes=filepath.stat().st_size,
        )

    def list_exports(self) -> List[Dict[str, Any]]:
        """
        List all export files in the export directory.

        Returns:
            List of export file info dicts.
        """
        exports: List[Dict[str, Any]] = []
        for f in sorted(self._export_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if f.is_file():
                exports.append({
                    "filename": f.name,
                    "filepath": str(f),
                    "size_bytes": f.stat().st_size,
                    "created_at": datetime.fromtimestamp(f.stat().st_mtime),
                })
        return exports

    def get_export_path(self, filename: str) -> Optional[Path]:
        """
        Get the full path for an export file.

        Args:
            filename: The export filename.

        Returns:
            The ``Path`` if the file exists, else ``None``.
        """
        filepath = self._export_dir / filename
        return filepath if filepath.exists() else None
