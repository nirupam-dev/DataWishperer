"""
Visualization service — Post-processes and manages generated charts.

Handles chart file management, Plotly figure creation for interactive
charts in Streamlit, and chart theming.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.core.config import StorageSettings, get_settings
from backend.core.logging_config import get_logger

logger = get_logger(__name__)

# Dark theme configuration for Plotly charts (used by Streamlit frontend)
PLOTLY_DARK_THEME: Dict[str, Any] = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"family": "Inter, sans-serif", "color": "#F0F0F5"},
    "colorway": [
        "#6C5CE7", "#00CEC9", "#FD79A8", "#FDCB6E",
        "#55EFC4", "#A29BFE", "#FF7675", "#74B9FF",
    ],
}


class VisualizationService:
    """
    Manages chart generation artifacts and theming.

    This service does not generate charts directly (the sandbox does that).
    Instead, it manages chart files, applies consistent theming, and
    provides chart metadata for the UI.

    Args:
        storage_settings: Storage configuration.
    """

    def __init__(self, storage_settings: Optional[StorageSettings] = None) -> None:
        settings = get_settings()
        self._storage = storage_settings or settings.storage
        self._charts_dir = self._storage.charts_path
        self._charts_dir.mkdir(parents=True, exist_ok=True)

    def get_chart_path(self, chart_filename: str) -> Optional[Path]:
        """
        Resolve and validate a chart file path.

        Args:
            chart_filename: The chart filename or full path.

        Returns:
            The ``Path`` if the file exists, else ``None``.
        """
        # Handle both full paths and filenames
        path = Path(chart_filename)
        if path.exists():
            return path

        # Try in charts directory
        chart_path = self._charts_dir / path.name
        return chart_path if chart_path.exists() else None

    def list_charts(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all chart files in the charts directory.

        Args:
            session_id: Optional filter (not yet implemented).

        Returns:
            List of chart metadata dicts.
        """
        charts: List[Dict[str, Any]] = []
        for f in sorted(
            self._charts_dir.glob("*.png"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            charts.append({
                "filename": f.name,
                "filepath": str(f),
                "size_bytes": f.stat().st_size,
            })
        return charts

    def cleanup_old_charts(self, keep_count: int = 100) -> int:
        """
        Delete old chart files, keeping only the most recent N.

        Args:
            keep_count: Number of most recent charts to keep.

        Returns:
            Number of charts deleted.
        """
        all_charts = sorted(
            self._charts_dir.glob("*.png"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        to_delete = all_charts[keep_count:]
        for chart in to_delete:
            chart.unlink()

        if to_delete:
            logger.info("Cleaned up %d old chart files", len(to_delete))

        return len(to_delete)

    @staticmethod
    def get_plotly_theme() -> Dict[str, Any]:
        """
        Return the Plotly dark theme configuration.

        Used by the Streamlit frontend for interactive chart rendering.

        Returns:
            Theme configuration dict.
        """
        return PLOTLY_DARK_THEME.copy()

    def get_chart_count(self) -> int:
        """Return the total number of chart files."""
        return len(list(self._charts_dir.glob("*.png")))
