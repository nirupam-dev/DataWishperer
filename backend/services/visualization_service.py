"""
Visualization service — Intelligent chart management and rendering.

Integrates the visualization engine (selector, generator, explainer, exporter)
with the agent pipeline. Handles chart file management, Plotly figure creation
for interactive charts in Streamlit, and chart theming.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.core.config import StorageSettings, get_settings
from backend.core.logging_config import get_logger
from backend.visualization.chart_selector import ChartSelector, ChartSpec, ChartType
from backend.visualization.chart_generator import ChartGenerator
from backend.visualization.chart_themes import ChartThemeManager, build_plotly_layout, DARK_THEME
from backend.visualization.chart_explainer import ChartExplainer
from backend.visualization.chart_export import ChartExporter

logger = get_logger(__name__)

# Plotly dark theme for backward compatibility
PLOTLY_DARK_THEME: Dict[str, Any] = build_plotly_layout(DARK_THEME)


class VisualizationService:
    """
    Manages intelligent chart selection, generation, and export.

    This service integrates the full visualization pipeline:
        1. ChartSelector: Determines optimal chart type from question + data
        2. ChartGenerator: Renders matplotlib PNG + Plotly interactive JSON
        3. ChartExplainer: Generates automatic chart explanations
        4. ChartExporter: Multi-format download (PNG, HTML)

    It also manages chart file lifecycle (listing, cleanup) and provides
    Plotly theme configuration for the Streamlit frontend.

    Args:
        storage_settings: Storage configuration.
    """

    def __init__(self, storage_settings: Optional[StorageSettings] = None) -> None:
        settings = get_settings()
        self._storage = storage_settings or settings.storage
        self._charts_dir = self._storage.charts_path
        self._charts_dir.mkdir(parents=True, exist_ok=True)

        # Initialize visualization engine components
        self._theme_manager = ChartThemeManager()
        self._selector = ChartSelector()
        self._generator = ChartGenerator(self._theme_manager)
        self._explainer = ChartExplainer()
        self._exporter = ChartExporter(str(self._storage.export_path))

    @property
    def selector(self) -> ChartSelector:
        """Expose the chart selector for direct use."""
        return self._selector

    @property
    def generator(self) -> ChartGenerator:
        """Expose the chart generator."""
        return self._generator

    @property
    def theme_manager(self) -> ChartThemeManager:
        """Expose the theme manager."""
        return self._theme_manager

    # ── Intelligent Chart Pipeline ───────────────────────────────────────

    def auto_generate_chart(
        self,
        df: pd.DataFrame,
        question: str,
        file_metadata: Any,
    ) -> Dict[str, Any]:
        """
        Full pipeline: select chart type → generate → explain.

        This is the primary entry point for AI-driven chart generation.

        Args:
            df: The source DataFrame.
            question: The user's question.
            file_metadata: FileMetadata for column analysis.

        Returns:
            Dict with: chart_path, chart_type, plotly_json, explanation,
            title, reasoning, confidence.
        """
        # Step 1: Select optimal chart type
        spec = self._selector.select(question=question, file_metadata=file_metadata)
        logger.info(
            "Chart selected: %s (confidence=%.2f) — %s",
            spec.chart_type.value,
            spec.confidence,
            spec.reasoning[:80],
        )

        # Step 2: Generate the chart
        result = self._generator.generate(
            df=df, spec=spec, output_dir=str(self._charts_dir)
        )

        # Step 3: Generate explanation
        explanation = self._explainer.explain(df=df, spec=spec)
        result["explanation"] = explanation

        return result

    def generate_chart_from_spec(
        self,
        df: pd.DataFrame,
        spec: ChartSpec,
    ) -> Dict[str, Any]:
        """
        Generate a chart from an explicit ChartSpec.

        Used when the caller wants to override automatic selection.

        Args:
            df: The source DataFrame.
            spec: An explicit chart specification.

        Returns:
            Dict with chart_path, plotly_json, explanation, etc.
        """
        result = self._generator.generate(
            df=df, spec=spec, output_dir=str(self._charts_dir)
        )
        result["explanation"] = self._explainer.explain(df=df, spec=spec)
        return result

    # ── Chart File Management ────────────────────────────────────────────

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

    # ── Export Support ───────────────────────────────────────────────────

    def export_chart(
        self,
        chart_path: str,
        format: str = "png",
        plotly_json: Optional[Dict] = None,
    ) -> Optional[str]:
        """
        Export a chart in the requested format.

        Args:
            chart_path: Path to the source PNG chart.
            format: Export format ('png', 'html').
            plotly_json: Plotly JSON data for HTML export.

        Returns:
            Path to the exported file, or None on failure.
        """
        if format == "png":
            return self._exporter.export_png(chart_path)
        elif format == "html" and plotly_json:
            return self._exporter.export_plotly_html(plotly_json)
        return None

    # ── Theme Access ─────────────────────────────────────────────────────

    @staticmethod
    def get_plotly_theme() -> Dict[str, Any]:
        """
        Return the Plotly dark theme configuration.

        Used by the Streamlit frontend for interactive chart rendering.

        Returns:
            Theme configuration dict.
        """
        return build_plotly_layout(DARK_THEME)

    def get_chart_count(self) -> int:
        """Return the total number of chart files."""
        return len(list(self._charts_dir.glob("*.png")))
