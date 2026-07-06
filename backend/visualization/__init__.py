"""
Visualization Engine — Intelligent chart selection, rendering, and export.

This module provides the complete visualization pipeline:
    - ChartSelector: Rule-based intelligent chart type selection
    - ChartGenerator: Unified rendering engine for all chart types
    - ChartThemeManager: Dark theme and publication-quality styling
    - ChartExplainer: AI-powered chart explanations
    - ChartExporter: Multi-format chart download support
"""

from backend.visualization.chart_selector import ChartSelector, ChartSpec, ChartType
from backend.visualization.chart_themes import ChartThemeManager
from backend.visualization.chart_generator import ChartGenerator
from backend.visualization.chart_explainer import ChartExplainer
from backend.visualization.chart_export import ChartExporter

__all__ = [
    "ChartSelector",
    "ChartSpec",
    "ChartType",
    "ChartThemeManager",
    "ChartGenerator",
    "ChartExplainer",
    "ChartExporter",
]
