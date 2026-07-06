"""
Tests for the intelligent visualization engine.

Tests chart selection, generation, explanation, and export
without requiring LLM or Ollama.
"""

from __future__ import annotations

import os
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.visualization.chart_selector import ChartSelector, ChartSpec, ChartType
from backend.visualization.chart_themes import (
    ChartThemeManager, DARK_THEME, LIGHT_THEME, build_plotly_layout,
)
from backend.visualization.chart_generator import ChartGenerator
from backend.visualization.chart_explainer import ChartExplainer
from backend.visualization.chart_export import ChartExporter
from backend.models.schemas import FileMetadata, ColumnInfo


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _build_metadata(columns: list[ColumnInfo], row_count: int = 100) -> FileMetadata:
    """Build a FileMetadata for testing."""
    return FileMetadata(
        original_name="test.csv",
        stored_path="/tmp/test.csv",
        row_count=row_count,
        col_count=len(columns),
        file_size_bytes=1024,
        memory_usage_mb=0.1,
        columns=columns,
    )


def _make_column(name: str, dtype: str, unique: int = 10, nulls: int = 0, n: int = 100) -> ColumnInfo:
    return ColumnInfo(
        name=name, dtype=dtype, non_null_count=n - nulls,
        null_count=nulls, unique_count=unique, sample_values=["a", "b"],
    )


def _sample_df() -> pd.DataFrame:
    """Create a sample DataFrame for chart generation tests."""
    np.random.seed(42)
    return pd.DataFrame({
        "category": np.random.choice(["A", "B", "C", "D", "E"], 100),
        "revenue": np.random.uniform(100, 10000, 100).round(2),
        "quantity": np.random.randint(1, 50, 100),
        "price": np.random.normal(50, 15, 100).round(2),
        "date": pd.date_range("2024-01-01", periods=100, freq="D"),
    })


# ── Chart Selector Tests ────────────────────────────────────────────────────

class TestChartSelector:
    """Tests for intelligent chart type selection."""

    def setup_method(self):
        self.selector = ChartSelector()

    def test_bar_chart_for_category_vs_numeric(self):
        meta = _build_metadata([
            _make_column("category", "object", unique=5),
            _make_column("revenue", "float64", unique=95),
        ])
        spec = self.selector.select("Show revenue by category", meta)
        assert spec.chart_type in (ChartType.BAR, ChartType.HORIZONTAL_BAR)

    def test_pie_chart_for_proportion_request(self):
        meta = _build_metadata([
            _make_column("region", "object", unique=5),
            _make_column("sales", "float64", unique=90),
        ])
        spec = self.selector.select("Show pie chart of sales by region", meta)
        assert spec.chart_type == ChartType.PIE

    def test_histogram_for_distribution(self):
        meta = _build_metadata([
            _make_column("price", "float64", unique=95),
        ])
        spec = self.selector.select("Show the distribution of prices", meta)
        assert spec.chart_type == ChartType.HISTOGRAM

    def test_scatter_for_relationship(self):
        meta = _build_metadata([
            _make_column("height", "float64", unique=90),
            _make_column("weight", "float64", unique=85),
        ])
        spec = self.selector.select("scatter plot of height vs weight", meta)
        assert spec.chart_type == ChartType.SCATTER

    def test_correlation_matrix(self):
        meta = _build_metadata([
            _make_column("a", "float64", unique=90),
            _make_column("b", "float64", unique=85),
            _make_column("c", "float64", unique=80),
        ])
        spec = self.selector.select("Show the correlation matrix", meta)
        assert spec.chart_type == ChartType.CORRELATION_MATRIX

    def test_box_plot_for_outliers(self):
        meta = _build_metadata([
            _make_column("salary", "float64", unique=90),
            _make_column("dept", "object", unique=5),
        ])
        spec = self.selector.select("Show a box plot of salary by department", meta)
        assert spec.chart_type == ChartType.BOX_PLOT

    def test_violin_plot(self):
        meta = _build_metadata([
            _make_column("score", "float64", unique=80),
            _make_column("group", "object", unique=4),
        ])
        spec = self.selector.select("Show violin plot of scores by group", meta)
        assert spec.chart_type == ChartType.VIOLIN_PLOT

    def test_line_chart_for_trend(self):
        meta = _build_metadata([
            _make_column("date", "datetime64", unique=100),
            _make_column("sales", "float64", unique=95),
        ])
        spec = self.selector.select("Show the trend of sales over time", meta)
        assert spec.chart_type == ChartType.LINE

    def test_heatmap_request(self):
        meta = _build_metadata([
            _make_column("a", "float64", unique=90),
            _make_column("b", "float64", unique=85),
        ])
        spec = self.selector.select("Show a heatmap", meta)
        assert spec.chart_type == ChartType.HEATMAP

    def test_spec_has_confidence(self):
        meta = _build_metadata([
            _make_column("cat", "object", unique=5),
            _make_column("val", "float64", unique=95),
        ])
        spec = self.selector.select("Compare values by category", meta)
        assert 0.0 <= spec.confidence <= 1.0

    def test_spec_has_reasoning(self):
        meta = _build_metadata([
            _make_column("x", "float64", unique=90),
            _make_column("y", "float64", unique=85),
        ])
        spec = self.selector.select("scatter plot", meta)
        assert len(spec.reasoning) > 0

    def test_fallback_for_no_columns(self):
        meta = _build_metadata([])
        spec = self.selector.select("Show something", meta)
        assert spec.chart_type is not None
        assert spec.confidence <= 0.5


# ── Chart Theme Tests ────────────────────────────────────────────────────────

class TestChartThemes:
    """Tests for chart theming system."""

    def test_dark_theme_defaults(self):
        assert DARK_THEME.name == "dark"
        assert DARK_THEME.dpi == 300
        assert len(DARK_THEME.palette) > 0

    def test_light_theme_exists(self):
        assert LIGHT_THEME.name == "light"
        assert LIGHT_THEME.bg_color == "#FFFFFF"

    def test_plotly_layout_generation(self):
        layout = build_plotly_layout(DARK_THEME)
        assert layout["template"] == "plotly_dark"
        assert "colorway" in layout
        assert len(layout["colorway"]) > 0

    def test_theme_manager_create_figure(self):
        import matplotlib
        matplotlib.use("Agg")
        mgr = ChartThemeManager()
        fig, ax = mgr.create_figure()
        assert fig is not None
        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_theme_manager_palette(self):
        mgr = ChartThemeManager()
        colors = mgr.get_palette(5)
        assert len(colors) == 5
        # More than available should cycle
        colors = mgr.get_palette(20)
        assert len(colors) == 20

    def test_theme_switch(self):
        mgr = ChartThemeManager()
        assert mgr.theme.name == "dark"
        mgr.set_theme("light")
        assert mgr.theme.name == "light"
        mgr.set_theme("dark")
        assert mgr.theme.name == "dark"


# ── Chart Generator Tests ───────────────────────────────────────────────────

class TestChartGenerator:
    """Tests for chart rendering engine."""

    def setup_method(self):
        self.generator = ChartGenerator()
        self.df = _sample_df()
        self.tmpdir = tempfile.mkdtemp()

    def _gen(self, chart_type: ChartType, **kwargs) -> dict:
        spec = ChartSpec(chart_type=chart_type, title="Test Chart", **kwargs)
        return self.generator.generate(self.df, spec, self.tmpdir)

    def test_bar_chart_generation(self):
        result = self._gen(ChartType.BAR, x_column="category", y_column="revenue")
        assert Path(result["chart_path"]).exists()
        assert result["chart_type"] == "bar"

    def test_pie_chart_generation(self):
        result = self._gen(ChartType.PIE, x_column="category", y_column="revenue", limit=5)
        assert Path(result["chart_path"]).exists()

    def test_histogram_generation(self):
        result = self._gen(ChartType.HISTOGRAM, x_column="price")
        assert Path(result["chart_path"]).exists()

    def test_scatter_generation(self):
        result = self._gen(ChartType.SCATTER, x_column="revenue", y_column="quantity")
        assert Path(result["chart_path"]).exists()

    def test_correlation_generation(self):
        result = self._gen(ChartType.CORRELATION_MATRIX)
        assert Path(result["chart_path"]).exists()

    def test_box_plot_generation(self):
        result = self._gen(ChartType.BOX_PLOT, x_column="category", y_column="revenue")
        assert Path(result["chart_path"]).exists()

    def test_violin_plot_generation(self):
        result = self._gen(ChartType.VIOLIN_PLOT, x_column="category", y_column="revenue")
        assert Path(result["chart_path"]).exists()

    def test_line_chart_generation(self):
        result = self._gen(ChartType.LINE, x_column="date", y_column="revenue")
        assert Path(result["chart_path"]).exists()

    def test_horizontal_bar_generation(self):
        result = self._gen(ChartType.HORIZONTAL_BAR, x_column="category", y_column="revenue")
        assert Path(result["chart_path"]).exists()

    def test_plotly_json_generated(self):
        result = self._gen(ChartType.BAR, x_column="category", y_column="revenue")
        assert "plotly_json" in result
        assert "data" in result["plotly_json"]
        assert "layout" in result["plotly_json"]

    def test_heatmap_generation(self):
        result = self._gen(ChartType.HEATMAP)
        assert Path(result["chart_path"]).exists()


# ── Chart Explainer Tests ────────────────────────────────────────────────────

class TestChartExplainer:
    """Tests for automatic chart explanations."""

    def setup_method(self):
        self.explainer = ChartExplainer()
        self.df = _sample_df()

    def test_bar_explanation(self):
        spec = ChartSpec(chart_type=ChartType.BAR, x_column="category",
                         y_column="revenue", reasoning="Test reasoning")
        explanation = self.explainer.explain(self.df, spec)
        assert len(explanation) > 20
        assert "bar chart" in explanation.lower() or "📊" in explanation

    def test_histogram_explanation(self):
        spec = ChartSpec(chart_type=ChartType.HISTOGRAM, x_column="price")
        explanation = self.explainer.explain(self.df, spec)
        assert "mean" in explanation.lower() or "distribution" in explanation.lower()

    def test_scatter_explanation(self):
        spec = ChartSpec(chart_type=ChartType.SCATTER, x_column="revenue", y_column="quantity")
        explanation = self.explainer.explain(self.df, spec)
        assert "correlation" in explanation.lower() or "relationship" in explanation.lower()

    def test_correlation_explanation(self):
        spec = ChartSpec(chart_type=ChartType.CORRELATION_MATRIX)
        explanation = self.explainer.explain(self.df, spec)
        assert "correlation" in explanation.lower()

    def test_box_explanation(self):
        spec = ChartSpec(chart_type=ChartType.BOX_PLOT, y_column="revenue")
        explanation = self.explainer.explain(self.df, spec)
        assert "median" in explanation.lower() or "distribution" in explanation.lower()

    def test_explanation_includes_reasoning(self):
        spec = ChartSpec(chart_type=ChartType.BAR, x_column="category",
                         y_column="revenue", reasoning="Selected bar for comparison")
        explanation = self.explainer.explain(self.df, spec)
        assert "Why this chart?" in explanation


# ── Chart Export Tests ───────────────────────────────────────────────────────

class TestChartExporter:
    """Tests for chart export functionality."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.exporter = ChartExporter(self.tmpdir)

    def test_plotly_html_export(self):
        plotly_data = {
            "data": [{"type": "bar", "x": ["A", "B"], "y": [1, 2]}],
            "layout": {"title": {"text": "Test"}},
        }
        path = self.exporter.export_plotly_html(plotly_data, "test_chart.html")
        assert Path(path).exists()
        content = Path(path).read_text()
        assert "Plotly" in content
        assert "DataWhisperer" in content

    def test_available_formats(self):
        formats = self.exporter.get_available_formats(has_plotly=True)
        assert "png" in formats
        assert "html" in formats

    def test_formats_without_plotly(self):
        formats = self.exporter.get_available_formats(has_plotly=False)
        assert "png" in formats
        assert "html" not in formats


# ── Sandbox Whitelist Tests ──────────────────────────────────────────────────

class TestSandboxWhitelist:
    """Verify visualization libraries are properly whitelisted."""

    def test_seaborn_whitelisted(self):
        from backend.sandbox.restrictions import ALLOWED_MODULES, ALLOWED_MODULE_ROOTS
        assert "seaborn" in ALLOWED_MODULES
        assert "sns" in ALLOWED_MODULES
        assert "seaborn" in ALLOWED_MODULE_ROOTS

    def test_plotly_whitelisted(self):
        from backend.sandbox.restrictions import ALLOWED_MODULES, ALLOWED_MODULE_ROOTS
        assert "plotly" in ALLOWED_MODULES
        assert "plotly.express" in ALLOWED_MODULES
        assert "plotly.graph_objects" in ALLOWED_MODULES
        assert "plotly" in ALLOWED_MODULE_ROOTS

    def test_scipy_whitelisted(self):
        from backend.sandbox.restrictions import ALLOWED_MODULES, ALLOWED_MODULE_ROOTS
        assert "scipy" in ALLOWED_MODULES
        assert "scipy.stats" in ALLOWED_MODULES
        assert "scipy" in ALLOWED_MODULE_ROOTS

    def test_dangerous_modules_still_blocked(self):
        from backend.sandbox.restrictions import BLOCKED_MODULES
        assert "os" in BLOCKED_MODULES
        assert "subprocess" in BLOCKED_MODULES
        assert "socket" in BLOCKED_MODULES


# ── Integration Tests ────────────────────────────────────────────────────────

class TestVisualizationServiceIntegration:
    """Integration tests for the full visualization pipeline."""

    def test_service_init(self):
        """Verify VisualizationService initializes without errors."""
        from backend.services.visualization_service import VisualizationService
        svc = VisualizationService()
        assert svc.selector is not None
        assert svc.generator is not None
        assert svc.theme_manager is not None

    def test_plotly_theme_backward_compat(self):
        """Verify backward-compatible Plotly theme dict."""
        from backend.services.visualization_service import VisualizationService
        theme = VisualizationService.get_plotly_theme()
        assert "template" in theme
        assert "colorway" in theme
        assert theme["template"] == "plotly_dark"

    def test_auto_generate_chart(self):
        """Full pipeline: select → generate → explain."""
        from backend.services.visualization_service import VisualizationService
        svc = VisualizationService()
        df = _sample_df()
        meta = _build_metadata([
            _make_column("category", "object", unique=5),
            _make_column("revenue", "float64", unique=95),
            _make_column("quantity", "int64", unique=45),
            _make_column("price", "float64", unique=90),
            _make_column("date", "datetime64", unique=100),
        ])
        result = svc.auto_generate_chart(df, "Show revenue by category", meta)
        assert "chart_path" in result
        assert "explanation" in result
        assert "chart_type" in result
        assert Path(result["chart_path"]).exists()
        assert len(result["explanation"]) > 10
