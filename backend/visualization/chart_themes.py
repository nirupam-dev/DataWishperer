"""
Chart Themes — Dark theme system and publication-quality figure styling.

Provides a centralized theming engine that produces consistent, beautiful
charts across all chart types and rendering backends (matplotlib + Plotly).

Design Philosophy:
    - Every chart should look publication-ready out of the box
    - Dark theme is the default (matches modern data tools aesthetic)
    - Colorblind-safe palettes are used throughout
    - Typography uses Inter/system fonts for crisp rendering
    - High DPI (300) for print-quality exports

Theme Hierarchy:
    1. Base theme (dark/light mode colors)
    2. Chart-specific overrides (e.g., heatmap colormaps)
    3. Publication-quality post-processing (tight layout, font sizes)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


# ── Color Palettes ───────────────────────────────────────────────────────────

# Primary palette: curated for dark backgrounds, colorblind-safe
PALETTE_PRIMARY: List[str] = [
    "#6C5CE7",  # Soft purple
    "#00CEC9",  # Teal
    "#FD79A8",  # Pink
    "#FDCB6E",  # Gold
    "#55EFC4",  # Mint
    "#A29BFE",  # Lavender
    "#FF7675",  # Coral
    "#74B9FF",  # Sky blue
    "#E17055",  # Burnt orange
    "#81ECEC",  # Light teal
    "#FAB1A0",  # Peach
    "#DFE6E9",  # Silver
]

# Sequential palette for heatmaps and gradients
PALETTE_SEQUENTIAL: str = "viridis"

# Diverging palette for correlation matrices
PALETTE_DIVERGING: str = "RdBu_r"

# Categorical palette — expanded for more categories
PALETTE_EXTENDED: List[str] = PALETTE_PRIMARY + [
    "#636E72",  # Gray
    "#B2BEC3",  # Light gray
    "#2D3436",  # Charcoal
    "#D63031",  # Red
    "#0984E3",  # Blue
    "#00B894",  # Green
    "#E84393",  # Hot pink
    "#FFEAA7",  # Light gold
]


# ── Theme Dataclass ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ChartTheme:
    """
    Complete theme configuration for chart rendering.

    Attributes:
        name: Theme identifier.
        bg_color: Figure background color (RGBA).
        plot_bg_color: Axes/plot area background color.
        text_color: Primary text color.
        grid_color: Grid line color (with alpha).
        accent_color: Highlight/accent color.
        font_family: Primary font stack.
        title_size: Title font size.
        label_size: Axis label font size.
        tick_size: Tick label font size.
        legend_size: Legend text font size.
        annotation_size: Annotation text font size.
        line_width: Default line width.
        dpi: Render resolution.
        palette: Color cycle for data series.
        figsize_default: Default figure size (width, height).
        figsize_wide: Wide figure size (for heatmaps, correlation).
        figsize_tall: Tall figure size (for horizontal bars).
    """

    name: str = "dark"
    bg_color: str = "#1a1a2e"
    plot_bg_color: str = "#16213e"
    text_color: str = "#F0F0F5"
    grid_color: str = "rgba(240, 240, 245, 0.08)"
    grid_color_mpl: str = "#2a2a4a"
    accent_color: str = "#6C5CE7"
    font_family: str = "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
    title_size: int = 18
    label_size: int = 13
    tick_size: int = 11
    legend_size: int = 11
    annotation_size: int = 10
    line_width: float = 2.0
    dpi: int = 300
    palette: List[str] = field(default_factory=lambda: PALETTE_PRIMARY.copy())
    figsize_default: Tuple[int, int] = (12, 7)
    figsize_wide: Tuple[int, int] = (14, 8)
    figsize_tall: Tuple[int, int] = (10, 10)


# ── Pre-built Themes ────────────────────────────────────────────────────────

DARK_THEME = ChartTheme(
    name="dark",
    bg_color="#1a1a2e",
    plot_bg_color="#16213e",
    text_color="#F0F0F5",
    grid_color="rgba(240, 240, 245, 0.08)",
    grid_color_mpl="#2a2a4a",
    accent_color="#6C5CE7",
)

LIGHT_THEME = ChartTheme(
    name="light",
    bg_color="#FFFFFF",
    plot_bg_color="#F8F9FA",
    text_color="#2D3436",
    grid_color="rgba(45, 52, 54, 0.1)",
    grid_color_mpl="#E0E0E0",
    accent_color="#6C5CE7",
)


# ── Plotly Theme Configuration ───────────────────────────────────────────────

def build_plotly_layout(theme: Optional[ChartTheme] = None) -> Dict[str, Any]:
    """
    Build a Plotly layout configuration dict from a ChartTheme.

    This layout is applied to every Plotly figure for consistent styling.
    Uses transparent backgrounds so the chart integrates with the UI.

    Args:
        theme: The chart theme. Defaults to DARK_THEME.

    Returns:
        Plotly layout dict ready for fig.update_layout().
    """
    t = theme or DARK_THEME
    return {
        "template": "plotly_dark" if t.name == "dark" else "plotly_white",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {
            "family": t.font_family,
            "color": t.text_color,
            "size": t.tick_size,
        },
        "title": {
            "font": {"size": t.title_size, "color": t.text_color},
            "x": 0.5,
            "xanchor": "center",
        },
        "xaxis": {
            "gridcolor": t.grid_color,
            "zerolinecolor": t.grid_color,
            "title": {"font": {"size": t.label_size}},
            "tickfont": {"size": t.tick_size},
        },
        "yaxis": {
            "gridcolor": t.grid_color,
            "zerolinecolor": t.grid_color,
            "title": {"font": {"size": t.label_size}},
            "tickfont": {"size": t.tick_size},
        },
        "colorway": t.palette,
        "legend": {
            "font": {"size": t.legend_size},
            "bgcolor": "rgba(0,0,0,0)",
        },
        "hoverlabel": {
            "bgcolor": t.plot_bg_color,
            "font_size": t.annotation_size,
            "font_color": t.text_color,
        },
        "margin": {"l": 60, "r": 30, "t": 60, "b": 60},
    }


# ── Chart Theme Manager ─────────────────────────────────────────────────────

class ChartThemeManager:
    """
    Manages chart theming across matplotlib and Plotly.

    Provides methods to apply themes to matplotlib figures and
    generate Plotly layout configurations. Thread-safe — uses
    per-figure styling rather than global rcParams mutation.

    Usage:
        manager = ChartThemeManager()  # Dark theme by default
        fig, ax = manager.create_figure()
        # ... plot data ...
        manager.finalize_figure(fig, title="My Chart")

        plotly_layout = manager.get_plotly_layout()
    """

    def __init__(self, theme: Optional[ChartTheme] = None) -> None:
        self._theme = theme or DARK_THEME

    @property
    def theme(self) -> ChartTheme:
        """Return the active theme."""
        return self._theme

    def set_theme(self, theme_name: str) -> None:
        """
        Switch between pre-built themes.

        Args:
            theme_name: 'dark' or 'light'.
        """
        if theme_name == "light":
            self._theme = LIGHT_THEME
        else:
            self._theme = DARK_THEME

    def create_figure(
        self,
        figsize: Optional[Tuple[int, int]] = None,
        subplot_kw: Optional[Dict[str, Any]] = None,
        nrows: int = 1,
        ncols: int = 1,
    ) -> Tuple[matplotlib.figure.Figure, Any]:
        """
        Create a themed matplotlib figure.

        Returns a figure with dark background, styled axes, and the
        correct font/color configuration. Does NOT mutate global rcParams.

        Args:
            figsize: Override figure size. Defaults to theme default.
            subplot_kw: Additional subplot keyword arguments.
            nrows: Number of subplot rows.
            ncols: Number of subplot columns.

        Returns:
            Tuple of (Figure, Axes or array of Axes).
        """
        t = self._theme
        size = figsize or t.figsize_default

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=size,
            subplot_kw=subplot_kw,
        )

        # Apply theme to figure
        fig.patch.set_facecolor(t.bg_color)

        # Apply theme to each axis
        ax_list = [axes] if not hasattr(axes, "__iter__") else np.array(axes).flatten()
        for ax in ax_list:
            ax.set_facecolor(t.plot_bg_color)
            ax.tick_params(colors=t.text_color, labelsize=t.tick_size)
            ax.xaxis.label.set_color(t.text_color)
            ax.yaxis.label.set_color(t.text_color)
            ax.title.set_color(t.text_color)
            for spine in ax.spines.values():
                spine.set_color(t.grid_color_mpl)
                spine.set_linewidth(0.5)
            ax.grid(True, alpha=0.15, color=t.grid_color_mpl, linestyle="--")

        return fig, axes

    def finalize_figure(
        self,
        fig: matplotlib.figure.Figure,
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        source_text: Optional[str] = None,
    ) -> None:
        """
        Apply final publication-quality touches to a figure.

        Adds title, subtitle, source annotation, and tight layout.
        Call this AFTER all plotting is done, BEFORE savefig.

        Args:
            fig: The matplotlib figure.
            title: Main chart title.
            subtitle: Optional subtitle below the title.
            source_text: Optional source/footer annotation.
        """
        t = self._theme

        if title:
            fig.suptitle(
                title,
                fontsize=t.title_size,
                fontweight="bold",
                color=t.text_color,
                y=0.98,
            )

        if subtitle:
            fig.text(
                0.5, 0.94,
                subtitle,
                ha="center",
                fontsize=t.annotation_size,
                color=t.text_color,
                alpha=0.7,
                style="italic",
            )

        if source_text:
            fig.text(
                0.99, 0.01,
                source_text,
                ha="right",
                fontsize=8,
                color=t.text_color,
                alpha=0.4,
            )

        fig.tight_layout(rect=[0, 0.02, 1, 0.95] if title else [0, 0, 1, 1])

    def get_palette(self, n: int) -> List[str]:
        """
        Get N colors from the theme palette, cycling if needed.

        Args:
            n: Number of colors needed.

        Returns:
            List of hex color strings.
        """
        palette = self._theme.palette
        if n <= len(palette):
            return palette[:n]
        # Cycle through the palette
        return [palette[i % len(palette)] for i in range(n)]

    def get_colormap(self, n: int) -> List[str]:
        """
        Get N colors from a matplotlib colormap for gradients.

        Args:
            n: Number of colors.

        Returns:
            List of hex color strings from viridis-like colormap.
        """
        cmap = plt.colormaps["plasma"]
        colors = [mcolors.to_hex(cmap(i / max(n - 1, 1))) for i in range(n)]
        return colors

    def get_plotly_layout(self) -> Dict[str, Any]:
        """Return the Plotly layout configuration for this theme."""
        return build_plotly_layout(self._theme)

    def get_plotly_colorscale(self, chart_type: str = "default") -> str:
        """
        Get the appropriate Plotly colorscale for a chart type.

        Args:
            chart_type: The chart type ('heatmap', 'correlation', 'default').

        Returns:
            Plotly colorscale name.
        """
        if chart_type in ("correlation", "correlation_matrix"):
            return "RdBu_r"
        elif chart_type == "heatmap":
            return "Viridis"
        return "Plasma"

    def save_figure(
        self,
        fig: matplotlib.figure.Figure,
        path: str,
        dpi: Optional[int] = None,
        transparent: bool = False,
    ) -> str:
        """
        Save a figure with publication-quality settings.

        Args:
            fig: The matplotlib figure.
            path: Output file path.
            dpi: Override DPI. Defaults to theme DPI (300).
            transparent: If True, save with transparent background.

        Returns:
            The output file path.
        """
        fig.savefig(
            path,
            dpi=dpi or self._theme.dpi,
            bbox_inches="tight",
            facecolor=fig.get_facecolor() if not transparent else "none",
            edgecolor="none",
            pad_inches=0.3,
        )
        plt.close(fig)
        return path
