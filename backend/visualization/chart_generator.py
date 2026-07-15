"""
Chart Generator — Unified rendering engine for all chart types.

Generates both static matplotlib (PNG) and Plotly interactive (JSON)
charts from a ChartSpec produced by ChartSelector.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backend.core.logging_config import get_logger
from backend.visualization.chart_selector import ChartSpec, ChartType
from backend.visualization.chart_themes import ChartThemeManager, PALETTE_PRIMARY

logger = get_logger(__name__)


class ChartGenerator:
    """
    Renders charts from ChartSpec objects.

    Produces publication-quality matplotlib PNGs and optional Plotly JSON
    for interactive rendering in Streamlit.
    """

    def __init__(self, theme_manager: Optional[ChartThemeManager] = None) -> None:
        self._theme = theme_manager or ChartThemeManager()

    def generate(
        self,
        df: pd.DataFrame,
        spec: ChartSpec,
        output_dir: str,
    ) -> Dict[str, Any]:
        """
        Generate a chart from data and specification.

        Returns dict with keys: chart_path, plotly_json, chart_type, title.
        """
        chart_id = uuid4().hex[:12]
        png_path = str(Path(output_dir) / f"chart_{chart_id}.png")

        dispatch = {
            ChartType.BAR: self._render_bar,
            ChartType.HORIZONTAL_BAR: self._render_bar,
            ChartType.PIE: self._render_pie,
            ChartType.HISTOGRAM: self._render_histogram,
            ChartType.SCATTER: self._render_scatter,
            ChartType.HEATMAP: self._render_heatmap,
            ChartType.CORRELATION_MATRIX: self._render_correlation,
            ChartType.BOX_PLOT: self._render_box,
            ChartType.VIOLIN_PLOT: self._render_violin,
            ChartType.LINE: self._render_line,
            ChartType.AREA: self._render_line,
        }

        render_fn = dispatch.get(spec.chart_type, self._render_bar)

        try:
            fig, plotly_data = render_fn(df, spec)
            self._theme.finalize_figure(fig, title=spec.title)
            self._theme.save_figure(fig, png_path)

            result = {
                "chart_path": png_path,
                "chart_type": spec.chart_type.value,
                "title": spec.title,
                "reasoning": spec.reasoning,
                "confidence": spec.confidence,
            }
            if plotly_data:
                result["plotly_json"] = plotly_data

            logger.info("Chart generated: %s at %s", spec.chart_type.value, png_path)
            return result

        except Exception as e:
            logger.error("Chart generation failed: %s", e)
            plt.close("all")
            raise

    # ── Bar Chart ────────────────────────────────────────────────────────

    def _render_bar(
        self, df: pd.DataFrame, spec: ChartSpec
    ) -> Tuple[matplotlib.figure.Figure, Optional[Dict]]:
        x_col, y_col = spec.x_column, spec.y_column

        if x_col and y_col and x_col in df.columns and y_col in df.columns:
            agg_map = {"mean": "mean", "sum": "sum", "count": "count",
                       "max": "max", "min": "min"}
            agg_fn = agg_map.get(spec.aggregation, "mean")
            data = df.groupby(x_col)[y_col].agg(agg_fn).sort_values(ascending=False)
        elif x_col and x_col in df.columns:
            data = df[x_col].value_counts()
        else:
            data = df.iloc[:, 0].value_counts()

        data = data.head(spec.limit)
        colors = self._theme.get_palette(len(data))
        fig, ax = self._theme.create_figure()

        if spec.chart_type == ChartType.HORIZONTAL_BAR:
            # Glow layer
            ax.barh(range(len(data)), data.values, height=0.7,
                    color=[c + "18" for c in colors], zorder=1)
            # Main bars
            bars = ax.barh(range(len(data)), data.values, height=0.55,
                           color=colors, edgecolor="none", alpha=0.92, zorder=3)
            ax.set_yticks(range(len(data)))
            ax.set_yticklabels([str(l)[:25] for l in data.index], fontsize=10)
            ax.set_xlabel(spec.y_label or y_col or "Value")
            # Value labels
            max_val = max(data.values) if len(data) > 0 else 1
            for bar, val in zip(bars, data.values):
                label = f"${val:,.0f}" if val >= 100 else f"{val:,.1f}"
                ax.text(bar.get_width() + max_val * 0.02,
                        bar.get_y() + bar.get_height() / 2,
                        label, va="center", fontsize=10, fontweight="bold",
                        color="#FFFFFF")
        else:
            # Glow layer behind bars
            ax.bar(range(len(data)), data.values, width=0.7,
                   color=[c + "18" for c in colors], zorder=1)
            # Main bars
            bars = ax.bar(range(len(data)), data.values, color=colors, width=0.55,
                          edgecolor="none", alpha=0.92, zorder=3)
            ax.set_xticks(range(len(data)))
            ax.set_xticklabels([str(l)[:15] for l in data.index],
                               rotation=45, ha="right", fontsize=10)
            ax.set_ylabel(spec.y_label or y_col or "Value")
            # Bold value labels on top
            if len(data) <= 15:
                max_val = max(data.values) if len(data) > 0 else 1
                for bar, val in zip(bars, data.values):
                    label = f"${val:,.0f}" if val >= 100 else f"{val:,.1f}"
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + max_val * 0.02,
                            label, ha="center", va="bottom", fontsize=10,
                            fontweight="bold", color="#FFFFFF")

        plotly = self._bar_to_plotly(data, spec)
        return fig, plotly

    def _bar_to_plotly(self, data: pd.Series, spec: ChartSpec) -> Dict:
        layout = self._theme.get_plotly_layout()
        layout["title"]["text"] = spec.title
        orient = "h" if spec.chart_type == ChartType.HORIZONTAL_BAR else "v"
        trace = {
            "type": "bar",
            "x": list(data.values) if orient == "h" else [str(i) for i in data.index],
            "y": [str(i) for i in data.index] if orient == "h" else list(data.values),
            "orientation": orient,
            "marker": {"color": self._theme.get_colormap(len(data))},
        }
        return {"data": [trace], "layout": layout}

    # ── Pie / Donut Chart ──────────────────────────────────────────────────

    def _render_pie(
        self, df: pd.DataFrame, spec: ChartSpec
    ) -> Tuple[matplotlib.figure.Figure, Optional[Dict]]:
        x_col, y_col = spec.x_column, spec.y_column

        if x_col and y_col and x_col in df.columns and y_col in df.columns:
            data = df.groupby(x_col)[y_col].sum().nlargest(spec.limit)
        elif x_col and x_col in df.columns:
            data = df[x_col].value_counts().head(spec.limit)
        else:
            data = df.iloc[:, 0].value_counts().head(spec.limit)

        colors = self._theme.get_palette(len(data))
        fig, ax = self._theme.create_figure()

        # Premium donut chart
        wedges, texts, autotexts = ax.pie(
            data.values, labels=[str(l)[:20] for l in data.index],
            colors=colors, autopct="%1.0f%%", startangle=90,
            counterclock=False, pctdistance=0.75,
            wedgeprops={"width": 0.4, "edgecolor": self._theme.theme.bg_color,
                        "linewidth": 2.5},
            textprops={"color": self._theme.theme.text_color, "fontsize": 11},
        )
        for at in autotexts:
            at.set_fontsize(10)
            at.set_color("white")
            at.set_fontweight("bold")

        # Center text with total
        total = data.values.sum()
        center_label = f"${total:,.0f}" if total >= 100 else f"{total:,.1f}"
        ax.text(0, 0, f"Total\n{center_label}", ha="center", va="center",
                fontsize=14, fontweight="bold", color="#FFFFFF")

        plotly = {
            "data": [{"type": "pie", "labels": [str(i) for i in data.index],
                      "values": list(data.values),
                      "marker": {"colors": colors},
                      "textinfo": "percent+label", "hole": 0.45}],
            "layout": {**self._theme.get_plotly_layout(), "title": {"text": spec.title}},
        }
        return fig, plotly

    # ── Histogram ────────────────────────────────────────────────────────

    def _render_histogram(
        self, df: pd.DataFrame, spec: ChartSpec
    ) -> Tuple[matplotlib.figure.Figure, Optional[Dict]]:
        col = spec.x_column
        if col and col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").dropna()
        else:
            num_cols = df.select_dtypes(include="number").columns
            col = num_cols[0] if len(num_cols) > 0 else df.columns[0]
            values = pd.to_numeric(df[col], errors="coerce").dropna()

        n_bins = spec.extra_params.get("bins", 30)
        fig, ax = self._theme.create_figure()
        n, bins, patches = ax.hist(values, bins=n_bins, color=PALETTE_PRIMARY[0],
                                    edgecolor=self._theme.theme.bg_color,
                                    linewidth=0.8, alpha=0.88, zorder=3)

        # Gradient coloring based on frequency
        cm = plt.colormaps["plasma"]
        norm = plt.Normalize(n.min(), n.max())
        for count, p in zip(n, patches):
            p.set_facecolor(cm(norm(count)))
            p.set_alpha(0.88)

        ax.set_xlabel(col, fontsize=self._theme.theme.label_size)
        ax.set_ylabel("Frequency", fontsize=self._theme.theme.label_size)

        # Statistical annotation lines
        mean_val, median_val = values.mean(), values.median()
        ax.axvline(mean_val, color="#F72585", linestyle="--", linewidth=2,
                   alpha=0.9, label=f"Mean: {mean_val:,.2f}", zorder=4)
        ax.axvline(median_val, color="#06D6A0", linestyle="--", linewidth=2,
                   alpha=0.9, label=f"Median: {median_val:,.2f}", zorder=4)
        ax.legend(fontsize=10, facecolor="#1A1A2E", edgecolor="none",
                  labelcolor=self._theme.theme.text_color, framealpha=0.85)

        plotly = {
            "data": [{"type": "histogram", "x": list(values), "nbinsx": n_bins,
                      "marker": {"color": PALETTE_PRIMARY[0], "line": {"width": 0.5, "color": "white"}}}],
            "layout": {**self._theme.get_plotly_layout(),
                       "title": {"text": spec.title},
                       "xaxis": {"title": {"text": col}},
                       "yaxis": {"title": {"text": "Frequency"}}},
        }
        return fig, plotly

    # ── Scatter Plot ─────────────────────────────────────────────────────

    def _render_scatter(
        self, df: pd.DataFrame, spec: ChartSpec
    ) -> Tuple[matplotlib.figure.Figure, Optional[Dict]]:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        x_col = spec.x_column if spec.x_column in df.columns else (num_cols[0] if num_cols else df.columns[0])
        y_col = spec.y_column if spec.y_column and spec.y_column in df.columns else (num_cols[1] if len(num_cols) > 1 else num_cols[0])

        x = pd.to_numeric(df[x_col], errors="coerce")
        y = pd.to_numeric(df[y_col], errors="coerce")
        mask = x.notna() & y.notna()
        x, y = x[mask], y[mask]

        fig, ax = self._theme.create_figure()
        alpha = spec.extra_params.get("alpha", 0.6)

        if spec.color_column and spec.color_column in df.columns:
            groups = df.loc[mask, spec.color_column]
            for i, (name, grp) in enumerate(df.loc[mask].groupby(spec.color_column)):
                c = PALETTE_PRIMARY[i % len(PALETTE_PRIMARY)]
                ax.scatter(x[grp.index], y[grp.index], c=c, alpha=alpha, s=40,
                           label=str(name)[:20], edgecolors="white", linewidth=0.3)
            ax.legend(fontsize=9, facecolor=self._theme.theme.plot_bg_color, edgecolor="none",
                      labelcolor=self._theme.theme.text_color)
        else:
            ax.scatter(x, y, c=PALETTE_PRIMARY[0], alpha=alpha, s=40,
                       edgecolors="white", linewidth=0.3)

        # Trendline
        if spec.extra_params.get("show_trendline") and len(x) > 2:
            try:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, p(x_line), "--", color="#FD79A8", linewidth=1.5, alpha=0.8)
            except Exception:
                pass

        ax.set_xlabel(x_col, fontsize=self._theme.theme.label_size)
        ax.set_ylabel(y_col, fontsize=self._theme.theme.label_size)

        plotly = {
            "data": [{"type": "scatter", "mode": "markers", "x": list(x), "y": list(y),
                      "marker": {"color": PALETTE_PRIMARY[0], "opacity": alpha, "size": 7}}],
            "layout": {**self._theme.get_plotly_layout(), "title": {"text": spec.title},
                       "xaxis": {"title": {"text": x_col}}, "yaxis": {"title": {"text": y_col}}},
        }
        return fig, plotly

    # ── Heatmap ──────────────────────────────────────────────────────────

    def _render_heatmap(
        self, df: pd.DataFrame, spec: ChartSpec
    ) -> Tuple[matplotlib.figure.Figure, Optional[Dict]]:
        num_df = df.select_dtypes(include="number")
        if num_df.empty:
            return self._render_bar(df, spec)

        if spec.x_column and spec.x_column in df.columns and len(spec.y_columns) > 0:
            pivot = df.pivot_table(values=spec.y_columns[0], index=spec.x_column,
                                    aggfunc="mean").head(20)
            data = pivot
        else:
            data = num_df.head(30).T

        fig, ax = self._theme.create_figure(figsize=self._theme.theme.figsize_wide)
        im = ax.imshow(data.values, cmap="viridis", aspect="auto")
        fig.colorbar(im, ax=ax, shrink=0.8)

        ax.set_xticks(range(data.shape[1]))
        ax.set_yticks(range(data.shape[0]))
        ax.set_xticklabels([str(c)[:12] for c in data.columns], rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels([str(r)[:15] for r in data.index], fontsize=9)

        if spec.extra_params.get("annotate") and data.shape[0] * data.shape[1] <= 100:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    val = data.values[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                                fontsize=8, color="white")

        plotly = {
            "data": [{"type": "heatmap", "z": data.values.tolist(),
                      "x": [str(c) for c in data.columns], "y": [str(r) for r in data.index],
                      "colorscale": "Viridis"}],
            "layout": {**self._theme.get_plotly_layout(), "title": {"text": spec.title}},
        }
        return fig, plotly

    # ── Correlation Matrix ───────────────────────────────────────────────

    def _render_correlation(
        self, df: pd.DataFrame, spec: ChartSpec
    ) -> Tuple[matplotlib.figure.Figure, Optional[Dict]]:
        num_df = df.select_dtypes(include="number")
        if num_df.shape[1] < 2:
            return self._render_bar(df, spec)

        cols = spec.y_columns if spec.y_columns else num_df.columns.tolist()
        cols = [c for c in cols if c in num_df.columns][:15]
        corr = num_df[cols].corr()

        fig, ax = self._theme.create_figure(figsize=self._theme.theme.figsize_wide)

        mask = np.triu(np.ones_like(corr, dtype=bool), k=1) if spec.extra_params.get("mask_upper") else None
        display = corr.copy()
        if mask is not None:
            display = display.where(~mask, np.nan)

        im = ax.imshow(display.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        fig.colorbar(im, ax=ax, shrink=0.8, label="Correlation")

        ax.set_xticks(range(len(cols)))
        ax.set_yticks(range(len(cols)))
        ax.set_xticklabels([str(c)[:12] for c in cols], rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels([str(c)[:12] for c in cols], fontsize=9)

        if spec.extra_params.get("annotate", True) and len(cols) <= 12:
            for i in range(len(cols)):
                for j in range(len(cols)):
                    val = display.values[i, j]
                    if not np.isnan(val):
                        color = "white" if abs(val) > 0.5 else self._theme.theme.text_color
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                                fontsize=8, color=color, fontweight="bold")

        plotly = {
            "data": [{"type": "heatmap", "z": corr.values.tolist(),
                      "x": [str(c) for c in cols], "y": [str(c) for c in cols],
                      "colorscale": "RdBu", "zmin": -1, "zmax": 1}],
            "layout": {**self._theme.get_plotly_layout(), "title": {"text": spec.title}},
        }
        return fig, plotly

    # ── Box Plot ─────────────────────────────────────────────────────────

    def _render_box(
        self, df: pd.DataFrame, spec: ChartSpec
    ) -> Tuple[matplotlib.figure.Figure, Optional[Dict]]:
        y_col = spec.y_column
        x_col = spec.x_column

        fig, ax = self._theme.create_figure()

        if x_col and x_col in df.columns and y_col and y_col in df.columns:
            groups = df.groupby(x_col)[y_col].apply(lambda g: g.dropna().tolist())
            groups = groups.head(10)
            bp = ax.boxplot(groups.values, patch_artist=True, tick_labels=[str(l)[:15] for l in groups.index],
                            medianprops={"color": "#FD79A8", "linewidth": 2},
                            whiskerprops={"color": self._theme.theme.text_color},
                            capprops={"color": self._theme.theme.text_color},
                            flierprops={"markeredgecolor": "#FF7675", "markersize": 4})
            colors = self._theme.get_palette(len(groups))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_xticklabels([str(l)[:15] for l in groups.index], rotation=45, ha="right")
        else:
            num_df = df.select_dtypes(include="number").iloc[:, :6]
            bp = ax.boxplot([num_df[c].dropna().values for c in num_df.columns],
                            patch_artist=True, tick_labels=[str(c)[:12] for c in num_df.columns],
                            medianprops={"color": "#FD79A8", "linewidth": 2},
                            whiskerprops={"color": self._theme.theme.text_color},
                            capprops={"color": self._theme.theme.text_color})
            colors = self._theme.get_palette(len(num_df.columns))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax.set_ylabel(y_col or "Value", fontsize=self._theme.theme.label_size)

        plotly = {
            "data": [{"type": "box", "y": list(df[y_col].dropna()) if y_col and y_col in df.columns else [],
                      "name": y_col or "Data",
                      "marker": {"color": PALETTE_PRIMARY[0]}}],
            "layout": {**self._theme.get_plotly_layout(), "title": {"text": spec.title}},
        }
        return fig, plotly

    # ── Violin Plot ──────────────────────────────────────────────────────

    def _render_violin(
        self, df: pd.DataFrame, spec: ChartSpec
    ) -> Tuple[matplotlib.figure.Figure, Optional[Dict]]:
        y_col = spec.y_column
        x_col = spec.x_column
        fig, ax = self._theme.create_figure()

        if x_col and x_col in df.columns and y_col and y_col in df.columns:
            groups = df.groupby(x_col)[y_col].apply(lambda g: g.dropna().tolist())
            groups = groups.head(8)
            data_list = [v for v in groups.values if len(v) > 0]
            labels = [str(l)[:15] for l in groups.index[:len(data_list)]]
            if data_list:
                parts = ax.violinplot(data_list, showmeans=True, showmedians=True)
                colors = self._theme.get_palette(len(data_list))
                for i, body in enumerate(parts.get("bodies", [])):
                    body.set_facecolor(colors[i])
                    body.set_alpha(0.7)
                for key in ("cbars", "cmins", "cmaxes", "cmeans", "cmedians"):
                    if key in parts:
                        parts[key].set_color(self._theme.theme.text_color)
                ax.set_xticks(range(1, len(labels) + 1))
                ax.set_xticklabels(labels, rotation=45, ha="right")
        else:
            num_df = df.select_dtypes(include="number").iloc[:, :5]
            data_list = [num_df[c].dropna().values for c in num_df.columns if len(num_df[c].dropna()) > 0]
            if data_list:
                parts = ax.violinplot(data_list, showmeans=True, showmedians=True)
                colors = self._theme.get_palette(len(data_list))
                for i, body in enumerate(parts.get("bodies", [])):
                    body.set_facecolor(colors[i])
                    body.set_alpha(0.7)
                ax.set_xticks(range(1, len(num_df.columns) + 1))
                ax.set_xticklabels([str(c)[:12] for c in num_df.columns], rotation=45, ha="right")

        ax.set_ylabel(y_col or "Value", fontsize=self._theme.theme.label_size)

        plotly = {
            "data": [{"type": "violin", "y": list(df[y_col].dropna()) if y_col and y_col in df.columns else [],
                      "name": y_col or "Data", "box": {"visible": True},
                      "meanline": {"visible": True},
                      "marker": {"color": PALETTE_PRIMARY[0]}}],
            "layout": {**self._theme.get_plotly_layout(), "title": {"text": spec.title}},
        }
        return fig, plotly

    # ── Line Chart ───────────────────────────────────────────────────────

    def _render_line(
        self, df: pd.DataFrame, spec: ChartSpec
    ) -> Tuple[matplotlib.figure.Figure, Optional[Dict]]:
        x_col = spec.x_column
        y_col = spec.y_column

        fig, ax = self._theme.create_figure()

        if x_col and x_col in df.columns:
            try:
                x_data = pd.to_datetime(df[x_col])
                sort_idx = x_data.argsort()
                x_data = x_data.iloc[sort_idx]
            except Exception:
                x_data = df[x_col].iloc[:200]
                sort_idx = range(len(x_data))
        else:
            x_data = range(len(df))
            sort_idx = range(len(df))

        if y_col and y_col in df.columns:
            y_data = pd.to_numeric(df[y_col], errors="coerce").iloc[sort_idx]
        else:
            num_cols = df.select_dtypes(include="number").columns
            y_col = num_cols[0] if len(num_cols) > 0 else df.columns[0]
            y_data = pd.to_numeric(df[y_col], errors="coerce").iloc[sort_idx]

        show_markers = spec.extra_params.get("show_markers", len(df) <= 100)
        marker = "o" if show_markers else ""
        line_color = PALETTE_PRIMARY[0]

        # Glow effect — thicker semi-transparent line behind
        ax.plot(x_data, y_data, color=line_color, linewidth=6, alpha=0.2, zorder=2)

        # Main line
        ax.plot(x_data, y_data, color=line_color, linewidth=2.5,
                marker=marker, markersize=6, markerfacecolor=line_color,
                markeredgecolor=self._theme.theme.bg_color,
                markeredgewidth=1.5, zorder=3)

        # Gradient fill beneath
        ax.fill_between(range(len(y_data)) if not hasattr(x_data, "dt") else x_data,
                        y_data, alpha=0.15, color=line_color)

        ax.set_xlabel(spec.x_label or x_col or "Index", fontsize=self._theme.theme.label_size)
        ax.set_ylabel(spec.y_label or y_col or "Value", fontsize=self._theme.theme.label_size)

        if hasattr(x_data, "dt") or (hasattr(x_data, "dtype") and "datetime" in str(x_data.dtype)):
            fig.autofmt_xdate()

        plotly = {
            "data": [{"type": "scatter", "mode": "lines+markers" if show_markers else "lines",
                      "x": [str(v) for v in x_data], "y": list(y_data),
                      "line": {"color": PALETTE_PRIMARY[0], "width": 2},
                      "fill": "tozeroy", "fillcolor": "rgba(108,92,231,0.1)"}],
            "layout": {**self._theme.get_plotly_layout(), "title": {"text": spec.title},
                       "xaxis": {"title": {"text": x_col or ""}},
                       "yaxis": {"title": {"text": y_col or ""}}},
        }
        return fig, plotly
