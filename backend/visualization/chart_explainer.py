"""
Chart Explainer — AI-powered automatic chart explanations.

Generates plain-English explanations for every chart, describing what
the visualization shows, key patterns, and statistical insights.
Works both with LLM (via the agent) and with a fast rule-based fallback.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
import numpy as np

from backend.core.logging_config import get_logger
from backend.visualization.chart_selector import ChartSpec, ChartType

logger = get_logger(__name__)


class ChartExplainer:
    """
    Generates automatic explanations for charts.

    Uses rule-based statistical analysis to produce explanations
    without requiring an LLM call. The LLM-based explanation is
    handled separately by the QueryChain (Stage 7).
    """

    def explain(self, df: pd.DataFrame, spec: ChartSpec) -> str:
        """
        Generate a plain-English explanation for a chart.

        Args:
            df: The source DataFrame.
            spec: The chart specification.

        Returns:
            A 2-4 sentence explanation string.
        """
        dispatch = {
            ChartType.BAR: self._explain_bar,
            ChartType.HORIZONTAL_BAR: self._explain_bar,
            ChartType.PIE: self._explain_pie,
            ChartType.HISTOGRAM: self._explain_histogram,
            ChartType.SCATTER: self._explain_scatter,
            ChartType.HEATMAP: self._explain_heatmap,
            ChartType.CORRELATION_MATRIX: self._explain_correlation,
            ChartType.BOX_PLOT: self._explain_box,
            ChartType.VIOLIN_PLOT: self._explain_violin,
            ChartType.LINE: self._explain_line,
            ChartType.AREA: self._explain_line,
        }

        fn = dispatch.get(spec.chart_type, self._explain_default)
        try:
            explanation = fn(df, spec)
            reasoning = f"\n\n🎯 **Why this chart?** {spec.reasoning}" if spec.reasoning else ""
            return explanation + reasoning
        except Exception as e:
            logger.warning("Chart explanation failed: %s", e)
            return self._explain_default(df, spec)

    def _explain_bar(self, df: pd.DataFrame, spec: ChartSpec) -> str:
        x, y = spec.x_column, spec.y_column
        if x and y and x in df.columns and y in df.columns:
            top = df.groupby(x)[y].mean().sort_values(ascending=False)
            top_name = str(top.index[0]) if len(top) > 0 else "N/A"
            top_val = top.values[0] if len(top) > 0 else 0
            n_groups = top.count()
            return (
                f"📊 This bar chart compares **{y}** across **{n_groups}** categories in **{x}**. "
                f"The highest value is **{top_name}** at **{top_val:,.2f}**. "
                f"The data ranges from {top.min():,.2f} to {top.max():,.2f}."
            )
        return f"📊 This bar chart shows the distribution of values across categories."

    def _explain_pie(self, df: pd.DataFrame, spec: ChartSpec) -> str:
        x, y = spec.x_column, spec.y_column
        if x and y and x in df.columns and y in df.columns:
            totals = df.groupby(x)[y].sum().sort_values(ascending=False).head(7)
            total = totals.sum()
            top_name = str(totals.index[0])
            top_pct = (totals.values[0] / total * 100) if total > 0 else 0
            return (
                f"🥧 This pie chart shows the proportional breakdown of **{y}** by **{x}**. "
                f"**{top_name}** holds the largest share at **{top_pct:.1f}%** of the total. "
                f"The chart covers {len(totals)} categories."
            )
        return "🥧 This pie chart shows proportional distribution across categories."

    def _explain_histogram(self, df: pd.DataFrame, spec: ChartSpec) -> str:
        col = spec.x_column
        if col and col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            mean, median, std = vals.mean(), vals.median(), vals.std()
            skew = "right-skewed" if mean > median else "left-skewed" if mean < median else "symmetric"
            return (
                f"📈 This histogram shows the distribution of **{col}**. "
                f"The mean is **{mean:,.2f}** and median is **{median:,.2f}** (distribution is {skew}). "
                f"Values range from {vals.min():,.2f} to {vals.max():,.2f} with σ={std:,.2f}."
            )
        return "📈 This histogram shows the frequency distribution of the data."

    def _explain_scatter(self, df: pd.DataFrame, spec: ChartSpec) -> str:
        x, y = spec.x_column, spec.y_column
        if x and y and x in df.columns and y in df.columns:
            xv = pd.to_numeric(df[x], errors="coerce")
            yv = pd.to_numeric(df[y], errors="coerce")
            mask = xv.notna() & yv.notna()
            if mask.sum() > 2:
                corr = xv[mask].corr(yv[mask])
                strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "weak"
                direction = "positive" if corr > 0 else "negative"
                return (
                    f"🔵 This scatter plot shows the relationship between **{x}** and **{y}**. "
                    f"There is a **{strength} {direction} correlation** (r={corr:.3f}). "
                    f"The plot contains {mask.sum():,} data points."
                )
        return "🔵 This scatter plot shows the relationship between two variables."

    def _explain_heatmap(self, df: pd.DataFrame, spec: ChartSpec) -> str:
        return "🗺️ This heatmap visualizes the intensity of values across a matrix. Darker/brighter cells indicate higher values."

    def _explain_correlation(self, df: pd.DataFrame, spec: ChartSpec) -> str:
        num_df = df.select_dtypes(include="number")
        if num_df.shape[1] >= 2:
            corr = num_df.corr()
            mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
            upper = corr.where(~mask).stack()
            if len(upper) > 0:
                strongest = upper.abs().idxmax()
                strongest_val = upper[strongest]
                return (
                    f"🔗 This correlation matrix shows pairwise correlations between **{num_df.shape[1]}** numeric columns. "
                    f"The strongest correlation is between **{strongest[0]}** and **{strongest[1]}** "
                    f"(r={strongest_val:.3f}). Values range from -1 (inverse) to +1 (perfect)."
                )
        return "🔗 This correlation matrix shows relationships between numeric variables."

    def _explain_box(self, df: pd.DataFrame, spec: ChartSpec) -> str:
        y = spec.y_column
        if y and y in df.columns:
            vals = pd.to_numeric(df[y], errors="coerce").dropna()
            q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
            iqr = q3 - q1
            outliers = ((vals < q1 - 1.5 * iqr) | (vals > q3 + 1.5 * iqr)).sum()
            return (
                f"📦 This box plot shows the distribution of **{y}**. "
                f"The median is **{vals.median():,.2f}**, IQR is [{q1:,.2f}, {q3:,.2f}]. "
                f"There are **{outliers}** outliers detected beyond 1.5×IQR."
            )
        return "📦 This box plot shows the distribution, quartiles, and outliers in the data."

    def _explain_violin(self, df: pd.DataFrame, spec: ChartSpec) -> str:
        y = spec.y_column
        if y and y in df.columns:
            vals = pd.to_numeric(df[y], errors="coerce").dropna()
            return (
                f"🎻 This violin plot shows the probability density of **{y}**. "
                f"The distribution has mean={vals.mean():,.2f}, median={vals.median():,.2f}. "
                f"Wider sections indicate more concentrated data points."
            )
        return "🎻 This violin plot shows the shape and density of the data distribution."

    def _explain_line(self, df: pd.DataFrame, spec: ChartSpec) -> str:
        y = spec.y_column
        if y and y in df.columns:
            vals = pd.to_numeric(df[y], errors="coerce").dropna()
            if len(vals) >= 2:
                trend = "upward" if vals.iloc[-1] > vals.iloc[0] else "downward"
                change = ((vals.iloc[-1] - vals.iloc[0]) / abs(vals.iloc[0]) * 100) if vals.iloc[0] != 0 else 0
                return (
                    f"📉 This line chart shows the trend of **{y}** over time. "
                    f"The overall trend is **{trend}** with a {abs(change):.1f}% change. "
                    f"Values range from {vals.min():,.2f} to {vals.max():,.2f}."
                )
        return "📉 This line chart shows how values change over time or sequence."

    def _explain_default(self, df: pd.DataFrame, spec: ChartSpec) -> str:
        return f"📊 This {spec.chart_type.value} visualization presents the data analysis results."
