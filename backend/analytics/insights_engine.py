"""
Insights Engine — Automatic insights, recommendations, and natural language summaries.

This is the "explain findings like a senior data scientist" module.
It takes raw analysis results and produces:
    - Ranked insights (surprising patterns, notable correlations, anomalies)
    - Actionable recommendations for next-step analysis
    - Natural language narrative summaries written in professional analyst style

All text generation is rule-based (no LLM needed). The LLM is only used
via the agent pipeline if the user wants deeper explanations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.core.logging_config import get_logger

logger = get_logger(__name__)


# ── Result Dataclasses ───────────────────────────────────────────────────────


@dataclass
class Insight:
    """A single automatically discovered insight."""

    category: str  # "correlation", "distribution", "outlier", "trend", "quality", "pattern"
    title: str
    description: str
    importance: float  # 0.0 - 1.0 (1.0 = most important)
    evidence: str  # statistical evidence
    emoji: str  # visual icon for the insight


@dataclass
class Recommendation:
    """An actionable recommendation for further analysis."""

    priority: int
    action: str
    rationale: str
    suggested_question: str  # question the user can ask the AI
    category: str  # "explore", "clean", "visualize", "model"


@dataclass
class InsightsReport:
    """Complete insights and recommendations report."""

    insights: List[Insight]
    recommendations: List[Recommendation]
    executive_summary: str  # senior data scientist-style narrative
    key_findings: List[str]  # bullet-point key findings


# ── Insights Engine ──────────────────────────────────────────────────────────


class InsightsEngine:
    """
    Generates automatic insights, recommendations, and NL summaries.

    Takes raw computation results from other analytics modules and
    produces senior data scientist-quality narratives and discoveries.

    Usage:
        engine = InsightsEngine()
        report = engine.generate_report(
            df=df,
            quality_report=quality_report,
            profile=dataset_profile,
            stats_report=statistical_report,
            predictive_report=predictive_report,
        )
    """

    def generate_report(
        self,
        df: pd.DataFrame,
        quality_report: Any = None,
        profile: Any = None,
        stats_report: Any = None,
        predictive_report: Any = None,
    ) -> InsightsReport:
        """
        Generate the complete insights report from all analysis modules.

        Args:
            df: The source DataFrame.
            quality_report: DataQualityReport from DataQualityAnalyzer.
            profile: DatasetProfile from DataProfiler.
            stats_report: StatisticalReport from StatisticalAnalyzer.
            predictive_report: PredictiveReport from PredictiveAnalyzer.

        Returns:
            An InsightsReport with ranked insights, recommendations, and summary.
        """
        insights: List[Insight] = []
        recommendations: List[Recommendation] = []

        # Mine insights from each report
        if quality_report:
            insights.extend(self._mine_quality_insights(quality_report))
            recommendations.extend(self._quality_recommendations(quality_report))

        if profile:
            insights.extend(self._mine_profile_insights(profile, df))

        if stats_report:
            insights.extend(self._mine_statistical_insights(stats_report))
            recommendations.extend(self._statistical_recommendations(stats_report))

        if predictive_report:
            insights.extend(self._mine_predictive_insights(predictive_report))
            recommendations.extend(self._predictive_recommendations(predictive_report))

        # Always mine basic DataFrame insights
        insights.extend(self._mine_dataframe_insights(df))

        # Sort insights by importance
        insights.sort(key=lambda i: i.importance, reverse=True)
        insights = insights[:20]  # Top 20 insights

        # Sort recommendations by priority
        recommendations.sort(key=lambda r: r.priority)
        recommendations = recommendations[:10]  # Top 10 recommendations

        # Generate summaries
        key_findings = self._extract_key_findings(insights)
        executive_summary = self._generate_executive_summary(
            df, insights, quality_report, profile, stats_report, predictive_report
        )

        return InsightsReport(
            insights=insights,
            recommendations=recommendations,
            executive_summary=executive_summary,
            key_findings=key_findings,
        )

    # ── Quality Insights ─────────────────────────────────────────────────

    def _mine_quality_insights(self, qr: Any) -> List[Insight]:
        """Extract insights from DataQualityReport."""
        insights: List[Insight] = []

        # Quality score insight
        score = getattr(qr, "overall_quality_score", 0)
        if score >= 90:
            insights.append(Insight(
                category="quality", title="Excellent Data Quality",
                description=f"Your dataset has a quality score of {score:.0f}/100, indicating very clean data suitable for analysis.",
                importance=0.3, evidence=f"Quality score: {score:.0f}/100", emoji="✅",
            ))
        elif score < 50:
            insights.append(Insight(
                category="quality", title="Data Quality Needs Attention",
                description=f"The quality score of {score:.0f}/100 suggests significant data issues that should be addressed before analysis.",
                importance=0.95, evidence=f"Quality score: {score:.0f}/100", emoji="⚠️",
            ))

        # Missing value insights
        missing = getattr(qr, "missing_values", [])
        critical_missing = [m for m in missing if getattr(m, "impact", "") in ("critical", "high")]
        if critical_missing:
            cols = ", ".join(getattr(m, "column", "?") for m in critical_missing[:3])
            insights.append(Insight(
                category="quality",
                title=f"{len(critical_missing)} Columns Have Critical Missing Data",
                description=f"Columns with significant missing values: {cols}. This may bias analysis results.",
                importance=0.85,
                evidence=f"{len(critical_missing)} columns with >10% missing values",
                emoji="🔴",
            ))

        # Duplicate insight
        dupes = getattr(qr, "duplicates", None)
        if dupes and getattr(dupes, "exact_duplicate_count", 0) > 0:
            count = dupes.exact_duplicate_count
            pct = getattr(dupes, "exact_duplicate_percentage", 0)
            insights.append(Insight(
                category="quality",
                title=f"{count} Duplicate Rows Detected",
                description=f"{pct:.1f}% of rows are exact duplicates. Removing them will improve analysis accuracy.",
                importance=0.7 if pct > 5 else 0.4,
                evidence=f"{count} duplicates ({pct:.1f}%)",
                emoji="📋",
            ))

        # Outlier insights
        outliers = getattr(qr, "outliers", [])
        severe = [o for o in outliers if getattr(o, "outlier_percentage", 0) > 5]
        if severe:
            insights.append(Insight(
                category="outlier",
                title=f"Significant Outliers in {len(severe)} Columns",
                description=f"Columns {', '.join(getattr(o, 'column', '?') for o in severe[:3])} contain >5% outlier values, which may affect statistical calculations.",
                importance=0.65,
                evidence=f"{len(severe)} columns with >5% outliers",
                emoji="📊",
            ))

        return insights

    def _quality_recommendations(self, qr: Any) -> List[Recommendation]:
        """Generate recommendations from quality report."""
        recs: List[Recommendation] = []

        missing = getattr(qr, "missing_values", [])
        critical = [m for m in missing if getattr(m, "impact", "") in ("critical", "high")]
        if critical:
            col = getattr(critical[0], "column", "the column")
            strategy = getattr(critical[0], "suggested_strategy", "fill_median")
            recs.append(Recommendation(
                priority=1, category="clean",
                action=f"Handle missing values in '{col}' using {strategy}",
                rationale=f"Critical missing data affects analysis reliability.",
                suggested_question=f"Clean the missing values in {col} and show the result",
            ))

        dupes = getattr(qr, "duplicates", None)
        if dupes and getattr(dupes, "exact_duplicate_count", 0) > 0:
            recs.append(Recommendation(
                priority=2, category="clean",
                action="Remove duplicate rows",
                rationale="Duplicates inflate counts and skew distributions.",
                suggested_question="How many duplicate rows are there? Remove them and show the cleaned data.",
            ))

        return recs

    # ── Profile Insights ─────────────────────────────────────────────────

    def _mine_profile_insights(self, profile: Any, df: pd.DataFrame) -> List[Insight]:
        """Extract insights from DatasetProfile."""
        insights: List[Insight] = []

        # Constant columns
        constants = getattr(profile, "constant_columns", [])
        if constants:
            insights.append(Insight(
                category="pattern",
                title=f"{len(constants)} Constant Columns Detected",
                description=f"Columns {', '.join(constants[:3])} have only one unique value and provide no analytical value.",
                importance=0.5,
                evidence=f"{len(constants)} columns with 1 unique value",
                emoji="🔒",
            ))

        # High cardinality warnings
        high_card = getattr(profile, "high_cardinality_columns", [])
        if high_card:
            insights.append(Insight(
                category="pattern",
                title=f"{len(high_card)} High-Cardinality Columns (Possible IDs)",
                description=f"Columns {', '.join(high_card[:3])} have nearly unique values — likely identifier columns.",
                importance=0.4,
                evidence=f"{len(high_card)} columns with >95% unique values",
                emoji="🔑",
            ))

        # Dataset size insight
        row_count = getattr(profile, "row_count", 0)
        col_count = getattr(profile, "col_count", 0)
        if row_count > 100000:
            insights.append(Insight(
                category="pattern",
                title="Large Dataset",
                description=f"With {row_count:,} rows and {col_count} columns, this is a substantial dataset. Consider sampling for exploratory analysis.",
                importance=0.3,
                evidence=f"{row_count:,} rows × {col_count} columns",
                emoji="📦",
            ))

        # Column type distribution
        col_profiles = getattr(profile, "column_profiles", [])
        skewed_cols = [p for p in col_profiles if getattr(p, "distribution_type", "") in ("right_skewed", "left_skewed")]
        if skewed_cols:
            insights.append(Insight(
                category="distribution",
                title=f"{len(skewed_cols)} Skewed Distributions Found",
                description=f"Columns {', '.join(getattr(p, 'name', '?') for p in skewed_cols[:3])} have skewed distributions. Consider log transformation for better analysis.",
                importance=0.5,
                evidence=f"{len(skewed_cols)} columns with |skewness| > 1",
                emoji="📈",
            ))

        return insights

    # ── Statistical Insights ─────────────────────────────────────────────

    def _mine_statistical_insights(self, sr: Any) -> List[Insight]:
        """Extract insights from StatisticalReport."""
        insights: List[Insight] = []

        corr = getattr(sr, "correlations", None)
        if corr:
            # Strong correlations
            top_pos = getattr(corr, "top_positive_pairs", [])
            top_neg = getattr(corr, "top_negative_pairs", [])

            for pair in (top_pos + top_neg)[:3]:
                r = getattr(pair, "correlation", 0)
                if abs(r) >= 0.7:
                    a = getattr(pair, "column_a", "?")
                    b = getattr(pair, "column_b", "?")
                    direction = "positive" if r > 0 else "negative"
                    insights.append(Insight(
                        category="correlation",
                        title=f"Strong {direction.title()} Correlation: {a} ↔ {b}",
                        description=f"There is a {getattr(pair, 'strength', 'strong')} {direction} correlation (r={r:+.3f}) between '{a}' and '{b}'. Changes in one variable are closely associated with changes in the other.",
                        importance=min(abs(r), 0.95),
                        evidence=f"Pearson r={r:+.3f}",
                        emoji="🔗",
                    ))

            # Multicollinearity warning
            high_corr = getattr(corr, "highly_correlated_columns", [])
            if len(high_corr) > 1:
                insights.append(Insight(
                    category="correlation",
                    title=f"Multicollinearity Detected ({len(high_corr)} pairs)",
                    description="Several feature pairs are highly correlated (|r| > 0.8). This can cause instability in regression models. Consider dropping one from each pair.",
                    importance=0.75,
                    evidence=f"{len(high_corr)} pairs with |r| > 0.8",
                    emoji="⚡",
                ))

        # Feature importance
        fi = getattr(sr, "feature_importance", None)
        if fi:
            rankings = getattr(fi, "rankings", [])
            if rankings:
                top = rankings[0]
                insights.append(Insight(
                    category="pattern",
                    title=f"Most Important Feature: {getattr(top, 'column', '?')}",
                    description=f"'{getattr(top, 'column', '?')}' has the highest importance score ({getattr(top, 'importance_score', 0):.3f}), making it the most predictive or informative feature.",
                    importance=0.6,
                    evidence=getattr(top, "reasoning", ""),
                    emoji="⭐",
                ))

        return insights

    def _statistical_recommendations(self, sr: Any) -> List[Recommendation]:
        """Generate recommendations from statistical report."""
        recs: List[Recommendation] = []

        corr = getattr(sr, "correlations", None)
        if corr:
            top_pairs = getattr(corr, "top_positive_pairs", []) + getattr(corr, "top_negative_pairs", [])
            strong = [p for p in top_pairs if abs(getattr(p, "correlation", 0)) > 0.5]
            if strong:
                p = strong[0]
                a, b = getattr(p, "column_a", "?"), getattr(p, "column_b", "?")
                recs.append(Recommendation(
                    priority=3, category="explore",
                    action=f"Investigate the relationship between '{a}' and '{b}'",
                    rationale=f"Strong correlation (r={getattr(p, 'correlation', 0):+.3f}) warrants deeper analysis.",
                    suggested_question=f"Create a scatter plot of {a} vs {b} and explain the relationship",
                ))

        dists = getattr(sr, "distribution_tests", [])
        non_normal = [d for d in dists if not getattr(d, "is_normal", True)]
        if non_normal:
            col = getattr(non_normal[0], "column", "?")
            recs.append(Recommendation(
                priority=4, category="explore",
                action=f"Examine the distribution of '{col}'",
                rationale="Non-normal distribution may require transformation for parametric tests.",
                suggested_question=f"Show a histogram of {col} and explain the distribution shape",
            ))

        return recs

    # ── Predictive Insights ──────────────────────────────────────────────

    def _mine_predictive_insights(self, pr: Any) -> List[Insight]:
        """Extract insights from PredictiveReport."""
        insights: List[Insight] = []

        ml = getattr(pr, "ml_result", None)
        if ml and getattr(ml, "score", 0) > 0:
            score = getattr(ml, "score", 0)
            target = getattr(ml, "target_column", "?")
            metric = getattr(ml, "metric_name", "score")
            if score > 0.7:
                insights.append(Insight(
                    category="pattern",
                    title=f"Predictable Target: '{target}'",
                    description=f"A basic model achieves {metric}={score:.3f} predicting '{target}', suggesting strong patterns in the data.",
                    importance=0.8,
                    evidence=f"{metric}={score:.3f}",
                    emoji="🎯",
                ))

        trends = getattr(pr, "trends", [])
        strong_trends = [t for t in trends if getattr(t, "trend_strength", 0) > 0.5]
        for t in strong_trends[:3]:
            col = getattr(t, "column", "?")
            direction = getattr(t, "trend_direction", "?")
            change = getattr(t, "change_percentage", 0)
            insights.append(Insight(
                category="trend",
                title=f"{direction.title()} Trend in '{col}'",
                description=f"'{col}' shows a clear {direction} trend with {abs(change):.1f}% change.",
                importance=0.65,
                evidence=f"R²={getattr(t, 'trend_strength', 0):.3f}, change={change:+.1f}%",
                emoji="📈" if direction == "increasing" else "📉",
            ))

        return insights

    def _predictive_recommendations(self, pr: Any) -> List[Recommendation]:
        """Generate recommendations from predictive report."""
        recs: List[Recommendation] = []

        ml = getattr(pr, "ml_result", None)
        if ml and getattr(ml, "score", 0) > 0.5:
            target = getattr(ml, "target_column", "?")
            recs.append(Recommendation(
                priority=5, category="model",
                action=f"Build a predictive model for '{target}'",
                rationale=f"Basic model shows {getattr(ml, 'metric_name', 'score')}={getattr(ml, 'score', 0):.3f} — a more sophisticated model could do better.",
                suggested_question=f"What are the most important factors that predict {target}?",
            ))

        trends = getattr(pr, "trends", [])
        forecastable = [t for t in trends if getattr(t, "trend_strength", 0) > 0.3]
        if forecastable:
            col = getattr(forecastable[0], "column", "?")
            recs.append(Recommendation(
                priority=6, category="explore",
                action=f"Forecast future values of '{col}'",
                rationale="Clear trend detected — extrapolation may be useful for planning.",
                suggested_question=f"What is the trend for {col} and what are the next predicted values?",
            ))

        return recs

    # ── DataFrame-Level Insights ─────────────────────────────────────────

    def _mine_dataframe_insights(self, df: pd.DataFrame) -> List[Insight]:
        """Extract basic insights directly from the DataFrame."""
        insights: List[Insight] = []

        # Check for highly imbalanced categorical columns
        for col in df.select_dtypes(include="object").columns:
            vc = df[col].value_counts()
            if len(vc) >= 2:
                top_pct = vc.iloc[0] / len(df) * 100
                if top_pct > 80:
                    insights.append(Insight(
                        category="distribution",
                        title=f"Imbalanced Category: '{col}'",
                        description=f"'{vc.index[0]}' dominates '{col}' at {top_pct:.0f}% of all values. This imbalance may affect grouping analysis.",
                        importance=0.45,
                        evidence=f"Mode '{vc.index[0]}' = {top_pct:.1f}%",
                        emoji="⚖️",
                    ))

        # Check for potential date columns stored as strings
        for col in df.select_dtypes(include="object").columns:
            sample = df[col].dropna().head(20).astype(str)
            try:
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().mean() > 0.8:
                    insights.append(Insight(
                        category="pattern",
                        title=f"'{col}' Appears to Be a Date Column",
                        description=f"The column '{col}' contains date-like values stored as text. Converting to datetime will enable time-series analysis.",
                        importance=0.5,
                        evidence=f"{parsed.notna().mean()*100:.0f}% parseable as dates",
                        emoji="📅",
                    ))
            except Exception:
                pass

        return insights

    # ── Key Findings ─────────────────────────────────────────────────────

    @staticmethod
    def _extract_key_findings(insights: List[Insight]) -> List[str]:
        """Extract bullet-point key findings from the top insights."""
        return [
            f"{i.emoji} **{i.title}**: {i.description}"
            for i in insights[:7]
        ]

    # ── Executive Summary ────────────────────────────────────────────────

    @staticmethod
    def _generate_executive_summary(
        df: pd.DataFrame,
        insights: List[Insight],
        quality_report: Any,
        profile: Any,
        stats_report: Any,
        predictive_report: Any,
    ) -> str:
        """
        Generate a senior data scientist-style executive summary.

        This reads like a professional analysis memo.
        """
        parts: List[str] = []

        # Opening
        row_count = len(df)
        col_count = len(df.columns)
        num_cols = len(df.select_dtypes(include="number").columns)
        cat_cols = len(df.select_dtypes(include="object").columns)

        parts.append(
            f"## 📊 Executive Summary\n\n"
            f"This dataset contains **{row_count:,} records** across **{col_count} variables** "
            f"({num_cols} numeric, {cat_cols} categorical). "
        )

        # Quality assessment
        if quality_report:
            score = getattr(quality_report, "overall_quality_score", 0)
            if score >= 80:
                parts.append(f"The data quality is **excellent** (score: {score:.0f}/100), requiring minimal preprocessing. ")
            elif score >= 60:
                parts.append(f"Data quality is **good** (score: {score:.0f}/100) but some cleaning is recommended. ")
            else:
                parts.append(f"**Data quality concerns** have been identified (score: {score:.0f}/100) — cleaning is strongly recommended before analysis. ")

        # Key statistical findings
        if stats_report:
            corr = getattr(stats_report, "correlations", None)
            if corr:
                top_pairs = getattr(corr, "top_positive_pairs", []) + getattr(corr, "top_negative_pairs", [])
                strong = [p for p in top_pairs if abs(getattr(p, "correlation", 0)) > 0.6]
                if strong:
                    p = strong[0]
                    parts.append(
                        f"\n\nThe strongest statistical relationship is between "
                        f"**{getattr(p, 'column_a', '?')}** and **{getattr(p, 'column_b', '?')}** "
                        f"(r={getattr(p, 'correlation', 0):+.3f}). "
                    )

        # Predictive findings
        if predictive_report:
            ml = getattr(predictive_report, "ml_result", None)
            if ml and getattr(ml, "score", 0) > 0.3:
                parts.append(
                    f"A preliminary model targeting **{getattr(ml, 'target_column', '?')}** "
                    f"achieved {getattr(ml, 'metric_name', 'score')}="
                    f"{getattr(ml, 'score', 0):.3f}, "
                    f"suggesting {'strong' if getattr(ml, 'score', 0) > 0.7 else 'moderate'} predictive potential. "
                )

        # Top insights summary
        important_insights = [i for i in insights if i.importance > 0.6]
        if important_insights:
            parts.append(
                f"\n\n### Key Findings\n\n"
                + "\n".join(
                    f"- {i.emoji} {i.title}: {i.description}"
                    for i in important_insights[:5]
                )
            )

        return "".join(parts)
