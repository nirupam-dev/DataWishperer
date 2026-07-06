"""
Analytics Orchestrator — Coordinates all analytics modules into a unified report.

Provides two entry points:
    - run_full_analysis(): Complete analysis (all modules, ~1-2 seconds)
    - run_quick_scan(): Fast subset (quality + profiling only, ~300ms)

The orchestrator is the main integration point between the analytics
engine and the rest of the DataWhisperer system.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.analytics.data_quality import DataQualityAnalyzer, DataQualityReport
from backend.analytics.data_profiler import DataProfiler, DatasetProfile
from backend.analytics.statistical import StatisticalAnalyzer, StatisticalReport
from backend.analytics.predictive import PredictiveAnalyzer, PredictiveReport
from backend.analytics.insights_engine import InsightsEngine, InsightsReport
from backend.core.logging_config import get_logger

logger = get_logger(__name__)


# ── Report Dataclass ─────────────────────────────────────────────────────────


@dataclass
class AnalyticsReport:
    """
    Complete analytics report combining all analysis modules.

    This is the top-level result consumed by the agent pipeline
    and the Streamlit UI.
    """

    quality: DataQualityReport
    profile: DatasetProfile
    statistics: Optional[StatisticalReport]
    predictive: Optional[PredictiveReport]
    insights: InsightsReport
    analysis_time_ms: float
    analysis_type: str  # "full" or "quick"

    def to_markdown(self) -> str:
        """Format the entire report as markdown for display."""
        sections: List[str] = []

        # Executive Summary
        sections.append(self.insights.executive_summary)

        # Key Findings
        if self.insights.key_findings:
            sections.append("\n### 🔍 Key Findings\n")
            for finding in self.insights.key_findings:
                sections.append(f"- {finding}")

        # Data Quality
        sections.append(f"\n### 🧹 Data Quality\n\n{self.quality.summary}")
        if self.quality.cleaning_actions:
            sections.append("\n**Recommended Actions:**")
            for action in self.quality.cleaning_actions[:5]:
                sections.append(f"- {action.action}")

        # Profile
        sections.append(
            f"\n### 📋 Dataset Profile\n\n"
            f"- **Rows:** {self.profile.row_count:,}\n"
            f"- **Columns:** {self.profile.col_count}\n"
            f"- **Memory:** {self.profile.memory_usage_mb:.1f} MB\n"
            f"- **Missing:** {self.profile.missing_percentage:.1f}%\n"
            f"- **Quality Score:** {self.profile.quality_score:.0f}/100"
        )

        # Statistics
        if self.statistics:
            sections.append(f"\n### 📈 Statistical Analysis\n\n{self.statistics.summary}")

        # Predictive
        if self.predictive:
            sections.append(f"\n### 🤖 Predictive Analysis\n\n{self.predictive.summary}")

        # Recommendations
        if self.insights.recommendations:
            sections.append("\n### 💡 Recommendations\n")
            for rec in self.insights.recommendations[:5]:
                sections.append(
                    f"- **{rec.action}** — {rec.rationale}\n"
                    f"  _Try asking:_ \"{rec.suggested_question}\""
                )

        # Footer
        sections.append(
            f"\n---\n_Analysis completed in {self.analysis_time_ms:.0f}ms_"
        )

        return "\n".join(sections)


# ── Orchestrator ─────────────────────────────────────────────────────────────


class AnalyticsOrchestrator:
    """
    Coordinates all analytics modules into a unified report.

    This is the single entry point for the agent and UI to run analytics.

    Usage:
        orchestrator = AnalyticsOrchestrator()
        report = orchestrator.run_full_analysis(df)
        markdown = report.to_markdown()
    """

    def __init__(self) -> None:
        self._quality = DataQualityAnalyzer()
        self._profiler = DataProfiler()
        self._statistical = StatisticalAnalyzer()
        self._predictive = PredictiveAnalyzer()
        self._insights = InsightsEngine()

    def run_full_analysis(self, df: pd.DataFrame) -> AnalyticsReport:
        """
        Run the complete analytics pipeline.

        Executes all modules: quality, profiling, statistics, predictive,
        and insights generation.

        Args:
            df: The input DataFrame (not modified).

        Returns:
            A complete AnalyticsReport with all sections.
        """
        start = time.time()

        logger.info(
            "Starting full analysis: %d rows × %d cols",
            df.shape[0], df.shape[1],
        )

        # Sample large datasets for performance
        analysis_df = self._prepare_df(df)

        # Run all modules
        quality = self._quality.analyze(analysis_df)
        profile = self._profiler.profile(analysis_df)
        statistics = self._statistical.analyze(analysis_df)
        predictive = self._predictive.analyze(analysis_df)

        # Generate insights from all reports
        insights = self._insights.generate_report(
            df=analysis_df,
            quality_report=quality,
            profile=profile,
            stats_report=statistics,
            predictive_report=predictive,
        )

        elapsed_ms = round((time.time() - start) * 1000, 2)

        logger.info("Full analysis complete: %.0fms", elapsed_ms)

        return AnalyticsReport(
            quality=quality,
            profile=profile,
            statistics=statistics,
            predictive=predictive,
            insights=insights,
            analysis_time_ms=elapsed_ms,
            analysis_type="full",
        )

    def run_quick_scan(self, df: pd.DataFrame) -> AnalyticsReport:
        """
        Run a fast quality + profiling scan only.

        Skips statistical and predictive analysis for speed.
        ~3x faster than full analysis.

        Args:
            df: The input DataFrame.

        Returns:
            An AnalyticsReport with quality and profile sections only.
        """
        start = time.time()

        analysis_df = self._prepare_df(df)
        quality = self._quality.analyze(analysis_df)
        profile = self._profiler.profile(analysis_df)

        insights = self._insights.generate_report(
            df=analysis_df,
            quality_report=quality,
            profile=profile,
        )

        elapsed_ms = round((time.time() - start) * 1000, 2)

        logger.info("Quick scan complete: %.0fms", elapsed_ms)

        return AnalyticsReport(
            quality=quality,
            profile=profile,
            statistics=None,
            predictive=None,
            insights=insights,
            analysis_time_ms=elapsed_ms,
            analysis_type="quick",
        )

    @staticmethod
    def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
        """Prepare the DataFrame for analysis (sample if too large)."""
        if len(df) > 100_000:
            logger.info(
                "Dataset has %d rows — sampling to 100K for analysis",
                len(df),
            )
            return df.sample(n=100_000, random_state=42)
        return df
