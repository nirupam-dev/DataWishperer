"""
Integration tests for the analytics pipeline.

Tests end-to-end flow through:
    - DataQualityAnalyzer → DataProfiler → StatisticalAnalyzer
    - PredictiveAnalyzer → InsightsEngine → Orchestrator
    - Markdown report generation
    - Module interaction correctness
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.analytics.data_quality import DataQualityAnalyzer
from backend.analytics.data_profiler import DataProfiler
from backend.analytics.statistical import StatisticalAnalyzer
from backend.analytics.predictive import PredictiveAnalyzer
from backend.analytics.insights_engine import InsightsEngine
from backend.analytics.orchestrator import AnalyticsOrchestrator


# ── Cross-Module Integration ────────────────────────────────────────────────


class TestCrossModuleIntegration:
    """Test that analytics modules work correctly together."""

    def setup_method(self):
        self.quality = DataQualityAnalyzer()
        self.profiler = DataProfiler()
        self.stats = StatisticalAnalyzer()
        self.pred = PredictiveAnalyzer()
        self.insights = InsightsEngine()

    def test_quality_feeds_insights(self, messy_df):
        """Quality report should feed into insights generation."""
        qr = self.quality.analyze(messy_df)
        profile = self.profiler.profile(messy_df)
        report = self.insights.generate_report(
            messy_df, quality_report=qr, profile=profile
        )
        assert len(report.insights) > 0
        # Insights should reference quality issues
        assert len(report.executive_summary) > 50

    def test_stats_feeds_insights(self, sample_df):
        """Statistical analysis should enrich insights."""
        qr = self.quality.analyze(sample_df)
        profile = self.profiler.profile(sample_df)
        sr = self.stats.analyze(sample_df)
        report = self.insights.generate_report(
            sample_df, quality_report=qr, profile=profile, stats_report=sr
        )
        assert len(report.key_findings) > 0

    def test_predictive_feeds_insights(self, time_series_df):
        """Predictive analysis should enrich insights."""
        qr = self.quality.analyze(time_series_df)
        profile = self.profiler.profile(time_series_df)
        sr = self.stats.analyze(time_series_df)
        pr = self.pred.analyze(time_series_df)
        report = self.insights.generate_report(
            time_series_df,
            quality_report=qr, profile=profile,
            stats_report=sr, predictive_report=pr,
        )
        assert len(report.recommendations) > 0

    def test_full_pipeline_no_data_loss(self, sample_df):
        """Running the full pipeline should not modify the input DataFrame."""
        df_copy = sample_df.copy()
        orchestrator = AnalyticsOrchestrator()
        orchestrator.run_full_analysis(sample_df)
        pd.testing.assert_frame_equal(sample_df, df_copy)


# ── Orchestrator Integration ────────────────────────────────────────────────


class TestOrchestratorIntegration:
    """Test the orchestrator end-to-end."""

    def setup_method(self):
        self.orchestrator = AnalyticsOrchestrator()

    def test_full_analysis_produces_all_sections(self, sample_df):
        report = self.orchestrator.run_full_analysis(sample_df)
        assert report.quality is not None
        assert report.profile is not None
        assert report.statistics is not None
        assert report.predictive is not None
        assert report.insights is not None
        assert report.analysis_time_ms > 0
        assert report.analysis_type == "full"

    def test_quick_scan_omits_heavy_modules(self, sample_df):
        report = self.orchestrator.run_quick_scan(sample_df)
        assert report.quality is not None
        assert report.profile is not None
        assert report.statistics is None
        assert report.predictive is None
        assert report.analysis_type == "quick"

    def test_markdown_report_completeness(self, sample_df):
        report = self.orchestrator.run_full_analysis(sample_df)
        md = report.to_markdown()
        # Must have all major sections
        assert "Executive Summary" in md
        assert "Data Quality" in md
        assert "Dataset Profile" in md
        assert "Statistical Analysis" in md
        assert "Predictive Analysis" in md
        assert "ms" in md  # timing footer

    def test_markdown_report_not_empty(self, sample_df):
        report = self.orchestrator.run_full_analysis(sample_df)
        md = report.to_markdown()
        assert len(md) > 500

    def test_messy_data_detects_all_issues(self, messy_df):
        report = self.orchestrator.run_full_analysis(messy_df)
        assert report.quality.total_missing_cells > 0
        assert report.quality.duplicates.exact_duplicate_count > 0
        assert len(report.quality.cleaning_actions) > 0
        assert report.quality.overall_quality_score < 95

    def test_time_series_detects_trends(self, time_series_df):
        report = self.orchestrator.run_full_analysis(time_series_df)
        assert report.predictive is not None
        assert len(report.predictive.trends) > 0
        # Sales column should show increasing trend
        sales_trend = next(
            (t for t in report.predictive.trends if t.column == "sales"),
            None,
        )
        assert sales_trend is not None
        assert sales_trend.trend_direction == "increasing"


# ── Data Flow Correctness ──────────────────────────────────────────────────


class TestDataFlowCorrectness:
    """Verify that data flows correctly between modules."""

    def test_quality_score_consistency(self, sample_df):
        """Quality score from quality module and profiler should be similar."""
        quality = DataQualityAnalyzer().analyze(sample_df)
        profile = DataProfiler().profile(sample_df)
        # Both should indicate high quality for clean data
        assert quality.overall_quality_score > 70
        assert profile.quality_score > 70

    def test_column_counts_match(self, sample_df):
        """Column counts should be consistent across all modules."""
        quality = DataQualityAnalyzer().analyze(sample_df)
        profile = DataProfiler().profile(sample_df)
        assert len(quality.missing_values) == sample_df.shape[1]
        assert len(profile.column_profiles) == sample_df.shape[1]
        assert profile.col_count == sample_df.shape[1]

    def test_insights_reference_actual_data(self, messy_df):
        """Insights should reference real column names and statistics."""
        quality = DataQualityAnalyzer().analyze(messy_df)
        profile = DataProfiler().profile(messy_df)
        insights = InsightsEngine().generate_report(
            messy_df, quality_report=quality, profile=profile
        )
        summary = insights.executive_summary
        # Should mention the actual data dimensions
        assert str(messy_df.shape[0]) in summary or "record" in summary.lower()


# ── Edge Case Integration ──────────────────────────────────────────────────


class TestEdgeCaseIntegration:
    """Test the full pipeline with edge-case inputs."""

    def test_empty_dataframe(self, empty_df):
        orchestrator = AnalyticsOrchestrator()
        report = orchestrator.run_full_analysis(empty_df)
        assert report.quality.overall_quality_score == 0

    def test_single_column_dataframe(self, single_column_df):
        orchestrator = AnalyticsOrchestrator()
        report = orchestrator.run_full_analysis(single_column_df)
        assert report.profile is not None
        assert report.profile.col_count == 1

    def test_all_nulls_dataframe(self, all_nulls_df):
        orchestrator = AnalyticsOrchestrator()
        report = orchestrator.run_full_analysis(all_nulls_df)
        assert report.quality.total_missing_percentage > 90

    def test_wide_dataframe(self, wide_df):
        orchestrator = AnalyticsOrchestrator()
        report = orchestrator.run_full_analysis(wide_df)
        assert report.profile.col_count == 50
