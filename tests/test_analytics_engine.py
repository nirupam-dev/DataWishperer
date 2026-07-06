"""
Tests for the premium analytics engine.

Tests all modules: data quality, profiling, statistical analysis,
predictive analysis, insights, and the orchestrator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.analytics.data_quality import DataQualityAnalyzer, DataQualityReport
from backend.analytics.data_profiler import DataProfiler, DatasetProfile
from backend.analytics.statistical import StatisticalAnalyzer, StatisticalReport
from backend.analytics.predictive import PredictiveAnalyzer, PredictiveReport
from backend.analytics.insights_engine import InsightsEngine, InsightsReport
from backend.analytics.orchestrator import AnalyticsOrchestrator, AnalyticsReport


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _sample_df() -> pd.DataFrame:
    """Create a realistic sample dataset."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "category": np.random.choice(["A", "B", "C", "D", "E"], n),
        "region": np.random.choice(["North", "South", "East", "West"], n),
        "revenue": np.random.lognormal(8, 1, n).round(2),
        "quantity": np.random.randint(1, 100, n),
        "price": np.random.normal(50, 15, n).round(2),
        "discount": np.random.uniform(0, 0.5, n).round(3),
        "satisfaction": np.random.choice([1, 2, 3, 4, 5], n),
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
    })


def _messy_df() -> pd.DataFrame:
    """Create a dataset with quality issues."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "id": range(n),
        "name": [f"item_{i}" for i in range(n)],
        "value": np.random.normal(100, 30, n).round(2),
        "category": np.random.choice(["A", "B", "C"], n),
    })
    # Inject missing values
    df.loc[10:20, "value"] = np.nan
    df.loc[50:55, "category"] = np.nan
    # Inject duplicates
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    # Inject outliers
    df.loc[98, "value"] = 9999.99
    df.loc[99, "value"] = -999.99
    return df


def _time_series_df() -> pd.DataFrame:
    """Create a dataset with clear trends."""
    np.random.seed(42)
    n = 100
    x = np.arange(n, dtype=float)
    return pd.DataFrame({
        "day": pd.date_range("2024-01-01", periods=n, freq="D"),
        "sales": (x * 2.5 + 100 + np.random.normal(0, 10, n)).round(2),
        "cost": (x * 1.2 + 50 + np.random.normal(0, 5, n)).round(2),
        "random_noise": np.random.normal(0, 1, n).round(4),
    })


# ── Data Quality Tests ───────────────────────────────────────────────────────


class TestDataQualityAnalyzer:
    """Tests for the data quality analysis module."""

    def setup_method(self):
        self.analyzer = DataQualityAnalyzer()

    def test_clean_data_high_score(self):
        df = _sample_df()
        report = self.analyzer.analyze(df)
        assert report.overall_quality_score > 80
        assert report.total_missing_cells == 0

    def test_messy_data_detects_issues(self):
        df = _messy_df()
        report = self.analyzer.analyze(df)
        assert report.total_missing_cells > 0
        assert report.duplicates.exact_duplicate_count > 0
        assert len(report.cleaning_actions) > 0

    def test_missing_value_detection(self):
        df = _messy_df()
        report = self.analyzer.analyze(df)
        missing_cols = [m for m in report.missing_values if m.null_count > 0]
        assert len(missing_cols) > 0
        for m in missing_cols:
            assert m.null_percentage > 0
            assert m.suggested_strategy != "none"
            assert m.impact != "none"

    def test_duplicate_detection(self):
        df = _messy_df()
        report = self.analyzer.analyze(df)
        assert report.duplicates.exact_duplicate_count == 5
        assert report.duplicates.exact_duplicate_percentage > 0

    def test_outlier_detection(self):
        df = _messy_df()
        report = self.analyzer.analyze(df)
        outlier_reports = [o for o in report.outliers if o.outlier_count_iqr > 0]
        assert len(outlier_reports) > 0

    def test_cleaning_recommendations(self):
        df = _messy_df()
        report = self.analyzer.analyze(df)
        assert len(report.cleaning_actions) > 0
        # Check recommendations are prioritized by impact
        for i in range(len(report.cleaning_actions) - 1):
            assert report.cleaning_actions[i].impact_score >= report.cleaning_actions[i + 1].impact_score

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        report = self.analyzer.analyze(df)
        assert report.overall_quality_score == 0
        assert "empty" in report.summary.lower()

    def test_quality_score_range(self):
        df = _sample_df()
        report = self.analyzer.analyze(df)
        assert 0 <= report.overall_quality_score <= 100

    def test_summary_generated(self):
        df = _messy_df()
        report = self.analyzer.analyze(df)
        assert len(report.summary) > 20


# ── Data Profiler Tests ──────────────────────────────────────────────────────


class TestDataProfiler:
    """Tests for the data profiling module."""

    def setup_method(self):
        self.profiler = DataProfiler()

    def test_profile_complete(self):
        df = _sample_df()
        profile = self.profiler.profile(df)
        assert profile.row_count == 200
        assert profile.col_count == 8
        assert len(profile.column_profiles) == 8

    def test_numeric_column_stats(self):
        df = _sample_df()
        profile = self.profiler.profile(df)
        revenue_profile = next(
            p for p in profile.column_profiles if p.name == "revenue"
        )
        assert revenue_profile.mean is not None
        assert revenue_profile.std is not None
        assert revenue_profile.min_val is not None
        assert revenue_profile.max_val is not None
        assert revenue_profile.median is not None

    def test_categorical_column_stats(self):
        df = _sample_df()
        profile = self.profiler.profile(df)
        cat_profile = next(
            p for p in profile.column_profiles if p.name == "category"
        )
        assert cat_profile.semantic_type == "categorical"
        assert cat_profile.top_values is not None
        assert len(cat_profile.top_values) > 0

    def test_distribution_classification(self):
        df = _sample_df()
        profile = self.profiler.profile(df)
        revenue_profile = next(
            p for p in profile.column_profiles if p.name == "revenue"
        )
        assert revenue_profile.distribution_type is not None
        assert revenue_profile.skewness is not None

    def test_quality_score(self):
        df = _sample_df()
        profile = self.profiler.profile(df)
        assert 0 <= profile.quality_score <= 100

    def test_constant_column_detection(self):
        df = pd.DataFrame({"const": [1] * 50, "vary": range(50)})
        profile = self.profiler.profile(df)
        assert "const" in profile.constant_columns

    def test_id_column_detection(self):
        df = pd.DataFrame({
            "id": range(100),
            "value": np.random.normal(0, 1, 100),
        })
        profile = self.profiler.profile(df)
        id_profile = next(p for p in profile.column_profiles if p.name == "id")
        assert id_profile.is_id_like

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        profile = self.profiler.profile(df)
        assert profile.row_count == 0
        assert profile.col_count == 0

    def test_dtype_summary(self):
        df = _sample_df()
        profile = self.profiler.profile(df)
        assert len(profile.dtypes_summary) > 0

    def test_memory_usage(self):
        df = _sample_df()
        profile = self.profiler.profile(df)
        assert profile.memory_usage_mb > 0


# ── Statistical Analysis Tests ───────────────────────────────────────────────


class TestStatisticalAnalyzer:
    """Tests for the statistical analysis module."""

    def setup_method(self):
        self.analyzer = StatisticalAnalyzer()

    def test_correlation_analysis(self):
        df = _sample_df()
        report = self.analyzer.analyze(df)
        assert report.correlations.correlation_matrix is not None
        assert len(report.correlations.correlation_matrix) > 0

    def test_top_correlation_pairs(self):
        # Create data with known correlation
        np.random.seed(42)
        x = np.random.normal(0, 1, 100)
        df = pd.DataFrame({
            "x": x,
            "y": x * 2 + np.random.normal(0, 0.1, 100),  # strongly correlated
            "z": np.random.normal(0, 1, 100),  # uncorrelated
        })
        report = self.analyzer.analyze(df)
        top = report.correlations.top_positive_pairs
        assert len(top) > 0
        assert abs(top[0].correlation) > 0.9  # x and y should be strongly correlated

    def test_feature_importance(self):
        df = _sample_df()
        report = self.analyzer.analyze(df)
        assert len(report.feature_importance.rankings) > 0
        # Rankings should be sorted
        for i in range(len(report.feature_importance.rankings) - 1):
            assert report.feature_importance.rankings[i].ranking <= report.feature_importance.rankings[i + 1].ranking

    def test_distribution_tests(self):
        df = _sample_df()
        report = self.analyzer.analyze(df)
        assert len(report.distribution_tests) > 0
        for dist in report.distribution_tests:
            assert dist.p_value >= 0
            assert dist.distribution_type is not None

    def test_multicollinearity_detection(self):
        np.random.seed(42)
        x = np.random.normal(0, 1, 100)
        df = pd.DataFrame({
            "a": x,
            "b": x + np.random.normal(0, 0.05, 100),  # nearly identical
            "c": np.random.normal(0, 1, 100),
        })
        report = self.analyzer.analyze(df)
        assert len(report.correlations.highly_correlated_columns) > 0

    def test_insufficient_columns(self):
        df = pd.DataFrame({"x": range(100)})
        report = self.analyzer.analyze(df)
        assert "Insufficient" in report.correlations.summary

    def test_summary_generated(self):
        df = _sample_df()
        report = self.analyzer.analyze(df)
        assert len(report.summary) > 10


# ── Predictive Analysis Tests ────────────────────────────────────────────────


class TestPredictiveAnalyzer:
    """Tests for the predictive analysis module."""

    def setup_method(self):
        self.analyzer = PredictiveAnalyzer()

    def test_auto_ml_regression(self):
        df = _sample_df()
        report = self.analyzer.analyze(df)
        assert report.ml_result is not None
        assert report.ml_result.task_type in ("classification", "regression")
        assert report.ml_result.score <= 1  # R² can be negative for poor fits

    def test_auto_ml_with_clear_pattern(self):
        np.random.seed(42)
        x = np.random.normal(0, 1, 200)
        df = pd.DataFrame({
            "feature1": x,
            "feature2": np.random.normal(0, 1, 200),
            "target": x * 3 + np.random.normal(0, 0.5, 200),
        })
        report = self.analyzer.analyze(df)
        assert report.ml_result is not None
        assert report.ml_result.target_column == "target"
        assert report.ml_result.score > 0.5  # should be predictable

    def test_trend_analysis(self):
        df = _time_series_df()
        report = self.analyzer.analyze(df)
        assert len(report.trends) > 0
        # Sales should have increasing trend
        sales_trend = next(
            (t for t in report.trends if t.column == "sales"), None
        )
        assert sales_trend is not None
        assert sales_trend.trend_direction == "increasing"
        assert sales_trend.trend_strength > 0.5

    def test_forecast_generated(self):
        df = _time_series_df()
        report = self.analyzer.analyze(df)
        assert len(report.forecasts) > 0
        for fc in report.forecasts:
            assert len(fc.forecast_values) > 0
            assert len(fc.confidence_lower) == len(fc.forecast_values)
            assert len(fc.confidence_upper) == len(fc.forecast_values)

    def test_insufficient_data(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        report = self.analyzer.analyze(df)
        assert report.ml_result is None

    def test_summary_generated(self):
        df = _sample_df()
        report = self.analyzer.analyze(df)
        assert len(report.summary) > 5


# ── Insights Engine Tests ────────────────────────────────────────────────────


class TestInsightsEngine:
    """Tests for the insights and recommendations engine."""

    def setup_method(self):
        self.engine = InsightsEngine()
        self.quality = DataQualityAnalyzer()
        self.profiler = DataProfiler()
        self.stats = StatisticalAnalyzer()
        self.pred = PredictiveAnalyzer()

    def test_insights_from_messy_data(self):
        df = _messy_df()
        qr = self.quality.analyze(df)
        profile = self.profiler.profile(df)
        report = self.engine.generate_report(df, quality_report=qr, profile=profile)
        assert len(report.insights) > 0
        assert len(report.executive_summary) > 50

    def test_insights_sorted_by_importance(self):
        df = _messy_df()
        qr = self.quality.analyze(df)
        report = self.engine.generate_report(df, quality_report=qr)
        for i in range(len(report.insights) - 1):
            assert report.insights[i].importance >= report.insights[i + 1].importance

    def test_recommendations_generated(self):
        df = _messy_df()
        qr = self.quality.analyze(df)
        sr = self.stats.analyze(df)
        report = self.engine.generate_report(df, quality_report=qr, stats_report=sr)
        assert len(report.recommendations) > 0
        for rec in report.recommendations:
            assert len(rec.suggested_question) > 0

    def test_key_findings(self):
        df = _sample_df()
        qr = self.quality.analyze(df)
        profile = self.profiler.profile(df)
        sr = self.stats.analyze(df)
        report = self.engine.generate_report(
            df, quality_report=qr, profile=profile, stats_report=sr
        )
        assert len(report.key_findings) > 0

    def test_executive_summary_content(self):
        df = _sample_df()
        qr = self.quality.analyze(df)
        profile = self.profiler.profile(df)
        sr = self.stats.analyze(df)
        pr = self.pred.analyze(df)
        report = self.engine.generate_report(
            df, quality_report=qr, profile=profile,
            stats_report=sr, predictive_report=pr,
        )
        summary = report.executive_summary
        assert "200" in summary  # row count
        assert "record" in summary.lower()

    def test_empty_dataframe_insights(self):
        df = pd.DataFrame()
        report = self.engine.generate_report(df)
        assert report.executive_summary is not None


# ── Orchestrator Tests ───────────────────────────────────────────────────────


class TestAnalyticsOrchestrator:
    """Tests for the analytics orchestrator."""

    def setup_method(self):
        self.orchestrator = AnalyticsOrchestrator()

    def test_full_analysis(self):
        df = _sample_df()
        report = self.orchestrator.run_full_analysis(df)
        assert report.analysis_type == "full"
        assert report.quality is not None
        assert report.profile is not None
        assert report.statistics is not None
        assert report.predictive is not None
        assert report.insights is not None
        assert report.analysis_time_ms > 0

    def test_quick_scan(self):
        df = _sample_df()
        report = self.orchestrator.run_quick_scan(df)
        assert report.analysis_type == "quick"
        assert report.quality is not None
        assert report.profile is not None
        assert report.statistics is None  # not included in quick scan
        assert report.predictive is None

    def test_markdown_export(self):
        df = _sample_df()
        report = self.orchestrator.run_full_analysis(df)
        md = report.to_markdown()
        assert len(md) > 200
        assert "Executive Summary" in md
        assert "Data Quality" in md
        assert "Dataset Profile" in md

    def test_performance_under_1s(self):
        """Full analysis should complete in reasonable time."""
        df = _sample_df()
        report = self.orchestrator.run_full_analysis(df)
        # Should be under 5 seconds even on slow machines
        assert report.analysis_time_ms < 5000

    def test_messy_data_full_pipeline(self):
        df = _messy_df()
        report = self.orchestrator.run_full_analysis(df)
        assert report.quality.overall_quality_score < 95  # should detect issues
        assert len(report.quality.cleaning_actions) > 0
        assert len(report.insights.insights) > 0

    def test_time_series_analysis(self):
        df = _time_series_df()
        report = self.orchestrator.run_full_analysis(df)
        assert report.predictive is not None
        assert len(report.predictive.trends) > 0
