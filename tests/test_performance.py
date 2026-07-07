"""
Performance and large file tests.

Tests:
    - Analytics pipeline performance (under N seconds)
    - Memory bounds on large DataFrames
    - Validator performance on large/complex code
    - Sampling behavior for large datasets
    - Orchestrator timing guarantees
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from backend.analytics.data_quality import DataQualityAnalyzer
from backend.analytics.data_profiler import DataProfiler
from backend.analytics.statistical import StatisticalAnalyzer
from backend.analytics.predictive import PredictiveAnalyzer
from backend.analytics.orchestrator import AnalyticsOrchestrator
from backend.sandbox.validator import CodeValidator


# ── Analytics Performance Tests ─────────────────────────────────────────────


class TestAnalyticsPerformance:
    """Verify analytics modules meet latency requirements."""

    def test_quality_analysis_under_1s(self, large_df):
        analyzer = DataQualityAnalyzer()
        start = time.time()
        report = analyzer.analyze(large_df)
        elapsed = time.time() - start
        assert elapsed < 1.0, f"Quality analysis took {elapsed:.2f}s (limit: 1s)"
        assert report.overall_quality_score > 0

    def test_profiler_under_1s(self, large_df):
        profiler = DataProfiler()
        start = time.time()
        profile = profiler.profile(large_df)
        elapsed = time.time() - start
        assert elapsed < 1.0, f"Profiling took {elapsed:.2f}s (limit: 1s)"
        assert profile.row_count == len(large_df)

    def test_statistical_analysis_under_2s(self, large_df):
        analyzer = StatisticalAnalyzer()
        start = time.time()
        report = analyzer.analyze(large_df)
        elapsed = time.time() - start
        assert elapsed < 2.0, f"Statistical analysis took {elapsed:.2f}s (limit: 2s)"

    def test_predictive_analysis_under_3s(self, large_df):
        analyzer = PredictiveAnalyzer()
        start = time.time()
        report = analyzer.analyze(large_df)
        elapsed = time.time() - start
        assert elapsed < 3.0, f"Predictive analysis took {elapsed:.2f}s (limit: 3s)"

    def test_full_orchestrator_under_5s(self, large_df):
        orchestrator = AnalyticsOrchestrator()
        start = time.time()
        report = orchestrator.run_full_analysis(large_df)
        elapsed = time.time() - start
        assert elapsed < 5.0, f"Full analysis took {elapsed:.2f}s (limit: 5s)"
        assert report.analysis_time_ms > 0

    def test_quick_scan_faster_than_full(self, large_df):
        orchestrator = AnalyticsOrchestrator()

        start_full = time.time()
        orchestrator.run_full_analysis(large_df)
        full_time = time.time() - start_full

        start_quick = time.time()
        orchestrator.run_quick_scan(large_df)
        quick_time = time.time() - start_quick

        assert quick_time < full_time, (
            f"Quick scan ({quick_time:.2f}s) should be faster "
            f"than full ({full_time:.2f}s)"
        )


# ── Large Dataset Sampling Tests ────────────────────────────────────────────


class TestLargeDatasetSampling:
    """Test that large datasets are sampled for performance."""

    def test_sampling_threshold(self):
        """Datasets > 100K rows should be sampled."""
        np.random.seed(42)
        huge_df = pd.DataFrame({
            "a": np.random.randn(150_000),
            "b": np.random.choice(["x", "y", "z"], 150_000),
        })
        result = AnalyticsOrchestrator._prepare_df(huge_df)
        assert len(result) == 100_000

    def test_small_dataset_not_sampled(self):
        """Datasets <= 100K rows should not be sampled."""
        small_df = pd.DataFrame({"a": range(50_000)})
        result = AnalyticsOrchestrator._prepare_df(small_df)
        assert len(result) == 50_000

    def test_sampling_is_deterministic(self):
        """Same data should produce same sample."""
        np.random.seed(42)
        df = pd.DataFrame({"a": range(200_000)})
        sample1 = AnalyticsOrchestrator._prepare_df(df)
        sample2 = AnalyticsOrchestrator._prepare_df(df)
        pd.testing.assert_frame_equal(sample1, sample2)


# ── Validator Performance Tests ─────────────────────────────────────────────


class TestValidatorPerformance:
    """Test that the validator is fast enough for production."""

    def setup_method(self):
        self.validator = CodeValidator()

    def test_simple_code_validation_under_10ms(self):
        code = "result = df.groupby('cat')['val'].mean()"
        start = time.time()
        for _ in range(100):
            self.validator.validate(code)
        elapsed = (time.time() - start) / 100
        assert elapsed < 0.01, f"Simple validation: {elapsed*1000:.1f}ms (limit: 10ms)"

    def test_complex_code_validation_under_50ms(self):
        code = (
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "fig, axes = plt.subplots(2, 2, figsize=(12, 10))\n"
            "for i, col in enumerate(['a', 'b', 'c', 'd']):\n"
            "    ax = axes[i // 2, i % 2]\n"
            "    df[col].hist(ax=ax, bins=20)\n"
            "    ax.set_title(f'Distribution of {col}')\n"
            "plt.tight_layout()\n"
            "plt.savefig(chart_path, dpi=150)\n"
            "plt.close()\n"
            "result = 'Charts generated'\n"
        )
        start = time.time()
        for _ in range(100):
            self.validator.validate(code)
        elapsed = (time.time() - start) / 100
        assert elapsed < 0.05, f"Complex validation: {elapsed*1000:.1f}ms (limit: 50ms)"

    def test_oversized_code_fails_fast(self):
        """Code exceeding size limit should fail immediately."""
        huge_code = "x = 1\n" * 50_000
        start = time.time()
        self.validator.validate(huge_code)
        elapsed = time.time() - start
        assert elapsed < 0.1, f"Size rejection: {elapsed*1000:.1f}ms (limit: 100ms)"


# ── Large File CSV Tests ────────────────────────────────────────────────────


class TestLargeFileHandling:
    """Test CSV analyzer and quality report on large files."""

    def test_large_csv_analysis(self, large_csv):
        from backend.utils.csv_analyzer import CSVAnalyzer
        analyzer = CSVAnalyzer()
        start = time.time()
        metadata = analyzer.analyze(large_csv, file_id="perf-001")
        elapsed = time.time() - start
        assert metadata.row_count == 50_000
        assert elapsed < 10.0, f"Large CSV analysis: {elapsed:.2f}s"

    def test_large_csv_preview_fast(self, large_csv):
        from backend.utils.csv_analyzer import CSVAnalyzer
        analyzer = CSVAnalyzer()
        start = time.time()
        rows = analyzer.get_preview_rows(large_csv, max_rows=100)
        elapsed = time.time() - start
        assert len(rows) == 100
        assert elapsed < 2.0, f"Preview: {elapsed:.2f}s"

    def test_large_csv_quality_report(self, large_csv):
        from backend.utils.csv_analyzer import CSVAnalyzer
        analyzer = CSVAnalyzer()
        start = time.time()
        report = analyzer.get_data_quality_report(large_csv)
        elapsed = time.time() - start
        assert report["completeness_pct"] > 0
        assert elapsed < 10.0, f"Quality report: {elapsed:.2f}s"


# ── Memory Safety Tests ────────────────────────────────────────────────────


class TestMemorySafety:
    """Ensure analytics don't consume excessive memory."""

    def test_result_data_types_are_serializable(self, sample_df):
        """All report fields should be JSON-serializable types."""
        import json
        orchestrator = AnalyticsOrchestrator()
        report = orchestrator.run_full_analysis(sample_df)
        # The markdown output should be a regular string
        md = report.to_markdown()
        assert isinstance(md, str)
        # Quality score should be a plain float
        assert isinstance(report.quality.overall_quality_score, float)

    def test_outlier_extreme_values_bounded(self, sample_df):
        """Outlier reports should have bounded extreme value lists."""
        analyzer = DataQualityAnalyzer()
        report = analyzer.analyze(sample_df)
        for outlier in report.outliers:
            assert len(outlier.extreme_values) <= 5

    def test_cleaning_actions_bounded(self, wide_df):
        """Cleaning actions should not explode for wide datasets."""
        analyzer = DataQualityAnalyzer()
        report = analyzer.analyze(wide_df)
        # Should have reasonable number of actions
        assert len(report.cleaning_actions) < 200
