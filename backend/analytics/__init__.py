"""
Premium Analytics Engine — AI-powered data analysis without LLM dependency.

This module provides deterministic, production-grade analytics that run
in <2 seconds on datasets up to 100K rows. All computations are pure
Python/pandas/numpy/scipy — no LLM calls needed.

Modules:
    - DataQualityAnalyzer: Missing values, duplicates, outliers, auto-cleaning
    - DataProfiler: Full column profiling, type inference, quality scoring
    - StatisticalAnalyzer: Correlations, feature importance, distributions
    - PredictiveAnalyzer: Basic ML, trend analysis, forecasting
    - InsightsEngine: Auto-insights, recommendations, NL summaries
    - AnalyticsOrchestrator: Coordinates all modules into a unified report
"""

from backend.analytics.data_quality import DataQualityAnalyzer
from backend.analytics.data_profiler import DataProfiler
from backend.analytics.statistical import StatisticalAnalyzer
from backend.analytics.predictive import PredictiveAnalyzer
from backend.analytics.insights_engine import InsightsEngine
from backend.analytics.orchestrator import AnalyticsOrchestrator

__all__ = [
    "DataQualityAnalyzer",
    "DataProfiler",
    "StatisticalAnalyzer",
    "PredictiveAnalyzer",
    "InsightsEngine",
    "AnalyticsOrchestrator",
]
