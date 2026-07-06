"""
Chart Selector — Intelligent, rule-based chart type selection engine.

Analyzes the user's question, dataset characteristics, and column metadata
to deterministically select the optimal chart type. This runs in <1ms and
avoids the latency of an LLM call for chart selection.

Selection Algorithm:
    1. Intent detection: Parse question for visualization keywords
    2. Column analysis: Classify relevant columns by dtype
    3. Rule matching: Apply statistical heuristics to select chart type
    4. Fallback: Return a sensible default if no rule matches

Design Principles:
    - Deterministic: same input → same output, always
    - Fast: no LLM calls, pure Python logic
    - Testable: every rule is unit-testable
    - Extensible: add new chart types by adding rules
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from backend.core.logging_config import get_logger
from backend.models.schemas import FileMetadata, ColumnInfo

logger = get_logger(__name__)


# ── Chart Type Enumeration ───────────────────────────────────────────────────

class ChartType(str, Enum):
    """Supported chart types for the visualization engine."""

    BAR = "bar"
    HORIZONTAL_BAR = "horizontal_bar"
    PIE = "pie"
    HISTOGRAM = "histogram"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    CORRELATION_MATRIX = "correlation_matrix"
    BOX_PLOT = "box_plot"
    VIOLIN_PLOT = "violin_plot"
    LINE = "line"
    AREA = "area"


# ── Chart Specification ──────────────────────────────────────────────────────

@dataclass
class ChartSpec:
    """
    Complete specification for rendering a chart.

    Produced by ChartSelector, consumed by ChartGenerator.
    Contains all information needed to render the chart without
    re-analyzing the data.

    Attributes:
        chart_type: The selected chart type.
        title: Auto-generated chart title.
        x_column: Column to use for x-axis (or groups).
        y_column: Column(s) to use for y-axis (or values).
        color_column: Optional column for color/hue encoding.
        x_label: X-axis label.
        y_label: Y-axis label.
        aggregation: Aggregation method ('mean', 'sum', 'count', etc.).
        sort_by: Sort method ('value', 'label', None).
        limit: Max number of items to display.
        orientation: 'vertical' or 'horizontal'.
        show_values: Whether to annotate bars/slices with values.
        confidence: Selection confidence (0.0–1.0).
        reasoning: Human-readable explanation of why this chart was chosen.
        extra_params: Additional chart-specific parameters.
    """

    chart_type: ChartType
    title: str = ""
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    y_columns: List[str] = field(default_factory=list)
    color_column: Optional[str] = None
    x_label: str = ""
    y_label: str = ""
    aggregation: str = "mean"
    sort_by: Optional[str] = "value"
    limit: int = 20
    orientation: str = "vertical"
    show_values: bool = True
    confidence: float = 0.0
    reasoning: str = ""
    extra_params: Dict[str, Any] = field(default_factory=dict)


# ── Intent Keywords ──────────────────────────────────────────────────────────

_INTENT_PATTERNS: Dict[str, List[str]] = {
    "distribution": [
        r"\bdistribution\b", r"\bspread\b", r"\bhistogram\b",
        r"\bfrequency\b", r"\bdensity\b", r"\bskew\b",
    ],
    "comparison": [
        r"\bcompare\b", r"\bcomparison\b", r"\bvs\b", r"\bversus\b",
        r"\btop\b", r"\bbottom\b", r"\branking\b", r"\brank\b",
        r"\bhighest\b", r"\blowest\b", r"\bmost\b", r"\bleast\b",
    ],
    "trend": [
        r"\btrend\b", r"\bover time\b", r"\bmonthly\b", r"\byearly\b",
        r"\bweekly\b", r"\bdaily\b", r"\btime series\b", r"\bgrowth\b",
        r"\bchange\b", r"\bprogression\b",
    ],
    "relationship": [
        r"\brelationship\b", r"\bcorrelat\w*\b", r"\bscatter\b",
        r"\bvs\b", r"\bassociat\w*\b", r"\bconnection\b",
        r"\bdepend\w*\b",
    ],
    "proportion": [
        r"\bproportion\b", r"\bpercentage\b", r"\bshare\b",
        r"\bcomposition\b", r"\bbreakdown\b", r"\bpie\b",
        r"\bratio\b",
    ],
    "correlation": [
        r"\bcorrelation\s*matrix\b", r"\bcorrelation\b",
        r"\bheatmap\b", r"\bcorrelations\b",
    ],
    "box_plot": [
        r"\bbox\s*plot\b", r"\bboxplot\b", r"\boutlier\b",
        r"\bquartile\b", r"\bmedian\b", r"\biqr\b",
    ],
    "violin": [
        r"\bviolin\b", r"\bdensity\s+comparison\b",
    ],
    "bar": [
        r"\bbar\s*chart\b", r"\bbar\s*graph\b", r"\bbar\s*plot\b",
    ],
    "line": [
        r"\bline\s*chart\b", r"\bline\s*graph\b", r"\bline\s*plot\b",
    ],
    "scatter": [
        r"\bscatter\s*plot\b", r"\bscatter\s*chart\b",
    ],
    "pie": [
        r"\bpie\s*chart\b", r"\bpie\s*graph\b",
    ],
    "heatmap": [
        r"\bheatmap\b", r"\bheat\s*map\b",
    ],
}


# ── Column Type Classification ───────────────────────────────────────────────

def _classify_columns(
    metadata: FileMetadata,
) -> Dict[str, List[ColumnInfo]]:
    """
    Classify columns into semantic categories based on dtype and cardinality.

    Returns:
        Dict with keys: 'numeric', 'categorical', 'datetime', 'text', 'boolean'.
    """
    result: Dict[str, List[ColumnInfo]] = {
        "numeric": [],
        "categorical": [],
        "datetime": [],
        "text": [],
        "boolean": [],
    }

    for col in metadata.columns:
        dtype_lower = col.dtype.lower()

        if any(t in dtype_lower for t in ("int", "float", "number", "decimal")):
            # High-cardinality numeric → stay numeric
            # Low-cardinality numeric → might be categorical
            if col.unique_count <= 10 and col.unique_count < metadata.row_count * 0.05:
                result["categorical"].append(col)
            else:
                result["numeric"].append(col)
        elif any(t in dtype_lower for t in ("datetime", "date", "timestamp")):
            result["datetime"].append(col)
        elif "bool" in dtype_lower:
            result["boolean"].append(col)
        elif "object" in dtype_lower or "string" in dtype_lower or "category" in dtype_lower:
            # Object columns: check cardinality to distinguish
            # categorical (low cardinality) vs free text (high cardinality)
            if col.unique_count <= 50:
                result["categorical"].append(col)
            else:
                result["text"].append(col)
        else:
            # Default: treat as categorical if low cardinality, text otherwise
            if col.unique_count <= 30:
                result["categorical"].append(col)
            else:
                result["text"].append(col)

    return result


def _detect_intent(question: str) -> List[Tuple[str, float]]:
    """
    Detect the user's visualization intent from question keywords.

    Returns a ranked list of (intent, confidence) tuples, sorted by
    confidence descending.
    """
    question_lower = question.lower()
    intents: List[Tuple[str, float]] = []

    for intent, patterns in _INTENT_PATTERNS.items():
        match_count = sum(
            1 for p in patterns
            if re.search(p, question_lower)
        )
        if match_count > 0:
            # More keyword matches → higher confidence
            confidence = min(match_count / len(patterns) + 0.3, 1.0)
            intents.append((intent, confidence))

    return sorted(intents, key=lambda x: x[1], reverse=True)


# ── Chart Selector ───────────────────────────────────────────────────────────

class ChartSelector:
    """
    Intelligent, rule-based chart type selection engine.

    Analyzes the question intent, column types, and data characteristics
    to select the optimal chart type. Deterministic and fast (<1ms).

    Usage:
        selector = ChartSelector()
        spec = selector.select(
            question="Show me revenue by category",
            file_metadata=metadata,
        )
        # spec.chart_type == ChartType.BAR
        # spec.x_column == "category"
        # spec.y_column == "revenue"

    Selection Priority:
        1. Explicit chart type mentioned in question
        2. Intent-based selection (trend → line, distribution → histogram)
        3. Data-driven heuristic (column types + cardinality)
    """

    def select(
        self,
        question: str,
        file_metadata: FileMetadata,
        x_col_hint: Optional[str] = None,
        y_col_hint: Optional[str] = None,
    ) -> ChartSpec:
        """
        Select the optimal chart type for a question and dataset.

        Args:
            question: The user's natural language question.
            file_metadata: Complete dataset metadata.
            x_col_hint: Optional explicit x-column override.
            y_col_hint: Optional explicit y-column override.

        Returns:
            A ChartSpec with the selected chart type and configuration.
        """
        # Step 1: Classify columns
        col_classes = _classify_columns(file_metadata)
        n_numeric = len(col_classes["numeric"])
        n_categorical = len(col_classes["categorical"])
        n_datetime = len(col_classes["datetime"])

        # Step 2: Detect intent from question
        intents = _detect_intent(question)
        primary_intent = intents[0][0] if intents else None
        intent_confidence = intents[0][1] if intents else 0.0

        logger.debug(
            "Chart selection: intents=%s, numeric=%d, categorical=%d, datetime=%d",
            intents[:3],
            n_numeric,
            n_categorical,
            n_datetime,
        )

        # Step 3: Apply selection rules (priority order)

        # Rule 0: Explicit chart type request
        spec = self._check_explicit_request(question, col_classes, file_metadata)
        if spec:
            return spec

        # Rule 1: Correlation matrix
        if primary_intent == "correlation" and n_numeric >= 2:
            return self._build_correlation_spec(col_classes, file_metadata)

        # Rule 2: Heatmap (explicit request)
        if primary_intent == "heatmap":
            return self._build_heatmap_spec(col_classes, file_metadata)

        # Rule 3: Violin plot (explicit request)
        if primary_intent == "violin" and n_numeric >= 1:
            return self._build_violin_spec(col_classes, file_metadata, question)

        # Rule 4: Box plot (explicit or outlier detection)
        if primary_intent == "box_plot" and n_numeric >= 1:
            return self._build_box_spec(col_classes, file_metadata, question)

        # Rule 5: Time series → Line chart
        if primary_intent == "trend" and n_datetime >= 1 and n_numeric >= 1:
            return self._build_line_spec(col_classes, file_metadata, question)

        # Rule 6: Proportion → Pie chart (only if few categories)
        if primary_intent == "proportion" and n_categorical >= 1:
            cat_col = col_classes["categorical"][0]
            if cat_col.unique_count <= 7:
                return self._build_pie_spec(col_classes, file_metadata, question)

        # Rule 7: Relationship between 2 numeric → Scatter
        if primary_intent == "relationship" and n_numeric >= 2:
            return self._build_scatter_spec(col_classes, file_metadata, question)

        # Rule 8: Distribution of single numeric → Histogram
        if primary_intent == "distribution" and n_numeric >= 1:
            return self._build_histogram_spec(col_classes, file_metadata, question)

        # Rule 9: Comparison → Bar chart
        if primary_intent == "comparison" and n_categorical >= 1 and n_numeric >= 1:
            return self._build_bar_spec(col_classes, file_metadata, question)

        # ── Data-driven fallback rules (no clear intent) ────────────
        # Fallback A: One categorical + one numeric → Bar
        if n_categorical >= 1 and n_numeric >= 1:
            cat_col = col_classes["categorical"][0]
            if cat_col.unique_count <= 15:
                return self._build_bar_spec(col_classes, file_metadata, question)

        # Fallback B: Time + numeric → Line
        if n_datetime >= 1 and n_numeric >= 1:
            return self._build_line_spec(col_classes, file_metadata, question)

        # Fallback C: Two numeric → Scatter
        if n_numeric >= 2:
            return self._build_scatter_spec(col_classes, file_metadata, question)

        # Fallback D: Single numeric → Histogram
        if n_numeric >= 1:
            return self._build_histogram_spec(col_classes, file_metadata, question)

        # Fallback E: All categorical → Bar (count-based)
        if n_categorical >= 1:
            return self._build_bar_spec(
                col_classes, file_metadata, question, aggregation="count"
            )

        # Last resort: correlation matrix if enough numeric columns
        if n_numeric >= 3:
            return self._build_correlation_spec(col_classes, file_metadata)

        # Absolute fallback
        return ChartSpec(
            chart_type=ChartType.BAR,
            title="Data Overview",
            confidence=0.2,
            reasoning="Default chart type — could not determine optimal visualization.",
        )

    # ── Explicit Chart Request Detection ─────────────────────────────────

    def _check_explicit_request(
        self,
        question: str,
        col_classes: Dict[str, List[ColumnInfo]],
        metadata: FileMetadata,
    ) -> Optional[ChartSpec]:
        """Check if the user explicitly requested a chart type."""
        q = question.lower()

        if re.search(r"\bpie\s*(chart|graph|plot)?\b", q):
            return self._build_pie_spec(col_classes, metadata, question)
        if re.search(r"\bscatter\s*(plot|chart|graph)?\b", q):
            return self._build_scatter_spec(col_classes, metadata, question)
        if re.search(r"\bhistogram\b", q):
            return self._build_histogram_spec(col_classes, metadata, question)
        if re.search(r"\bviolin\s*(plot|chart)?\b", q):
            return self._build_violin_spec(col_classes, metadata, question)
        if re.search(r"\bbox\s*(plot|chart)?\b", q):
            return self._build_box_spec(col_classes, metadata, question)
        if re.search(r"\b(heatmap|heat\s*map)\b", q):
            return self._build_heatmap_spec(col_classes, metadata)
        if re.search(r"\bcorrelation\s*(matrix|plot|chart|heatmap)?\b", q):
            return self._build_correlation_spec(col_classes, metadata)
        if re.search(r"\bline\s*(chart|graph|plot)?\b", q):
            return self._build_line_spec(col_classes, metadata, question)
        if re.search(r"\bbar\s*(chart|graph|plot)?\b", q):
            return self._build_bar_spec(col_classes, metadata, question)

        return None

    # ── Chart-Specific Spec Builders ─────────────────────────────────────

    def _build_bar_spec(
        self,
        col_classes: Dict[str, List[ColumnInfo]],
        metadata: FileMetadata,
        question: str,
        aggregation: str = "mean",
    ) -> ChartSpec:
        """Build a bar chart specification."""
        cat_cols = col_classes["categorical"]
        num_cols = col_classes["numeric"]

        x_col = cat_cols[0].name if cat_cols else None
        y_col = num_cols[0].name if num_cols else None

        # Determine if horizontal bars are better (long labels or many categories)
        n_cats = cat_cols[0].unique_count if cat_cols else 0
        orientation = "horizontal" if n_cats > 10 else "vertical"
        chart_type = ChartType.HORIZONTAL_BAR if orientation == "horizontal" else ChartType.BAR

        # Auto-detect aggregation from question
        q = question.lower()
        if any(w in q for w in ("total", "sum")):
            aggregation = "sum"
        elif any(w in q for w in ("count", "how many", "number of")):
            aggregation = "count"
        elif any(w in q for w in ("average", "mean", "avg")):
            aggregation = "mean"
        elif any(w in q for w in ("max", "maximum", "highest")):
            aggregation = "max"
        elif any(w in q for w in ("min", "minimum", "lowest")):
            aggregation = "min"

        title = f"{aggregation.title()} of {y_col or 'values'} by {x_col or 'category'}"

        return ChartSpec(
            chart_type=chart_type,
            title=title.title(),
            x_column=x_col,
            y_column=y_col,
            x_label=x_col or "",
            y_label=y_col or "Count",
            aggregation=aggregation,
            orientation=orientation,
            limit=20,
            show_values=n_cats <= 15,
            confidence=0.8 if cat_cols and num_cols else 0.5,
            reasoning=(
                f"Bar chart selected: {n_cats} categories in '{x_col}' vs "
                f"numeric '{y_col}' with {aggregation} aggregation."
            ),
        )

    def _build_pie_spec(
        self,
        col_classes: Dict[str, List[ColumnInfo]],
        metadata: FileMetadata,
        question: str,
    ) -> ChartSpec:
        """Build a pie chart specification."""
        cat_cols = col_classes["categorical"]
        num_cols = col_classes["numeric"]

        x_col = cat_cols[0].name if cat_cols else None
        y_col = num_cols[0].name if num_cols else None
        n_cats = cat_cols[0].unique_count if cat_cols else 0

        # Pie charts should have few slices for readability
        limit = min(n_cats, 7)

        return ChartSpec(
            chart_type=ChartType.PIE,
            title=f"Distribution of {y_col or 'values'} by {x_col or 'category'}",
            x_column=x_col,
            y_column=y_col,
            aggregation="sum",
            limit=limit,
            show_values=True,
            confidence=0.7 if n_cats <= 7 else 0.4,
            reasoning=(
                f"Pie chart selected: {n_cats} categories — shows proportional "
                f"breakdown of '{y_col}' across '{x_col}'."
            ),
            extra_params={"show_pct": True, "explode_top": True},
        )

    def _build_histogram_spec(
        self,
        col_classes: Dict[str, List[ColumnInfo]],
        metadata: FileMetadata,
        question: str,
    ) -> ChartSpec:
        """Build a histogram specification."""
        num_cols = col_classes["numeric"]
        col = num_cols[0] if num_cols else None
        col_name = col.name if col else "values"

        # Auto-determine bin count using Sturges' rule approximation
        n_bins = min(max(int(metadata.row_count ** 0.5), 10), 50)

        return ChartSpec(
            chart_type=ChartType.HISTOGRAM,
            title=f"Distribution of {col_name}",
            x_column=col_name,
            x_label=col_name,
            y_label="Frequency",
            confidence=0.85 if num_cols else 0.3,
            reasoning=(
                f"Histogram selected: shows the frequency distribution of "
                f"numeric column '{col_name}' across {n_bins} bins."
            ),
            extra_params={"bins": n_bins, "show_kde": True},
        )

    def _build_scatter_spec(
        self,
        col_classes: Dict[str, List[ColumnInfo]],
        metadata: FileMetadata,
        question: str,
    ) -> ChartSpec:
        """Build a scatter plot specification."""
        num_cols = col_classes["numeric"]
        cat_cols = col_classes["categorical"]

        x_col = num_cols[0].name if len(num_cols) >= 1 else None
        y_col = num_cols[1].name if len(num_cols) >= 2 else None
        color_col = cat_cols[0].name if cat_cols and cat_cols[0].unique_count <= 10 else None

        return ChartSpec(
            chart_type=ChartType.SCATTER,
            title=f"{y_col or '?'} vs {x_col or '?'}",
            x_column=x_col,
            y_column=y_col,
            color_column=color_col,
            x_label=x_col or "",
            y_label=y_col or "",
            confidence=0.85 if len(num_cols) >= 2 else 0.3,
            reasoning=(
                f"Scatter plot selected: shows relationship between "
                f"'{x_col}' and '{y_col}'"
                + (f", colored by '{color_col}'" if color_col else "")
                + "."
            ),
            extra_params={"alpha": 0.6, "show_trendline": True},
        )

    def _build_heatmap_spec(
        self,
        col_classes: Dict[str, List[ColumnInfo]],
        metadata: FileMetadata,
    ) -> ChartSpec:
        """Build a heatmap specification."""
        num_cols = col_classes["numeric"]
        cat_cols = col_classes["categorical"]

        y_cols = [c.name for c in num_cols]
        x_col = cat_cols[0].name if cat_cols else None

        return ChartSpec(
            chart_type=ChartType.HEATMAP,
            title="Heatmap",
            x_column=x_col,
            y_columns=y_cols,
            confidence=0.7,
            reasoning=(
                f"Heatmap selected: visualizes intensity patterns across "
                f"{len(y_cols)} numeric columns."
            ),
            extra_params={"annotate": len(y_cols) <= 10},
        )

    def _build_correlation_spec(
        self,
        col_classes: Dict[str, List[ColumnInfo]],
        metadata: FileMetadata,
    ) -> ChartSpec:
        """Build a correlation matrix specification."""
        num_cols = col_classes["numeric"]
        y_cols = [c.name for c in num_cols]

        return ChartSpec(
            chart_type=ChartType.CORRELATION_MATRIX,
            title="Correlation Matrix",
            y_columns=y_cols,
            confidence=0.9 if len(num_cols) >= 3 else 0.5,
            reasoning=(
                f"Correlation matrix selected: shows pairwise correlations "
                f"between {len(num_cols)} numeric columns."
            ),
            extra_params={
                "annotate": len(num_cols) <= 12,
                "method": "pearson",
                "mask_upper": True,
            },
        )

    def _build_box_spec(
        self,
        col_classes: Dict[str, List[ColumnInfo]],
        metadata: FileMetadata,
        question: str,
    ) -> ChartSpec:
        """Build a box plot specification."""
        num_cols = col_classes["numeric"]
        cat_cols = col_classes["categorical"]

        y_col = num_cols[0].name if num_cols else None
        x_col = cat_cols[0].name if cat_cols and cat_cols[0].unique_count <= 10 else None

        return ChartSpec(
            chart_type=ChartType.BOX_PLOT,
            title=f"Distribution of {y_col or 'values'}"
                  + (f" by {x_col}" if x_col else ""),
            x_column=x_col,
            y_column=y_col,
            x_label=x_col or "",
            y_label=y_col or "",
            confidence=0.8 if num_cols else 0.3,
            reasoning=(
                f"Box plot selected: shows median, quartiles, and outliers "
                f"for '{y_col}'"
                + (f" grouped by '{x_col}'" if x_col else "")
                + "."
            ),
            extra_params={"show_points": metadata.row_count <= 500},
        )

    def _build_violin_spec(
        self,
        col_classes: Dict[str, List[ColumnInfo]],
        metadata: FileMetadata,
        question: str,
    ) -> ChartSpec:
        """Build a violin plot specification."""
        num_cols = col_classes["numeric"]
        cat_cols = col_classes["categorical"]

        y_col = num_cols[0].name if num_cols else None
        x_col = cat_cols[0].name if cat_cols and cat_cols[0].unique_count <= 10 else None

        return ChartSpec(
            chart_type=ChartType.VIOLIN_PLOT,
            title=f"Distribution of {y_col or 'values'}"
                  + (f" by {x_col}" if x_col else ""),
            x_column=x_col,
            y_column=y_col,
            x_label=x_col or "",
            y_label=y_col or "",
            confidence=0.8 if num_cols else 0.3,
            reasoning=(
                f"Violin plot selected: shows probability density and "
                f"distribution shape for '{y_col}'"
                + (f" across '{x_col}' groups" if x_col else "")
                + "."
            ),
            extra_params={"inner": "box", "split": False},
        )

    def _build_line_spec(
        self,
        col_classes: Dict[str, List[ColumnInfo]],
        metadata: FileMetadata,
        question: str,
    ) -> ChartSpec:
        """Build a line chart specification."""
        datetime_cols = col_classes["datetime"]
        num_cols = col_classes["numeric"]
        cat_cols = col_classes["categorical"]

        x_col = datetime_cols[0].name if datetime_cols else (
            cat_cols[0].name if cat_cols else None
        )
        y_col = num_cols[0].name if num_cols else None

        return ChartSpec(
            chart_type=ChartType.LINE,
            title=f"{y_col or 'values'} Over Time",
            x_column=x_col,
            y_column=y_col,
            x_label=x_col or "Time",
            y_label=y_col or "Value",
            confidence=0.85 if datetime_cols and num_cols else 0.5,
            reasoning=(
                f"Line chart selected: shows trend of '{y_col}' over "
                f"'{x_col}' (time series)."
            ),
            extra_params={"show_markers": metadata.row_count <= 100},
        )
