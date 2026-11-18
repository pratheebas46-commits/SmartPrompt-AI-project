"""Module 9: Interactive Visualization Engine for SmartPrompt AI.

Generates interactive Plotly Figure objects from Module 6 AnalysisResult,
Module 4 AnalyticalTaskPlan, and Module 7 AnomalyReport outputs.
Completely decoupled from Streamlit; handles empty and unsuitable data safely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from config.logging_config import get_logger
from modules.analysis_engine import AnalysisResult, _safe_numeric
from modules.anomaly_detector import AnomalyReport
from modules.intent_detector import AnalyticalIntent
from modules.prompt_optimizer import AnalyticalTaskPlan

logger = get_logger(__name__)

SUPPORTED_CHART_TYPES: frozenset[str] = frozenset(
    {"bar", "horizontal_bar", "line", "pie", "histogram", "scatter", "heatmap"}
)


# ---------------------------------------------------------------------------
# Exceptions & Structured Result
# ---------------------------------------------------------------------------


class VisualizationError(Exception):
    """Base exception for Plotly visualization generation errors."""


@dataclass(frozen=True)
class VisualizationResult:
    """Container pairing a generated Plotly Figure with metadata and suitability status."""

    is_renderable: bool
    chart_type: str
    title: str
    figure: go.Figure | None = None
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_renderable": self.is_renderable,
            "chart_type": self.chart_type,
            "title": self.title,
            "reason": self.reason,
            "metadata": self.metadata,
            "figure_json": self.figure.to_json() if self.figure is not None else None,
        }


# ---------------------------------------------------------------------------
# Individual Plotly Chart Builders
# ---------------------------------------------------------------------------


def _apply_common_layout(
    fig: go.Figure,
    title: str,
    x_label: str = "",
    y_label: str = "",
) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
        margin={"l": 50, "r": 30, "t": 60, "b": 50},
        hovermode="closest",
    )
    return fig


def create_bar_chart(
    categories: list[Any],
    values: list[Any],
    title: str,
    x_label: str,
    y_label: str,
    horizontal: bool = False,
) -> VisualizationResult:
    """Build a vertical or horizontal Plotly bar chart, handling empty/invalid data safely."""
    pairs = [(c, v) for c, v in zip(categories, values) if c is not None and v is not None]
    if not pairs:
        return VisualizationResult(
            is_renderable=False,
            chart_type="horizontal_bar" if horizontal else "bar",
            title=title,
            figure=None,
            reason="No valid non-null category/value pairs available for bar chart.",
        )

    cats, vals = zip(*pairs)
    if horizontal:
        fig = go.Figure(
            go.Bar(
                x=list(vals),
                y=[str(c) for c in cats],
                orientation="h",
                marker_color="#2563EB",
            )
        )
        fig.update_layout(yaxis={"autorange": "reversed"})
        _apply_common_layout(fig, title=title, x_label=y_label, y_label=x_label)
        chart_type = "horizontal_bar"
    else:
        fig = go.Figure(
            go.Bar(
                x=[str(c) for c in cats],
                y=list(vals),
                marker_color="#2563EB",
            )
        )
        _apply_common_layout(fig, title=title, x_label=x_label, y_label=y_label)
        chart_type = "bar"

    return VisualizationResult(is_renderable=True, chart_type=chart_type, title=title, figure=fig)


def create_line_chart(
    x_values: list[Any],
    y_values: list[Any],
    title: str,
    x_label: str,
    y_label: str,
) -> VisualizationResult:
    """Build a Plotly line chart for trend and time-series visualization."""
    pairs = [(x, y) for x, y in zip(x_values, y_values) if x is not None and y is not None]
    if len(pairs) < 2:
        return VisualizationResult(
            is_renderable=False,
            chart_type="line",
            title=title,
            figure=None,
            reason="Line chart requires at least 2 valid data points.",
        )

    xs, ys = zip(*pairs)
    fig = go.Figure(
        go.Scatter(
            x=list(xs),
            y=list(ys),
            mode="lines+markers",
            line={"color": "#0EA5E9", "width": 2.5},
            marker={"size": 7},
        )
    )
    _apply_common_layout(fig, title=title, x_label=x_label, y_label=y_label)
    return VisualizationResult(is_renderable=True, chart_type="line", title=title, figure=fig)


def create_pie_chart(
    labels: list[Any],
    values: list[Any],
    title: str,
    max_slices: int = 8,
) -> VisualizationResult:
    """Build a Plotly pie chart only when part-of-whole data is non-negative and concise."""
    pairs = [(lbl, val) for lbl, val in zip(labels, values) if lbl is not None and val is not None]
    if len(pairs) < 2:
        return VisualizationResult(
            is_renderable=False,
            chart_type="pie",
            title=title,
            figure=None,
            reason="Pie chart requires at least 2 categories.",
        )
    if len(pairs) > max_slices:
        return VisualizationResult(
            is_renderable=False,
            chart_type="pie",
            title=title,
            figure=None,
            reason=f"Pie chart is unsuitable for {len(pairs)} categories (max {max_slices}); use bar chart instead.",
        )

    clean_labels: list[str] = []
    clean_vals: list[float] = []
    for lbl, val in pairs:
        try:
            num = float(val)
        except (TypeError, ValueError):
            return VisualizationResult(
                is_renderable=False,
                chart_type="pie",
                title=title,
                figure=None,
                reason="Pie chart values must be numeric.",
            )
        if num < 0:
            return VisualizationResult(
                is_renderable=False,
                chart_type="pie",
                title=title,
                figure=None,
                reason="Pie chart is unsuitable for negative values.",
            )
        clean_labels.append(str(lbl))
        clean_vals.append(num)

    if sum(clean_vals) <= 0:
        return VisualizationResult(
            is_renderable=False,
            chart_type="pie",
            title=title,
            figure=None,
            reason="Pie chart requires a positive total sum.",
        )

    fig = go.Figure(
        go.Pie(
            labels=clean_labels,
            values=clean_vals,
            hole=0.3,
            textinfo="label+percent",
        )
    )
    fig.update_layout(title={"text": title, "x": 0.02, "xanchor": "left"}, template="plotly_white")
    return VisualizationResult(is_renderable=True, chart_type="pie", title=title, figure=fig)


def create_histogram(
    values: list[float] | pd.Series | None = None,
    bin_labels: list[str] | None = None,
    bin_counts: list[int] | None = None,
    title: str = "Distribution Histogram",
    x_label: str = "Value",
    y_label: str = "Count",
    nbins: int = 20,
) -> VisualizationResult:
    """Build a Plotly histogram from raw numeric values or pre-binned counts."""
    if bin_labels is not None and bin_counts is not None:
        if not bin_labels or len(bin_labels) != len(bin_counts) or sum(bin_counts) == 0:
            return VisualizationResult(
                is_renderable=False,
                chart_type="histogram",
                title=title,
                figure=None,
                reason="Histogram bins are empty.",
            )
        fig = go.Figure(go.Bar(x=bin_labels, y=bin_counts, marker_color="#6366F1"))
        _apply_common_layout(fig, title=title, x_label=x_label, y_label=y_label)
        return VisualizationResult(is_renderable=True, chart_type="histogram", title=title, figure=fig)

    if values is None:
        return VisualizationResult(
            is_renderable=False,
            chart_type="histogram",
            title=title,
            figure=None,
            reason="No numeric values provided for histogram.",
        )

    series = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if series.empty:
        return VisualizationResult(
            is_renderable=False,
            chart_type="histogram",
            title=title,
            figure=None,
            reason="All values are null or non-numeric for histogram.",
        )

    fig = go.Figure(go.Histogram(x=series.tolist(), nbinsx=nbins, marker_color="#6366F1"))
    _apply_common_layout(fig, title=title, x_label=x_label, y_label=y_label)
    return VisualizationResult(is_renderable=True, chart_type="histogram", title=title, figure=fig)


def create_scatter_plot(
    x_values: list[Any],
    y_values: list[Any],
    title: str,
    x_label: str,
    y_label: str,
    labels: list[int] | None = None,
) -> VisualizationResult:
    """Build a Plotly scatter plot (optionally coloring anomaly labels: 1=normal, -1=anomaly)."""
    pairs = [
        (x, y, labels[idx] if labels and idx < len(labels) else 1)
        for idx, (x, y) in enumerate(zip(x_values, y_values))
        if x is not None and y is not None
    ]
    if len(pairs) < 2:
        return VisualizationResult(
            is_renderable=False,
            chart_type="scatter",
            title=title,
            figure=None,
            reason="Scatter plot requires at least 2 valid numeric pairs.",
        )

    if labels is not None:
        normal_pts = [(x, y) for x, y, lbl in pairs if lbl != -1]
        anom_pts = [(x, y) for x, y, lbl in pairs if lbl == -1]
        fig = go.Figure()
        if normal_pts:
            fig.add_trace(
                go.Scatter(
                    x=[p[0] for p in normal_pts],
                    y=[p[1] for p in normal_pts],
                    mode="markers",
                    name="Normal",
                    marker={"color": "#2563EB", "size": 7, "opacity": 0.75},
                )
            )
        if anom_pts:
            fig.add_trace(
                go.Scatter(
                    x=[p[0] for p in anom_pts],
                    y=[p[1] for p in anom_pts],
                    mode="markers",
                    name="Anomaly",
                    marker={"color": "#DC2626", "size": 10, "symbol": "diamond"},
                )
            )
    else:
        fig = go.Figure(
            go.Scatter(
                x=[p[0] for p in pairs],
                y=[p[1] for p in pairs],
                mode="markers",
                marker={"color": "#2563EB", "size": 8, "opacity": 0.8},
            )
        )

    _apply_common_layout(fig, title=title, x_label=x_label, y_label=y_label)
    return VisualizationResult(is_renderable=True, chart_type="scatter", title=title, figure=fig)


def create_heatmap(
    x_labels: list[str],
    y_labels: list[str],
    z_matrix: list[list[float | None]],
    title: str = "Correlation Heatmap",
    x_label: str = "Features",
    y_label: str = "Features",
) -> VisualizationResult:
    """Build a Plotly heatmap for correlation matrices or 2D tabular grids."""
    if not x_labels or not y_labels or not z_matrix:
        return VisualizationResult(
            is_renderable=False,
            chart_type="heatmap",
            title=title,
            figure=None,
            reason="Heatmap matrix is empty.",
        )

    fig = go.Figure(
        go.Heatmap(
            z=z_matrix,
            x=x_labels,
            y=y_labels,
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
        )
    )
    _apply_common_layout(fig, title=title, x_label=x_label, y_label=y_label)
    return VisualizationResult(is_renderable=True, chart_type="heatmap", title=title, figure=fig)


# ---------------------------------------------------------------------------
# Automatic Chart Selection & Analysis Integration
# ---------------------------------------------------------------------------


def select_chart_type(
    analysis_result: AnalysisResult,
    plan: AnalyticalTaskPlan | None = None,
    df: pd.DataFrame | None = None,
    prefer_pie: bool = False,
) -> str:
    """Automatically choose the most appropriate chart type for an analytical result."""
    intent = analysis_result.intent
    table = analysis_result.result_table

    if intent == AnalyticalIntent.TREND_ANALYSIS.value:
        return "line"
    if intent == AnalyticalIntent.DISTRIBUTION_ANALYSIS.value:
        return "histogram"
    if intent == AnalyticalIntent.RANKING.value:
        return "horizontal_bar"
    if intent == AnalyticalIntent.CORRELATION.value:
        cols = analysis_result.numerical_evidence.columns_used
        if len(cols) == 2 and df is not None and all(c in df.columns for c in cols):
            return "scatter"
        return "heatmap"
    if intent == AnalyticalIntent.ANOMALY_ANALYSIS.value:
        return "scatter"
    if intent in {AnalyticalIntent.AGGREGATION.value, AnalyticalIntent.COMPARISON.value}:
        agg = analysis_result.summary_metrics.get("aggregation")
        if (
            prefer_pie
            and agg in {"sum", "count"}
            and 2 <= len(table) <= 6
            and all((row.get("primary_value") or 0) >= 0 for row in table)
        ):
            return "pie"
        return "bar"
    if intent == AnalyticalIntent.MISSING_VALUE_ANALYSIS.value:
        return "bar"
    if plan and plan.recommended_chart_type in SUPPORTED_CHART_TYPES:
        return plan.recommended_chart_type
    return "bar"


def generate_chart_for_analysis(
    analysis_result: AnalysisResult,
    plan: AnalyticalTaskPlan | None = None,
    df: pd.DataFrame | None = None,
    chart_type_override: str | None = None,
    prefer_pie: bool = False,
) -> VisualizationResult:
    """Generate an interactive Plotly Figure directly from Module 6 AnalysisResult."""
    if analysis_result.status != "completed":
        return VisualizationResult(
            is_renderable=False,
            chart_type="none",
            title=f"Analysis ({analysis_result.intent})",
            figure=None,
            reason=f"Cannot visualize analysis with status '{analysis_result.status}'.",
        )

    table = analysis_result.result_table
    if not table:
        return VisualizationResult(
            is_renderable=False,
            chart_type="none",
            title=f"Analysis ({analysis_result.intent})",
            figure=None,
            reason="Analytical result table is empty.",
        )

    chosen = chart_type_override or select_chart_type(
        analysis_result=analysis_result,
        plan=plan,
        df=df,
        prefer_pie=prefer_pie,
    )
    if chosen not in SUPPORTED_CHART_TYPES:
        return VisualizationResult(
            is_renderable=False,
            chart_type=chosen,
            title=f"Unsupported Chart ({chosen})",
            figure=None,
            reason=f"Unsupported chart type '{chosen}'. Supported: {sorted(SUPPORTED_CHART_TYPES)}.",
        )

    intent = analysis_result.intent
    summary = analysis_result.summary_metrics

    if intent == AnalyticalIntent.TREND_ANALYSIS.value:
        metric = summary.get("metric_column", "Value")
        agg = summary.get("aggregation", "sum")
        gran = summary.get("granularity", "month")
        return create_line_chart(
            x_values=[row.get("period") for row in table],
            y_values=[row.get("value") for row in table],
            title=f"{gran.title()}ly {agg.title()} of {metric}",
            x_label=summary.get("date_column", "Period"),
            y_label=f"{agg}({metric})",
        )

    if intent == AnalyticalIntent.DISTRIBUTION_ANALYSIS.value:
        col = summary.get("column", "Metric")
        if df is not None and col in df.columns and chosen == "histogram":
            s = _safe_numeric(df, col).dropna()
            return create_histogram(
                values=s,
                title=f"Distribution of {col}",
                x_label=col,
                y_label="Frequency",
            )
        labels = [f"{row.get('bin_start')}–{row.get('bin_end')}" for row in table]
        counts = [int(row.get("count", 0)) for row in table]
        return create_histogram(
            bin_labels=labels,
            bin_counts=counts,
            title=f"Distribution of {col}",
            x_label=col,
            y_label="Frequency",
        )

    if intent == AnalyticalIntent.CORRELATION.value:
        cols = analysis_result.numerical_evidence.columns_used
        if chosen == "scatter" and len(cols) >= 2 and df is not None:
            c1, c2 = cols[0], cols[1]
            pair_df = pd.DataFrame({c1: _safe_numeric(df, c1), c2: _safe_numeric(df, c2)}).dropna().head(1000)
            r_val = table[0].get("pearson_r") if table else None
            return create_scatter_plot(
                x_values=pair_df[c1].tolist(),
                y_values=pair_df[c2].tolist(),
                title=f"{c1} vs {c2} (r = {r_val})",
                x_label=c1,
                y_label=c2,
            )
        # Build correlation matrix from pair rows
        if len(cols) >= 2:
            lookup: dict[tuple[str, str], float | None] = {}
            for row in table:
                a, b, r = row.get("column_a"), row.get("column_b"), row.get("pearson_r")
                if a and b:
                    lookup[(a, b)] = r
                    lookup[(b, a)] = r
            matrix = [
                [1.0 if r_col == c_col else lookup.get((r_col, c_col)) for c_col in cols]
                for r_col in cols
            ]
            return create_heatmap(
                x_labels=cols,
                y_labels=cols,
                z_matrix=matrix,
                title="Pearson Correlation Matrix",
            )

    if intent == AnalyticalIntent.MISSING_VALUE_ANALYSIS.value:
        return create_bar_chart(
            categories=[row.get("column") for row in table],
            values=[row.get("missing_pct") for row in table],
            title="Missing Data Percentage by Column",
            x_label="Column",
            y_label="Missing (%)",
            horizontal=False,
        )

    if intent in {
        AnalyticalIntent.RANKING.value,
        AnalyticalIntent.AGGREGATION.value,
        AnalyticalIntent.COMPARISON.value,
    }:
        group_col = summary.get("group_column")
        metric_col = summary.get("metric_column", "Value")
        agg = summary.get("aggregation", "sum")

        if group_col and group_col in table[0]:
            cats = [row.get(group_col) for row in table]
            vals = [row.get("primary_value", row.get("count")) for row in table]
            title = f"{agg.title()} of {metric_col} by {group_col}"
            if chosen == "pie":
                return create_pie_chart(labels=cats, values=vals, title=title)
            return create_bar_chart(
                categories=cats,
                values=vals,
                title=title,
                x_label=group_col,
                y_label=f"{agg}({metric_col})",
                horizontal=(chosen == "horizontal_bar"),
            )

        if "compared_columns" in summary and "column" in table[0]:
            cats = [row.get("column") for row in table]
            vals = [row.get("mean") for row in table]
            return create_bar_chart(
                categories=cats,
                values=vals,
                title="Comparison of Column Means",
                x_label="Column",
                y_label="Mean",
                horizontal=False,
            )

    if intent == AnalyticalIntent.DATASET_SUMMARY.value and "column" in table[0]:
        return create_bar_chart(
            categories=[row.get("column") for row in table],
            values=[row.get("missing_pct") for row in table],
            title="Dataset Column Missingness Overview",
            x_label="Column",
            y_label="Missing (%)",
            horizontal=False,
        )

    return VisualizationResult(
        is_renderable=False,
        chart_type=chosen,
        title=f"Analysis ({intent})",
        figure=None,
        reason=f"Result shape for intent '{intent}' is not suitable for a {chosen} chart.",
    )


def generate_anomaly_chart(
    anomaly_report: AnomalyReport,
    df: pd.DataFrame | None = None,
    x_feature: str | None = None,
    y_feature: str | None = None,
) -> VisualizationResult:
    """Generate a Plotly scatter plot highlighting Isolation Forest anomalies vs normal observations."""
    if anomaly_report.rows_analyzed == 0 or not anomaly_report.anomaly_scores:
        return VisualizationResult(
            is_renderable=False,
            chart_type="scatter",
            title="Anomaly Detection",
            figure=None,
            reason="Anomaly report contains no analyzed rows.",
        )

    features = anomaly_report.features_used
    if df is not None and len(features) >= 2:
        fx = x_feature if (x_feature in features) else features[0]
        fy = y_feature if (y_feature in features and y_feature != fx) else features[1]
        clean_df = df[features].dropna() if anomaly_report.missing_strategy == "drop" else df[features]
        if len(clean_df) == len(anomaly_report.row_labels):
            return create_scatter_plot(
                x_values=_safe_numeric(clean_df, fx).tolist(),
                y_values=_safe_numeric(clean_df, fy).tolist(),
                title=f"Isolation Forest Anomalies: {fx} vs {fy}",
                x_label=fx,
                y_label=fy,
                labels=anomaly_report.row_labels,
            )

    # Fallback: Row Index vs Anomaly Score
    row_positions = list(range(len(anomaly_report.anomaly_scores)))
    return create_scatter_plot(
        x_values=row_positions,
        y_values=anomaly_report.anomaly_scores,
        title="Isolation Forest Anomaly Scores by Observation",
        x_label="Observation Index",
        y_label="Anomaly Score (higher = more anomalous)",
        labels=anomaly_report.row_labels,
    )


class VisualizationEngine:
    """Stateful Streamlit-independent service wrapper for Module 9."""

    def create_analysis_chart(
        self,
        analysis_result: AnalysisResult,
        plan: AnalyticalTaskPlan | None = None,
        df: pd.DataFrame | None = None,
        chart_type_override: str | None = None,
        prefer_pie: bool = False,
    ) -> VisualizationResult:
        return generate_chart_for_analysis(
            analysis_result=analysis_result,
            plan=plan,
            df=df,
            chart_type_override=chart_type_override,
            prefer_pie=prefer_pie,
        )

    def create_anomaly_chart(
        self,
        anomaly_report: AnomalyReport,
        df: pd.DataFrame | None = None,
        x_feature: str | None = None,
        y_feature: str | None = None,
    ) -> VisualizationResult:
        return generate_anomaly_chart(
            anomaly_report=anomaly_report,
            df=df,
            x_feature=x_feature,
            y_feature=y_feature,
        )
