"""Unit tests for Module 9: Interactive Visualization Engine (modules/visualization.py)."""

from __future__ import annotations

import sys

import pandas as pd
import plotly.graph_objects as go
import pytest

from modules.analysis_engine import execute_analysis
from modules.anomaly_detector import AnomalyConfig, detect_anomalies
from modules.dataset_profiler import profile_dataframe
from modules.prompt_optimizer import optimize_prompt
from modules.visualization import (
    SUPPORTED_CHART_TYPES,
    VisualizationEngine,
    create_bar_chart,
    create_heatmap,
    create_histogram,
    create_line_chart,
    create_pie_chart,
    create_scatter_plot,
    generate_anomaly_chart,
    generate_chart_for_analysis,
    select_chart_type,
)


@pytest.fixture
def sales_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Order_Date": [
                "2025-01-05",
                "2025-01-20",
                "2025-02-10",
                "2025-02-22",
                "2025-03-08",
                "2025-03-19",
                "2025-04-04",
                "2025-04-18",
            ],
            "Product": [
                "Laptop",
                "Mouse",
                "Laptop",
                "Monitor",
                "Mouse",
                "Monitor",
                "Keyboard",
                "Laptop",
            ],
            "Revenue": [1200.0, 50.0, 1300.0, 400.0, 75.0, 420.0, 120.0, 1450.0],
            "Units": [2, 5, 2, 1, 6, 1, 3, 3],
        }
    )


def test_supported_chart_types_constant() -> None:
    assert {"bar", "horizontal_bar", "line", "pie", "histogram", "scatter", "heatmap"}.issubset(
        SUPPORTED_CHART_TYPES
    )


def test_streamlit_is_not_imported() -> None:
    assert "streamlit" not in sys.modules


def test_create_bar_chart_vertical_and_horizontal() -> None:
    res_v = create_bar_chart(
        categories=["Laptop", "Monitor", "Mouse"],
        values=[3950, 820, 125],
        title="Revenue by Product",
        x_label="Product",
        y_label="Revenue",
        horizontal=False,
    )
    assert res_v.is_renderable is True
    assert res_v.chart_type == "bar"
    assert isinstance(res_v.figure, go.Figure)
    assert res_v.figure.layout.title.text == "Revenue by Product"
    assert res_v.figure.layout.xaxis.title.text == "Product"
    assert res_v.figure.layout.yaxis.title.text == "Revenue"

    res_h = create_bar_chart(
        categories=["Laptop", "Monitor"],
        values=[3950, 820],
        title="Top Products",
        x_label="Product",
        y_label="Revenue",
        horizontal=True,
    )
    assert res_h.is_renderable is True
    assert res_h.chart_type == "horizontal_bar"
    assert res_h.figure is not None


def test_create_line_chart_and_insufficient_points() -> None:
    ok = create_line_chart(
        x_values=["2025-01-31", "2025-02-28", "2025-03-31"],
        y_values=[1250.0, 1700.0, 495.0],
        title="Monthly Revenue Trend",
        x_label="Order_Date",
        y_label="sum(Revenue)",
    )
    assert ok.is_renderable is True
    assert ok.chart_type == "line"
    assert isinstance(ok.figure, go.Figure)

    empty = create_line_chart(["2025-01-31"], [100.0], "Too Short", "Date", "Value")
    assert empty.is_renderable is False
    assert empty.figure is None


def test_create_pie_chart_suitability_guards() -> None:
    ok = create_pie_chart(["A", "B", "C"], [40, 35, 25], "Share by Category")
    assert ok.is_renderable is True
    assert ok.chart_type == "pie"
    assert isinstance(ok.figure, go.Figure)

    neg = create_pie_chart(["A", "B"], [100, -20], "Negative Share")
    assert neg.is_renderable is False
    assert "negative" in (neg.reason or "").lower()

    too_many = create_pie_chart([f"C{i}" for i in range(12)], [10] * 12, "Crowded Pie", max_slices=8)
    assert too_many.is_renderable is False


def test_create_histogram_raw_and_prebinned() -> None:
    raw = create_histogram(values=[10, 12, 15, 22, 30, 35], title="Value Histogram", x_label="Revenue")
    assert raw.is_renderable is True
    assert raw.chart_type == "histogram"
    assert isinstance(raw.figure, go.Figure)

    binned = create_histogram(bin_labels=["0-10", "10-20"], bin_counts=[3, 5], title="Pre-binned")
    assert binned.is_renderable is True

    empty = create_histogram(values=[])
    assert empty.is_renderable is False


def test_create_scatter_plot_and_heatmap() -> None:
    scatter = create_scatter_plot(
        x_values=[1, 2, 3, 4],
        y_values=[10, 20, 30, 100],
        title="Units vs Revenue",
        x_label="Units",
        y_label="Revenue",
        labels=[1, 1, 1, -1],
    )
    assert scatter.is_renderable is True
    assert len(scatter.figure.data) == 2  # Normal and Anomaly traces

    heatmap = create_heatmap(
        x_labels=["Revenue", "Units"],
        y_labels=["Revenue", "Units"],
        z_matrix=[[1.0, 0.85], [0.85, 1.0]],
        title="Correlation Matrix",
    )
    assert heatmap.is_renderable is True
    assert heatmap.chart_type == "heatmap"


def test_auto_chart_selection_and_generation_across_intents(sales_df: pd.DataFrame) -> None:
    profile = profile_dataframe(sales_df, dataset_name="sales.csv")
    engine = VisualizationEngine()

    # 1. Ranking -> horizontal_bar
    rank_opt = optimize_prompt("Top products by total revenue", profile)
    rank_res = execute_analysis(sales_df, rank_opt.task_plan, profile=profile)
    assert select_chart_type(rank_res, rank_opt.task_plan, sales_df) == "horizontal_bar"
    rank_viz = engine.create_analysis_chart(rank_res, plan=rank_opt.task_plan, df=sales_df)
    assert rank_viz.is_renderable is True
    assert isinstance(rank_viz.figure, go.Figure)

    # 2. Trend -> line
    trend_opt = optimize_prompt("Show monthly trend of revenue over order_date", profile)
    trend_res = execute_analysis(sales_df, trend_opt.task_plan, profile=profile)
    assert select_chart_type(trend_res, trend_opt.task_plan, sales_df) == "line"
    trend_viz = generate_chart_for_analysis(trend_res, plan=trend_opt.task_plan, df=sales_df)
    assert trend_viz.is_renderable is True
    assert trend_viz.chart_type == "line"

    # 3. Correlation -> scatter (2 columns)
    corr_opt = optimize_prompt("Check correlation between Revenue and Units", profile)
    corr_res = execute_analysis(sales_df, corr_opt.task_plan, profile=profile)
    corr_viz = generate_chart_for_analysis(corr_res, plan=corr_opt.task_plan, df=sales_df)
    assert corr_viz.is_renderable is True
    assert corr_viz.chart_type == "scatter"

    # 4. Distribution -> histogram
    dist_opt = optimize_prompt("Show the distribution of Revenue", profile)
    dist_res = execute_analysis(sales_df, dist_opt.task_plan, profile=profile)
    dist_viz = generate_chart_for_analysis(dist_res, plan=dist_opt.task_plan, df=sales_df)
    assert dist_viz.is_renderable is True
    assert dist_viz.chart_type == "histogram"

    # 5. Unknown / unsupported analysis handles gracefully
    unk_opt = optimize_prompt("What is the weather in Tokyo?", profile)
    unk_res = execute_analysis(sales_df, unk_opt.task_plan, profile=profile)
    unk_viz = generate_chart_for_analysis(unk_res, plan=unk_opt.task_plan, df=sales_df)
    assert unk_viz.is_renderable is False
    assert unk_viz.figure is None


def test_generate_anomaly_chart_with_isolation_forest() -> None:
    df = pd.DataFrame(
        {
            "Amount": [100, 105, 98, 102, 110, 95, 101, 99, 103, 5000],
            "Qty": [1, 1, 2, 1, 2, 1, 1, 2, 1, 50],
        }
    )
    report = detect_anomalies(df, config=AnomalyConfig(contamination=0.1, random_state=42))
    viz = generate_anomaly_chart(report, df=df)
    assert viz.is_renderable is True
    assert viz.chart_type == "scatter"
    assert isinstance(viz.figure, go.Figure)
