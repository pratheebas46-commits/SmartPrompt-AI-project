"""Unit tests for Module 6: Automated EDA & Query Execution Engine (modules/eda_engine.py)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from modules.dataset_profiler import profile_dataframe
from modules.eda_engine import (
    AutomatedEDAReport,
    EDAEngine,
    EDAResult,
    InsufficientDataForEDAError,
    InvalidEDAPlanError,
    execute_analytical_plan,
    run_automated_eda,
)
from modules.prompt_optimizer import optimize_prompt


@pytest.fixture()
def sales_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Order_Date": [
                "2026-01-05",
                "2026-01-20",
                "2026-02-10",
                "2026-02-22",
                "2026-03-04",
                "2026-03-18",
            ],
            "Product": ["Laptop", "Phone", "Laptop", "Tablet", "Phone", "Monitor"],
            "Region": ["North", "South", "North", "East", "West", "North"],
            "Revenue": [1500.0, 800.0, 1700.0, 600.0, 900.0, 400.0],
            "Units_Sold": [3, 4, 4, 2, 5, 1],
        }
    )


def test_execute_ranking_plan_computes_exact_totals_and_chart(sales_df: pd.DataFrame) -> None:
    profile = profile_dataframe(sales_df, dataset_name="sales.csv")
    opt = optimize_prompt("Which product generated the highest revenue?", profile)

    result = execute_analytical_plan(sales_df, opt.task_plan, profile=profile)
    assert isinstance(result, EDAResult)
    assert result.intent == "ranking"
    assert result.summary_metrics["top_group"] == "Laptop"
    assert result.summary_metrics["top_value"] == 3200.0
    assert result.summary_metrics["grand_total_sum"] == 5900.0
    assert len(result.chart_specs) == 1
    assert result.chart_specs[0].chart_type == "horizontal_bar"

    serialized = json.dumps(result.to_dict(), allow_nan=False)
    assert "Laptop" in serialized


def test_execute_trend_and_correlation_and_distribution(sales_df: pd.DataFrame) -> None:
    profile = profile_dataframe(sales_df, dataset_name="sales.csv")

    trend_opt = optimize_prompt("Show monthly revenue trend over time", profile)
    trend_res = execute_analytical_plan(sales_df, trend_opt.task_plan, profile=profile)
    assert trend_res.summary_metrics["period_count"] == 3
    assert trend_res.chart_specs[0].chart_type == "line"

    corr_opt = optimize_prompt("What is the correlation between Revenue and Units_Sold?", profile)
    corr_res = execute_analytical_plan(sales_df, corr_opt.task_plan, profile=profile)
    assert corr_res.summary_metrics["pair_count"] == 1
    assert corr_res.chart_specs[0].chart_type == "scatter"

    dist_opt = optimize_prompt("Show distribution of Revenue", profile)
    dist_res = execute_analytical_plan(sales_df, dist_opt.task_plan, profile=profile)
    assert dist_res.summary_metrics["count"] == 6
    assert dist_res.chart_specs[0].chart_type == "histogram"


def test_execute_filter_and_comparison_plans(sales_df: pd.DataFrame) -> None:
    profile = profile_dataframe(sales_df, dataset_name="sales.csv")

    filt_opt = optimize_prompt("Filter rows where Revenue > 1000", profile)
    filt_res = execute_analytical_plan(sales_df, filt_opt.task_plan, profile=profile)
    assert filt_res.summary_metrics["matched_rows"] == 2
    assert filt_res.summary_metrics["metric_sum"] == 3200.0

    comp_opt = optimize_prompt("Compare average revenue across Region", profile)
    comp_res = execute_analytical_plan(sales_df, comp_opt.task_plan, profile=profile)
    assert comp_res.summary_metrics["highest_mean_group"] == "North"


def test_run_automated_eda_full_dataset(sales_df: pd.DataFrame) -> None:
    engine = EDAEngine()
    report = engine.run_full_eda(sales_df, dataset_name="sales.csv")

    assert isinstance(report, AutomatedEDAReport)
    assert report.row_count == 6
    assert len(report.numeric_summary) == 2
    assert "Product" in report.categorical_breakdowns
    assert "Revenue" in report.correlation_pearson
    assert len(report.chart_specs) >= 2

    serialized = json.dumps(report.to_dict(), allow_nan=False)
    assert "Pearson Correlation Heatmap" in serialized


def test_empty_dataframe_and_missing_column_errors(sales_df: pd.DataFrame) -> None:
    profile = profile_dataframe(sales_df, dataset_name="sales.csv")
    opt = optimize_prompt("Top products by revenue", profile)

    with pytest.raises(InsufficientDataForEDAError):
        run_automated_eda(pd.DataFrame())

    broken_df = sales_df.drop(columns=["Revenue"])
    with pytest.raises(InvalidEDAPlanError):
        execute_analytical_plan(broken_df, opt.task_plan)
