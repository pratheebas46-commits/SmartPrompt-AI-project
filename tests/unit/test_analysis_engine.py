"""Unit tests for Module 6: Automated Data Analysis Engine (modules/analysis_engine.py)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from modules.analysis_engine import (
    AnalysisEngine,
    AnalysisResult,
    InvalidAnalysisColumnError,
    UnsupportedAnalysisTaskError,
    execute_analysis,
)
from modules.dataset_profiler import profile_dataframe
from modules.prompt_optimizer import AnalyticalTaskPlan, optimize_prompt


@pytest.fixture()
def deterministic_sales_df() -> pd.DataFrame:
    """Dataset with hand-verifiable exact sums, means, medians, rankings, trends, and correlations."""
    return pd.DataFrame(
        {
            "Order_Date": [
                "2026-01-05",
                "2026-01-15",
                "2026-02-05",
                "2026-02-20",
                "2026-03-10",
                "2026-03-25",
            ],
            "Product": ["Laptop", "Phone", "Laptop", "Tablet", "Phone", "Monitor"],
            "Region": ["North", "South", "North", "East", "West", "North"],
            "Revenue": [1000.0, 400.0, 2000.0, 600.0, 800.0, 200.0],  # Total = 5000.0
            "Units": [2, 2, 4, 3, 4, 1],  # Total = 16
        }
    )


def test_exact_aggregations_sum_mean_median_count_min_max(deterministic_sales_df: pd.DataFrame) -> None:
    profile = profile_dataframe(deterministic_sales_df, dataset_name="known_sales.csv")
    engine = AnalysisEngine()

    for agg, expected in [
        ("sum", 5000.0),
        ("mean", round(5000.0 / 6.0, 6)),
        ("median", 700.0),  # sorted: [200, 400, 600, 800, 1000, 2000] -> (600+800)/2 = 700
        ("count", 6),
        ("min", 200.0),
        ("max", 2000.0),
    ]:
        plan = AnalyticalTaskPlan(
            intent="aggregation",
            can_execute=True,
            target_columns=["Revenue"],
            metric_column="Revenue",
            metric_columns=["Revenue"],
            group_column=None,
            date_column=None,
            time_granularity=None,
            aggregation=agg,
            sort_order=None,
            limit=None,
            filters=[],
            recommended_chart_type="summary_table",
            execution_steps=[],
            quality_caveats=[],
        )
        res = engine.analyze(deterministic_sales_df, plan, profile=profile)
        assert res.status == "completed"
        assert res.summary_metrics["value"] == pytest.approx(expected, rel=1e-5)
        assert res.numerical_evidence.totals["Revenue"] == 5000.0


def test_groupby_and_product_ranking_known_answers(deterministic_sales_df: pd.DataFrame) -> None:
    profile = profile_dataframe(deterministic_sales_df, dataset_name="known_sales.csv")
    opt = optimize_prompt("Which product generated the highest revenue?", profile)
    res = execute_analysis(deterministic_sales_df, opt.task_plan, profile=profile)

    assert isinstance(res, AnalysisResult)
    assert res.summary_metrics["top_group"] == "Laptop"
    assert res.summary_metrics["top_value"] == 3000.0
    assert res.summary_metrics["grand_total_sum"] == 5000.0
    # Laptop = 3000/5000 = 60%, Phone = 1200/5000 = 24%, Tablet = 12%, Monitor = 4%
    assert res.numerical_evidence.percentages["Product=Laptop"] == 60.0
    assert res.numerical_evidence.percentages["Product=Phone"] == 24.0


def test_trend_correlation_distribution_filtering_and_comparison(deterministic_sales_df: pd.DataFrame) -> None:
    profile = profile_dataframe(deterministic_sales_df, dataset_name="known_sales.csv")

    # 1. Trend by month: Jan=1400, Feb=2600, Mar=1000
    trend_opt = optimize_prompt("Show monthly revenue trend over time", profile)
    trend_res = execute_analysis(deterministic_sales_df, trend_opt.task_plan, profile=profile)
    assert [r["value"] for r in trend_res.result_table] == [1400.0, 2600.0, 1000.0]

    # 2. Correlation between Revenue and Units
    corr_opt = optimize_prompt("What is the correlation between Revenue and Units?", profile)
    corr_res = execute_analysis(deterministic_sales_df, corr_opt.task_plan, profile=profile)
    expected_r = deterministic_sales_df["Revenue"].corr(deterministic_sales_df["Units"])
    assert corr_res.result_table[0]["pearson_r"] == pytest.approx(expected_r, abs=1e-5)

    # 3. Distribution
    dist_opt = optimize_prompt("Show distribution of Revenue", profile)
    dist_res = execute_analysis(deterministic_sales_df, dist_opt.task_plan, profile=profile)
    assert dist_res.summary_metrics["q25"] == 450.0
    assert dist_res.summary_metrics["q75"] == 950.0

    # 4. Filtering: Revenue > 600 -> rows [1000, 2000, 800] (sum = 3800)
    filt_opt = optimize_prompt("Filter rows where Revenue > 600", profile)
    filt_res = execute_analysis(deterministic_sales_df, filt_opt.task_plan, profile=profile)
    assert filt_res.summary_metrics["matched_rows"] == 3
    assert filt_res.numerical_evidence.totals["Revenue"] == 3800.0

    # 5. Comparison across Region: North has (1000+2000+200)/3 = 1066.6667
    comp_opt = optimize_prompt("Compare average revenue across Region", profile)
    comp_res = execute_analysis(deterministic_sales_df, comp_opt.task_plan, profile=profile)
    assert comp_res.summary_metrics["top_group"] == "North"
    assert comp_res.summary_metrics["top_value"] == pytest.approx(1066.666667, rel=1e-4)

    serialized = json.dumps(comp_res.to_dict(), allow_nan=False)
    assert "North" in serialized


def test_unsupported_tasks_and_code_injection_prevented(deterministic_sales_df: pd.DataFrame) -> None:
    profile = profile_dataframe(deterministic_sales_df, dataset_name="known_sales.csv")

    # Unknown intent returns safe unsupported status without raising or inventing numbers
    unknown_plan = AnalyticalTaskPlan(
        intent="unknown",
        can_execute=False,
        target_columns=[],
        metric_column=None,
        metric_columns=[],
        group_column=None,
        date_column=None,
        time_granularity=None,
        aggregation=None,
        sort_order=None,
        limit=None,
        filters=[],
        recommended_chart_type="summary_table",
        execution_steps=[],
        quality_caveats=[],
    )
    res_unknown = execute_analysis(deterministic_sales_df, unknown_plan, profile=profile)
    assert res_unknown.status == "unsupported"
    assert res_unknown.result_table == []

    # Disallowed aggregation / code injection string must be blocked by whitelist
    malicious_plan = AnalyticalTaskPlan(
        intent="aggregation",
        can_execute=True,
        target_columns=["Revenue"],
        metric_column="Revenue",
        metric_columns=["Revenue"],
        group_column=None,
        date_column=None,
        time_granularity=None,
        aggregation="__import__('os').system('echo pwned')",
        sort_order=None,
        limit=None,
        filters=[],
        recommended_chart_type="summary_table",
        execution_steps=[],
        quality_caveats=[],
    )
    with pytest.raises(UnsupportedAnalysisTaskError):
        execute_analysis(deterministic_sales_df, malicious_plan)

    # Missing column rejected safely
    missing_col_plan = AnalyticalTaskPlan(
        intent="aggregation",
        can_execute=True,
        target_columns=["NonExistent"],
        metric_column="NonExistent",
        metric_columns=["NonExistent"],
        group_column=None,
        date_column=None,
        time_granularity=None,
        aggregation="sum",
        sort_order=None,
        limit=None,
        filters=[],
        recommended_chart_type="summary_table",
        execution_steps=[],
        quality_caveats=[],
    )
    with pytest.raises(InvalidAnalysisColumnError):
        execute_analysis(deterministic_sales_df, missing_col_plan)
