"""Unit tests and sample sales dataset demonstration for Module 4 (modules/prompt_optimizer.py)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from modules.dataset_profiler import DatasetProfileResult, profile_dataframe
from modules.prompt_optimizer import (
    EmptyPromptError,
    PromptOptimizationResult,
    PromptOptimizer,
    optimize_prompt,
)


@pytest.fixture()
def sample_sales_profile() -> DatasetProfileResult:
    """Sample sales dataset containing missing values, duplicates, and mixed column types."""
    sales_df = pd.DataFrame(
        {
            "Order_Date": [
                "2026-01-05",
                "2026-01-12",
                "2026-02-03",
                "2026-02-18",
                "2026-03-07",
                "2026-03-07",
            ],
            "Product": ["Laptop", "Phone", "Tablet", "Laptop", "Monitor", "Monitor"],
            "Region": ["North", "South", "East", "West", "North", "North"],
            "Revenue": [1450.0, 820.0, None, 1600.0, 490.0, 490.0],
            "Units_Sold": [2, 4, 3, 2, 1, 1],
        }
    )
    return profile_dataframe(sales_df, dataset_name="sample_sales.csv")


def test_optimize_vague_ranking_prompt_on_sales_dataset(sample_sales_profile: DatasetProfileResult) -> None:
    result = optimize_prompt("Top products?", sample_sales_profile)

    assert isinstance(result, PromptOptimizationResult)
    assert result.detected_intent == "ranking"
    assert result.rewritten_by == "local_rule_engine"
    assert result.task_plan.can_execute is True
    assert result.task_plan.group_column == "Product"
    assert result.task_plan.metric_column == "Revenue"
    assert result.task_plan.aggregation == "sum"
    assert result.task_plan.sort_order == "descending"
    assert result.task_plan.limit == 5
    assert "Product" in result.improved_prompt
    assert "Revenue" in result.improved_prompt
    assert len(result.missing_context) >= 1
    assert len(result.improvement_reasons) >= 1
    assert len(result.task_plan.quality_caveats) >= 1  # Missing Revenue & duplicate row caveats


def test_optimize_trend_prompt_adds_date_and_granularity(sample_sales_profile: DatasetProfileResult) -> None:
    result = optimize_prompt("Show revenue trend", sample_sales_profile)

    assert result.detected_intent == "trend_analysis"
    assert result.task_plan.date_column == "Order_Date"
    assert result.task_plan.time_granularity == "month"
    assert result.task_plan.metric_column == "Revenue"
    assert result.task_plan.recommended_chart_type == "line_chart"
    assert "Order_Date" in result.referenced_columns


def test_flags_unsupported_metrics_and_unavailable_columns(sample_sales_profile: DatasetProfileResult) -> None:
    result = optimize_prompt(
        "Which product had the highest profit margin by customer_tier?",
        sample_sales_profile,
    )

    # Unavailable terms must be flagged and excluded from referenced_columns
    lowered_unsupported = {u.lower() for u in result.unsupported_references}
    assert "profit" in lowered_unsupported or "customer_tier" in lowered_unsupported
    assert len(result.warnings) >= 1
    existing_cols = set(sample_sales_profile.column_profiles.keys())
    assert set(result.referenced_columns).issubset(existing_cols)
    assert set(result.task_plan.target_columns).issubset(existing_cols)


def test_offline_operation_without_api_key(monkeypatch, sample_sales_profile: DatasetProfileResult) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "offline")
    monkeypatch.setenv("GEMINI_API_KEY", "")

    optimizer = PromptOptimizer(use_online_llm=True)
    result = optimizer.optimize("Compare average revenue across Region", sample_sales_profile)

    assert result.rewritten_by == "local_rule_engine"
    assert result.detected_intent == "comparison"
    assert result.task_plan.aggregation == "mean"
    assert result.task_plan.group_column == "Region"
    assert result.task_plan.metric_column == "Revenue"

    serialized = json.dumps(result.to_dict(), allow_nan=False)
    assert "sample_sales.csv" in serialized


def test_empty_prompt_rejected(sample_sales_profile: DatasetProfileResult) -> None:
    with pytest.raises(EmptyPromptError):
        optimize_prompt("   ", sample_sales_profile)
