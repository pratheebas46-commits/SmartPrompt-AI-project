"""Unit tests for Module 3: Natural Language Analytical Intent Detection (modules/intent_detector.py)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from modules.dataset_profiler import DatasetProfileResult, profile_dataframe
from modules.intent_detector import (
    AnalyticalIntent,
    AnalyticalTask,
    IntentDetector,
    InvalidQuestionError,
    detect_intent,
)


@pytest.fixture()
def sales_profile() -> DatasetProfileResult:
    """Build a DatasetProfileResult using Module 1 for realistic schema-aware intent tests."""
    df = pd.DataFrame(
        {
            "Order_Date": ["2026-01-01", "2026-01-15", "2026-02-01", "2026-02-15"],
            "Product": ["Laptop", "Phone", "Tablet", "Laptop"],
            "Region": ["North", "South", "East", "West"],
            "Revenue": [1200.0, 800.0, 600.0, 1500.0],
            "Discount": [50.0, 20.0, 15.0, 75.0],
            "Is_Returned": [False, True, False, False],
        }
    )
    return profile_dataframe(df, dataset_name="sales_data.csv")


def test_prompt_example_ranking_highest_revenue_product(sales_profile: DatasetProfileResult) -> None:
    task = detect_intent("Which product generated the highest revenue?", sales_profile)

    assert isinstance(task, AnalyticalTask)
    assert task.intent == AnalyticalIntent.RANKING.value
    assert task.group_column == "Product"
    assert task.metric_column == "Revenue"
    assert task.aggregation == "sum"
    assert task.sort_order == "descending"
    assert task.is_ambiguous is False
    assert task.is_fallback is False


def test_dataset_summary_intent(sales_profile: DatasetProfileResult) -> None:
    task = detect_intent("Give me an overview summary of the dataset.", sales_profile)
    assert task.intent == AnalyticalIntent.DATASET_SUMMARY.value
    assert task.confidence > 0.4


def test_comparison_intent(sales_profile: DatasetProfileResult) -> None:
    task = detect_intent("Compare average revenue across Region.", sales_profile)
    assert task.intent == AnalyticalIntent.COMPARISON.value
    assert task.metric_column == "Revenue"
    assert task.group_column == "Region"
    assert task.aggregation == "mean"


def test_trend_analysis_intent_and_date_granularity(sales_profile: DatasetProfileResult) -> None:
    task = detect_intent("Show monthly revenue trend over time.", sales_profile)
    assert task.intent == AnalyticalIntent.TREND_ANALYSIS.value
    assert task.metric_column == "Revenue"
    assert task.date_column == "Order_Date"
    assert task.time_granularity == "month"


def test_aggregation_intent(sales_profile: DatasetProfileResult) -> None:
    task = detect_intent("What is the total revenue per region?", sales_profile)
    assert task.intent == AnalyticalIntent.AGGREGATION.value
    assert task.metric_column == "Revenue"
    assert task.group_column == "Region"
    assert task.aggregation == "sum"


def test_correlation_intent(sales_profile: DatasetProfileResult) -> None:
    task = detect_intent("What is the correlation between Revenue and Discount?", sales_profile)
    assert task.intent == AnalyticalIntent.CORRELATION.value
    assert set(task.metric_columns) == {"Revenue", "Discount"}
    assert task.is_ambiguous is False


def test_distribution_analysis_intent(sales_profile: DatasetProfileResult) -> None:
    task = detect_intent("Show the distribution and histogram of Revenue.", sales_profile)
    assert task.intent == AnalyticalIntent.DISTRIBUTION_ANALYSIS.value
    assert task.metric_column == "Revenue"


def test_anomaly_analysis_intent(sales_profile: DatasetProfileResult) -> None:
    task = detect_intent("Detect anomalies and outliers in Revenue.", sales_profile)
    assert task.intent == AnalyticalIntent.ANOMALY_ANALYSIS.value
    assert task.metric_column == "Revenue"


def test_filtering_intent_numeric_and_categorical(sales_profile: DatasetProfileResult) -> None:
    task = detect_intent("Filter rows where Revenue > 1000 for North.", sales_profile)
    assert task.intent == AnalyticalIntent.FILTERING.value
    assert any(f.column == "Revenue" and f.operator == ">" and f.value == 1000 for f in task.filters)
    assert any(f.column == "Region" and f.operator == "==" and f.value == "North" for f in task.filters)


def test_missing_value_analysis_intent(sales_profile: DatasetProfileResult) -> None:
    task = detect_intent("Which columns have missing values?", sales_profile)
    assert task.intent == AnalyticalIntent.MISSING_VALUE_ANALYSIS.value


def test_ambiguous_question_detection(sales_profile: DatasetProfileResult) -> None:
    # Vague question
    vague_task = detect_intent("Analyze data", sales_profile)
    assert vague_task.is_ambiguous is True
    assert len(vague_task.ambiguity_reasons) > 0

    # Correlation missing second numerical column
    corr_task = detect_intent("Check correlation of Revenue", sales_profile)
    assert corr_task.intent == AnalyticalIntent.CORRELATION.value
    assert corr_task.is_ambiguous is True
    assert len(corr_task.clarification_questions) > 0


def test_unknown_question_fallback_and_serialization(sales_profile: DatasetProfileResult) -> None:
    detector = IntentDetector()
    task = detector.detect("What is the weather in Tokyo tomorrow?", sales_profile)

    assert task.intent == AnalyticalIntent.UNKNOWN.value
    assert task.is_fallback is True
    assert task.confidence == 0.0
    assert task.fallback_reason is not None
    assert len(task.suggested_questions) > 0

    serialized = json.dumps(task.to_dict(), allow_nan=False)
    assert "unknown" in serialized

    with pytest.raises(InvalidQuestionError):
        detector.detect("   ", sales_profile)
