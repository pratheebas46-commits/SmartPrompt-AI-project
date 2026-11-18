"""Unit tests for Module 5: ML-Based Prompt Quality Evaluator (modules/prompt_quality_model.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from modules.dataset_profiler import profile_dataframe
from modules.prompt_quality_model import (
    InvalidPromptInputError,
    ModelNotFoundError,
    PromptQualityClass,
    PromptQualityEvaluator,
    build_labeled_prompt_dataset,
    evaluate_prompt_quality,
    extract_prompt_features,
    load_prompt_quality_model,
    save_prompt_quality_model,
    train_prompt_quality_model,
)


def test_labeled_dataset_is_balanced_and_has_distinct_prompts() -> None:
    samples = build_labeled_prompt_dataset()
    labels = [s.quality_label for s in samples]
    texts = [s.prompt_text.strip().lower() for s in samples]

    assert labels.count(PromptQualityClass.LOW.value) >= 25
    assert labels.count(PromptQualityClass.MEDIUM.value) >= 25
    assert labels.count(PromptQualityClass.HIGH.value) >= 25
    assert len(set(texts)) == len(texts)  # Zero duplicate prompts to prevent train/test leakage


def test_feature_extraction_reflects_specificity_context_clarity_and_columns() -> None:
    cols = ["order_date", "product", "region", "revenue"]

    low_feats = extract_prompt_features("analyze stuff", available_columns=cols)
    high_feats = extract_prompt_features(
        "Rank the top 5 products by total revenue grouped by region monthly.",
        available_columns=cols,
    )

    assert high_feats.specificity_score > low_feats.specificity_score
    assert high_feats.context_score > low_feats.context_score
    assert high_feats.intent_clarity_score > low_feats.intent_clarity_score
    assert high_feats.column_relevance_score > low_feats.column_relevance_score
    assert "revenue" in high_feats.matched_columns


def test_train_evaluate_save_and_load_model_with_joblib(tmp_path: Path) -> None:
    model_file = tmp_path / "prompt_quality_test.joblib"
    bundle = train_prompt_quality_model(test_size=0.25, random_state=42, save_path=model_file)

    assert model_file.is_file()
    report = bundle.evaluation_report
    assert report.train_size > report.test_size > 0
    assert 0.0 <= report.accuracy <= 1.0
    assert 0.0 <= report.precision_weighted <= 1.0
    assert 0.0 <= report.recall_weighted <= 1.0
    assert 0.0 <= report.f1_weighted <= 1.0
    assert set(report.classes) == {"Low", "Medium", "High"}
    assert len(report.confusion_matrix) == 3

    # Reload persisted model via Joblib for offline inference
    loaded_bundle = load_prompt_quality_model(model_path=model_file)
    assert loaded_bundle.evaluation_report.accuracy == report.accuracy

    sales_df = pd.DataFrame(
        {
            "order_date": ["2026-01-01", "2026-02-01"],
            "product": ["Laptop", "Phone"],
            "region": ["North", "South"],
            "revenue": [1200.0, 800.0],
        }
    )
    profile = profile_dataframe(sales_df, dataset_name="sales.csv")

    low_pred = evaluate_prompt_quality("analyze data", profile=profile, model_bundle=loaded_bundle)
    high_pred = evaluate_prompt_quality(
        "Rank the top 5 products by total revenue grouped by region over order_date monthly.",
        profile=profile,
        model_bundle=loaded_bundle,
    )

    assert low_pred.predicted_class in {"Low", "Medium", "High"}
    assert high_pred.predicted_class in {"Low", "Medium", "High"}
    assert high_pred.quality_score > low_pred.quality_score
    assert abs(sum(low_pred.class_probabilities.values()) - 1.0) < 1e-2
    assert len(low_pred.recommendations) >= 1

    serialized = json.dumps(high_pred.to_dict(), allow_nan=False)
    assert "predicted_class" in serialized


def test_missing_model_and_invalid_prompt_errors(tmp_path: Path) -> None:
    missing_path = tmp_path / "does_not_exist.joblib"
    with pytest.raises(ModelNotFoundError):
        load_prompt_quality_model(model_path=missing_path, auto_train_if_missing=False)

    evaluator = PromptQualityEvaluator(model_path=missing_path)
    report = evaluator.train(save=True)
    assert missing_path.is_file()
    assert report.test_size > 0

    with pytest.raises(InvalidPromptInputError):
        evaluator.predict("   ")
