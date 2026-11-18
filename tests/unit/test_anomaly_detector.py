"""Unit tests for Module 7: ML-Based Anomaly Detection (modules/anomaly_detector.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modules.anomaly_detector import (
    INTERPRETATION_DISCLAIMER,
    AnomalyConfig,
    AnomalyDetector,
    AnomalyReport,
    InsufficientAnomalyDataError,
    InvalidAnomalyConfigError,
    detect_anomalies,
)


@pytest.fixture()
def mixed_anomaly_df() -> pd.DataFrame:
    """Create a dataset with 38 tight normal rows and 2 extreme numerical anomalies."""
    rng = np.random.default_rng(42)
    normal_rev = rng.normal(loc=100.0, scale=5.0, size=38)
    normal_qty = rng.normal(loc=10.0, scale=1.5, size=38)

    # Inject 2 extreme unusual records at indices 38 and 39 (plus one missing value at row 5)
    revenues = np.concatenate([normal_rev, [9500.0, 12000.0]])
    quantities = np.concatenate([normal_qty, [450.0, 600.0]])
    revenues[5] = np.nan

    return pd.DataFrame(
        {
            "category": ["Standard"] * 40,
            "constant_col": [1.0] * 40,
            "revenue": revenues,
            "quantity": quantities,
        }
    )


def test_isolation_forest_detects_injected_anomalies_and_handles_missing(
    mixed_anomaly_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    detector = AnomalyDetector(models_dir=tmp_path)
    config = AnomalyConfig(
        feature_columns=["revenue", "quantity", "constant_col"],
        contamination=0.05,  # 5% of 40 = 2 anomalies expected
        missing_strategy="median",
        n_estimators=100,
        random_state=42,
    )
    report = detector.detect(
        mixed_anomaly_df,
        config=config,
        persist_filename="iso_forest_test.joblib",
    )

    assert isinstance(report, AnomalyReport)
    assert report.algorithm == "IsolationForest"
    assert report.features_used == ["revenue", "quantity"]
    assert "constant_col" in report.excluded_features
    assert report.rows_analyzed == 40
    assert report.anomaly_count == 2
    assert report.anomaly_percentage == 5.0
    # Extreme rows 38 and 39 must be flagged
    assert set(report.anomalous_row_indices) == {38, 39}
    assert len(report.row_labels) == 40
    assert len(report.anomaly_scores) == 40
    assert report.anomalous_records[0].top_deviating_features[0]["feature"] in {"revenue", "quantity"}

    # Disclaimer & honest unsupervised diagnostics (no fabricated precision/recall)
    assert "not automatically" in report.interpretation_note.lower()
    assert report.interpretation_note == INTERPRETATION_DISCLAIMER
    assert report.diagnostics.evaluation_type == "unsupervised_score_distribution"
    assert report.diagnostics.score_separation_margin is not None
    assert report.diagnostics.score_separation_margin > 0
    assert "precision and recall are not reported" in report.diagnostics.note.lower()

    assert Path(report.model_artifact_path).is_file()
    serialized = json.dumps(report.to_dict(), allow_nan=False)
    assert "IsolationForest" in serialized


def test_drop_missing_strategy_and_configurable_contamination(mixed_anomaly_df: pd.DataFrame) -> None:
    config = AnomalyConfig(
        feature_columns=["revenue", "quantity"],
        contamination=0.10,
        missing_strategy="drop",
    )
    report = detect_anomalies(mixed_anomaly_df, config=config)

    assert report.rows_dropped_for_missing == 1
    assert report.rows_analyzed == 39
    assert report.anomaly_count == 4  # ~10% of 39
    assert 38 in report.anomalous_row_indices
    assert 39 in report.anomalous_row_indices


def test_invalid_anomaly_configs_and_insufficient_data_errors() -> None:
    small_df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    with pytest.raises(InsufficientAnomalyDataError):
        detect_anomalies(small_df)

    valid_df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]})
    with pytest.raises(InvalidAnomalyConfigError):
        detect_anomalies(valid_df, AnomalyConfig(contamination=0.8))

    with pytest.raises(InvalidAnomalyConfigError):
        detect_anomalies(valid_df, AnomalyConfig(missing_strategy="magic"))

    with pytest.raises(InvalidAnomalyConfigError):
        detect_anomalies(valid_df, AnomalyConfig(feature_columns=["missing_col"]))
