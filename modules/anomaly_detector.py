"""Module 7: ML-Based Anomaly Detection using Scikit-learn Isolation Forest.

Detects multivariate and univariate numerical anomalies offline using
Scikit-learn IsolationForest with Median/Mean imputation and StandardScaler,
reports affected row indices, scores, percentages, feature deviations, and
unsupervised score diagnostics without fabricating precision or recall.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.logging_config import get_logger
from config.settings import get_settings
from modules.dataset_profiler import DatasetProfileResult

logger = get_logger(__name__)

INTERPRETATION_DISCLAIMER = (
    "Anomalous values detected by Isolation Forest indicate observations that deviate "
    "statistically from the majority distribution. An anomaly is NOT automatically a data "
    "error, defect, or fraud and must be reviewed with domain context."
)


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------


class AnomalyDetectionError(Exception):
    """Base exception for ML anomaly detection errors."""


class InvalidAnomalyConfigError(AnomalyDetectionError):
    """Raised when contamination, missing strategy, or feature selection is invalid."""


class InsufficientAnomalyDataError(AnomalyDetectionError):
    """Raised when there are insufficient valid numerical rows or features to fit Isolation Forest."""


# ---------------------------------------------------------------------------
# Structured Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnomalyConfig:
    """User-configurable parameters for Isolation Forest anomaly detection."""

    feature_columns: list[str] | None = None
    contamination: float | str = 0.05  # Float in (0.0, 0.5] or "auto"
    missing_strategy: str = "median"  # "median" | "mean" | "drop"
    n_estimators: int = 100
    random_state: int = 42
    max_anomalous_records_returned: int = 100

    def validate(self) -> None:
        if isinstance(self.contamination, str):
            if self.contamination.strip().lower() != "auto":
                raise InvalidAnomalyConfigError(
                    "contamination must be 'auto' or a float in (0.0, 0.5]."
                )
        elif isinstance(self.contamination, (int, float)):
            c_val = float(self.contamination)
            if not (0.0 < c_val <= 0.5):
                raise InvalidAnomalyConfigError(
                    f"contamination must be in (0.0, 0.5], got {self.contamination}."
                )
        else:
            raise InvalidAnomalyConfigError("contamination must be 'auto' or a float in (0.0, 0.5].")

        if self.missing_strategy not in {"median", "mean", "drop"}:
            raise InvalidAnomalyConfigError(
                f"missing_strategy must be 'median', 'mean', or 'drop', got '{self.missing_strategy}'."
            )

        if self.n_estimators < 10:
            raise InvalidAnomalyConfigError("n_estimators must be at least 10.")


@dataclass(frozen=True)
class AnomalousRecord:
    """Detailed record for a single row flagged as anomalous."""

    row_index: int
    raw_label: int  # -1 for anomaly
    anomaly_score: float  # Higher = more anomalous (negated decision_function)
    decision_function_score: float
    feature_values: dict[str, float | None]
    top_deviating_features: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UnsupervisedDiagnostics:
    """Honest unsupervised evaluation metrics (never fabricates precision/recall)."""

    evaluation_type: str
    decision_score_min: float
    decision_score_p05: float
    decision_score_median: float
    decision_score_max: float
    mean_score_normal: float | None
    mean_score_anomaly: float | None
    score_separation_margin: float | None
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnomalyReport:
    """Complete structured output from Module 7 ML Anomaly Detection."""

    algorithm: str
    contamination_setting: float | str
    missing_strategy: str
    features_used: list[str]
    excluded_features: list[str]
    total_rows_input: int
    rows_analyzed: int
    rows_dropped_for_missing: int
    anomaly_count: int
    normal_count: int
    anomaly_percentage: float
    anomalous_row_indices: list[int]
    row_labels: list[int]  # 1 = normal, -1 = anomaly (aligned with analyzed indices)
    anomaly_scores: list[float]  # Higher = more anomalous (aligned with analyzed indices)
    anomalous_records: list[AnomalousRecord]
    diagnostics: UnsupervisedDiagnostics
    interpretation_note: str
    warnings: list[str]
    model_artifact_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "contamination_setting": self.contamination_setting,
            "missing_strategy": self.missing_strategy,
            "features_used": self.features_used,
            "excluded_features": self.excluded_features,
            "total_rows_input": self.total_rows_input,
            "rows_analyzed": self.rows_analyzed,
            "rows_dropped_for_missing": self.rows_dropped_for_missing,
            "anomaly_count": self.anomaly_count,
            "normal_count": self.normal_count,
            "anomaly_percentage": self.anomaly_percentage,
            "anomalous_row_indices": self.anomalous_row_indices,
            "row_labels": self.row_labels,
            "anomaly_scores": self.anomaly_scores,
            "anomalous_records": [r.to_dict() for r in self.anomalous_records],
            "diagnostics": self.diagnostics.to_dict(),
            "interpretation_note": self.interpretation_note,
            "warnings": self.warnings,
            "model_artifact_path": self.model_artifact_path,
        }


# ---------------------------------------------------------------------------
# Feature Preparation & Detection Implementation
# ---------------------------------------------------------------------------


def _select_numerical_features(
    df: pd.DataFrame,
    requested_features: list[str] | None,
    profile: DatasetProfileResult | None,
) -> tuple[list[str], list[str], list[str]]:
    """Resolve valid non-constant numerical features and explain any exclusions."""
    warnings: list[str] = []
    excluded: list[str] = []

    if requested_features is not None:
        if not requested_features:
            raise InvalidAnomalyConfigError("feature_columns cannot be an empty list.")
        missing = [c for c in requested_features if c not in df.columns]
        if missing:
            raise InvalidAnomalyConfigError(f"Requested feature column(s) not found: {missing}.")
        candidates = list(dict.fromkeys(requested_features))
    elif profile is not None:
        candidates = [c for c in profile.numerical_columns if c in df.columns]
    else:
        candidates = [
            str(c)
            for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
        ]

    usable: list[str] = []
    for col in candidates:
        if pd.api.types.is_bool_dtype(df[col]):
            excluded.append(col)
            warnings.append(f"Excluded boolean column '{col}' from numerical anomaly detection.")
            continue

        numeric_s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        non_null = numeric_s.dropna()
        if non_null.empty:
            excluded.append(col)
            warnings.append(f"Excluded all-null column '{col}' from anomaly detection.")
            continue
        if non_null.nunique() <= 1:
            excluded.append(col)
            warnings.append(f"Excluded zero-variance constant column '{col}' from anomaly detection.")
            continue
        usable.append(col)

    if not usable:
        raise InsufficientAnomalyDataError(
            "No usable non-constant numerical features available for Isolation Forest anomaly detection."
        )

    return usable, excluded, warnings


def detect_anomalies(
    df: pd.DataFrame,
    config: AnomalyConfig | None = None,
    profile: DatasetProfileResult | None = None,
    save_model_path: str | Path | None = None,
) -> AnomalyReport:
    """Run Scikit-learn Isolation Forest anomaly detection on numerical dataset features."""
    if not isinstance(df, pd.DataFrame):
        raise AnomalyDetectionError("Input 'df' must be a pandas DataFrame.")
    if df.empty:
        raise InsufficientAnomalyDataError("Cannot run anomaly detection on an empty DataFrame.")

    cfg = config or AnomalyConfig()
    cfg.validate()

    features, excluded, warnings = _select_numerical_features(df, cfg.feature_columns, profile)
    total_rows_input = int(len(df))

    work_df = pd.DataFrame(
        {
            col: pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            for col in features
        },
        index=df.index,
    )

    rows_dropped = 0
    if cfg.missing_strategy == "drop":
        before = len(work_df)
        work_df = work_df.dropna(axis=0, how="any")
        rows_dropped = before - len(work_df)
        if rows_dropped > 0:
            warnings.append(f"Dropped {rows_dropped} row(s) containing missing feature values.")

    rows_analyzed = int(len(work_df))
    if rows_analyzed < 5:
        raise InsufficientAnomalyDataError(
            f"At least 5 valid rows are required for Isolation Forest, got {rows_analyzed}."
        )

    steps: list[tuple[str, Any]] = []
    if cfg.missing_strategy in {"median", "mean"}:
        steps.append(("imputer", SimpleImputer(strategy=cfg.missing_strategy)))
    steps.append(("scaler", StandardScaler()))
    steps.append(
        (
            "detector",
            IsolationForest(
                n_estimators=cfg.n_estimators,
                contamination=cfg.contamination,
                random_state=cfg.random_state,
            ),
        )
    )

    pipeline = Pipeline(steps=steps)
    pipeline.fit(work_df)

    raw_labels = pipeline.predict(work_df)  # 1 = normal, -1 = anomaly
    decision_scores = pipeline.decision_function(work_df)  # lower = more anomalous
    anomaly_scores = -decision_scores  # higher = more anomalous

    anomaly_mask = raw_labels == -1
    normal_mask = raw_labels == 1
    anomaly_count = int(anomaly_mask.sum())
    normal_count = int(normal_mask.sum())
    anomaly_pct = round((anomaly_count / rows_analyzed) * 100.0, 2)

    analyzed_indices = [int(idx) for idx in work_df.index.tolist()]
    anomalous_indices = [
        int(idx) for idx, is_anom in zip(analyzed_indices, anomaly_mask) if is_anom
    ]

    # Compute feature-level robust z-deviations for interpretability of flagged anomalies
    medians = work_df.median()
    stds = work_df.std(ddof=0).replace(0, np.nan)

    anomalous_records: list[AnomalousRecord] = []
    anom_positions = np.where(anomaly_mask)[0]
    # Order anomalous records from highest anomaly score to lowest
    sorted_anom_positions = sorted(anom_positions, key=lambda pos: float(anomaly_scores[pos]), reverse=True)

    for pos in sorted_anom_positions[: cfg.max_anomalous_records_returned]:
        idx = analyzed_indices[pos]
        row_vals = work_df.iloc[pos]
        feature_dict = {
            col: (None if pd.isna(row_vals[col]) else round(float(row_vals[col]), 4))
            for col in features
        }
        deviations: list[dict[str, Any]] = []
        for col in features:
            val = row_vals[col]
            col_std = stds[col]
            if pd.notna(val) and pd.notna(col_std) and col_std > 0:
                z_dev = abs(float(val) - float(medians[col])) / float(col_std)
                deviations.append(
                    {
                        "feature": col,
                        "value": round(float(val), 4),
                        "feature_median": round(float(medians[col]), 4),
                        "std_deviations_from_median": round(z_dev, 2),
                    }
                )
        deviations.sort(key=lambda d: d["std_deviations_from_median"], reverse=True)

        anomalous_records.append(
            AnomalousRecord(
                row_index=idx,
                raw_label=int(raw_labels[pos]),
                anomaly_score=round(float(anomaly_scores[pos]), 6),
                decision_function_score=round(float(decision_scores[pos]), 6),
                feature_values=feature_dict,
                top_deviating_features=deviations[:3],
            )
        )

    mean_norm = round(float(decision_scores[normal_mask].mean()), 6) if normal_count > 0 else None
    mean_anom = round(float(decision_scores[anomaly_mask].mean()), 6) if anomaly_count > 0 else None
    sep_margin = (
        round(mean_norm - mean_anom, 6)
        if (mean_norm is not None and mean_anom is not None)
        else None
    )

    diagnostics = UnsupervisedDiagnostics(
        evaluation_type="unsupervised_score_distribution",
        decision_score_min=round(float(np.min(decision_scores)), 6),
        decision_score_p05=round(float(np.percentile(decision_scores, 5)), 6),
        decision_score_median=round(float(np.median(decision_scores)), 6),
        decision_score_max=round(float(np.max(decision_scores)), 6),
        mean_score_normal=mean_norm,
        mean_score_anomaly=mean_anom,
        score_separation_margin=sep_margin,
        note=(
            "Isolation Forest is unsupervised; precision and recall are not reported "
            "because ground-truth anomaly labels are not available."
        ),
    )

    saved_path_str: str | None = None
    if save_model_path is not None:
        target = Path(save_model_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": pipeline, "features": features}, target)
        saved_path_str = str(target)
        logger.info("Saved IsolationForest pipeline to '%s'.", target)

    logger.info(
        "IsolationForest completed: rows=%d, features=%s, anomalies=%d (%.2f%%).",
        rows_analyzed,
        features,
        anomaly_count,
        anomaly_pct,
    )

    return AnomalyReport(
        algorithm="IsolationForest",
        contamination_setting=cfg.contamination,
        missing_strategy=cfg.missing_strategy,
        features_used=features,
        excluded_features=excluded,
        total_rows_input=total_rows_input,
        rows_analyzed=rows_analyzed,
        rows_dropped_for_missing=rows_dropped,
        anomaly_count=anomaly_count,
        normal_count=normal_count,
        anomaly_percentage=anomaly_pct,
        anomalous_row_indices=anomalous_indices,
        row_labels=[int(x) for x in raw_labels.tolist()],
        anomaly_scores=[round(float(x), 6) for x in anomaly_scores.tolist()],
        anomalous_records=anomalous_records,
        diagnostics=diagnostics,
        interpretation_note=INTERPRETATION_DISCLAIMER,
        warnings=warnings,
        model_artifact_path=saved_path_str,
    )


class AnomalyDetector:
    """Stateful offline service wrapper for Module 7 Isolation Forest anomaly detection."""

    def __init__(self, models_dir: str | Path | None = None) -> None:
        settings = get_settings()
        self.models_dir = Path(models_dir) if models_dir is not None else settings.models_dir

    def detect(
        self,
        df: pd.DataFrame,
        config: AnomalyConfig | None = None,
        profile: DatasetProfileResult | None = None,
        persist_filename: str | None = None,
    ) -> AnomalyReport:
        save_path = (self.models_dir / persist_filename) if persist_filename else None
        return detect_anomalies(
            df=df,
            config=config,
            profile=profile,
            save_model_path=save_path,
        )
