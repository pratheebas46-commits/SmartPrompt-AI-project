"""Pydantic request schemas for SmartPrompt AI FastAPI endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DatasetSourceMixin(BaseModel):
    """Identifies either an uploaded server session dataset_id or inline JSON records."""

    dataset_id: str | None = Field(
        default=None,
        description="Identifier returned by POST /api/dataset/profile.",
    )
    records: list[dict[str, Any]] | None = Field(
        default=None,
        description="Optional inline row records when running statelessly without an upload session.",
    )
    dataset_name: str = Field(
        default="dataset",
        description="Optional dataset name used when inline records are provided.",
    )


class CleanDatasetRequest(DatasetSourceMixin):
    """Request payload for POST /api/dataset/clean."""

    numeric_missing_strategy: str = Field(default="none")
    categorical_missing_strategy: str = Field(default="none")
    column_missing_strategies: dict[str, str] = Field(default_factory=dict)
    constant_fill_value: Any = Field(default="Unknown")
    numeric_constant_fill_value: float | int = Field(default=0)
    drop_column_missing_threshold: float = Field(default=0.5, gt=0.0, le=1.0)

    remove_duplicates: bool = Field(default=False)
    duplicate_subset: list[str] | None = Field(default=None)
    duplicate_keep: str | bool = Field(default="first")

    type_conversions: dict[str, str] = Field(default_factory=dict)
    datetime_columns: list[str] = Field(default_factory=list)
    datetime_format: str | None = Field(default=None)

    numeric_scaling: str = Field(default="none")
    numeric_columns_to_preprocess: list[str] | None = Field(default=None)

    strip_categorical_whitespace: bool = Field(default=False)
    categorical_case: str = Field(default="none")
    categorical_encoding: str = Field(default="none")
    categorical_columns_to_preprocess: list[str] | None = Field(default=None)

    reprofile_after_cleaning: bool = Field(default=True)
    update_active_dataset: bool = Field(
        default=True,
        description="Whether to set the cleaned DataFrame as the active dataset in session store.",
    )


class PromptOptimizeRequest(DatasetSourceMixin):
    """Request payload for POST /api/prompt/optimize."""

    prompt: str = Field(..., min_length=1, description="Original natural-language analytical question.")
    use_online_llm: bool = Field(
        default=False,
        description="Optional online LLM enhancement (requires EXECUTION_MODE='online' and valid key).",
    )


class PromptScoreRequest(BaseModel):
    """Request payload for POST /api/prompt/score."""

    prompt: str = Field(..., min_length=1, description="Analytical prompt to score.")
    dataset_id: str | None = Field(default=None)
    available_columns: list[str] | None = Field(default=None)


class AnalysisRunRequest(DatasetSourceMixin):
    """Request payload for POST /api/analysis/run."""

    prompt: str = Field(..., min_length=1, description="Analytical question to optimize, execute, and validate.")
    use_online_llm: bool = Field(default=False)
    chart_type_override: str | None = Field(default=None)
    prefer_pie: bool = Field(default=False)
    save_to_history: bool = Field(default=True)
    run_anomaly_detection: bool = Field(default=False)
    anomaly_contamination: float | str = Field(default=0.05)


class AnomalyDetectRequest(DatasetSourceMixin):
    """Request payload for POST /api/anomaly/detect."""

    feature_columns: list[str] | None = Field(default=None)
    contamination: float | str = Field(default=0.05)
    missing_strategy: str = Field(default="median")
    n_estimators: int = Field(default=100, ge=10, le=500)
    x_feature: str | None = Field(default=None)
    y_feature: str | None = Field(default=None)


class AnalyticalClaimInput(BaseModel):
    """Structured or text claim submitted to POST /api/analysis/validate."""

    claim_type: str
    description: str = ""
    column: str | None = None
    secondary_column: str | None = None
    group_column: str | None = None
    group_value: Any = None
    claimed_value: float | int | None = None
    claimed_entity: Any = None
    sort_order: str = "descending"
    aggregation: str = "sum"
    filters: list[dict[str, Any]] = Field(default_factory=list)
    statement_text: str | None = None


class AnalysisValidateRequest(DatasetSourceMixin):
    """Request payload for POST /api/analysis/validate."""

    analysis_id: str | None = Field(
        default=None,
        description="Optional stored analysis run ID in server session cache to validate.",
    )
    prompt: str | None = Field(
        default=None,
        description="Optional prompt to execute and validate in one call if analysis_id is omitted.",
    )
    claims: list[AnalyticalClaimInput] | None = Field(
        default=None,
        description="Explicit structured or text claims to validate against the dataset.",
    )
    abs_tol: float = Field(default=1e-3, ge=0.0)
    rel_tol: float = Field(default=1e-3, ge=0.0)


class ReportExportRequest(DatasetSourceMixin):
    """Request payload for POST /api/report/export."""

    export_format: str = Field(default="pdf", description="'pdf' or 'csv'.")
    analysis_id: str | None = Field(default=None)
    report_title: str = Field(default="SmartPrompt AI — Analytical Report")
    additional_insights: list[str] | None = Field(default=None)


class RegisterRequest(BaseModel):
    """Request payload for POST /api/auth/register."""

    full_name: str = Field(..., min_length=2, max_length=120)
    username: str = Field(..., min_length=3, max_length=32)
    email: str = Field(..., min_length=5, max_length=255)
    password: str = Field(..., min_length=8, max_length=128)
    confirm_password: str = Field(..., min_length=8, max_length=128)


class LoginRequest(BaseModel):
    """Request payload for POST /api/auth/login."""

    identifier: str = Field(..., min_length=1, description="Username or email address.")
    password: str = Field(..., min_length=1)
