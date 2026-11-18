"""Pydantic response schemas for SmartPrompt AI FastAPI endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Consistent error payload returned across all exception handlers."""

    error: str
    detail: str
    status_code: int


class HealthResponse(BaseModel):
    """Response schema for GET /health."""

    status: str
    app_name: str
    execution_mode: str
    database_connected: bool
    database_backend: str
    online_ai_enabled: bool
    active_dataset_sessions: int
    warnings: list[str] = Field(default_factory=list)


class DatasetProfileResponse(BaseModel):
    """Response schema for POST /api/dataset/profile."""

    dataset_id: str
    profile: dict[str, Any]
    preview_rows: list[dict[str, Any]]
    columns: list[str]


class DatasetCleanResponse(BaseModel):
    """Response schema for POST /api/dataset/clean."""

    dataset_id: str
    cleaning_report: dict[str, Any]
    updated_profile: dict[str, Any] | None
    preview_rows: list[dict[str, Any]]
    columns: list[str]


class PromptOptimizeResponse(BaseModel):
    """Response schema for POST /api/prompt/optimize."""

    dataset_id: str | None
    optimization: dict[str, Any]


class PromptScoreResponse(BaseModel):
    """Response schema for POST /api/prompt/score."""

    prediction: dict[str, Any]


class AnalysisRunResponse(BaseModel):
    """Response schema for POST /api/analysis/run."""

    dataset_id: str | None
    analysis_id: str | None
    prompt_optimization: dict[str, Any]
    analysis_result: dict[str, Any]
    validation_report: dict[str, Any]
    visualization: dict[str, Any]
    anomaly_report: dict[str, Any] | None = None
    verified_insights: list[str] = Field(default_factory=list)
    excluded_unsupported_claims: list[str] = Field(default_factory=list)


class AnomalyDetectResponse(BaseModel):
    """Response schema for POST /api/anomaly/detect."""

    dataset_id: str | None
    anomaly_report: dict[str, Any]
    visualization: dict[str, Any]


class AnalysisValidateResponse(BaseModel):
    """Response schema for POST /api/analysis/validate."""

    dataset_id: str | None
    validation_report: dict[str, Any]
    verified_insights: list[str] = Field(default_factory=list)
    excluded_unsupported_claims: list[str] = Field(default_factory=list)


class HistoryListResponse(BaseModel):
    """Response schema for GET /api/history."""

    total_count: int
    items: list[dict[str, Any]]


class HistoryItemResponse(BaseModel):
    """Response schema for GET /api/history/{analysis_id}."""

    item: dict[str, Any]


class HistoryDeleteResponse(BaseModel):
    """Response schema for DELETE /api/history/{analysis_id}."""

    deleted: bool
    analysis_id: str


class UserPublicResponse(BaseModel):
    """Public representation of an authenticated user (never includes password hash)."""

    user_id: str
    full_name: str
    username: str
    email: str
    created_at: str


class AuthTokenResponse(BaseModel):
    """Response schema for login and registration endpoints."""

    access_token: str | None = None
    token_type: str = "bearer"
    user: UserPublicResponse
    message: str


class LogoutResponse(BaseModel):
    """Response schema for POST /api/auth/logout."""

    logged_out: bool
    message: str
