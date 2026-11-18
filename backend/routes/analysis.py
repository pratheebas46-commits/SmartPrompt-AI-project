"""Automated analysis, anomaly detection, self-validation, and report export routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import Response

from backend.routes.dependencies import get_analytics_service
from backend.schemas.requests import (
    AnalysisRunRequest,
    AnalysisValidateRequest,
    AnomalyDetectRequest,
    ReportExportRequest,
)
from backend.schemas.responses import (
    AnalysisRunResponse,
    AnalysisValidateResponse,
    AnomalyDetectResponse,
)
from backend.services.analytics_service import AnalyticsService

router = APIRouter(tags=["Analysis, Validation & Reports"])


@router.post(
    "/api/analysis/run",
    response_model=AnalysisRunResponse,
    summary="Optimize prompt, execute deterministic analysis, validate, visualize, and save history",
)
def run_analysis_endpoint(
    payload: AnalysisRunRequest,
    service: AnalyticsService = Depends(get_analytics_service),
) -> AnalysisRunResponse:
    """Execute Modules 4, 6, 8, 9, optional 7, 10, and 11 in one backend pipeline call."""
    result = service.run_full_analysis(payload)
    return AnalysisRunResponse(**result)


@router.post(
    "/api/anomaly/detect",
    response_model=AnomalyDetectResponse,
    summary="Detect numerical anomalies using Scikit-learn Isolation Forest",
)
def detect_anomalies_endpoint(
    payload: AnomalyDetectRequest,
    service: AnalyticsService = Depends(get_analytics_service),
) -> AnomalyDetectResponse:
    """Run Module 7 Isolation Forest anomaly detection and Module 9 scatter plot generation."""
    result = service.run_anomaly_detection(payload)
    return AnomalyDetectResponse(**result)


@router.post(
    "/api/analysis/validate",
    response_model=AnalysisValidateResponse,
    summary="Cross-validate analytical results or claims against raw dataset calculations",
)
def validate_analysis_endpoint(
    payload: AnalysisValidateRequest,
    service: AnalyticsService = Depends(get_analytics_service),
) -> AnalysisValidateResponse:
    """Run Module 8 evidence-based self-validation on results or explicit claims."""
    result = service.validate_results_or_claims(payload)
    return AnalysisValidateResponse(**result)


@router.post(
    "/api/report/export",
    summary="Generate downloadable CSV or PDF analytical report",
)
def export_report_endpoint(
    payload: ReportExportRequest,
    service: AnalyticsService = Depends(get_analytics_service),
) -> Response:
    """Build and stream Module 11 CSV or ReportLab PDF report bytes."""
    content, media_type, filename = service.export_report(payload)
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
