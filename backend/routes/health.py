"""Root, health check, readiness probe, and sanitized runtime settings routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Response, status

from backend.routes.dependencies import get_analytics_service
from backend.schemas.responses import HealthResponse
from backend.services.analytics_service import AnalyticsService

router = APIRouter(tags=["System & Health"])


@router.get("/", summary="Root service status")
def root_endpoint(service: AnalyticsService = Depends(get_analytics_service)) -> dict[str, Any]:
    """Lightweight root endpoint linking to OpenAPI docs and health check."""
    return {
        "service": service.settings.app_name,
        "status": "running",
        "execution_mode": service.settings.execution_mode.value,
        "docs_url": "/docs",
        "health_url": "/health",
    }


@router.get("/health", response_model=HealthResponse, summary="Check backend and database health")
def health_check(service: AnalyticsService = Depends(get_analytics_service)) -> HealthResponse:
    """Return service health, active execution mode, and database connectivity status."""
    return HealthResponse(**service.get_health_status())


@router.get("/ready", summary="Container readiness probe")
def readiness_check(
    response: Response,
    service: AnalyticsService = Depends(get_analytics_service),
) -> dict[str, Any]:
    """Return HTTP 200 when database connectivity is healthy, or 503 if unavailable."""
    health = service.get_health_status()
    if not health.get("database_connected"):
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"ready": bool(health.get("database_connected")), "mode": health.get("execution_mode")}


@router.get("/api/settings", summary="Retrieve sanitized non-secret runtime configuration")
def get_sanitized_settings(service: AnalyticsService = Depends(get_analytics_service)) -> dict[str, Any]:
    """Expose secret-free server configuration for dashboard transparency."""
    return service.settings.sanitized_summary()
