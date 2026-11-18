"""FastAPI dependency helpers for accessing application services."""

from __future__ import annotations

from fastapi import Request

from backend.services.analytics_service import AnalyticsService


def get_analytics_service(request: Request) -> AnalyticsService:
    """Retrieve the shared AnalyticsService instance from FastAPI app state."""
    return request.app.state.analytics_service
