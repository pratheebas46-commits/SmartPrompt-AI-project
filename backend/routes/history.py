"""Analysis history retrieval and deletion routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from backend.routes.dependencies import get_analytics_service
from backend.schemas.responses import (
    HistoryDeleteResponse,
    HistoryItemResponse,
    HistoryListResponse,
)
from backend.services.analytics_service import AnalyticsService

router = APIRouter(prefix="/api/history", tags=["Analysis History"])


@router.get(
    "",
    response_model=HistoryListResponse,
    summary="List stored analysis history records",
)
def list_history_endpoint(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    dataset_fingerprint: str | None = Query(default=None),
    detected_intent: str | None = Query(default=None),
    validation_status: str | None = Query(default=None),
    service: AnalyticsService = Depends(get_analytics_service),
) -> HistoryListResponse:
    """Retrieve stored analysis metadata history from SQLite/PostgreSQL."""
    result = service.list_history(
        limit=limit,
        offset=offset,
        dataset_fingerprint=dataset_fingerprint,
        detected_intent=detected_intent,
        validation_status=validation_status,
    )
    return HistoryListResponse(**result)


@router.get(
    "/{analysis_id}",
    response_model=HistoryItemResponse,
    summary="Retrieve a single analysis history record",
)
def get_history_item_endpoint(
    analysis_id: str,
    service: AnalyticsService = Depends(get_analytics_service),
) -> HistoryItemResponse:
    """Fetch metadata for one analysis run by analysis_id."""
    result = service.get_history_item(analysis_id)
    return HistoryItemResponse(**result)


@router.delete(
    "/{analysis_id}",
    response_model=HistoryDeleteResponse,
    summary="Delete an analysis history record",
)
def delete_history_item_endpoint(
    analysis_id: str,
    service: AnalyticsService = Depends(get_analytics_service),
) -> HistoryDeleteResponse:
    """Delete a stored analysis history entry by analysis_id."""
    result = service.delete_history_item(analysis_id)
    return HistoryDeleteResponse(**result)
