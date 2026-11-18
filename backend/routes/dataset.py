"""Dataset upload, profiling, cleaning, preview, and session cleanup routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, UploadFile

from backend.routes.dependencies import get_analytics_service
from backend.schemas.requests import CleanDatasetRequest
from backend.schemas.responses import DatasetCleanResponse, DatasetProfileResponse
from backend.services.analytics_service import AnalyticsService
from modules.dataset_profiler import (
    SUPPORTED_EXTENSIONS,
    FileSizeLimitError,
    UnsupportedFileFormatError,
)

router = APIRouter(prefix="/api/dataset", tags=["Dataset Management"])

_CHUNK_SIZE = 1024 * 1024  # 1 MB streaming read cap guard


@router.post(
    "/profile",
    response_model=DatasetProfileResponse,
    summary="Upload and profile a CSV or Excel dataset",
)
async def profile_dataset_endpoint(
    file: UploadFile = File(..., description="CSV (.csv) or Excel (.xlsx, .xls) dataset file"),
    sheet_name: str | None = Form(default=None, description="Optional Excel worksheet name"),
    service: AnalyticsService = Depends(get_analytics_service),
) -> DatasetProfileResponse:
    """Validate uploaded file format/size safely, load in memory, and close upload temp buffer."""
    safe_name = Path(file.filename or "").name
    ext = Path(safe_name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        await file.close()
        raise UnsupportedFileFormatError(
            f"Unsupported file format '{ext or 'none'}'. Allowed: {sorted(SUPPORTED_EXTENSIONS)}."
        )

    max_bytes = service.settings.max_upload_size_mb * 1024 * 1024
    chunks: list[bytes] = []
    total_size = 0
    try:
        while True:
            chunk = await file.read(_CHUNK_SIZE)
            if not chunk:
                break
            total_size += len(chunk)
            if total_size > max_bytes:
                raise FileSizeLimitError(
                    f"Uploaded file '{safe_name}' exceeds maximum allowed size of "
                    f"{service.settings.max_upload_size_mb} MB."
                )
            chunks.append(chunk)
    finally:
        await file.close()

    raw_bytes = b"".join(chunks)
    result = service.profile_uploaded_file(
        file_bytes=raw_bytes,
        filename=safe_name,
        sheet_name=sheet_name or None,
    )
    return DatasetProfileResponse(**result)


@router.post(
    "/clean",
    response_model=DatasetCleanResponse,
    summary="Clean and preprocess the active dataset",
)
def clean_dataset_endpoint(
    payload: CleanDatasetRequest,
    service: AnalyticsService = Depends(get_analytics_service),
) -> DatasetCleanResponse:
    """Apply user-selected Module 2 cleaning strategies without mutating original data."""
    result = service.clean_session_dataset(payload)
    return DatasetCleanResponse(**result)


@router.get(
    "/{dataset_id}/preview",
    summary="Retrieve dataset profile and sample preview rows",
)
def preview_dataset_endpoint(
    dataset_id: str,
    max_rows: int = 50,
    service: AnalyticsService = Depends(get_analytics_service),
) -> dict[str, Any]:
    """Return active DataFrame preview rows and latest schema profile."""
    return service.get_dataset_preview(dataset_id=dataset_id, max_rows=max(1, min(max_rows, 200)))


@router.delete(
    "/{dataset_id}",
    summary="Release an active dataset session from memory",
)
def delete_dataset_session_endpoint(
    dataset_id: str,
    service: AnalyticsService = Depends(get_analytics_service),
) -> dict[str, Any]:
    """Explicitly clear uploaded dataset memory when a user finishes a session."""
    deleted = service.delete_dataset_session(dataset_id=dataset_id)
    return {"deleted": deleted, "dataset_id": dataset_id}
