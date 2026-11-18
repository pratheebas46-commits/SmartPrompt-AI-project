"""FastAPI route for Unstructured Data to Structured Data Converter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Response, UploadFile
from pydantic import BaseModel, Field

from backend.routes.dependencies import get_analytics_service
from backend.services.analytics_service import AnalyticsService
from modules.unstructured_converter import (
    ConvertedDocumentResult,
    convert_unstructured_bytes,
    export_converted_to_excel_bytes,
)

router = APIRouter(prefix="/api/converter", tags=["Unstructured Data Converter"])

_CHUNK_SIZE = 1024 * 1024


class ConverterExportExcelRequest(BaseModel):
    """Request payload for POST /api/converter/export-excel."""

    source_file_name: str
    source_file_type: str = "TXT"
    file_size_bytes: int = 0
    extraction_strategy: str = "Extracted"
    columns: list[str]
    rows: list[dict[str, str]]
    raw_text_preview: str = ""
    raw_character_count: int = 0
    missing_cell_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    extracted_at: str = ""


@router.post("/extract", summary="Extract and structure data from PDF, DOCX, or TXT files")
async def extract_unstructured_endpoint(
    file: UploadFile = File(..., description="PDF (.pdf), Word (.docx), or Text (.txt) document"),
    service: AnalyticsService = Depends(get_analytics_service),
) -> dict[str, Any]:
    """Validate uploaded unstructured file, extract content, and return structured rows/columns."""
    safe_name = Path(file.filename or "").name
    max_mb = service.settings.max_upload_size_mb
    max_bytes = max_mb * 1024 * 1024

    chunks: list[bytes] = []
    total_size = 0
    try:
        while True:
            chunk = await file.read(_CHUNK_SIZE)
            if not chunk:
                break
            total_size += len(chunk)
            if total_size > max_bytes:
                raise ValueError(f"Uploaded file '{safe_name}' exceeds maximum allowed size of {max_mb} MB.")
            chunks.append(chunk)
    finally:
        await file.close()

    raw_bytes = b"".join(chunks)
    result = convert_unstructured_bytes(raw_bytes=raw_bytes, filename=safe_name, max_upload_mb=max_mb)
    return result.to_dict()


@router.post("/export-excel", summary="Download structured conversion preview as Excel (.xlsx)")
def export_converted_excel_endpoint(payload: ConverterExportExcelRequest) -> Response:
    """Serialize reviewed structured conversion output into a genuine downloadable .xlsx workbook."""
    result = ConvertedDocumentResult(
        source_file_name=payload.source_file_name,
        source_file_type=payload.source_file_type,
        file_size_bytes=payload.file_size_bytes,
        extraction_strategy=payload.extraction_strategy,
        columns=payload.columns,
        rows=payload.rows,
        raw_text_preview=payload.raw_text_preview,
        raw_character_count=payload.raw_character_count,
        missing_cell_count=payload.missing_cell_count,
        warnings=payload.warnings,
        extracted_at=payload.extracted_at,
    )
    excel_bytes = export_converted_to_excel_bytes(result)
    stem = Path(payload.source_file_name).stem or "converted_data"
    out_name = f"{stem}_structured.xlsx"
    return Response(
        content=excel_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
    )
