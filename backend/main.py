"""Module 12: FastAPI Application Entrypoint for SmartPrompt AI."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.routes import analysis, auth, converter, dataset, health, history, prompt
from backend.schemas.responses import ErrorResponse
from backend.services.analytics_service import AnalyticsService
from backend.services.session_store import DatasetSessionNotFoundError
from config.logging_config import get_logger, setup_logging
from config.settings import Settings, get_settings
from database.db_manager import (
    AuthenticationError,
    DatabaseConnectionError,
    DatabaseManager,
    DuplicateUserError,
    RecordNotFoundError,
)
from modules.analysis_engine import (
    AnalysisEngineError,
    InsufficientAnalysisDataError,
    InvalidAnalysisColumnError,
    UnsupportedAnalysisTaskError,
)
from modules.anomaly_detector import (
    AnomalyDetectionError,
    InsufficientAnomalyDataError,
    InvalidAnomalyConfigError,
)
from modules.data_cleaner import (
    ColumnNotFoundError,
    DataCleaningError,
    InvalidCleaningStrategyError,
)
from modules.dataset_profiler import (
    DatasetLoadError,
    DatasetProfilerError,
    EmptyDatasetError,
    FileSizeLimitError,
    UnsupportedFileFormatError,
)
from modules.intent_detector import IntentDetectionError, InvalidQuestionError
from modules.prompt_optimizer import EmptyPromptError, PromptOptimizationError
from modules.prompt_quality_model import InvalidPromptInputError, PromptQualityModelError
from modules.report_generator import ReportGenerationError
from modules.unstructured_converter import (
    EmptyOrCorruptedDocumentError,
    UnstructuredConverterError,
    UnsupportedDocumentFormatError,
)
from modules.validator import ValidationError

logger = get_logger(__name__)


def _error_response(status_code: int, error: str, detail: str) -> JSONResponse:
    body = ErrorResponse(error=error, detail=detail, status_code=status_code)
    return JSONResponse(status_code=status_code, content=body.model_dump())


def _register_exception_handlers(app: FastAPI) -> None:
    """Map domain exceptions consistently to HTTP status codes and ErrorResponse payloads."""

    @app.exception_handler(DatasetSessionNotFoundError)
    async def session_not_found_handler(_: Request, exc: DatasetSessionNotFoundError) -> JSONResponse:
        return _error_response(status.HTTP_404_NOT_FOUND, "DatasetSessionNotFound", str(exc))

    @app.exception_handler(RecordNotFoundError)
    async def record_not_found_handler(_: Request, exc: RecordNotFoundError) -> JSONResponse:
        return _error_response(status.HTTP_404_NOT_FOUND, "RecordNotFound", str(exc))

    @app.exception_handler(DuplicateUserError)
    async def duplicate_user_handler(_: Request, exc: DuplicateUserError) -> JSONResponse:
        return _error_response(status.HTTP_409_CONFLICT, "DuplicateUser", str(exc))

    @app.exception_handler(AuthenticationError)
    async def authentication_error_handler(_: Request, exc: AuthenticationError) -> JSONResponse:
        return _error_response(status.HTTP_401_UNAUTHORIZED, "AuthenticationFailed", str(exc))

    @app.exception_handler(UnsupportedFileFormatError)
    async def unsupported_format_handler(_: Request, exc: UnsupportedFileFormatError) -> JSONResponse:
        return _error_response(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "UnsupportedFileFormat", str(exc))

    @app.exception_handler(FileSizeLimitError)
    async def file_size_handler(_: Request, exc: FileSizeLimitError) -> JSONResponse:
        return _error_response(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "FileSizeLimitExceeded", str(exc))

    @app.exception_handler(EmptyDatasetError)
    async def empty_dataset_handler(_: Request, exc: EmptyDatasetError) -> JSONResponse:
        return _error_response(status.HTTP_400_BAD_REQUEST, "EmptyOrMissingDataset", str(exc))

    @app.exception_handler(DatasetLoadError)
    async def dataset_load_handler(_: Request, exc: DatasetLoadError) -> JSONResponse:
        return _error_response(status.HTTP_400_BAD_REQUEST, "DatasetLoadError", str(exc))

    @app.exception_handler(DatasetProfilerError)
    async def profiler_handler(_: Request, exc: DatasetProfilerError) -> JSONResponse:
        return _error_response(status.HTTP_400_BAD_REQUEST, "DatasetProfilerError", str(exc))

    @app.exception_handler(ColumnNotFoundError)
    @app.exception_handler(InvalidCleaningStrategyError)
    @app.exception_handler(DataCleaningError)
    async def cleaning_handler(_: Request, exc: DataCleaningError) -> JSONResponse:
        return _error_response(status.HTTP_400_BAD_REQUEST, type(exc).__name__, str(exc))

    @app.exception_handler(InvalidQuestionError)
    @app.exception_handler(IntentDetectionError)
    @app.exception_handler(EmptyPromptError)
    @app.exception_handler(PromptOptimizationError)
    @app.exception_handler(InvalidPromptInputError)
    @app.exception_handler(PromptQualityModelError)
    async def prompt_handler(_: Request, exc: Exception) -> JSONResponse:
        return _error_response(status.HTTP_400_BAD_REQUEST, type(exc).__name__, str(exc))

    @app.exception_handler(InvalidAnalysisColumnError)
    @app.exception_handler(UnsupportedAnalysisTaskError)
    @app.exception_handler(InsufficientAnalysisDataError)
    @app.exception_handler(AnalysisEngineError)
    async def analysis_handler(_: Request, exc: AnalysisEngineError) -> JSONResponse:
        return _error_response(status.HTTP_400_BAD_REQUEST, type(exc).__name__, str(exc))

    @app.exception_handler(InvalidAnomalyConfigError)
    @app.exception_handler(InsufficientAnomalyDataError)
    @app.exception_handler(AnomalyDetectionError)
    async def anomaly_handler(_: Request, exc: AnomalyDetectionError) -> JSONResponse:
        return _error_response(status.HTTP_400_BAD_REQUEST, type(exc).__name__, str(exc))

    @app.exception_handler(ValidationError)
    @app.exception_handler(ReportGenerationError)
    @app.exception_handler(UnsupportedDocumentFormatError)
    @app.exception_handler(EmptyOrCorruptedDocumentError)
    @app.exception_handler(UnstructuredConverterError)
    @app.exception_handler(ValueError)
    async def validation_or_value_handler(_: Request, exc: Exception) -> JSONResponse:
        return _error_response(status.HTTP_400_BAD_REQUEST, type(exc).__name__, str(exc))

    @app.exception_handler(DatabaseConnectionError)
    async def db_error_handler(_: Request, exc: DatabaseConnectionError) -> JSONResponse:
        return _error_response(status.HTTP_503_SERVICE_UNAVAILABLE, "DatabaseConnectionError", str(exc))

    @app.exception_handler(Exception)
    async def fallback_exception_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled backend exception: %s", exc)
        return _error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "InternalServerError",
            "An unexpected internal error occurred while processing your request.",
        )


def create_app(
    settings: Settings | None = None,
    db_manager: DatabaseManager | None = None,
) -> FastAPI:
    """Application factory creating a configured FastAPI instance."""
    cfg = settings or get_settings()
    setup_logging(settings=cfg)

    service = AnalyticsService(settings=cfg, db_manager=db_manager)

    @asynccontextmanager
    async def lifespan(fastapi_app: FastAPI) -> AsyncIterator[None]:
        logger.info(
            "Starting SmartPrompt AI FastAPI backend (mode=%s, host=%s, port=%s).",
            cfg.execution_mode.value,
            cfg.api_host,
            cfg.api_port,
        )
        yield
        cleared = fastapi_app.state.analytics_service.session_store.clear_all()
        fastapi_app.state.analytics_service.db_manager.dispose()
        logger.info("SmartPrompt AI FastAPI backend shut down cleanly (cleared %d session(s)).", cleared)

    fastapi_app = FastAPI(
        title="SmartPrompt AI API",
        description=(
            "Dataset-Aware, Self-Validating Intelligent Data Analyst REST API. "
            "Provides CSV/Excel profiling, non-destructive cleaning, NLP intent detection, "
            "dataset-aware prompt optimization, ML prompt quality scoring, deterministic EDA, "
            "Isolation Forest anomaly detection, evidence-based validation, Plotly visualizations, "
            "SQLAlchemy history, and ReportLab PDF/CSV exports."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    fastapi_app.state.analytics_service = service

    # Safe environment-based CORS setup
    allowed_origins = list(cfg.cors_allowed_origins)
    allow_creds = "*" not in allowed_origins
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=allow_creds,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    _register_exception_handlers(fastapi_app)

    fastapi_app.include_router(health.router)
    fastapi_app.include_router(auth.router)
    fastapi_app.include_router(dataset.router)
    fastapi_app.include_router(converter.router)
    fastapi_app.include_router(prompt.router)
    fastapi_app.include_router(analysis.router)
    fastapi_app.include_router(history.router)

    return fastapi_app


app = create_app()
