"""Prompt optimization and ML prompt quality evaluation routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.routes.dependencies import get_analytics_service
from backend.schemas.requests import PromptOptimizeRequest, PromptScoreRequest
from backend.schemas.responses import PromptOptimizeResponse, PromptScoreResponse
from backend.services.analytics_service import AnalyticsService

router = APIRouter(prefix="/api/prompt", tags=["SmartPrompt Engine"])


@router.post(
    "/optimize",
    response_model=PromptOptimizeResponse,
    summary="Optimize a user prompt using dataset metadata and intent detection",
)
def optimize_prompt_endpoint(
    payload: PromptOptimizeRequest,
    service: AnalyticsService = Depends(get_analytics_service),
) -> PromptOptimizeResponse:
    """Run Module 3 intent detection and Module 4 dataset-aware prompt optimization."""
    result = service.optimize_user_prompt(payload)
    return PromptOptimizeResponse(**result)


@router.post(
    "/score",
    response_model=PromptScoreResponse,
    summary="Evaluate analytical prompt quality using the ML classifier",
)
def score_prompt_endpoint(
    payload: PromptScoreRequest,
    service: AnalyticsService = Depends(get_analytics_service),
) -> PromptScoreResponse:
    """Run Module 5 Scikit-learn TF-IDF + LogisticRegression prompt quality prediction."""
    result = service.score_user_prompt(payload)
    return PromptScoreResponse(**result)
