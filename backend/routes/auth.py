"""Authentication routes for SmartPrompt AI (/api/auth/*)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, status

from backend.routes.dependencies import get_analytics_service
from backend.schemas.requests import LoginRequest, RegisterRequest
from backend.schemas.responses import AuthTokenResponse, LogoutResponse, UserPublicResponse
from backend.services.analytics_service import AnalyticsService
from database.db_manager import AuthenticationError

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


def _extract_bearer_token(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header is required.",
        )
    parts = authorization.strip().split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format. Expected 'Bearer <token>'.",
        )
    return parts[1].strip()


@router.post("/register", response_model=AuthTokenResponse, status_code=status.HTTP_201_CREATED)
def register_user(
    payload: RegisterRequest,
    service: AnalyticsService = Depends(get_analytics_service),
) -> AuthTokenResponse:
    """Register a new user account with salted PBKDF2 password hashing."""
    if payload.password != payload.confirm_password:
        raise ValueError("Password and confirm password do not match.")

    user = service.db_manager.register_user(
        full_name=payload.full_name,
        username=payload.username,
        email=payload.email,
        password=payload.password,
    )
    return AuthTokenResponse(
        access_token=None,
        user=UserPublicResponse(**user.to_dict()),
        message=f"Account created successfully for {user.full_name}. Please sign in.",
    )


@router.post("/login", response_model=AuthTokenResponse)
def login_user(
    payload: LoginRequest,
    service: AnalyticsService = Depends(get_analytics_service),
) -> AuthTokenResponse:
    """Authenticate by username or email and return a bearer session token."""
    user, raw_token = service.db_manager.authenticate_user(
        identifier=payload.identifier,
        password=payload.password,
    )
    return AuthTokenResponse(
        access_token=raw_token,
        token_type="bearer",
        user=UserPublicResponse(**user.to_dict()),
        message=f"Welcome back, {user.full_name}.",
    )


@router.get("/me", response_model=UserPublicResponse)
def get_current_user(
    authorization: str | None = Header(default=None),
    service: AnalyticsService = Depends(get_analytics_service),
) -> UserPublicResponse:
    """Return the authenticated user's profile for an active bearer session."""
    token = _extract_bearer_token(authorization)
    user = service.db_manager.get_user_by_session_token(token)
    if user is None:
        raise AuthenticationError("Session is invalid or has expired. Please log in again.")
    return UserPublicResponse(**user.to_dict())


@router.post("/logout", response_model=LogoutResponse)
def logout_user(
    authorization: str | None = Header(default=None),
    service: AnalyticsService = Depends(get_analytics_service),
) -> LogoutResponse:
    """Revoke the current bearer session token."""
    token = _extract_bearer_token(authorization)
    revoked = service.db_manager.revoke_session_token(token)
    return LogoutResponse(
        logged_out=revoked,
        message="You have been logged out." if revoked else "Session already closed.",
    )
