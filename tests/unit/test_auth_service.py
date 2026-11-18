"""Unit & FastAPI tests for SmartPrompt AI user registration, login, session persistence, and logout."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from backend.main import create_app
from config.settings import ExecutionMode, Settings
from database.db_manager import (
    AuthenticationError,
    DatabaseManager,
    DuplicateUserError,
    hash_password,
    verify_password,
)


def _make_test_db(tmp_path: Path) -> DatabaseManager:
    db_file = tmp_path / "auth_test.db"
    cfg = Settings(execution_mode=ExecutionMode.OFFLINE, database_url=f"sqlite:///{db_file}")
    return DatabaseManager(settings=cfg)


def test_password_hashing_salted_and_verified() -> None:
    h1 = hash_password("Analyst2025!")
    h2 = hash_password("Analyst2025!")
    assert h1 != h2  # unique random salt per hash
    assert "Analyst2025!" not in h1
    assert h1.startswith("pbkdf2_sha256$")
    assert verify_password("Analyst2025!", h1) is True
    assert verify_password("WrongPass123", h1) is False


def test_db_register_duplicate_login_and_logout(tmp_path: Path) -> None:
    db = _make_test_db(tmp_path)
    try:
        user = db.register_user(
            full_name="Aarav Mehta",
            username="AaravM",
            email="Aarav@Example.COM",
            password="Password123",
        )
        assert user.username == "aaravm"
        assert user.email == "aarav@example.com"

        with pytest.raises(DuplicateUserError):
            db.register_user("Other User", "aaravm", "other@example.com", "Password123")

        with pytest.raises(DuplicateUserError):
            db.register_user("Other User", "otheruser", "aarav@example.com", "Password123")

        with pytest.raises(AuthenticationError):
            db.authenticate_user("aaravm", "WrongPass99")

        auth_user, token = db.authenticate_user("aarav@example.com", "Password123")
        assert auth_user.user_id == user.user_id
        assert db.get_user_by_session_token(token) is not None

        assert db.revoke_session_token(token) is True
        assert db.get_user_by_session_token(token) is None
    finally:
        db.dispose()


def test_fastapi_auth_lifecycle(tmp_path: Path) -> None:
    db = _make_test_db(tmp_path)
    cfg = Settings(execution_mode=ExecutionMode.OFFLINE, database_url=db.database_url)
    app = create_app(settings=cfg, db_manager=db)

    with TestClient(app) as client:
        # Unauthenticated /me fails
        unauth_resp = client.get("/api/auth/me")
        assert unauth_resp.status_code == 401

        # Register
        reg_resp = client.post(
            "/api/auth/register",
            json={
                "full_name": "Priya Nair",
                "username": "priya_nair",
                "email": "priya@smartprompt.ai",
                "password": "StrongPass123",
                "confirm_password": "StrongPass123",
            },
        )
        assert reg_resp.status_code == 201

        # Duplicate rejected cleanly with 409
        dup_resp = client.post(
            "/api/auth/register",
            json={
                "full_name": "Priya Clone",
                "username": "priya_nair",
                "email": "clone@smartprompt.ai",
                "password": "StrongPass123",
                "confirm_password": "StrongPass123",
            },
        )
        assert dup_resp.status_code == 409

        # Bad password fails with 401
        bad_login = client.post(
            "/api/auth/login",
            json={"identifier": "priya_nair", "password": "WrongPassword1"},
        )
        assert bad_login.status_code == 401

        # Valid login returns bearer token
        good_login = client.post(
            "/api/auth/login",
            json={"identifier": "priya@smartprompt.ai", "password": "StrongPass123"},
        )
        assert good_login.status_code == 200
        token = good_login.json()["access_token"]
        assert token

        # Authenticated /me succeeds
        me_resp = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me_resp.status_code == 200
        assert me_resp.json()["username"] == "priya_nair"

        # Logout revokes token
        logout_resp = client.post("/api/auth/logout", headers={"Authorization": f"Bearer {token}"})
        assert logout_resp.status_code == 200
        assert logout_resp.json()["logged_out"] is True

        # Token rejected after logout
        me_after = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me_after.status_code == 401
