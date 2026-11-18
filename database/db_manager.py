"""Module 10: Database Management and Analysis History for SmartPrompt AI.

Uses SQLAlchemy 2.0 to persist analysis history and dataset metadata (without storing
raw uploaded datasets by default) in SQLite for local offline execution and optional
PostgreSQL for online deployment via environment-based configuration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
from pathlib import Path
import re
import secrets
from typing import Any
import uuid

from sqlalchemy import Boolean, DateTime, Integer, String, Text, create_engine, delete, func, or_, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from config.logging_config import get_logger
from config.mode_manager import ModeManager
from config.settings import Settings, get_settings

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------


class DatabaseManagerError(Exception):
    """Base exception for database management and history operations."""


class DatabaseConnectionError(DatabaseManagerError):
    """Raised when connecting to or querying the configured database fails."""


class RecordNotFoundError(DatabaseManagerError):
    """Raised when a requested analysis history record does not exist."""


class DuplicateUserError(DatabaseManagerError):
    """Raised when registering a username or email that already exists."""


class AuthenticationError(DatabaseManagerError):
    """Raised when user credentials or session tokens are invalid."""


# ---------------------------------------------------------------------------
# SQLAlchemy ORM Schema
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    """SQLAlchemy Declarative Base for SmartPrompt AI metadata tables."""


class AnalysisHistoryORM(Base):
    """Stores analytical query history and dataset metadata (no raw dataset payloads)."""

    __tablename__ = "analysis_history"

    analysis_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    dataset_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    dataset_fingerprint: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    dataset_row_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    dataset_column_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    dataset_metadata_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    original_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    optimized_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    detected_intent: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    analysis_summary_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    validation_status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    anomaly_summary_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    execution_mode: Mapped[str] = mapped_column(String(32), nullable=False, default="offline")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )


class UserORM(Base):
    """Stores SmartPrompt AI registered user accounts with salted PBKDF2 hashes."""

    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    full_name: Mapped[str] = mapped_column(String(120), nullable=False)
    username: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class UserSessionORM(Base):
    """Stores hashed bearer session tokens for authenticated users."""

    __tablename__ = "user_sessions"

    token_hash: Mapped[str] = mapped_column(String(128), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    revoked: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


# ---------------------------------------------------------------------------
# Pydantic-Free Portable Record Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalysisHistoryRecord:
    """Clean domain record returned by DatabaseManager read/write methods."""

    analysis_id: str
    dataset_name: str
    dataset_fingerprint: str
    dataset_row_count: int
    dataset_column_count: int
    dataset_metadata: dict[str, Any]
    original_prompt: str
    optimized_prompt: str
    detected_intent: str
    analysis_summary: dict[str, Any]
    validation_status: str
    anomaly_summary: dict[str, Any] | None
    execution_mode: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UserRecord:
    """Public user account record (never contains password hash)."""

    user_id: str
    full_name: str
    username: str
    email: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_.-]{3,32}$")
_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
_PBKDF2_ITERATIONS = 120_000


def hash_password(password: str, salt_hex: str | None = None) -> str:
    """Hash a plaintext password using salted PBKDF2-HMAC-SHA256."""
    if not password:
        raise ValueError("Password cannot be empty.")
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${salt.hex()}${derived.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Constant-time verification of a password against a stored PBKDF2 hash."""
    if not password or not stored_hash:
        return False
    parts = stored_hash.split("$")
    if len(parts) != 4 or parts[0] != "pbkdf2_sha256":
        return False
    try:
        iterations = int(parts[1])
        salt = bytes.fromhex(parts[2])
        expected = bytes.fromhex(parts[3])
    except ValueError:
        return False
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(candidate, expected)


def _hash_session_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def validate_registration_inputs(
    full_name: str,
    username: str,
    email: str,
    password: str,
) -> tuple[str, str, str]:
    """Validate and normalize user registration fields, returning (full_name, username, email)."""
    clean_name = full_name.strip()
    clean_user = username.strip().lower()
    clean_email = email.strip().lower()

    if len(clean_name) < 2:
        raise ValueError("Full name must be at least 2 characters.")
    if not _USERNAME_RE.match(clean_user):
        raise ValueError("Username must be 3-32 characters using letters, numbers, '.', '-', or '_'.")
    if not _EMAIL_RE.match(clean_email):
        raise ValueError("Please enter a valid email address.")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters long.")
    if not any(ch.isalpha() for ch in password) or not any(ch.isdigit() for ch in password):
        raise ValueError("Password must include at least one letter and one number.")
    return clean_name, clean_user, clean_email


def _mask_db_credentials(db_url: str) -> str:
    """Redact username/password credentials before logging any database URL."""
    return re.sub(r"://([^:@/]+):([^@/]+)@", r"://***:***@", db_url)


def _orm_to_record(row: AnalysisHistoryORM) -> AnalysisHistoryRecord:
    created = row.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return AnalysisHistoryRecord(
        analysis_id=row.analysis_id,
        dataset_name=row.dataset_name,
        dataset_fingerprint=row.dataset_fingerprint,
        dataset_row_count=row.dataset_row_count,
        dataset_column_count=row.dataset_column_count,
        dataset_metadata=json.loads(row.dataset_metadata_json or "{}"),
        original_prompt=row.original_prompt,
        optimized_prompt=row.optimized_prompt,
        detected_intent=row.detected_intent,
        analysis_summary=json.loads(row.analysis_summary_json or "{}"),
        validation_status=row.validation_status,
        anomaly_summary=json.loads(row.anomaly_summary_json) if row.anomaly_summary_json else None,
        execution_mode=row.execution_mode,
        created_at=created.isoformat(),
    )


def _user_orm_to_record(row: UserORM) -> UserRecord:
    created = row.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return UserRecord(
        user_id=row.user_id,
        full_name=row.full_name,
        username=row.username,
        email=row.email,
        created_at=created.isoformat(),
    )


# ---------------------------------------------------------------------------
# Database Manager Service
# ---------------------------------------------------------------------------


class DatabaseManager:
    """Manages SQLAlchemy connections, schema initialization, and analysis history CRUD."""

    def __init__(
        self,
        database_url: str | None = None,
        settings: Settings | None = None,
        auto_initialize: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        capabilities = ModeManager(self.settings).resolve(strict_online=False)
        self.execution_mode = capabilities.mode.value
        self.database_url = database_url or capabilities.database_url
        self.is_sqlite = self.database_url.startswith("sqlite")

        if self.is_sqlite and not self.database_url.endswith(":memory:"):
            raw_path = self.database_url.replace("sqlite:///", "", 1)
            Path(raw_path).parent.mkdir(parents=True, exist_ok=True)

        self.engine: Engine = self._create_engine_safe(self.database_url)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False, expire_on_commit=False)

        if auto_initialize:
            self.initialize_schema()

    def _create_engine_safe(self, db_url: str) -> Engine:
        try:
            connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
            engine_kwargs: dict[str, Any] = {"connect_args": connect_args, "future": True}
            if not db_url.startswith("sqlite"):
                engine_kwargs["pool_pre_ping"] = True
                engine_kwargs["pool_recycle"] = 300
                engine_kwargs["pool_size"] = self.settings.db_pool_size
                engine_kwargs["max_overflow"] = self.settings.db_max_overflow
            return create_engine(db_url, **engine_kwargs)
        except Exception as exc:
            masked = _mask_db_credentials(db_url)
            logger.error("Failed to create SQLAlchemy engine for '%s': %s", masked, exc)
            raise DatabaseConnectionError(f"Unable to configure database engine for '{masked}': {exc}") from exc

    def initialize_schema(self) -> None:
        """Create required database tables idempotently."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database schema initialized (%s).", _mask_db_credentials(self.database_url))
        except SQLAlchemyError as exc:
            masked = _mask_db_credentials(self.database_url)
            logger.error("Schema initialization failed for '%s': %s", masked, exc)
            raise DatabaseConnectionError(f"Database schema initialization failed: {exc}") from exc

    def health_check(self) -> dict[str, Any]:
        """Verify active database connectivity with a lightweight SELECT 1 probe."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return {
                "connected": True,
                "backend": "sqlite" if self.is_sqlite else self.database_url.split("://", 1)[0],
                "execution_mode": self.execution_mode,
            }
        except SQLAlchemyError as exc:
            raise DatabaseConnectionError(f"Database health check failed: {exc}") from exc

    def save_analysis(
        self,
        dataset_name: str,
        dataset_fingerprint: str,
        original_prompt: str,
        optimized_prompt: str,
        detected_intent: str,
        analysis_summary: dict[str, Any],
        validation_status: str,
        dataset_metadata: dict[str, Any] | None = None,
        dataset_row_count: int = 0,
        dataset_column_count: int = 0,
        anomaly_summary: dict[str, Any] | None = None,
        analysis_id: str | None = None,
    ) -> AnalysisHistoryRecord:
        """Persist a new analysis history entry (metadata only, no raw dataset rows)."""
        record_id = analysis_id or str(uuid.uuid4())
        meta_json = json.dumps(dataset_metadata or {}, allow_nan=False, default=str)
        summary_json = json.dumps(analysis_summary or {}, allow_nan=False, default=str)
        anom_json = json.dumps(anomaly_summary, allow_nan=False, default=str) if anomaly_summary is not None else None

        orm_obj = AnalysisHistoryORM(
            analysis_id=record_id,
            dataset_name=dataset_name,
            dataset_fingerprint=dataset_fingerprint,
            dataset_row_count=int(dataset_row_count),
            dataset_column_count=int(dataset_column_count),
            dataset_metadata_json=meta_json,
            original_prompt=original_prompt,
            optimized_prompt=optimized_prompt,
            detected_intent=detected_intent,
            analysis_summary_json=summary_json,
            validation_status=validation_status,
            anomaly_summary_json=anom_json,
            execution_mode=self.execution_mode,
            created_at=datetime.now(timezone.utc),
        )

        try:
            with self.SessionLocal() as session:
                session.add(orm_obj)
                session.commit()
                session.refresh(orm_obj)
                logger.info("Saved analysis history record '%s' (dataset='%s').", record_id, dataset_name)
                return _orm_to_record(orm_obj)
        except SQLAlchemyError as exc:
            logger.error("Failed to save analysis history '%s': %s", record_id, exc)
            raise DatabaseConnectionError(f"Failed to persist analysis history: {exc}") from exc

    def get_analysis(self, analysis_id: str) -> AnalysisHistoryRecord | None:
        """Retrieve a single analysis history record by ID, or None if missing."""
        try:
            with self.SessionLocal() as session:
                row = session.get(AnalysisHistoryORM, analysis_id)
                return _orm_to_record(row) if row is not None else None
        except SQLAlchemyError as exc:
            raise DatabaseConnectionError(f"Failed to retrieve analysis '{analysis_id}': {exc}") from exc

    def list_analyses(
        self,
        limit: int = 50,
        offset: int = 0,
        dataset_fingerprint: str | None = None,
        detected_intent: str | None = None,
        validation_status: str | None = None,
    ) -> list[AnalysisHistoryRecord]:
        """List previous analyses ordered newest-first with optional filters."""
        try:
            with self.SessionLocal() as session:
                stmt = select(AnalysisHistoryORM)
                if dataset_fingerprint:
                    stmt = stmt.where(AnalysisHistoryORM.dataset_fingerprint == dataset_fingerprint)
                if detected_intent:
                    stmt = stmt.where(AnalysisHistoryORM.detected_intent == detected_intent)
                if validation_status:
                    stmt = stmt.where(AnalysisHistoryORM.validation_status == validation_status)
                stmt = stmt.order_by(AnalysisHistoryORM.created_at.desc()).offset(offset).limit(limit)
                rows = session.scalars(stmt).all()
                return [_orm_to_record(r) for r in rows]
        except SQLAlchemyError as exc:
            raise DatabaseConnectionError(f"Failed to list analysis history: {exc}") from exc

    def count_analyses(self, dataset_fingerprint: str | None = None) -> int:
        """Count stored analysis records."""
        try:
            with self.SessionLocal() as session:
                stmt = select(func.count()).select_from(AnalysisHistoryORM)
                if dataset_fingerprint:
                    stmt = stmt.where(AnalysisHistoryORM.dataset_fingerprint == dataset_fingerprint)
                return int(session.scalar(stmt) or 0)
        except SQLAlchemyError as exc:
            raise DatabaseConnectionError(f"Failed to count analysis history: {exc}") from exc

    def delete_analysis(self, analysis_id: str) -> bool:
        """Delete a single analysis history entry by ID. Returns True if deleted."""
        try:
            with self.SessionLocal() as session:
                row = session.get(AnalysisHistoryORM, analysis_id)
                if row is None:
                    return False
                session.delete(row)
                session.commit()
                logger.info("Deleted analysis history record '%s'.", analysis_id)
                return True
        except SQLAlchemyError as exc:
            raise DatabaseConnectionError(f"Failed to delete analysis '{analysis_id}': {exc}") from exc

    def clear_history(self, dataset_fingerprint: str | None = None) -> int:
        """Delete all history records (or all for a specific dataset). Returns deleted count."""
        try:
            with self.SessionLocal() as session:
                stmt = delete(AnalysisHistoryORM)
                if dataset_fingerprint:
                    stmt = stmt.where(AnalysisHistoryORM.dataset_fingerprint == dataset_fingerprint)
                result = session.execute(stmt)
                session.commit()
                deleted_count = int(result.rowcount or 0)
                logger.info("Cleared %d analysis history record(s).", deleted_count)
                return deleted_count
        except SQLAlchemyError as exc:
            raise DatabaseConnectionError(f"Failed to clear analysis history: {exc}") from exc

    # -----------------------------------------------------------------------
    # User Authentication & Session Persistence
    # -----------------------------------------------------------------------

    def register_user(
        self,
        full_name: str,
        username: str,
        email: str,
        password: str,
    ) -> UserRecord:
        """Create a new user account with salted PBKDF2 password hashing."""
        clean_name, clean_user, clean_email = validate_registration_inputs(
            full_name=full_name,
            username=username,
            email=email,
            password=password,
        )
        password_digest = hash_password(password)
        user_id = f"USR-{uuid.uuid4().hex[:12].upper()}"

        try:
            with self.SessionLocal() as session:
                existing = session.scalar(
                    select(UserORM).where(or_(UserORM.username == clean_user, UserORM.email == clean_email))
                )
                if existing is not None:
                    if existing.username == clean_user:
                        raise DuplicateUserError(f"Username '{clean_user}' is already registered.")
                    raise DuplicateUserError(f"Email '{clean_email}' is already registered.")

                user_row = UserORM(
                    user_id=user_id,
                    full_name=clean_name,
                    username=clean_user,
                    email=clean_email,
                    password_hash=password_digest,
                    created_at=datetime.now(timezone.utc),
                )
                session.add(user_row)
                session.commit()
                session.refresh(user_row)
                logger.info("Registered user '%s' (%s).", clean_user, user_id)
                return _user_orm_to_record(user_row)
        except DuplicateUserError:
            raise
        except IntegrityError as exc:
            raise DuplicateUserError("Username or email is already registered.") from exc
        except SQLAlchemyError as exc:
            raise DatabaseConnectionError(f"Failed to register user: {exc}") from exc

    def authenticate_user(self, identifier: str, password: str) -> tuple[UserRecord, str]:
        """Verify username or email + password and issue a persisted session token."""
        clean_identifier = identifier.strip().lower()
        if not clean_identifier or not password:
            raise AuthenticationError("Username/email and password are required.")

        try:
            with self.SessionLocal() as session:
                user_row = session.scalar(
                    select(UserORM).where(
                        or_(UserORM.username == clean_identifier, UserORM.email == clean_identifier)
                    )
                )
                if user_row is None or not verify_password(password, user_row.password_hash):
                    raise AuthenticationError("Invalid username/email or password.")

                raw_token = secrets.token_urlsafe(32)
                now = datetime.now(timezone.utc)
                sess_row = UserSessionORM(
                    token_hash=_hash_session_token(raw_token),
                    user_id=user_row.user_id,
                    created_at=now,
                    expires_at=now + timedelta(hours=72),
                    revoked=False,
                )
                session.add(sess_row)
                session.commit()
                return _user_orm_to_record(user_row), raw_token
        except AuthenticationError:
            raise
        except SQLAlchemyError as exc:
            raise DatabaseConnectionError(f"Authentication failed: {exc}") from exc

    def get_user_by_session_token(self, raw_token: str) -> UserRecord | None:
        """Resolve an active, unrevoked, unexpired bearer token to its UserRecord."""
        if not raw_token or not raw_token.strip():
            return None
        digest = _hash_session_token(raw_token.strip())
        now = datetime.now(timezone.utc)
        try:
            with self.SessionLocal() as session:
                sess_row = session.get(UserSessionORM, digest)
                if sess_row is None or sess_row.revoked:
                    return None
                expires = sess_row.expires_at
                if expires.tzinfo is None:
                    expires = expires.replace(tzinfo=timezone.utc)
                if expires <= now:
                    return None
                user_row = session.get(UserORM, sess_row.user_id)
                return _user_orm_to_record(user_row) if user_row is not None else None
        except SQLAlchemyError as exc:
            raise DatabaseConnectionError(f"Session verification failed: {exc}") from exc

    def revoke_session_token(self, raw_token: str) -> bool:
        """Revoke an existing session token on logout."""
        if not raw_token or not raw_token.strip():
            return False
        digest = _hash_session_token(raw_token.strip())
        try:
            with self.SessionLocal() as session:
                sess_row = session.get(UserSessionORM, digest)
                if sess_row is None:
                    return False
                sess_row.revoked = True
                session.commit()
                return True
        except SQLAlchemyError as exc:
            raise DatabaseConnectionError(f"Failed to revoke session: {exc}") from exc

    def dispose(self) -> None:
        """Close pooled connections cleanly."""
        self.engine.dispose()
