"""Central settings management for SmartPrompt AI (Offline & Online modes)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
import tempfile

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - fallback when python-dotenv is not yet installed
    load_dotenv = None


BASE_DIR: Path = Path(__file__).resolve().parent.parent


class ExecutionMode(str, Enum):
    """Supported application execution modes."""

    OFFLINE = "offline"
    ONLINE = "online"


def _fallback_load_env_file(env_path: Path) -> None:
    """Lightweight .env loader used only when python-dotenv is unavailable."""
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_environment(env_file: Path | None = None) -> None:
    """Load environment variables from a .env file without overriding OS env vars."""
    target_file = env_file or (BASE_DIR / ".env")
    if load_dotenv is not None:
        load_dotenv(dotenv_path=target_file, override=False)
    else:
        _fallback_load_env_file(target_file)


def _parse_int(env_key: str, default: int) -> int:
    raw_value = os.getenv(env_key)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable '{env_key}' must be an integer, got '{raw_value}'.") from exc


def _parse_bool(env_key: str, default: bool) -> bool:
    raw_value = os.getenv(env_key)
    if raw_value is None or raw_value.strip() == "":
        return default
    norm = raw_value.strip().lower()
    if norm in {"1", "true", "yes", "on"}:
        return True
    if norm in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Environment variable '{env_key}' must be a boolean, got '{raw_value}'.")


def _resolve_path(raw_path: str, base_dir: Path) -> Path:
    candidate = Path(raw_path)
    return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()


def _parse_execution_mode(raw_mode: str) -> ExecutionMode:
    normalized = raw_mode.strip().lower()
    try:
        return ExecutionMode(normalized)
    except ValueError as exc:
        valid = ", ".join(mode.value for mode in ExecutionMode)
        raise ValueError(f"Invalid EXECUTION_MODE '{raw_mode}'. Must be one of: {valid}.") from exc


def normalize_database_url(raw_url: str) -> str:
    """Normalize cloud PostgreSQL URLs (postgres:// -> postgresql+psycopg2://) for SQLAlchemy 2.0."""
    url = raw_url.strip()
    if not url:
        return ""
    if url.startswith("postgres://"):
        return "postgresql+psycopg2://" + url[len("postgres://") :]
    if url.startswith("postgresql://"):
        return "postgresql+psycopg2://" + url[len("postgresql://") :]
    return url


@dataclass(frozen=True)
class Settings:
    """Immutable central configuration object for SmartPrompt AI."""

    base_dir: Path = BASE_DIR
    app_name: str = "SmartPrompt AI"
    app_env: str = "development"
    execution_mode: ExecutionMode = ExecutionMode.OFFLINE
    strict_online_mode: bool = False

    # Server & Deployment
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_base_url: str = "http://127.0.0.1:8000"
    streamlit_host: str = "127.0.0.1"
    streamlit_port: int = 8501
    app_url: str = ""
    cors_allowed_origins: tuple[str, ...] = (
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    )

    # Database
    sqlite_db_path: Path = field(default_factory=lambda: (BASE_DIR / "data" / "smartprompt_ai.db").resolve())
    database_url: str = ""
    db_pool_size: int = 5
    db_max_overflow: int = 10

    # AI Provider (Server-side only)
    ai_provider: str = "local"
    gemini_api_key: str = ""
    online_ai_model: str = "gemini-2.5-flash"
    ai_request_timeout_sec: int = 20

    # Storage, Temp & Upload Limits
    data_dir: Path = field(default_factory=lambda: (BASE_DIR / "data").resolve())
    models_dir: Path = field(default_factory=lambda: (BASE_DIR / "ml_models").resolve())
    reports_dir: Path = field(default_factory=lambda: (BASE_DIR / "reports").resolve())
    temp_dir: Path = field(default_factory=lambda: (Path(tempfile.gettempdir()) / "smartprompt_ai").resolve())
    max_upload_size_mb: int = 50
    session_ttl_seconds: int = 1800
    max_active_sessions: int = 30

    # Logging
    log_level: str = "INFO"
    enable_file_logging: bool = True
    log_dir: Path = field(default_factory=lambda: (BASE_DIR / "logs").resolve())
    log_file_name: str = "smartprompt_ai.log"
    log_max_bytes: int = 5 * 1024 * 1024
    log_backup_count: int = 3

    @property
    def is_offline(self) -> bool:
        return self.execution_mode == ExecutionMode.OFFLINE

    @property
    def is_online(self) -> bool:
        return self.execution_mode == ExecutionMode.ONLINE

    @property
    def effective_database_url(self) -> str:
        """Return SQLAlchemy connection URL appropriate for the active mode."""
        normalized_remote = normalize_database_url(self.database_url)
        if self.is_online and normalized_remote:
            return normalized_remote
        return f"sqlite:///{self.sqlite_db_path.as_posix()}"

    @property
    def has_valid_online_ai_key(self) -> bool:
        """Return True if a non-placeholder server-side AI key is present."""
        key = self.gemini_api_key.strip()
        return bool(key) and key != "MY_GEMINI_API_KEY"

    @property
    def log_file_path(self) -> Path:
        return self.log_dir / self.log_file_name

    def ensure_directories(self) -> None:
        """Create required runtime directories safely without assuming persistent local disk in online mode."""
        if self.is_online:
            # In online/cloud environments, use ephemeral temp workspace; avoid requiring repo local paths.
            for directory in (self.temp_dir, self.models_dir):
                directory.mkdir(parents=True, exist_ok=True)
            if not normalize_database_url(self.database_url):
                self.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)
            if self.enable_file_logging:
                self.log_dir.mkdir(parents=True, exist_ok=True)
            return

        directories = [
            self.data_dir,
            self.data_dir / "raw",
            self.data_dir / "processed",
            self.data_dir / "sample",
            self.models_dir,
            self.reports_dir,
            self.temp_dir,
            self.sqlite_db_path.parent,
        ]
        if self.enable_file_logging:
            directories.append(self.log_dir)
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def sanitized_summary(self) -> dict[str, object]:
        """Return a safe, secret-free diagnostic snapshot of current settings."""
        db_scheme = self.effective_database_url.split("://", 1)[0]
        return {
            "app_name": self.app_name,
            "app_env": self.app_env,
            "execution_mode": self.execution_mode.value,
            "strict_online_mode": self.strict_online_mode,
            "api_host": self.api_host,
            "api_port": self.api_port,
            "api_base_url": self.api_base_url,
            "streamlit_host": self.streamlit_host,
            "streamlit_port": self.streamlit_port,
            "cors_allowed_origins": list(self.cors_allowed_origins),
            "database_backend": db_scheme,
            "ai_provider": self.ai_provider,
            "online_ai_configured": self.has_valid_online_ai_key,
            "online_ai_model": self.online_ai_model,
            "max_upload_size_mb": self.max_upload_size_mb,
            "session_ttl_seconds": self.session_ttl_seconds,
            "enable_file_logging": self.enable_file_logging,
            "log_level": self.log_level,
        }

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> "Settings":
        """Construct a validated Settings instance from environment variables."""
        load_environment(env_file=env_file)

        mode = _parse_execution_mode(os.getenv("EXECUTION_MODE", ExecutionMode.OFFLINE.value))
        default_root = (
            (Path(tempfile.gettempdir()) / "smartprompt_ai").resolve()
            if mode == ExecutionMode.ONLINE
            else BASE_DIR
        )

        sqlite_default = (
            str(default_root / "smartprompt_ai.db")
            if mode == ExecutionMode.ONLINE
            else "data/smartprompt_ai.db"
        )
        sqlite_path = _resolve_path(os.getenv("SQLITE_DB_PATH", sqlite_default), BASE_DIR)
        data_dir = _resolve_path(os.getenv("DATA_DIR", str(default_root / "data")), BASE_DIR)
        models_dir = _resolve_path(os.getenv("MODELS_DIR", str(default_root / "ml_models")), BASE_DIR)
        reports_dir = _resolve_path(os.getenv("REPORTS_DIR", str(default_root / "reports")), BASE_DIR)
        log_dir = _resolve_path(os.getenv("LOG_DIR", str(default_root / "logs")), BASE_DIR)
        temp_dir = _resolve_path(
            os.getenv("TEMP_DIR", str((Path(tempfile.gettempdir()) / "smartprompt_ai").resolve())),
            BASE_DIR,
        )

        log_level = os.getenv("LOG_LEVEL", "INFO").strip().upper()
        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if log_level not in valid_log_levels:
            raise ValueError(f"Invalid LOG_LEVEL '{log_level}'. Must be one of: {sorted(valid_log_levels)}.")

        max_upload_mb = _parse_int("MAX_UPLOAD_SIZE_MB", 50)
        if max_upload_mb <= 0:
            raise ValueError("MAX_UPLOAD_SIZE_MB must be a positive integer.")

        session_ttl = _parse_int("SESSION_TTL_SECONDS", 1800)
        if session_ttl <= 0:
            raise ValueError("SESSION_TTL_SECONDS must be a positive integer.")

        max_sessions = _parse_int("MAX_ACTIVE_SESSIONS", 30)
        if max_sessions <= 0:
            raise ValueError("MAX_ACTIVE_SESSIONS must be a positive integer.")

        raw_cors = os.getenv(
            "CORS_ALLOWED_ORIGINS",
            "http://localhost:8501,http://127.0.0.1:8501,http://localhost:3000,http://127.0.0.1:3000",
        )
        cors_origins = tuple(origin.strip() for origin in raw_cors.split(",") if origin.strip())
        app_url_val = os.getenv("APP_URL", "").strip()
        if app_url_val and app_url_val != "MY_APP_URL" and app_url_val not in cors_origins:
            cors_origins = (*cors_origins, app_url_val)

        default_host = "0.0.0.0" if mode == ExecutionMode.ONLINE else "127.0.0.1"
        api_host = os.getenv("API_HOST", default_host).strip() or default_host
        api_port = _parse_int("PORT", _parse_int("API_PORT", 8000))
        default_api_url = f"http://{'127.0.0.1' if api_host == '0.0.0.0' else api_host}:{api_port}"
        api_base_url = os.getenv("API_BASE_URL", default_api_url).strip() or default_api_url
        streamlit_host = os.getenv("STREAMLIT_HOST", default_host).strip() or default_host

        return cls(
            base_dir=BASE_DIR,
            app_name=os.getenv("APP_NAME", "SmartPrompt AI").strip() or "SmartPrompt AI",
            app_env=os.getenv("APP_ENV", "production" if mode == ExecutionMode.ONLINE else "development").strip(),
            execution_mode=mode,
            strict_online_mode=_parse_bool("STRICT_ONLINE_MODE", False),
            api_host=api_host,
            api_port=api_port,
            api_base_url=api_base_url,
            streamlit_host=streamlit_host,
            streamlit_port=_parse_int("STREAMLIT_PORT", 8501),
            app_url=app_url_val,
            cors_allowed_origins=cors_origins,
            sqlite_db_path=sqlite_path,
            database_url=normalize_database_url(os.getenv("DATABASE_URL", "")),
            db_pool_size=_parse_int("DB_POOL_SIZE", 5),
            db_max_overflow=_parse_int("DB_MAX_OVERFLOW", 10),
            ai_provider=os.getenv("AI_PROVIDER", "local").strip().lower() or "local",
            gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
            online_ai_model=os.getenv("ONLINE_AI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash",
            ai_request_timeout_sec=_parse_int("AI_REQUEST_TIMEOUT_SEC", 20),
            data_dir=data_dir,
            models_dir=models_dir,
            reports_dir=reports_dir,
            temp_dir=temp_dir,
            max_upload_size_mb=max_upload_mb,
            session_ttl_seconds=session_ttl,
            max_active_sessions=max_sessions,
            log_level=log_level,
            enable_file_logging=_parse_bool("ENABLE_FILE_LOGGING", mode == ExecutionMode.OFFLINE),
            log_dir=log_dir,
            log_file_name=os.getenv("LOG_FILE_NAME", "smartprompt_ai.log").strip() or "smartprompt_ai.log",
            log_max_bytes=_parse_int("LOG_MAX_BYTES", 5 * 1024 * 1024),
            log_backup_count=_parse_int("LOG_BACKUP_COUNT", 3),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings instance for the current process."""
    return Settings.from_env()


def reload_settings(env_file: Path | None = None) -> Settings:
    """Clear settings cache and reload configuration from environment."""
    get_settings.cache_clear()
    return Settings.from_env(env_file=env_file)
