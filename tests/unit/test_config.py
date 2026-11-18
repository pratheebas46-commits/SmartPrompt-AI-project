"""Unit tests for Module 14: Offline and Online Execution Configuration and Logging."""

from __future__ import annotations

import logging
from pathlib import Path
import tempfile

from config.logging_config import get_logger, setup_logging
from config.mode_manager import ModeConfigurationError, ModeManager
from config.settings import ExecutionMode, Settings, normalize_database_url


def test_default_offline_settings_work_without_internet(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "offline")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "ml_models"))
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "data" / "test.db"))
    monkeypatch.setenv("ENABLE_FILE_LOGGING", "true")

    settings = Settings.from_env(env_file=tmp_path / "missing.env")
    caps = ModeManager(settings).resolve()

    assert settings.is_offline is True
    assert settings.api_host == "127.0.0.1"
    assert caps.mode == ExecutionMode.OFFLINE
    assert caps.database_backend == "sqlite"
    assert caps.use_external_ai is False
    assert caps.ai_provider == "local"
    assert caps.requires_local_storage is True
    assert (tmp_path / "data" / "raw").is_dir()
    assert (tmp_path / "ml_models").is_dir()


def test_online_mode_uses_temp_paths_and_normalizes_postgres_url(monkeypatch, tmp_path: Path) -> None:
    for var in ("DATA_DIR", "MODELS_DIR", "REPORTS_DIR", "LOG_DIR", "SQLITE_DB_PATH", "ENABLE_FILE_LOGGING"):
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setenv("EXECUTION_MODE", "online")
    monkeypatch.setenv("DATABASE_URL", "postgres://user:pass@db.example.com:5432/smartprompt")
    monkeypatch.setenv("GEMINI_API_KEY", "")

    settings = Settings.from_env(env_file=tmp_path / "missing.env")
    caps = ModeManager(settings).resolve(strict_online=False)

    assert settings.is_online is True
    assert settings.api_host == "0.0.0.0"
    assert settings.enable_file_logging is False
    assert settings.effective_database_url == "postgresql+psycopg2://user:pass@db.example.com:5432/smartprompt"
    assert str(Path(tempfile.gettempdir())) in str(settings.models_dir)
    assert caps.requires_local_storage is False
    assert caps.allow_local_fallback is True
    assert caps.use_external_ai is False
    assert normalize_database_url("postgresql://u:p@h/d") == "postgresql+psycopg2://u:p@h/d"


def test_online_mode_fallback_and_strict_validation(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "online")
    monkeypatch.setenv("STRICT_ONLINE_MODE", "true")
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("GEMINI_API_KEY", "MY_GEMINI_API_KEY")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "ml_models"))
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "data" / "test.db"))

    settings = Settings.from_env(env_file=tmp_path / "missing.env")
    manager = ModeManager(settings)

    caps = manager.resolve(strict_online=False)
    assert caps.mode == ExecutionMode.ONLINE
    assert caps.database_backend == "sqlite"
    assert caps.use_external_ai is False
    assert len(caps.warnings) == 2

    try:
        manager.resolve()  # Picks up settings.strict_online_mode == True
        raised = False
    except ModeConfigurationError:
        raised = True
    assert raised is True


def test_sanitized_summary_hides_api_key(monkeypatch, tmp_path: Path) -> None:
    secret_value = "super-secret-api-key-12345"
    monkeypatch.setenv("EXECUTION_MODE", "online")
    monkeypatch.setenv("GEMINI_API_KEY", secret_value)
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg2://user:pass@localhost:5432/db")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "ml_models"))
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))

    settings = Settings.from_env(env_file=tmp_path / "missing.env")
    summary = settings.sanitized_summary()
    caps = ModeManager(settings).resolve(strict_online=True)

    assert secret_value not in str(summary)
    assert summary["online_ai_configured"] is True
    assert caps.use_external_ai is True
    assert caps.database_backend == "postgresql+psycopg2"


def test_logging_setup_creates_log_file_and_supports_stdout_only(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "offline")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("ENABLE_FILE_LOGGING", "true")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "ml_models"))
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "data" / "test.db"))

    settings = Settings.from_env(env_file=tmp_path / "missing.env")
    root_logger = setup_logging(settings, force_reconfigure=True)
    second_call = setup_logging(settings, force_reconfigure=False)

    assert root_logger is second_call
    assert len(root_logger.handlers) == 2

    child_logger = get_logger("test_module")
    child_logger.info("Skeleton logging verification message.")
    for handler in root_logger.handlers:
        handler.flush()

    assert settings.log_file_path.is_file()
    assert "Skeleton logging verification message." in settings.log_file_path.read_text(encoding="utf-8")
    assert logging.getLogger("smartprompt_ai.test_module").name == child_logger.name

    # Cloud / online stdout-only logging mode
    monkeypatch.setenv("ENABLE_FILE_LOGGING", "false")
    cloud_settings = Settings.from_env(env_file=tmp_path / "missing.env")
    cloud_logger = setup_logging(cloud_settings, force_reconfigure=True)
    assert len(cloud_logger.handlers) == 1
