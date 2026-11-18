"""Centralized logging configuration for SmartPrompt AI."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import sys

from config.settings import Settings, get_settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_ROOT_LOGGER_NAME = "smartprompt_ai"


def setup_logging(settings: Settings | None = None, force_reconfigure: bool = False) -> logging.Logger:
    """Configure console logging and optional rotating file logging safely."""
    cfg = settings or get_settings()
    cfg.ensure_directories()

    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    level = getattr(logging, cfg.log_level, logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers and not force_reconfigure:
        return logger

    if force_reconfigure:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if cfg.enable_file_logging:
        try:
            file_handler = RotatingFileHandler(
                filename=cfg.log_file_path,
                maxBytes=cfg.log_max_bytes,
                backupCount=cfg.log_backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError as exc:
            logger.warning("File logging unavailable (%s); continuing with stdout logging.", exc)

    logger.debug(
        "Logging initialized in %s mode (level=%s, file_logging=%s).",
        cfg.execution_mode.value,
        cfg.log_level,
        cfg.enable_file_logging,
    )
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the SmartPrompt AI namespace."""
    root_logger = logging.getLogger(_ROOT_LOGGER_NAME)
    if not root_logger.handlers:
        setup_logging()
    suffix = name.removeprefix(f"{_ROOT_LOGGER_NAME}.")
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{suffix}")
