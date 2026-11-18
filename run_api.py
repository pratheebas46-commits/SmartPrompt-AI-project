"""Convenience launcher for SmartPrompt AI FastAPI backend (localhost & cloud deployment)."""

from __future__ import annotations

import importlib.util
import os
import sys

if importlib.util.find_spec("uvicorn") is None or importlib.util.find_spec("fastapi") is None:
    print(
        "FastAPI/Uvicorn dependencies are not installed in this Python environment.\n"
        "Run: pip install -r requirements.txt",
        file=sys.stderr,
    )
    raise SystemExit(1)

import uvicorn

from config.settings import get_settings


def main() -> None:
    settings = get_settings()
    host = os.getenv("API_HOST", settings.api_host)
    port = int(os.getenv("PORT", os.getenv("API_PORT", str(settings.api_port))))
    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=settings.app_env == "development",
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
