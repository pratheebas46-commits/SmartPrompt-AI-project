"""Convenience launcher for SmartPrompt AI Streamlit frontend (`app.py`)."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

from config.settings import get_settings


def main() -> None:
    if importlib.util.find_spec("streamlit") is None:
        print(
            "Streamlit is not installed in this Python environment.\n"
            "Run: pip install -r requirements.txt",
            file=sys.stderr,
        )
        raise SystemExit(1)

    settings = get_settings()
    app_path = Path(__file__).resolve().parent / "app.py"
    port = os.getenv("PORT", os.getenv("STREAMLIT_PORT", str(settings.streamlit_port)))
    host = os.getenv("STREAMLIT_HOST", settings.streamlit_host)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--server.address",
        host,
        "--server.headless",
        "true",
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
