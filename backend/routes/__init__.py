"""FastAPI route package exports for SmartPrompt AI."""

from backend.routes import analysis, auth, converter, dataset, health, history, prompt

__all__ = ["analysis", "auth", "converter", "dataset", "health", "history", "prompt"]
