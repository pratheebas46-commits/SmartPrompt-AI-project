"""Thread-safe server-side session store with TTL eviction for active datasets."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from threading import RLock
import time
from typing import Any
import uuid

import pandas as pd

from modules.analysis_engine import AnalysisResult
from modules.anomaly_detector import AnomalyReport
from modules.dataset_profiler import DatasetProfileResult
from modules.prompt_optimizer import PromptOptimizationResult
from modules.validator import ValidationReport


class DatasetSessionNotFoundError(Exception):
    """Raised when a requested dataset_id is missing or expired from server memory."""


@dataclass
class DatasetSession:
    """In-memory working state for an active uploaded dataset."""

    dataset_id: str
    dataframe: pd.DataFrame
    profile: DatasetProfileResult
    original_dataframe: pd.DataFrame
    last_accessed: float = field(default_factory=time.monotonic)
    latest_prompt_result: PromptOptimizationResult | None = None
    latest_analysis_result: AnalysisResult | None = None
    latest_validation_report: ValidationReport | None = None
    latest_anomaly_report: AnomalyReport | None = None
    analysis_artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)


class DatasetSessionStore:
    """Bounded thread-safe store with automatic TTL cleanup of dataset sessions."""

    def __init__(self, max_sessions: int = 30, ttl_seconds: int = 1800) -> None:
        self._max_sessions = max_sessions
        self._ttl_seconds = ttl_seconds
        self._lock = RLock()
        self._sessions: OrderedDict[str, DatasetSession] = OrderedDict()

    def cleanup_expired(self) -> int:
        """Evict expired sessions to reclaim memory cleanly."""
        now = time.monotonic()
        removed = 0
        with self._lock:
            expired_ids = [
                sid
                for sid, sess in self._sessions.items()
                if (now - sess.last_accessed) > self._ttl_seconds
            ]
            for sid in expired_ids:
                self._sessions.pop(sid, None)
                removed += 1
        return removed

    def create_session(
        self,
        df: pd.DataFrame,
        profile: DatasetProfileResult,
        dataset_id: str | None = None,
    ) -> DatasetSession:
        self.cleanup_expired()
        sid = dataset_id or str(uuid.uuid4())
        session = DatasetSession(
            dataset_id=sid,
            dataframe=df.copy(deep=True),
            profile=profile,
            original_dataframe=df.copy(deep=True),
        )
        with self._lock:
            self._sessions[sid] = session
            self._sessions.move_to_end(sid)
            while len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)
        return session

    def get_session(self, dataset_id: str) -> DatasetSession:
        self.cleanup_expired()
        with self._lock:
            session = self._sessions.get(dataset_id)
            if session is None:
                raise DatasetSessionNotFoundError(
                    f"Dataset session '{dataset_id}' not found or expired. Please upload a dataset first."
                )
            session.last_accessed = time.monotonic()
            self._sessions.move_to_end(dataset_id)
            return session

    def update_dataframe(
        self,
        dataset_id: str,
        df: pd.DataFrame,
        profile: DatasetProfileResult,
    ) -> DatasetSession:
        with self._lock:
            session = self.get_session(dataset_id)
            session.dataframe = df.copy(deep=True)
            session.profile = profile
            session.last_accessed = time.monotonic()
            return session

    def record_run_artifacts(
        self,
        dataset_id: str,
        analysis_id: str | None,
        prompt_result: PromptOptimizationResult | None = None,
        analysis_result: AnalysisResult | None = None,
        validation_report: ValidationReport | None = None,
        anomaly_report: AnomalyReport | None = None,
    ) -> None:
        with self._lock:
            session = self.get_session(dataset_id)
            if prompt_result is not None:
                session.latest_prompt_result = prompt_result
            if analysis_result is not None:
                session.latest_analysis_result = analysis_result
            if validation_report is not None:
                session.latest_validation_report = validation_report
            if anomaly_report is not None:
                session.latest_anomaly_report = anomaly_report
            if analysis_id:
                session.analysis_artifacts[analysis_id] = {
                    "prompt_result": prompt_result or session.latest_prompt_result,
                    "analysis_result": analysis_result or session.latest_analysis_result,
                    "validation_report": validation_report or session.latest_validation_report,
                    "anomaly_report": anomaly_report or session.latest_anomaly_report,
                }

    def delete_session(self, dataset_id: str) -> bool:
        with self._lock:
            removed = self._sessions.pop(dataset_id, None)
            return removed is not None

    def clear_all(self) -> int:
        with self._lock:
            count = len(self._sessions)
            self._sessions.clear()
            return count

    def count_sessions(self) -> int:
        self.cleanup_expired()
        with self._lock:
            return len(self._sessions)
