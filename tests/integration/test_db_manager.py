"""Integration tests for Module 10: Database Management and Analysis History (database/db_manager.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from config.settings import Settings
from database.db_manager import (
    DatabaseConnectionError,
    DatabaseManager,
)


def test_sqlite_schema_init_and_crud_lifecycle(tmp_path: Path) -> None:
    db_file = tmp_path / "smartprompt_history.db"
    db_url = f"sqlite:///{db_file}"

    manager = DatabaseManager(database_url=db_url, auto_initialize=True)
    health = manager.health_check()
    assert health["connected"] is True
    assert health["backend"] == "sqlite"
    assert db_file.exists()

    rec1 = manager.save_analysis(
        dataset_name="sales_2025.csv",
        dataset_fingerprint="fp_abc123",
        original_prompt="Top products?",
        optimized_prompt="Rank the top 5 highest values of sum(Revenue) grouped by Product.",
        detected_intent="ranking",
        analysis_summary={"top_group": "Laptop", "top_value": 4500.0},
        validation_status="Verified",
        dataset_metadata={"numerical_columns": ["Revenue"], "categorical_columns": ["Product"]},
        dataset_row_count=120,
        dataset_column_count=5,
        anomaly_summary={"anomaly_count": 2, "anomaly_percentage": 1.67},
    )
    assert rec1.analysis_id
    assert rec1.dataset_name == "sales_2025.csv"
    assert rec1.dataset_fingerprint == "fp_abc123"
    assert rec1.validation_status == "Verified"
    assert rec1.dataset_row_count == 120
    # Confirm metadata-only persistence (no raw dataset rows stored)
    assert "raw_rows" not in rec1.to_dict()
    assert "dataframe" not in rec1.to_dict()

    rec2 = manager.save_analysis(
        dataset_name="hr_attrition.xlsx",
        dataset_fingerprint="fp_xyz789",
        original_prompt="Show salary distribution",
        optimized_prompt="Analyze statistical distribution of Salary.",
        detected_intent="distribution_analysis",
        analysis_summary={"mean": 68000.0, "median": 65000.0},
        validation_status="Warning",
        dataset_row_count=50,
        dataset_column_count=8,
    )

    fetched = manager.get_analysis(rec1.analysis_id)
    assert fetched is not None
    assert fetched.optimized_prompt == rec1.optimized_prompt
    assert fetched.anomaly_summary == {"anomaly_count": 2, "anomaly_percentage": 1.67}

    assert manager.count_analyses() == 2
    assert manager.count_analyses(dataset_fingerprint="fp_abc123") == 1

    all_records = manager.list_analyses()
    assert len(all_records) == 2
    assert all_records[0].analysis_id == rec2.analysis_id  # newest-first

    filtered_intent = manager.list_analyses(detected_intent="ranking")
    assert len(filtered_intent) == 1
    assert filtered_intent[0].analysis_id == rec1.analysis_id

    deleted = manager.delete_analysis(rec1.analysis_id)
    assert deleted is True
    assert manager.get_analysis(rec1.analysis_id) is None
    assert manager.delete_analysis("nonexistent-id") is False
    assert manager.count_analyses() == 1

    cleared = manager.clear_history()
    assert cleared == 1
    assert manager.count_analyses() == 0
    manager.dispose()


def test_environment_based_sqlite_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sqlite_path = tmp_path / "env_configured.db"
    monkeypatch.setenv("EXECUTION_MODE", "offline")
    monkeypatch.setenv("SQLITE_DB_PATH", str(sqlite_path))

    settings = Settings.from_env()
    manager = DatabaseManager(settings=settings)
    assert manager.is_sqlite is True
    assert sqlite_path.exists()
    manager.dispose()


def test_invalid_database_url_raises_connection_error() -> None:
    with pytest.raises(DatabaseConnectionError):
        DatabaseManager(database_url="invalid_driver://bad_host/bad_db", auto_initialize=True)
