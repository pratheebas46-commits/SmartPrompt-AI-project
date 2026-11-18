"""End-to-end integration test verifying Module 13 frontend API client against Module 12 FastAPI backend."""

from __future__ import annotations

import ast
from pathlib import Path

import httpx
import pytest

from backend.main import create_app
from config.settings import ExecutionMode, Settings
from frontend.api_client import BackendConnectionError, SmartPromptAPIClient


def test_frontend_has_no_direct_analytical_module_imports() -> None:
    """Verify app.py and frontend/api_client.py rely exclusively on the FastAPI backend."""
    for filepath in ("app.py", "frontend/api_client.py"):
        tree = ast.parse(Path(filepath).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("modules"), f"{filepath} must not import {alias.name}"
                    assert not alias.name.startswith("database"), f"{filepath} must not import {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("modules"), f"{filepath} must not import {node.module}"
                assert not node.module.startswith("database"), f"{filepath} must not import {node.module}"


def test_frontend_handles_backend_disconnection_gracefully() -> None:
    client = SmartPromptAPIClient(base_url="http://127.0.0.1:59999", timeout_sec=1.0)
    with pytest.raises(BackendConnectionError):
        client.health()


def test_full_upload_to_report_download_workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise complete frontend->backend workflow via ASGI transport from upload to report download."""
    settings = Settings(
        base_dir=tmp_path,
        execution_mode=ExecutionMode.OFFLINE,
        sqlite_db_path=tmp_path / "e2e.db",
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "ml_models",
        reports_dir=tmp_path / "reports",
        log_dir=tmp_path / "logs",
    )
    app = create_app(settings=settings)
    transport = httpx.ASGITransport(app=app)

    original_client_cls = httpx.Client

    def _patched_client(*args, **kwargs):
        kwargs["transport"] = transport
        return original_client_cls(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", _patched_client)

    api = SmartPromptAPIClient(base_url="http://testserver")

    # 1. Health & sanitized settings
    health = api.health()
    assert health["status"] == "healthy"
    settings_summary = api.get_settings_summary()
    assert "gemini_api_key" not in settings_summary

    # 2. Upload & profile CSV
    csv_bytes = (
        b"Order_Date,Product,Revenue,Units\n"
        b"2025-01-05,Laptop,1200,2\n"
        b"2025-01-12,Mouse,50,5\n"
        b"2025-02-01,Laptop,1300,2\n"
        b"2025-02-15,Monitor,400,1\n"
        b"2025-03-03,Mouse,60,6\n"
        b"2025-03-20,Laptop,4800,8\n"
    )
    prof = api.profile_dataset(csv_bytes, "sales_e2e.csv")
    dataset_id = prof["dataset_id"]
    assert prof["profile"]["row_count"] == 6

    # 3. Optimize & score prompt
    opt = api.optimize_prompt(dataset_id, "Top products?")["optimization"]
    assert opt["detected_intent"] == "ranking"
    score_orig = api.score_prompt(opt["original_prompt"], dataset_id=dataset_id)["prediction"]
    score_opt = api.score_prompt(opt["improved_prompt"], dataset_id=dataset_id)["prediction"]
    assert score_opt["quality_score"] >= score_orig["quality_score"]

    # 4. Run full analysis + validation + visualization + history persistence
    run = api.run_analysis(
        {
            "dataset_id": dataset_id,
            "prompt": "Rank top 3 products by total Revenue",
            "save_to_history": True,
        }
    )
    analysis_id = run["analysis_id"]
    assert run["analysis_result"]["status"] == "completed"
    assert run["validation_report"]["overall_status"] in {"Verified", "Warning"}
    assert run["visualization"]["is_renderable"] is True

    # 5. Anomaly detection
    anom = api.detect_anomalies({"dataset_id": dataset_id, "contamination": 0.2})
    assert anom["anomaly_report"]["algorithm"] == "IsolationForest"

    # 6. History inspection
    hist = api.list_history()
    assert hist["total_count"] == 1
    assert api.get_history_item(analysis_id)["item"]["analysis_id"] == analysis_id

    # 7. Download PDF & CSV reports
    pdf_bytes, pdf_name = api.export_report(dataset_id=dataset_id, export_format="pdf", analysis_id=analysis_id)
    csv_report, csv_name = api.export_report(dataset_id=dataset_id, export_format="csv", analysis_id=analysis_id)
    assert pdf_bytes.startswith(b"%PDF")
    assert pdf_name.endswith(".pdf")
    assert b"sales_e2e.csv" in csv_report
    assert csv_name.endswith(".csv")
