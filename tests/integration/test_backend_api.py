"""Integration tests for Module 12: FastAPI Backend using TestClient."""

from __future__ import annotations

import io
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from backend.main import create_app
from config.settings import ExecutionMode, Settings


@pytest.fixture
def api_client(tmp_path: Path) -> TestClient:
    db_file = tmp_path / "test_api.db"
    models_dir = tmp_path / "ml_models"
    settings = Settings(
        base_dir=tmp_path,
        execution_mode=ExecutionMode.OFFLINE,
        sqlite_db_path=db_file,
        data_dir=tmp_path / "data",
        models_dir=models_dir,
        reports_dir=tmp_path / "reports",
        log_dir=tmp_path / "logs",
        cors_allowed_origins=("http://localhost:8501", "https://demo.smartprompt.ai"),
    )
    app = create_app(settings=settings)
    with TestClient(app) as client:
        yield client


def _sample_csv_bytes() -> bytes:
    csv_content = (
        "Order_Date,Product,Region,Revenue,Units\n"
        "2025-01-05,Laptop,North,1200,2\n"
        "2025-01-18,Mouse,South,50,5\n"
        "2025-02-09,Laptop,North,1300,2\n"
        "2025-02-22,Monitor,East,400,1\n"
        "2025-03-11,Mouse,South,75,6\n"
        "2025-03-25,Laptop,West,5000,8\n"
        "2025-03-25,Laptop,West,5000,8\n"
        "2025-04-07,Keyboard,East,,3\n"
    )
    return csv_content.encode("utf-8")


def test_health_and_openapi_docs(api_client: TestClient) -> None:
    resp = api_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["database_connected"] is True
    assert body["execution_mode"] == "offline"
    assert body["online_ai_enabled"] is False

    docs = api_client.get("/docs")
    assert docs.status_code == 200
    openapi = api_client.get("/openapi.json")
    assert openapi.status_code == 200
    assert "/api/analysis/run" in openapi.json()["paths"]


def test_cors_allows_configured_origin(api_client: TestClient) -> None:
    resp = api_client.options(
        "/health",
        headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:8501"


def test_invalid_file_upload_and_missing_session_errors(api_client: TestClient) -> None:
    bad_ext = api_client.post(
        "/api/dataset/profile",
        files={"file": ("malware.txt", b"hello world", "text/plain")},
    )
    assert bad_ext.status_code == 415
    assert bad_ext.json()["error"] == "UnsupportedFileFormat"

    empty_csv = api_client.post(
        "/api/dataset/profile",
        files={"file": ("empty.csv", b"", "text/csv")},
    )
    assert empty_csv.status_code == 413
    assert empty_csv.json()["error"] == "FileSizeLimitExceeded"

    missing_sess = api_client.post(
        "/api/prompt/optimize",
        json={"dataset_id": "non-existent-id", "prompt": "Top products?"},
    )
    assert missing_sess.status_code == 404
    assert missing_sess.json()["error"] == "DatasetSessionNotFound"


def test_full_api_workflow_profile_clean_prompt_analyze_validate_anomaly_history_report(
    api_client: TestClient,
) -> None:
    # 1. Profile uploaded CSV
    prof_resp = api_client.post(
        "/api/dataset/profile",
        files={"file": ("sales.csv", io.BytesIO(_sample_csv_bytes()), "text/csv")},
    )
    assert prof_resp.status_code == 200
    prof_data = prof_resp.json()
    dataset_id = prof_data["dataset_id"]
    assert prof_data["profile"]["row_count"] == 8
    assert "Revenue" in prof_data["columns"]

    # 2. Clean dataset (deduplicate + median impute numerical missing)
    clean_resp = api_client.post(
        "/api/dataset/clean",
        json={
            "dataset_id": dataset_id,
            "remove_duplicates": True,
            "numeric_missing_strategy": "median",
        },
    )
    assert clean_resp.status_code == 200
    clean_data = clean_resp.json()
    assert clean_data["cleaning_report"]["duplicates_removed"] == 1
    assert clean_data["cleaning_report"]["missing_values_after"] == 0
    assert clean_data["updated_profile"]["row_count"] == 7

    # 3. Optimize prompt
    opt_resp = api_client.post(
        "/api/prompt/optimize",
        json={"dataset_id": dataset_id, "prompt": "Top products?"},
    )
    assert opt_resp.status_code == 200
    opt_data = opt_resp.json()["optimization"]
    assert opt_data["detected_intent"] == "ranking"
    assert "Product" in opt_data["referenced_columns"]

    # 4. Score prompt quality (ML classifier)
    score_resp = api_client.post(
        "/api/prompt/score",
        json={"dataset_id": dataset_id, "prompt": "Top products?"},
    )
    assert score_resp.status_code == 200
    pred = score_resp.json()["prediction"]
    assert pred["predicted_class"] in {"Low", "Medium", "High"}
    assert 0.0 <= pred["quality_score"] <= 100.0

    # 5. Run end-to-end analysis (Module 4 + 6 + 8 + 9 + 10)
    run_resp = api_client.post(
        "/api/analysis/run",
        json={
            "dataset_id": dataset_id,
            "prompt": "Rank top 3 products by total Revenue",
            "save_to_history": True,
        },
    )
    assert run_resp.status_code == 200
    run_data = run_resp.json()
    analysis_id = run_data["analysis_id"]
    assert analysis_id
    assert run_data["analysis_result"]["status"] == "completed"
    assert run_data["analysis_result"]["summary_metrics"]["top_group"] == "Laptop"
    assert run_data["validation_report"]["overall_status"] in {"Verified", "Warning"}
    assert run_data["visualization"]["is_renderable"] is True
    assert run_data["visualization"]["figure"] is not None

    # 6. Detect anomalies
    anom_resp = api_client.post(
        "/api/anomaly/detect",
        json={"dataset_id": dataset_id, "contamination": 0.15},
    )
    assert anom_resp.status_code == 200
    anom_data = anom_resp.json()
    assert anom_data["anomaly_report"]["algorithm"] == "IsolationForest"
    assert "NOT automatically a data error" in anom_data["anomaly_report"]["interpretation_note"]
    assert anom_data["visualization"]["is_renderable"] is True

    # 7. Explicit claim validation (one valid, one fabricated)
    val_resp = api_client.post(
        "/api/analysis/validate",
        json={
            "dataset_id": dataset_id,
            "claims": [
                {
                    "claim_type": "total",
                    "description": "Wrong total Revenue",
                    "column": "Revenue",
                    "claimed_value": 999999.0,
                }
            ],
        },
    )
    assert val_resp.status_code == 200
    val_data = val_resp.json()
    assert val_data["validation_report"]["overall_status"] == "Unsupported"
    assert len(val_data["excluded_unsupported_claims"]) >= 1

    # 8. History list, get, export CSV/PDF, and delete
    hist_list = api_client.get("/api/history")
    assert hist_list.status_code == 200
    assert hist_list.json()["total_count"] >= 1

    hist_item = api_client.get(f"/api/history/{analysis_id}")
    assert hist_item.status_code == 200
    assert hist_item.json()["item"]["analysis_id"] == analysis_id

    pdf_resp = api_client.post(
        "/api/report/export",
        json={"dataset_id": dataset_id, "analysis_id": analysis_id, "export_format": "pdf"},
    )
    assert pdf_resp.status_code == 200
    assert pdf_resp.headers["content-type"] == "application/pdf"
    assert pdf_resp.content.startswith(b"%PDF")

    csv_resp = api_client.post(
        "/api/report/export",
        json={"dataset_id": dataset_id, "analysis_id": analysis_id, "export_format": "csv"},
    )
    assert csv_resp.status_code == 200
    assert "sales.csv" in csv_resp.text

    del_resp = api_client.delete(f"/api/history/{analysis_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] is True

    missing_hist = api_client.get(f"/api/history/{analysis_id}")
    assert missing_hist.status_code == 404
