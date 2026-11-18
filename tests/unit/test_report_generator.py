"""Unit tests for Module 11: Report Generation (modules/report_generator.py)."""

from __future__ import annotations

import csv
import io
from pathlib import Path
import sys

import pandas as pd
import pytest

from modules.analysis_engine import execute_analysis
from modules.anomaly_detector import AnomalyConfig, detect_anomalies
from modules.dataset_profiler import profile_dataframe
from modules.prompt_optimizer import optimize_prompt
from modules.report_generator import (
    ReportGenerator,
    build_report_payload,
    export_report_to_csv,
    export_report_to_pdf,
    generate_report_bundle,
)
from modules.validator import AnalyticalClaim, validate_analysis_result, validate_claims


@pytest.fixture
def sample_pipeline_outputs() -> dict:
    df = pd.DataFrame(
        {
            "Order_Date": [
                "2025-01-10",
                "2025-01-15",
                "2025-02-05",
                "2025-02-18",
                "2025-03-02",
                "2025-03-25",
            ],
            "Product": ["Laptop", "Mouse", "Laptop", "Monitor", "Mouse", "Laptop"],
            "Revenue": [1200.0, 45.0, 1300.0, 420.0, 55.0, 5000.0],
        }
    )
    profile = profile_dataframe(df, dataset_name="sales_sample.csv")
    prompt_res = optimize_prompt("Top products by revenue", profile)
    analysis_res = execute_analysis(df, prompt_res.task_plan, profile=profile)
    validation_rep = validate_analysis_result(df, analysis_res, profile=profile)
    anomaly_rep = detect_anomalies(df, config=AnomalyConfig(contamination=0.2, random_state=42))
    return {
        "df": df,
        "profile": profile,
        "prompt_res": prompt_res,
        "analysis_res": analysis_res,
        "validation_rep": validation_rep,
        "anomaly_rep": anomaly_rep,
    }


def test_report_generator_is_independent_of_streamlit() -> None:
    assert "streamlit" not in sys.modules


def test_csv_and_pdf_export_contain_required_sections(sample_pipeline_outputs: dict, tmp_path: Path) -> None:
    out = sample_pipeline_outputs
    csv_file = tmp_path / "analytical_report.csv"
    pdf_file = tmp_path / "analytical_report.pdf"

    bundle = generate_report_bundle(
        profile=out["profile"],
        prompt_result=out["prompt_res"],
        analysis_result=out["analysis_res"],
        validation_report=out["validation_rep"],
        anomaly_report=out["anomaly_rep"],
        csv_output_path=csv_file,
        pdf_output_path=pdf_file,
    )

    assert bundle.csv_bytes is not None and len(bundle.csv_bytes) > 100
    assert bundle.pdf_bytes is not None and bundle.pdf_bytes.startswith(b"%PDF")
    assert csv_file.exists() and pdf_file.exists()

    csv_text = bundle.csv_bytes.decode("utf-8")
    rows = list(csv.reader(io.StringIO(csv_text)))
    sections = {r[0] for r in rows if r}
    assert "Dataset Summary" in sections
    assert "Prompt Context" in sections
    assert "Validation" in sections
    assert "Key Results" in sections
    assert "Statistical Insights" in sections
    assert "Anomaly Summary" in sections
    assert "sales_sample.csv" in csv_text
    assert out["prompt_res"].original_prompt in csv_text


def test_unsupported_claims_are_excluded_from_statistical_insights(sample_pipeline_outputs: dict) -> None:
    out = sample_pipeline_outputs
    df = out["df"]

    # Create a validation report containing an explicitly Unsupported claim
    bad_val_report = validate_claims(
        df,
        claims=[
            AnalyticalClaim(
                claim_type="total",
                description="Fabricated Revenue claim",
                column="Revenue",
                claimed_value=9999999.0,
            ),
            AnalyticalClaim(
                claim_type="column_reference",
                description="Missing Profit claim",
                column="Profit",
            ),
        ],
    )
    assert bad_val_report.overall_status == "Unsupported"

    payload = build_report_payload(
        profile=out["profile"],
        prompt_result=out["prompt_res"],
        analysis_result=out["analysis_res"],
        validation_report=bad_val_report,
        additional_insights=[
            "Profit grew 40% over Q1",  # references unsupported column 'Profit' -> must be excluded
            "Unsupported fabricated claim 9999999.0",  # must be excluded
        ],
    )

    assert payload.validation_status == "Unsupported"
    assert len(payload.excluded_unsupported_claims) >= 2
    insights_joined = " ".join(payload.statistical_insights).lower()
    assert "9999999" not in insights_joined
    assert "profit grew 40%" not in insights_joined


def test_pdf_handles_missing_and_corrupt_chart_images_gracefully(sample_pipeline_outputs: dict, tmp_path: Path) -> None:
    out = sample_pipeline_outputs
    generator = ReportGenerator()
    payload = generator.build_payload(
        profile=out["profile"],
        prompt_result=out["prompt_res"],
        analysis_result=out["analysis_res"],
        validation_report=out["validation_rep"],
        original_prompt="Check Revenue > 100 & Profit < 50",  # Tests XML character escaping in ReportLab
    )

    missing_path = tmp_path / "does_not_exist.png"
    corrupt_bytes = b"not-a-valid-png-image"

    pdf_bytes = export_report_to_pdf(
        payload,
        chart_images=[
            {"image": missing_path, "caption": "Missing Chart"},
            {"image": corrupt_bytes, "caption": "Corrupt Chart"},
            {"image": b"", "caption": "Empty Bytes Chart"},
        ],
    )
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 500
    csv_bytes = export_report_to_csv(payload)
    assert b"Check Revenue > 100 & Profit < 50" in csv_bytes
