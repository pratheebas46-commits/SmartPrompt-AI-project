"""Unit tests for Module 8: Evidence-Based Self-Validation Engine (modules/validator.py)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from modules.analysis_engine import execute_analysis
from modules.dataset_profiler import profile_dataframe
from modules.prompt_optimizer import optimize_prompt
from modules.validator import (
    VALIDATION_LIMITATION_NOTE,
    AnalyticalClaim,
    ResultValidator,
    ValidationReport,
    ValidationStatus,
    validate_analysis_result,
    validate_claims,
)


@pytest.fixture()
def validation_sales_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Product": ["Laptop", "Laptop", "Phone", "Phone", "Tablet", "Monitor"],
            "Region": ["North", "North", "South", "East", "West", "North"],
            "Revenue": [1200.0, 1800.0, 600.0, 400.0, 500.0, 500.0],  # Sum = 5000.0, Mean = 833.3333
            "Units": [2, 3, 2, 2, 1, 1],
        }
    )


def test_cross_validate_module6_analysis_result_verified(validation_sales_df: pd.DataFrame) -> None:
    profile = profile_dataframe(validation_sales_df, dataset_name="val_sales.csv")
    opt = optimize_prompt("Which product generated the highest revenue?", profile)
    analysis_res = execute_analysis(validation_sales_df, opt.task_plan, profile=profile)

    report = validate_analysis_result(validation_sales_df, analysis_res, profile=profile)
    assert isinstance(report, ValidationReport)
    assert report.overall_status == ValidationStatus.VERIFIED.value
    assert report.unsupported_count == 0
    assert report.verified_count >= 5
    assert report.limitations_note == VALIDATION_LIMITATION_NOTE

    serialized = json.dumps(report.to_dict(), allow_nan=False)
    assert "Verified" in serialized


def test_validate_correct_and_incorrect_claims_and_tolerance(validation_sales_df: pd.DataFrame) -> None:
    validator = ResultValidator(abs_tol=1e-2, rel_tol=1e-3)
    claims = [
        # Correct total
        AnalyticalClaim(claim_type="total", description="Total Revenue", column="Revenue", claimed_value=5000.0),
        # Correct mean within floating-point rounding tolerance (5000/6 = 833.3333...)
        AnalyticalClaim(claim_type="mean", description="Mean Revenue", column="Revenue", claimed_value=833.33),
        # Correct percentage share for Laptop (3000 / 5000 = 60.0%)
        AnalyticalClaim(
            claim_type="percentage",
            description="Laptop revenue share",
            column="Revenue",
            group_column="Product",
            group_value="Laptop",
            claimed_value=60.0,
        ),
        # Correct ranking
        AnalyticalClaim(
            claim_type="ranking",
            description="Top Product by Revenue",
            column="Revenue",
            group_column="Product",
            claimed_entity="Laptop",
            claimed_value=3000.0,
        ),
    ]
    report_ok = validator.validate_claims(validation_sales_df, claims)
    assert report_ok.overall_status == ValidationStatus.VERIFIED.value
    assert report_ok.verified_count == 4

    # Deliberately incorrect claims
    bad_claims = [
        AnalyticalClaim(claim_type="total", description="Inflated Revenue", column="Revenue", claimed_value=9999.0),
        AnalyticalClaim(
            claim_type="ranking",
            description="Wrong top product",
            column="Revenue",
            group_column="Product",
            claimed_entity="Monitor",
            claimed_value=3000.0,
        ),
        AnalyticalClaim(claim_type="column_reference", description="Missing column", column="Profit"),
    ]
    report_bad = validator.validate_claims(validation_sales_df, bad_claims)
    assert report_bad.overall_status == ValidationStatus.UNSUPPORTED.value
    assert report_bad.unsupported_count == 3
    assert report_bad.checks[0].calculated_value == 5000.0
    assert report_bad.checks[0].absolute_diff == 4999.0


def test_small_sample_warning_and_unparseable_prose_limitation(validation_sales_df: pd.DataFrame) -> None:
    # Subset for Phone has only 2 rows (< 5) -> Warning status even though math matches
    claims = [
        AnalyticalClaim(
            claim_type="mean",
            description="Mean Phone Revenue",
            column="Revenue",
            group_column="Product",
            group_value="Phone",
            claimed_value=500.0,
        ),
    ]
    warn_report = validate_claims(validation_sales_df, claims, min_sample_warning=5)
    assert warn_report.overall_status == ValidationStatus.WARNING.value
    assert warn_report.warning_count == 1

    # Open-ended prose & text claim with non-existent column -> Unsupported
    text_claims = [
        AnalyticalClaim(claim_type="text", description="Text total", statement_text="Total Revenue is 5000"),
        AnalyticalClaim(claim_type="text", description="Text missing col", statement_text="Total Profit is 1200"),
        AnalyticalClaim(
            claim_type="text",
            description="Unstructured prose",
            statement_text="Customer sentiment improved dramatically because of marketing.",
        ),
    ]
    text_report = validate_claims(validation_sales_df, text_claims)
    assert text_report.verified_count == 1
    assert text_report.unsupported_count == 2
    assert "does not claim complete verification" in text_report.limitations_note.lower()
