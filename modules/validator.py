"""Module 8: Evidence-Based Self-Validation Engine for SmartPrompt AI.

Independently cross-checks analytical results and structured/natural-language
claims against raw dataset computations to verify totals, averages, percentages,
rankings, correlations, and column references, flagging discrepancies and
unsupported assertions deterministically.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
import re
from typing import Any

import numpy as np
import pandas as pd

from config.logging_config import get_logger
from modules.analysis_engine import AnalysisResult, _apply_safe_filters, _safe_numeric
from modules.dataset_profiler import DatasetProfileResult

logger = get_logger(__name__)

VALIDATION_LIMITATION_NOTE = (
    "Deterministic self-validation verifies schema-grounded totals, averages, medians, "
    "counts, percentages, rankings, correlations, and column references against actual "
    "dataset computations. It does not claim complete verification of open-ended or "
    "ambiguous natural-language prose."
)


# ---------------------------------------------------------------------------
# Enums, Exceptions & Dataclasses
# ---------------------------------------------------------------------------


class ValidationStatus(str, Enum):
    VERIFIED = "Verified"
    WARNING = "Warning"
    UNSUPPORTED = "Unsupported"


class ValidationError(Exception):
    """Base exception for self-validation engine errors."""


@dataclass(frozen=True)
class AnalyticalClaim:
    """Structured analytical claim to verify independently against a DataFrame."""

    claim_type: str  # "total" | "mean" | "median" | "count" | "percentage" | "ranking" | "column_reference" | "correlation" | "text"
    description: str
    column: str | None = None
    secondary_column: str | None = None
    group_column: str | None = None
    group_value: Any = None
    claimed_value: float | int | None = None
    claimed_entity: Any = None
    sort_order: str = "descending"
    aggregation: str = "sum"
    filters: list[dict[str, Any]] = field(default_factory=list)
    statement_text: str | None = None


@dataclass(frozen=True)
class ValidationCheckResult:
    """Single deterministic verification check outcome."""

    check_name: str
    status: str  # "Verified" | "Warning" | "Unsupported"
    column: str | None
    reported_value: Any
    calculated_value: Any
    absolute_diff: float | None
    tolerance_used: float
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationReport:
    """Aggregated evidence-based validation audit report."""

    overall_status: str  # "Verified" | "Warning" | "Unsupported"
    total_checks: int
    verified_count: int
    warning_count: int
    unsupported_count: int
    checks: list[ValidationCheckResult]
    limitations_note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_status": self.overall_status,
            "total_checks": self.total_checks,
            "verified_count": self.verified_count,
            "warning_count": self.warning_count,
            "unsupported_count": self.unsupported_count,
            "checks": [c.to_dict() for c in self.checks],
            "limitations_note": self.limitations_note,
        }


# ---------------------------------------------------------------------------
# Tolerance & Comparison Helpers
# ---------------------------------------------------------------------------


def _values_match(
    reported: float | int,
    calculated: float | int,
    abs_tol: float = 1e-4,
    rel_tol: float = 1e-4,
) -> tuple[bool, float]:
    rep = float(reported)
    calc = float(calculated)
    diff = abs(rep - calc)
    matched = math.isclose(rep, calc, rel_tol=rel_tol, abs_tol=abs_tol)
    return matched, round(diff, 6)


def _overall_status_from_checks(checks: list[ValidationCheckResult]) -> str:
    if not checks:
        return ValidationStatus.UNSUPPORTED.value
    statuses = {c.status for c in checks}
    if ValidationStatus.UNSUPPORTED.value in statuses:
        return ValidationStatus.UNSUPPORTED.value
    if ValidationStatus.WARNING.value in statuses:
        return ValidationStatus.WARNING.value
    return ValidationStatus.VERIFIED.value


def _summarize_report(checks: list[ValidationCheckResult]) -> ValidationReport:
    verified = sum(1 for c in checks if c.status == ValidationStatus.VERIFIED.value)
    warnings = sum(1 for c in checks if c.status == ValidationStatus.WARNING.value)
    unsupported = sum(1 for c in checks if c.status == ValidationStatus.UNSUPPORTED.value)
    return ValidationReport(
        overall_status=_overall_status_from_checks(checks),
        total_checks=len(checks),
        verified_count=verified,
        warning_count=warnings,
        unsupported_count=unsupported,
        checks=checks,
        limitations_note=VALIDATION_LIMITATION_NOTE,
    )


# ---------------------------------------------------------------------------
# Independent Claim Verification Functions
# ---------------------------------------------------------------------------


def verify_column_references(
    df: pd.DataFrame,
    columns: list[str],
    profile: DatasetProfileResult | None = None,
) -> list[ValidationCheckResult]:
    """Verify that every referenced column exists in the active dataset."""
    available = set(df.columns)
    if profile is not None:
        available &= set(profile.column_profiles.keys())

    checks: list[ValidationCheckResult] = []
    for col in columns:
        if col in available:
            checks.append(
                ValidationCheckResult(
                    check_name=f"column_exists:{col}",
                    status=ValidationStatus.VERIFIED.value,
                    column=col,
                    reported_value=col,
                    calculated_value=col,
                    absolute_diff=0.0,
                    tolerance_used=0.0,
                    message=f"Column '{col}' verified in dataset schema.",
                )
            )
        else:
            checks.append(
                ValidationCheckResult(
                    check_name=f"column_exists:{col}",
                    status=ValidationStatus.UNSUPPORTED.value,
                    column=col,
                    reported_value=col,
                    calculated_value=None,
                    absolute_diff=None,
                    tolerance_used=0.0,
                    message=f"Unsupported column reference '{col}': column does not exist in dataset.",
                )
            )
    return checks


def verify_single_claim(
    df: pd.DataFrame,
    claim: AnalyticalClaim,
    abs_tol: float = 1e-2,
    rel_tol: float = 1e-3,
    min_sample_warning: int = 5,
) -> ValidationCheckResult:
    """Independently compute and verify a structured analytical claim against df."""
    if claim.claim_type == "column_reference":
        col = claim.column
        if col and col in df.columns:
            return ValidationCheckResult(
                check_name=f"column_reference:{col}",
                status=ValidationStatus.VERIFIED.value,
                column=col,
                reported_value=col,
                calculated_value=col,
                absolute_diff=0.0,
                tolerance_used=0.0,
                message=f"Column '{col}' exists in the dataset.",
            )
        return ValidationCheckResult(
            check_name=f"column_reference:{col}",
            status=ValidationStatus.UNSUPPORTED.value,
            column=col,
            reported_value=col,
            calculated_value=None,
            absolute_diff=None,
            tolerance_used=0.0,
            message=f"Referenced column '{col}' does not exist in the dataset.",
        )

    if claim.claim_type == "text" and claim.statement_text:
        return _verify_text_statement(df, claim.statement_text, abs_tol=abs_tol, rel_tol=rel_tol)

    # Apply any claim-level filters safely
    try:
        work_df = _apply_safe_filters(df, claim.filters) if claim.filters else df
    except Exception as exc:
        return ValidationCheckResult(
            check_name=f"claim:{claim.claim_type}",
            status=ValidationStatus.UNSUPPORTED.value,
            column=claim.column,
            reported_value=claim.claimed_value,
            calculated_value=None,
            absolute_diff=None,
            tolerance_used=abs_tol,
            message=f"Filter validation failed: {exc}",
        )

    col = claim.column
    if not col or col not in work_df.columns:
        return ValidationCheckResult(
            check_name=f"claim:{claim.claim_type}",
            status=ValidationStatus.UNSUPPORTED.value,
            column=col,
            reported_value=claim.claimed_value,
            calculated_value=None,
            absolute_diff=None,
            tolerance_used=abs_tol,
            message=f"Column '{col}' not found in dataset; claim is unsupported.",
        )

    # Optional group filter for subgroup claims (e.g., "Revenue in North")
    if claim.group_column and claim.group_value is not None and claim.claim_type != "ranking":
        if claim.group_column not in work_df.columns:
            return ValidationCheckResult(
                check_name=f"claim:{claim.claim_type}",
                status=ValidationStatus.UNSUPPORTED.value,
                column=claim.group_column,
                reported_value=claim.claimed_value,
                calculated_value=None,
                absolute_diff=None,
                tolerance_used=abs_tol,
                message=f"Group column '{claim.group_column}' not found in dataset.",
            )
        target_mask = (
            work_df[claim.group_column].astype(str).str.strip().str.lower()
            == str(claim.group_value).strip().lower()
        )
        sub_df = work_df[work_df[claim.group_column].notna() & target_mask]
    else:
        sub_df = work_df

    # 1. Ranking verification
    if claim.claim_type == "ranking":
        grp_col = claim.group_column
        if not grp_col or grp_col not in work_df.columns:
            return ValidationCheckResult(
                check_name="ranking_verification",
                status=ValidationStatus.UNSUPPORTED.value,
                column=col,
                reported_value=claim.claimed_entity,
                calculated_value=None,
                absolute_diff=None,
                tolerance_used=abs_tol,
                message=f"Grouping column '{grp_col}' missing for ranking verification.",
            )

        num_s = _safe_numeric(work_df, col)
        valid_mask = num_s.notna() & work_df[grp_col].notna()
        valid_df = work_df.loc[valid_mask].copy()
        valid_df[col] = num_s.loc[valid_mask]
        if valid_df.empty:
            return ValidationCheckResult(
                check_name="ranking_verification",
                status=ValidationStatus.UNSUPPORTED.value,
                column=col,
                reported_value=claim.claimed_entity,
                calculated_value=None,
                absolute_diff=None,
                tolerance_used=abs_tol,
                message=f"No valid values in '{col}' and '{grp_col}' to verify ranking.",
            )

        agg = claim.aggregation if claim.aggregation in {"sum", "mean", "median", "count", "min", "max"} else "sum"
        grouped = valid_df.groupby(grp_col, dropna=True)[col].agg(agg)
        descending = claim.sort_order != "ascending"
        sorted_series = grouped.sort_values(ascending=not descending)
        actual_top_entity = sorted_series.index[0]
        actual_top_val = round(float(sorted_series.iloc[0]), 6)

        entity_ok = str(actual_top_entity).strip().lower() == str(claim.claimed_entity).strip().lower()
        val_ok = True
        diff = 0.0
        if claim.claimed_value is not None:
            val_ok, diff = _values_match(claim.claimed_value, actual_top_val, abs_tol=abs_tol, rel_tol=rel_tol)

        if entity_ok and val_ok:
            return ValidationCheckResult(
                check_name=f"ranking:{grp_col}:{col}",
                status=ValidationStatus.VERIFIED.value,
                column=col,
                reported_value={"entity": claim.claimed_entity, "value": claim.claimed_value},
                calculated_value={"entity": actual_top_entity, "value": actual_top_val},
                absolute_diff=diff,
                tolerance_used=abs_tol,
                message=f"Verified top {grp_col} is '{actual_top_entity}' ({agg}={actual_top_val}).",
            )
        return ValidationCheckResult(
            check_name=f"ranking:{grp_col}:{col}",
            status=ValidationStatus.UNSUPPORTED.value,
            column=col,
            reported_value={"entity": claim.claimed_entity, "value": claim.claimed_value},
            calculated_value={"entity": actual_top_entity, "value": actual_top_val},
            absolute_diff=diff,
            tolerance_used=abs_tol,
            message=(
                f"Ranking discrepancy: reported ({claim.claimed_entity}, {claim.claimed_value}), "
                f"actual ({actual_top_entity}, {actual_top_val})."
            ),
        )

    # 2. Percentage share verification (group sum / total sum * 100)
    if claim.claim_type == "percentage":
        if claim.claimed_value is None:
            return ValidationCheckResult(
                check_name="percentage_verification",
                status=ValidationStatus.UNSUPPORTED.value,
                column=col,
                reported_value=None,
                calculated_value=None,
                absolute_diff=None,
                tolerance_used=abs_tol,
                message="No claimed percentage value provided.",
            )
        all_num = _safe_numeric(work_df, col).dropna()
        sub_num = _safe_numeric(sub_df, col).dropna()
        if all_num.empty or float(all_num.sum()) == 0.0 or sub_num.empty:
            return ValidationCheckResult(
                check_name=f"percentage:{col}",
                status=ValidationStatus.UNSUPPORTED.value,
                column=col,
                reported_value=claim.claimed_value,
                calculated_value=None,
                absolute_diff=None,
                tolerance_used=abs_tol,
                message="Cannot compute percentage share due to empty or zero-sum denominator.",
            )
        calc_pct = round(float(sub_num.sum()) / float(all_num.sum()) * 100.0, 6)
        matched, diff = _values_match(claim.claimed_value, calc_pct, abs_tol=abs_tol, rel_tol=rel_tol)
        status = ValidationStatus.VERIFIED.value if matched else ValidationStatus.UNSUPPORTED.value
        return ValidationCheckResult(
            check_name=f"percentage:{col}",
            status=status,
            column=col,
            reported_value=claim.claimed_value,
            calculated_value=calc_pct,
            absolute_diff=diff,
            tolerance_used=abs_tol,
            message=(
                f"Percentage share verified ({calc_pct:.2f}%)."
                if matched
                else f"Percentage discrepancy: reported {claim.claimed_value}%, calculated {calc_pct:.4f}%."
            ),
        )

    # 3. Correlation verification
    if claim.claim_type == "correlation":
        col2 = claim.secondary_column
        if not col2 or col2 not in work_df.columns or claim.claimed_value is None:
            return ValidationCheckResult(
                check_name="correlation_verification",
                status=ValidationStatus.UNSUPPORTED.value,
                column=col,
                reported_value=claim.claimed_value,
                calculated_value=None,
                absolute_diff=None,
                tolerance_used=abs_tol,
                message="Correlation verification requires two valid columns and a claimed r value.",
            )
        pair_df = pd.DataFrame({col: _safe_numeric(work_df, col), col2: _safe_numeric(work_df, col2)}).dropna()
        if len(pair_df) < 3 or pair_df[col].nunique() <= 1 or pair_df[col2].nunique() <= 1:
            return ValidationCheckResult(
                check_name=f"correlation:{col}:{col2}",
                status=ValidationStatus.UNSUPPORTED.value,
                column=col,
                reported_value=claim.claimed_value,
                calculated_value=None,
                absolute_diff=None,
                tolerance_used=abs_tol,
                message="Insufficient or constant paired observations for correlation.",
            )
        calc_r = round(float(pair_df[col].corr(pair_df[col2], method="pearson")), 6)
        matched, diff = _values_match(claim.claimed_value, calc_r, abs_tol=abs_tol, rel_tol=rel_tol)
        return ValidationCheckResult(
            check_name=f"correlation:{col}:{col2}",
            status=ValidationStatus.VERIFIED.value if matched else ValidationStatus.UNSUPPORTED.value,
            column=f"{col},{col2}",
            reported_value=claim.claimed_value,
            calculated_value=calc_r,
            absolute_diff=diff,
            tolerance_used=abs_tol,
            message=(
                f"Pearson correlation verified (r={calc_r})."
                if matched
                else f"Correlation discrepancy: reported {claim.claimed_value}, calculated {calc_r}."
            ),
        )

    # 4. Numerical scalar claims: total, mean, median, count
    if claim.claimed_value is None:
        return ValidationCheckResult(
            check_name=f"{claim.claim_type}:{col}",
            status=ValidationStatus.UNSUPPORTED.value,
            column=col,
            reported_value=None,
            calculated_value=None,
            absolute_diff=None,
            tolerance_used=abs_tol,
            message="Claim has no numerical value to verify.",
        )

    num_vals = _safe_numeric(sub_df, col).dropna()
    if num_vals.empty:
        return ValidationCheckResult(
            check_name=f"{claim.claim_type}:{col}",
            status=ValidationStatus.UNSUPPORTED.value,
            column=col,
            reported_value=claim.claimed_value,
            calculated_value=None,
            absolute_diff=None,
            tolerance_used=abs_tol,
            message=f"No valid numeric values in '{col}' for claim verification.",
        )

    if claim.claim_type == "total":
        calc_val = round(float(num_vals.sum()), 6)
    elif claim.claim_type == "mean":
        calc_val = round(float(num_vals.mean()), 6)
    elif claim.claim_type == "median":
        calc_val = round(float(num_vals.median()), 6)
    elif claim.claim_type == "count":
        calc_val = int(len(num_vals))
    else:
        return ValidationCheckResult(
            check_name=f"unsupported_claim_type:{claim.claim_type}",
            status=ValidationStatus.UNSUPPORTED.value,
            column=col,
            reported_value=claim.claimed_value,
            calculated_value=None,
            absolute_diff=None,
            tolerance_used=abs_tol,
            message=f"Claim type '{claim.claim_type}' is not supported for deterministic verification.",
        )

    matched, diff = _values_match(claim.claimed_value, calc_val, abs_tol=abs_tol, rel_tol=rel_tol)
    if not matched:
        return ValidationCheckResult(
            check_name=f"{claim.claim_type}:{col}",
            status=ValidationStatus.UNSUPPORTED.value,
            column=col,
            reported_value=claim.claimed_value,
            calculated_value=calc_val,
            absolute_diff=diff,
            tolerance_used=abs_tol,
            message=f"{claim.claim_type.title()} discrepancy on '{col}': reported {claim.claimed_value}, calculated {calc_val}.",
        )

    if len(num_vals) < min_sample_warning:
        return ValidationCheckResult(
            check_name=f"{claim.claim_type}:{col}",
            status=ValidationStatus.WARNING.value,
            column=col,
            reported_value=claim.claimed_value,
            calculated_value=calc_val,
            absolute_diff=diff,
            tolerance_used=abs_tol,
            message=(
                f"{claim.claim_type.title()} matches ({calc_val}), but sample size is small "
                f"(n={len(num_vals)} < {min_sample_warning})."
            ),
        )

    return ValidationCheckResult(
        check_name=f"{claim.claim_type}:{col}",
        status=ValidationStatus.VERIFIED.value,
        column=col,
        reported_value=claim.claimed_value,
        calculated_value=calc_val,
        absolute_diff=diff,
        tolerance_used=abs_tol,
        message=f"{claim.claim_type.title()} on '{col}' verified ({calc_val}).",
    )


def _verify_text_statement(
    df: pd.DataFrame,
    statement: str,
    abs_tol: float = 1e-2,
    rel_tol: float = 1e-3,
) -> ValidationCheckResult:
    """Parse simple structured quantitative statements; flag unsupported/unparseable prose safely."""
    text = statement.strip()
    patterns = [
        ("total", re.compile(r"\btotal\s+([a-z0-9_]+)\s+(?:is|=|equals)\s+(-?\d+(?:\.\d+)?)\b", re.I)),
        ("mean", re.compile(r"\b(?:average|mean)\s+([a-z0-9_]+)\s+(?:is|=|equals)\s+(-?\d+(?:\.\d+)?)\b", re.I)),
        ("median", re.compile(r"\bmedian\s+([a-z0-9_]+)\s+(?:is|=|equals)\s+(-?\d+(?:\.\d+)?)\b", re.I)),
        ("count", re.compile(r"\bcount\s+(?:of\s+)?([a-z0-9_]+)\s+(?:is|=|equals)\s+(\d+)\b", re.I)),
    ]

    col_lookup = {c.lower(): c for c in df.columns}
    for claim_type, pat in patterns:
        m = pat.search(text)
        if m:
            raw_col, raw_num = m.group(1), m.group(2)
            actual_col = col_lookup.get(raw_col.lower())
            val: int | float = float(raw_num) if "." in raw_num else int(raw_num)
            if not actual_col:
                return ValidationCheckResult(
                    check_name=f"text_claim:{claim_type}",
                    status=ValidationStatus.UNSUPPORTED.value,
                    column=raw_col,
                    reported_value=val,
                    calculated_value=None,
                    absolute_diff=None,
                    tolerance_used=abs_tol,
                    message=f"Claim references unavailable column '{raw_col}'.",
                )
            return verify_single_claim(
                df,
                AnalyticalClaim(
                    claim_type=claim_type,
                    description=text,
                    column=actual_col,
                    claimed_value=val,
                ),
                abs_tol=abs_tol,
                rel_tol=rel_tol,
            )

    return ValidationCheckResult(
        check_name="text_claim:unverified_prose",
        status=ValidationStatus.UNSUPPORTED.value,
        column=None,
        reported_value=text,
        calculated_value=None,
        absolute_diff=None,
        tolerance_used=abs_tol,
        message=(
            "Natural-language statement could not be deterministically parsed into a "
            "verifiable numerical claim against dataset columns."
        ),
    )


# ---------------------------------------------------------------------------
# Cross-Validation of Module 6 AnalysisResult
# ---------------------------------------------------------------------------


def validate_analysis_result(
    df: pd.DataFrame,
    analysis_result: AnalysisResult,
    profile: DatasetProfileResult | None = None,
    abs_tol: float = 1e-4,
    rel_tol: float = 1e-4,
    min_sample_warning: int = 5,
) -> ValidationReport:
    """Independently cross-validate a Module 6 AnalysisResult against raw DataFrame calculations."""
    checks: list[ValidationCheckResult] = []
    ev = analysis_result.numerical_evidence

    if analysis_result.status != "completed":
        checks.append(
            ValidationCheckResult(
                check_name="analysis_status",
                status=ValidationStatus.UNSUPPORTED.value,
                column=None,
                reported_value=analysis_result.status,
                calculated_value="completed",
                absolute_diff=None,
                tolerance_used=0.0,
                message=f"Analysis did not complete successfully (status='{analysis_result.status}').",
            )
        )
        return _summarize_report(checks)

    # 1. Verify column references
    checks.extend(verify_column_references(df, ev.columns_used, profile=profile))

    # Apply the exact same filters independently for cross-checking
    filters = analysis_result.plan_snapshot.get("filters") or []
    filtered_df = _apply_safe_filters(df, filters) if filters else df

    # 2. Verify row conservation
    if len(filtered_df) == ev.filtered_rows:
        checks.append(
            ValidationCheckResult(
                check_name="row_count_conservation",
                status=ValidationStatus.VERIFIED.value,
                column=None,
                reported_value=ev.filtered_rows,
                calculated_value=len(filtered_df),
                absolute_diff=0.0,
                tolerance_used=0.0,
                message=f"Filtered row count verified ({len(filtered_df)} rows).",
            )
        )
    else:
        checks.append(
            ValidationCheckResult(
                check_name="row_count_conservation",
                status=ValidationStatus.UNSUPPORTED.value,
                column=None,
                reported_value=ev.filtered_rows,
                calculated_value=len(filtered_df),
                absolute_diff=abs(ev.filtered_rows - len(filtered_df)),
                tolerance_used=0.0,
                message=f"Row count mismatch: reported {ev.filtered_rows}, actual {len(filtered_df)}.",
            )
        )

    # 3. Verify totals, averages, and medians
    for col, rep_total in ev.totals.items():
        checks.append(
            verify_single_claim(
                filtered_df,
                AnalyticalClaim(claim_type="total", description=f"Total {col}", column=col, claimed_value=rep_total),
                abs_tol=abs_tol,
                rel_tol=rel_tol,
                min_sample_warning=min_sample_warning,
            )
        )

    for col, rep_avg in ev.averages.items():
        checks.append(
            verify_single_claim(
                filtered_df,
                AnalyticalClaim(claim_type="mean", description=f"Mean {col}", column=col, claimed_value=rep_avg),
                abs_tol=abs_tol,
                rel_tol=rel_tol,
                min_sample_warning=min_sample_warning,
            )
        )

    for col, rep_med in ev.medians.items():
        checks.append(
            verify_single_claim(
                filtered_df,
                AnalyticalClaim(claim_type="median", description=f"Median {col}", column=col, claimed_value=rep_med),
                abs_tol=abs_tol,
                rel_tol=rel_tol,
                min_sample_warning=min_sample_warning,
            )
        )

    # 4. Verify percentage sum invariant when rankings exist
    if ev.percentages and ev.rankings:
        pct_sum = sum(float(v) for v in ev.percentages.values() if v is not None)
        matched_100, diff_100 = _values_match(pct_sum, 100.0, abs_tol=0.1, rel_tol=1e-2)
        checks.append(
            ValidationCheckResult(
                check_name="percentage_sum_invariant",
                status=ValidationStatus.VERIFIED.value if matched_100 else ValidationStatus.UNSUPPORTED.value,
                column=analysis_result.plan_snapshot.get("metric_column"),
                reported_value=round(pct_sum, 4),
                calculated_value=100.0,
                absolute_diff=diff_100,
                tolerance_used=0.1,
                message=(
                    f"Group percentage shares sum to {pct_sum:.2f}% (within tolerance of 100%)."
                    if matched_100
                    else f"Percentage shares sum to {pct_sum:.2f}%, deviating from 100%."
                ),
            )
        )

    # 5. Verify top ranking entity & value independently
    group_col = analysis_result.summary_metrics.get("group_column")
    metric_col = analysis_result.summary_metrics.get("metric_column")
    top_group = analysis_result.summary_metrics.get("top_group")
    top_value = analysis_result.summary_metrics.get("top_value")
    if group_col and metric_col and top_group is not None and top_value is not None:
        checks.append(
            verify_single_claim(
                filtered_df,
                AnalyticalClaim(
                    claim_type="ranking",
                    description=f"Top {group_col} by {metric_col}",
                    column=metric_col,
                    group_column=group_col,
                    claimed_entity=top_group,
                    claimed_value=top_value,
                    aggregation=analysis_result.summary_metrics.get("aggregation") or "sum",
                    sort_order=analysis_result.summary_metrics.get("sort_order") or "descending",
                ),
                abs_tol=abs_tol,
                rel_tol=rel_tol,
            )
        )

    # 6. Verify correlations
    for corr_row in ev.correlations:
        if corr_row.get("pearson_r") is not None:
            checks.append(
                verify_single_claim(
                    filtered_df,
                    AnalyticalClaim(
                        claim_type="correlation",
                        description=f"Pearson r({corr_row['column_a']}, {corr_row['column_b']})",
                        column=corr_row["column_a"],
                        secondary_column=corr_row["column_b"],
                        claimed_value=corr_row["pearson_r"],
                    ),
                    abs_tol=abs_tol,
                    rel_tol=rel_tol,
                )
            )

    return _summarize_report(checks)


def validate_claims(
    df: pd.DataFrame,
    claims: list[AnalyticalClaim],
    abs_tol: float = 1e-2,
    rel_tol: float = 1e-3,
    min_sample_warning: int = 5,
) -> ValidationReport:
    """Validate a list of structured or text analytical claims against actual dataset computations."""
    if not isinstance(df, pd.DataFrame):
        raise ValidationError("Input 'df' must be a pandas DataFrame.")

    checks = [
        verify_single_claim(
            df=df,
            claim=claim,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            min_sample_warning=min_sample_warning,
        )
        for claim in claims
    ]
    report = _summarize_report(checks)
    logger.info(
        "Validation completed: status=%s (verified=%d, warning=%d, unsupported=%d).",
        report.overall_status,
        report.verified_count,
        report.warning_count,
        report.unsupported_count,
    )
    return report


class ResultValidator:
    """Stateful service wrapper for Module 8 evidence-based self-validation."""

    def __init__(self, abs_tol: float = 1e-4, rel_tol: float = 1e-4) -> None:
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol

    def validate_analysis(
        self,
        df: pd.DataFrame,
        analysis_result: AnalysisResult,
        profile: DatasetProfileResult | None = None,
    ) -> ValidationReport:
        return validate_analysis_result(
            df=df,
            analysis_result=analysis_result,
            profile=profile,
            abs_tol=self.abs_tol,
            rel_tol=self.rel_tol,
        )

    def validate_claims(
        self,
        df: pd.DataFrame,
        claims: list[AnalyticalClaim],
    ) -> ValidationReport:
        return validate_claims(
            df=df,
            claims=claims,
            abs_tol=self.abs_tol,
            rel_tol=self.rel_tol,
        )
