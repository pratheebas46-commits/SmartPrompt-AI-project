"""Module 6: Automated Data Analysis Engine for SmartPrompt AI.

Executes structured analytical task plans deterministically using Pandas and NumPy
in offline mode without visualization/UI coupling and without arbitrary code execution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time
from typing import Any

import numpy as np
import pandas as pd

from config.logging_config import get_logger
from modules.dataset_profiler import DatasetProfileResult, profile_dataframe
from modules.intent_detector import AnalyticalIntent
from modules.prompt_optimizer import AnalyticalTaskPlan

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Strict Whitelists (Prevents Arbitrary Code Execution)
# ---------------------------------------------------------------------------

ALLOWED_AGGREGATIONS: frozenset[str] = frozenset(
    {"sum", "mean", "median", "count", "min", "max", "std"}
)

ALLOWED_FILTER_OPERATORS: frozenset[str] = frozenset(
    {"==", "!=", ">", ">=", "<", "<=", "contains"}
)

_GRANULARITY_FREQ: dict[str, str] = {
    "day": "D",
    "week": "W",
    "month": "ME",
    "quarter": "QE",
    "year": "YE",
}


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------


class AnalysisEngineError(Exception):
    """Base exception for automated data analysis engine errors."""


class UnsupportedAnalysisTaskError(AnalysisEngineError):
    """Raised when an operation, aggregation, or operator is outside the safe whitelist."""


class InvalidAnalysisColumnError(AnalysisEngineError):
    """Raised when an analytical task references a column absent from the DataFrame."""


class InsufficientAnalysisDataError(AnalysisEngineError):
    """Raised when the dataset or filtered slice lacks valid records to compute results."""


# ---------------------------------------------------------------------------
# Structured Output Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NumericalEvidence:
    """Explicit computed numerical facts used by downstream validation and insights."""

    columns_used: list[str]
    input_rows: int
    filtered_rows: int
    valid_metric_rows: int
    dropped_null_rows: int
    totals: dict[str, float | int] = field(default_factory=dict)
    averages: dict[str, float] = field(default_factory=dict)
    medians: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    min_max: dict[str, dict[str, float | int]] = field(default_factory=dict)
    percentages: dict[str, float] = field(default_factory=dict)
    rankings: list[dict[str, Any]] = field(default_factory=list)
    correlations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnalysisResult:
    """Complete structured output from Module 6 Automated Data Analysis Engine."""

    status: str  # "completed" | "unsupported" | "insufficient_data"
    intent: str
    dataset_name: str
    summary_metrics: dict[str, Any]
    result_table: list[dict[str, Any]]
    numerical_evidence: NumericalEvidence
    warnings: list[str]
    execution_time_ms: float
    plan_snapshot: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "intent": self.intent,
            "dataset_name": self.dataset_name,
            "summary_metrics": self.summary_metrics,
            "result_table": self.result_table,
            "numerical_evidence": self.numerical_evidence.to_dict(),
            "warnings": self.warnings,
            "execution_time_ms": self.execution_time_ms,
            "plan_snapshot": self.plan_snapshot,
        }


# ---------------------------------------------------------------------------
# Safe Conversion & Filtering Helpers
# ---------------------------------------------------------------------------


def _clean_scalar(value: Any, decimals: int = 6) -> Any:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        val = float(value)
        return None if (np.isnan(val) or np.isinf(val)) else round(val, decimals)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _records_from_df(df: pd.DataFrame, max_rows: int = 200) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in df.head(max_rows).to_dict(orient="records"):
        rows.append({str(k): _clean_scalar(v) for k, v in rec.items()})
    return rows


def _safe_numeric(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _validate_plan_safety(df: pd.DataFrame, plan: AnalyticalTaskPlan) -> None:
    """Strictly validate columns, aggregation, and filter operators before any computation."""
    available = set(df.columns)

    for col in plan.target_columns:
        if col not in available:
            raise InvalidAnalysisColumnError(f"Column '{col}' does not exist in dataset.")
    if plan.metric_column and plan.metric_column not in available:
        raise InvalidAnalysisColumnError(f"Metric column '{plan.metric_column}' does not exist in dataset.")
    for col in plan.metric_columns:
        if col not in available:
            raise InvalidAnalysisColumnError(f"Metric column '{col}' does not exist in dataset.")
    if plan.group_column and plan.group_column not in available:
        raise InvalidAnalysisColumnError(f"Group column '{plan.group_column}' does not exist in dataset.")
    if plan.date_column and plan.date_column not in available:
        raise InvalidAnalysisColumnError(f"Date column '{plan.date_column}' does not exist in dataset.")

    if plan.aggregation is not None:
        agg_norm = str(plan.aggregation).strip().lower()
        if agg_norm not in ALLOWED_AGGREGATIONS:
            raise UnsupportedAnalysisTaskError(
                f"Aggregation '{plan.aggregation}' is not permitted. Allowed: {sorted(ALLOWED_AGGREGATIONS)}."
            )

    for flt in plan.filters:
        col = flt.get("column")
        op = flt.get("operator")
        if col not in available:
            raise InvalidAnalysisColumnError(f"Filter column '{col}' does not exist in dataset.")
        if op not in ALLOWED_FILTER_OPERATORS:
            raise UnsupportedAnalysisTaskError(
                f"Filter operator '{op}' is not permitted. Allowed: {sorted(ALLOWED_FILTER_OPERATORS)}."
            )


def _apply_safe_filters(df: pd.DataFrame, filters: list[dict[str, Any]]) -> pd.DataFrame:
    """Filter DataFrame rows purely via boolean masking (no eval/query)."""
    if not filters:
        return df

    working = df.copy()
    for flt in filters:
        col = flt["column"]
        op = flt["operator"]
        val = flt.get("value")
        series = working[col]

        if op in {">", ">=", "<", "<="}:
            num_s = _safe_numeric(working, col)
            try:
                threshold = float(val)
            except (TypeError, ValueError) as exc:
                raise UnsupportedAnalysisTaskError(
                    f"Operator '{op}' requires a numeric value, got '{val}'."
                ) from exc
            if op == ">":
                mask = num_s > threshold
            elif op == ">=":
                mask = num_s >= threshold
            elif op == "<":
                mask = num_s < threshold
            else:
                mask = num_s <= threshold
            working = working[mask.fillna(False)]
        elif op == "==":
            if pd.api.types.is_numeric_dtype(series) and isinstance(val, (int, float)):
                working = working[_safe_numeric(working, col) == val]
            else:
                mask = series.astype(str).str.strip().str.lower() == str(val).strip().lower()
                working = working[series.notna() & mask]
        elif op == "!=":
            if pd.api.types.is_numeric_dtype(series) and isinstance(val, (int, float)):
                working = working[_safe_numeric(working, col) != val]
            else:
                mask = series.astype(str).str.strip().str.lower() != str(val).strip().lower()
                working = working[series.notna() & mask]
        elif op == "contains":
            mask = series.astype(str).str.lower().str.contains(str(val).strip().lower(), na=False, regex=False)
            working = working[series.notna() & mask]

    return working.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Analytical Operation Implementations
# ---------------------------------------------------------------------------


def _run_groupby_or_scalar(
    df: pd.DataFrame,
    plan: AnalyticalTaskPlan,
    input_rows: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], NumericalEvidence]:
    metric = plan.metric_column
    group = plan.group_column
    agg = (plan.aggregation or "sum").lower()
    descending = (plan.sort_order or "descending") == "descending"
    filtered_rows = len(df)

    # Count-only grouped query when no numeric metric is provided
    if (not metric or metric not in df.columns) and agg == "count" and group:
        valid_group = df.dropna(subset=[group])
        grouped = valid_group.groupby(group, dropna=True).size().reset_index(name="count")
        grouped = grouped.sort_values("count", ascending=not descending).reset_index(drop=True)
        total_cnt = int(grouped["count"].sum())
        if total_cnt > 0:
            grouped["pct_of_total"] = (grouped["count"] / total_cnt * 100.0).round(4)
        limited = grouped.head(plan.limit) if plan.limit else grouped

        pct_map = {
            str(r[group]): _clean_scalar(r["pct_of_total"], 4)
            for r in grouped.to_dict(orient="records")
        } if "pct_of_total" in grouped.columns else {}

        evidence = NumericalEvidence(
            columns_used=[group],
            input_rows=input_rows,
            filtered_rows=filtered_rows,
            valid_metric_rows=len(valid_group),
            dropped_null_rows=filtered_rows - len(valid_group),
            counts={"total_records": total_cnt, "group_count": int(len(grouped))},
            percentages=pct_map,
            rankings=_records_from_df(limited),
        )
        summary = {
            "group_column": group,
            "aggregation": "count",
            "total_records": total_cnt,
            "group_count": int(len(grouped)),
            "top_group": _clean_scalar(grouped.iloc[0][group]) if not grouped.empty else None,
            "top_value": int(grouped.iloc[0]["count"]) if not grouped.empty else None,
        }
        return summary, _records_from_df(limited), evidence

    if not metric:
        raise InsufficientAnalysisDataError("No numerical metric column specified for aggregation/ranking.")

    num_series = _safe_numeric(df, metric)
    valid_mask = num_series.notna()
    if group:
        valid_mask = valid_mask & df[group].notna()

    valid_rows = int(valid_mask.sum())
    dropped = filtered_rows - valid_rows
    if valid_rows == 0:
        raise InsufficientAnalysisDataError(f"Column '{metric}' has 0 valid numeric values after filtering.")

    work = df.loc[valid_mask].copy()
    work[metric] = num_series.loc[valid_mask]

    overall_sum = float(work[metric].sum())
    overall_mean = float(work[metric].mean())
    overall_median = float(work[metric].median())
    overall_min = float(work[metric].min())
    overall_max = float(work[metric].max())

    if group:
        grouped_raw = work.groupby(group, dropna=True)[metric].agg(
            ["count", "sum", "mean", "median", "min", "max", "std"]
        )
        grouped_df = grouped_raw.reset_index()
        grouped_df["primary_value"] = grouped_df[agg]
        grouped_df = grouped_df.sort_values("primary_value", ascending=not descending).reset_index(drop=True)

        if overall_sum != 0:
            grouped_df["pct_of_total_sum"] = (grouped_df["sum"] / overall_sum * 100.0).round(4)
        else:
            grouped_df["pct_of_total_sum"] = 0.0

        limited_df = grouped_df.head(plan.limit) if plan.limit else grouped_df
        top_row = grouped_df.iloc[0]

        pct_map = {
            f"{group}={r[group]}": _clean_scalar(r["pct_of_total_sum"], 4)
            for r in grouped_df.to_dict(orient="records")
        }
        evidence = NumericalEvidence(
            columns_used=[group, metric],
            input_rows=input_rows,
            filtered_rows=filtered_rows,
            valid_metric_rows=valid_rows,
            dropped_null_rows=dropped,
            totals={metric: _clean_scalar(overall_sum)},
            averages={metric: _clean_scalar(overall_mean)},
            medians={metric: _clean_scalar(overall_median)},
            counts={metric: valid_rows, "groups": int(len(grouped_df))},
            min_max={metric: {"min": _clean_scalar(overall_min), "max": _clean_scalar(overall_max)}},
            percentages=pct_map,
            rankings=_records_from_df(grouped_df[[group, "primary_value", "sum", "mean", "count", "pct_of_total_sum"]]),
        )
        summary = {
            "metric_column": metric,
            "group_column": group,
            "aggregation": agg,
            "sort_order": "descending" if descending else "ascending",
            "group_count": int(len(grouped_df)),
            "top_group": _clean_scalar(top_row[group]),
            "top_value": _clean_scalar(top_row["primary_value"]),
            "grand_total_sum": _clean_scalar(overall_sum),
            "overall_mean": _clean_scalar(overall_mean),
            "overall_median": _clean_scalar(overall_median),
            "overall_min": _clean_scalar(overall_min),
            "overall_max": _clean_scalar(overall_max),
        }
        return summary, _records_from_df(limited_df), evidence

    # Ungrouped scalar aggregation
    agg_val = float(getattr(work[metric], agg)())
    summary = {
        "metric_column": metric,
        "aggregation": agg,
        "value": _clean_scalar(agg_val),
        "count": valid_rows,
        "sum": _clean_scalar(overall_sum),
        "mean": _clean_scalar(overall_mean),
        "median": _clean_scalar(overall_median),
        "min": _clean_scalar(overall_min),
        "max": _clean_scalar(overall_max),
    }
    evidence = NumericalEvidence(
        columns_used=[metric],
        input_rows=input_rows,
        filtered_rows=filtered_rows,
        valid_metric_rows=valid_rows,
        dropped_null_rows=dropped,
        totals={metric: _clean_scalar(overall_sum)},
        averages={metric: _clean_scalar(overall_mean)},
        medians={metric: _clean_scalar(overall_median)},
        counts={metric: valid_rows},
        min_max={metric: {"min": _clean_scalar(overall_min), "max": _clean_scalar(overall_max)}},
    )
    table = [summary]
    return summary, table, evidence


def _run_comparison(
    df: pd.DataFrame,
    plan: AnalyticalTaskPlan,
    input_rows: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], NumericalEvidence]:
    if plan.group_column and plan.metric_column:
        summary, table, evidence = _run_groupby_or_scalar(df, plan, input_rows)
        if len(table) >= 2:
            vals = [row["primary_value"] for row in table if row.get("primary_value") is not None]
            if vals:
                summary["spread_max_minus_min"] = _clean_scalar(max(vals) - min(vals))
        return summary, table, evidence

    cols = [c for c in plan.metric_columns if c in df.columns]
    if len(cols) < 2:
        return _run_groupby_or_scalar(df, plan, input_rows)

    rows: list[dict[str, Any]] = []
    totals: dict[str, float | int] = {}
    averages: dict[str, float] = {}
    medians: dict[str, float] = {}
    counts: dict[str, int] = {}
    min_max: dict[str, dict[str, float | int]] = {}

    for col in cols:
        s = _safe_numeric(df, col).dropna()
        if s.empty:
            continue
        col_sum = _clean_scalar(s.sum())
        col_mean = _clean_scalar(s.mean())
        col_med = _clean_scalar(s.median())
        col_min = _clean_scalar(s.min())
        col_max = _clean_scalar(s.max())
        cnt = int(len(s))

        totals[col] = col_sum
        averages[col] = col_mean
        medians[col] = col_med
        counts[col] = cnt
        min_max[col] = {"min": col_min, "max": col_max}
        rows.append(
            {
                "column": col,
                "count": cnt,
                "sum": col_sum,
                "mean": col_mean,
                "median": col_med,
                "min": col_min,
                "max": col_max,
                "std": _clean_scalar(s.std(ddof=1)) if cnt > 1 else 0.0,
            }
        )

    evidence = NumericalEvidence(
        columns_used=cols,
        input_rows=input_rows,
        filtered_rows=len(df),
        valid_metric_rows=len(df),
        dropped_null_rows=0,
        totals=totals,
        averages=averages,
        medians=medians,
        counts=counts,
        min_max=min_max,
    )
    return {"compared_columns": cols, "column_count": len(rows)}, rows, evidence


def _run_trend(
    df: pd.DataFrame,
    plan: AnalyticalTaskPlan,
    input_rows: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], NumericalEvidence]:
    date_col = plan.date_column
    metric = plan.metric_column
    gran = (plan.time_granularity or "month").lower()
    agg = (plan.aggregation or "sum").lower()

    if not date_col or not metric:
        raise InsufficientAnalysisDataError("Trend analysis requires both a date column and a metric column.")

    try:
        dt = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
    except TypeError:
        dt = pd.to_datetime(df[date_col], errors="coerce")
    num = _safe_numeric(df, metric)

    valid_mask = dt.notna() & num.notna()
    valid_rows = int(valid_mask.sum())
    if valid_rows == 0:
        raise InsufficientAnalysisDataError(f"No valid ({date_col}, {metric}) pairs found.")

    work = pd.DataFrame({"_dt": dt.loc[valid_mask], metric: num.loc[valid_mask]}).sort_values("_dt")
    freq = _GRANULARITY_FREQ.get(gran, "ME")
    try:
        grouped = work.set_index("_dt")[metric].resample(freq).agg([agg, "count"])
    except ValueError:
        legacy = {"ME": "M", "QE": "Q", "YE": "Y"}.get(freq, freq)
        grouped = work.set_index("_dt")[metric].resample(legacy).agg([agg, "count"])

    grouped = grouped[grouped["count"] > 0].copy()
    grouped["period"] = grouped.index.strftime("%Y-%m-%d")
    grouped = grouped.rename(columns={agg: "value", "count": "record_count"}).reset_index(drop=True)
    grouped["delta"] = grouped["value"].diff()
    grouped["pct_change"] = grouped["value"].pct_change().replace([np.inf, -np.inf], np.nan) * 100.0

    overall_sum = float(work[metric].sum())
    overall_mean = float(work[metric].mean())
    first_val = float(grouped["value"].iloc[0])
    last_val = float(grouped["value"].iloc[-1])
    overall_growth_pct = ((last_val - first_val) / abs(first_val) * 100.0) if first_val != 0 else None

    summary = {
        "date_column": date_col,
        "metric_column": metric,
        "granularity": gran,
        "aggregation": agg,
        "period_count": int(len(grouped)),
        "grand_total_sum": _clean_scalar(overall_sum),
        "overall_mean": _clean_scalar(overall_mean),
        "start_period": _clean_scalar(grouped["period"].iloc[0]),
        "end_period": _clean_scalar(grouped["period"].iloc[-1]),
        "overall_change_pct": _clean_scalar(overall_growth_pct, 4),
    }
    evidence = NumericalEvidence(
        columns_used=[date_col, metric],
        input_rows=input_rows,
        filtered_rows=len(df),
        valid_metric_rows=valid_rows,
        dropped_null_rows=len(df) - valid_rows,
        totals={metric: _clean_scalar(overall_sum)},
        averages={metric: _clean_scalar(overall_mean)},
        counts={metric: valid_rows, "periods": int(len(grouped))},
        min_max={metric: {"min": _clean_scalar(work[metric].min()), "max": _clean_scalar(work[metric].max())}},
        percentages={"overall_change_pct": _clean_scalar(overall_growth_pct, 4)} if overall_growth_pct is not None else {},
    )
    return summary, _records_from_df(grouped), evidence


def _run_correlation(
    df: pd.DataFrame,
    plan: AnalyticalTaskPlan,
    profile: DatasetProfileResult | None,
    input_rows: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], NumericalEvidence, list[str]]:
    warnings: list[str] = []
    cols = [c for c in plan.metric_columns if c in df.columns]
    if len(cols) < 2 and profile is not None:
        cols = [c for c in profile.numerical_columns if c in df.columns]
    if len(cols) < 2:
        raise InsufficientAnalysisDataError("Correlation requires at least two numerical columns.")

    num_df = pd.DataFrame({c: _safe_numeric(df, c) for c in cols})
    active_cols = [c for c in cols if num_df[c].dropna().nunique() > 1]
    if len(active_cols) < 2:
        raise InsufficientAnalysisDataError("Fewer than two non-constant numerical columns available.")
    if len(active_cols) < len(cols):
        dropped = [c for c in cols if c not in active_cols]
        warnings.append(f"Excluded constant column(s) from correlation: {dropped}.")

    pearson = num_df[active_cols].corr(method="pearson")
    spearman = num_df[active_cols].corr(method="spearman")

    pair_rows: list[dict[str, Any]] = []
    for i, c1 in enumerate(active_cols):
        for c2 in active_cols[i + 1 :]:
            pair_n = int(num_df[[c1, c2]].dropna().shape[0])
            pair_rows.append(
                {
                    "column_a": c1,
                    "column_b": c2,
                    "pearson_r": _clean_scalar(pearson.loc[c1, c2]),
                    "spearman_rho": _clean_scalar(spearman.loc[c1, c2]),
                    "sample_pairs": pair_n,
                }
            )
    pair_rows.sort(key=lambda r: abs(r["pearson_r"] or 0.0), reverse=True)

    summary = {
        "columns_analyzed": active_cols,
        "pair_count": len(pair_rows),
        "strongest_pair": pair_rows[0] if pair_rows else None,
    }
    evidence = NumericalEvidence(
        columns_used=active_cols,
        input_rows=input_rows,
        filtered_rows=len(df),
        valid_metric_rows=int(num_df[active_cols].dropna(how="all").shape[0]),
        dropped_null_rows=0,
        correlations=pair_rows,
    )
    return summary, pair_rows, evidence, warnings


def _run_distribution(
    df: pd.DataFrame,
    plan: AnalyticalTaskPlan,
    input_rows: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], NumericalEvidence]:
    metric = plan.metric_column
    if not metric:
        raise InsufficientAnalysisDataError("Distribution analysis requires a numerical column.")

    s = _safe_numeric(df, metric)
    valid = s.dropna()
    valid_rows = int(len(valid))
    if valid_rows == 0:
        raise InsufficientAnalysisDataError(f"Column '{metric}' contains no valid numerical values.")

    q25 = float(valid.quantile(0.25))
    q50 = float(valid.quantile(0.50))
    q75 = float(valid.quantile(0.75))
    std_val = float(valid.std(ddof=1)) if valid_rows > 1 else 0.0
    skew_val = float(valid.skew()) if valid_rows > 2 and std_val > 0 else 0.0

    bin_count = min(20, max(5, int(np.sqrt(valid_rows))))
    counts, edges = np.histogram(valid.to_numpy(dtype=float), bins=bin_count)
    bins_table = [
        {
            "bin_start": _clean_scalar(edges[i]),
            "bin_end": _clean_scalar(edges[i + 1]),
            "count": int(counts[i]),
            "pct": round(int(counts[i]) / valid_rows * 100.0, 2),
        }
        for i in range(len(counts))
    ]

    summary = {
        "column": metric,
        "count": valid_rows,
        "sum": _clean_scalar(valid.sum()),
        "mean": _clean_scalar(valid.mean()),
        "median": _clean_scalar(q50),
        "std": _clean_scalar(std_val),
        "min": _clean_scalar(valid.min()),
        "q25": _clean_scalar(q25),
        "q75": _clean_scalar(q75),
        "iqr": _clean_scalar(q75 - q25),
        "max": _clean_scalar(valid.max()),
        "skewness": _clean_scalar(skew_val),
    }
    evidence = NumericalEvidence(
        columns_used=[metric],
        input_rows=input_rows,
        filtered_rows=len(df),
        valid_metric_rows=valid_rows,
        dropped_null_rows=len(df) - valid_rows,
        totals={metric: summary["sum"]},
        averages={metric: summary["mean"]},
        medians={metric: summary["median"]},
        counts={metric: valid_rows},
        min_max={metric: {"min": summary["min"], "max": summary["max"]}},
    )
    return summary, bins_table, evidence


def _run_filtering(
    df: pd.DataFrame,
    plan: AnalyticalTaskPlan,
    input_rows: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], NumericalEvidence]:
    matched_rows = len(df)
    matched_pct = round((matched_rows / input_rows) * 100.0, 4) if input_rows > 0 else 0.0

    totals: dict[str, float | int] = {}
    averages: dict[str, float] = {}
    medians: dict[str, float] = {}
    counts: dict[str, int] = {"matched_rows": matched_rows, "input_rows": input_rows}
    min_max: dict[str, dict[str, float | int]] = {}

    metric = plan.metric_column
    if metric and metric in df.columns and matched_rows > 0:
        s = _safe_numeric(df, metric).dropna()
        if not s.empty:
            totals[metric] = _clean_scalar(s.sum())
            averages[metric] = _clean_scalar(s.mean())
            medians[metric] = _clean_scalar(s.median())
            counts[metric] = int(len(s))
            min_max[metric] = {"min": _clean_scalar(s.min()), "max": _clean_scalar(s.max())}

    summary = {
        "matched_rows": matched_rows,
        "input_rows": input_rows,
        "matched_pct": matched_pct,
        "filters_applied": plan.filters,
        "metric_totals": totals,
        "metric_averages": averages,
    }
    evidence = NumericalEvidence(
        columns_used=plan.target_columns,
        input_rows=input_rows,
        filtered_rows=matched_rows,
        valid_metric_rows=matched_rows,
        dropped_null_rows=0,
        totals=totals,
        averages=averages,
        medians=medians,
        counts=counts,
        min_max=min_max,
        percentages={"matched_pct": matched_pct},
    )
    return summary, _records_from_df(df, max_rows=100), evidence


def _run_missing_values(df: pd.DataFrame, input_rows: int) -> tuple[dict[str, Any], list[dict[str, Any]], NumericalEvidence]:
    rows: list[dict[str, Any]] = []
    pct_map: dict[str, float] = {}
    cnt_map: dict[str, int] = {}
    for col in df.columns:
        miss = int(df[col].isna().sum())
        pct = round((miss / input_rows) * 100.0, 4) if input_rows > 0 else 0.0
        rows.append({"column": str(col), "missing_count": miss, "non_null_count": input_rows - miss, "missing_pct": pct})
        pct_map[str(col)] = pct
        cnt_map[str(col)] = miss
    rows.sort(key=lambda r: r["missing_count"], reverse=True)
    total_missing = sum(cnt_map.values())
    total_cells = input_rows * df.shape[1]
    overall_pct = round((total_missing / total_cells) * 100.0, 4) if total_cells > 0 else 0.0

    summary = {
        "total_cells": total_cells,
        "total_missing_cells": total_missing,
        "overall_missing_pct": overall_pct,
    }
    evidence = NumericalEvidence(
        columns_used=[str(c) for c in df.columns],
        input_rows=input_rows,
        filtered_rows=len(df),
        valid_metric_rows=len(df),
        dropped_null_rows=0,
        counts={"total_missing_cells": total_missing, **cnt_map},
        percentages={"overall_missing_pct": overall_pct, **pct_map},
    )
    return summary, rows, evidence


def _run_dataset_summary(
    df: pd.DataFrame,
    profile: DatasetProfileResult | None,
    input_rows: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], NumericalEvidence]:
    prof = profile or profile_dataframe(df, dataset_name="dataset")
    totals: dict[str, float | int] = {}
    averages: dict[str, float] = {}
    medians: dict[str, float] = {}
    counts: dict[str, int] = {"row_count": prof.row_count, "column_count": prof.column_count}
    min_max: dict[str, dict[str, float | int]] = {}
    rows: list[dict[str, Any]] = []

    for col in prof.numerical_columns:
        if col in df.columns:
            s = _safe_numeric(df, col).dropna()
            if not s.empty:
                totals[col] = _clean_scalar(s.sum())
                averages[col] = _clean_scalar(s.mean())
                medians[col] = _clean_scalar(s.median())
                counts[col] = int(len(s))
                min_max[col] = {"min": _clean_scalar(s.min()), "max": _clean_scalar(s.max())}

    for col, cp in prof.column_profiles.items():
        rows.append(
            {
                "column": col,
                "type": cp.inferred_type,
                "non_null_count": cp.non_null_count,
                "missing_pct": cp.missing_pct,
                "unique_count": cp.unique_count,
                "mean": averages.get(col),
                "total": totals.get(col),
            }
        )

    summary = {
        "row_count": prof.row_count,
        "column_count": prof.column_count,
        "quality_score": prof.quality_report.quality_score,
        "quality_grade": prof.quality_report.quality_grade,
        "numerical_columns": prof.numerical_columns,
        "categorical_columns": prof.categorical_columns,
        "datetime_columns": prof.datetime_columns,
    }
    evidence = NumericalEvidence(
        columns_used=list(df.columns),
        input_rows=input_rows,
        filtered_rows=len(df),
        valid_metric_rows=len(df),
        dropped_null_rows=0,
        totals=totals,
        averages=averages,
        medians=medians,
        counts=counts,
        min_max=min_max,
    )
    return summary, rows, evidence


# ---------------------------------------------------------------------------
# Public Module API
# ---------------------------------------------------------------------------


def execute_analysis(
    df: pd.DataFrame,
    plan: AnalyticalTaskPlan,
    profile: DatasetProfileResult | None = None,
    dataset_name: str = "dataset",
) -> AnalysisResult:
    """Execute a structured AnalyticalTaskPlan purely with Pandas & NumPy in offline mode."""
    if not isinstance(df, pd.DataFrame):
        raise AnalysisEngineError("Input 'df' must be a pandas DataFrame.")
    if not isinstance(plan, AnalyticalTaskPlan):
        raise AnalysisEngineError("Input 'plan' must be an AnalyticalTaskPlan instance.")

    start_ns = time.perf_counter_ns()
    resolved_name = profile.dataset_name if profile else dataset_name

    if df.empty:
        raise InsufficientAnalysisDataError("Cannot execute analysis on an empty DataFrame (0 rows).")

    _validate_plan_safety(df, plan)
    input_rows = int(len(df))

    if plan.intent == AnalyticalIntent.UNKNOWN.value:
        elapsed = round((time.perf_counter_ns() - start_ns) / 1_000_000.0, 3)
        return AnalysisResult(
            status="unsupported",
            intent=plan.intent,
            dataset_name=resolved_name,
            summary_metrics={"reason": "Analytical intent is unknown or unsupported."},
            result_table=[],
            numerical_evidence=NumericalEvidence(
                columns_used=[],
                input_rows=input_rows,
                filtered_rows=input_rows,
                valid_metric_rows=0,
                dropped_null_rows=0,
            ),
            warnings=["Task intent 'unknown' could not be mapped to a deterministic analysis operation."],
            execution_time_ms=elapsed,
            plan_snapshot=plan.to_dict(),
        )

    filtered_df = _apply_safe_filters(df, plan.filters)
    if filtered_df.empty and plan.intent != AnalyticalIntent.FILTERING.value:
        raise InsufficientAnalysisDataError("Applied filters matched 0 records.")

    warnings: list[str] = []
    intent = plan.intent

    if intent in {AnalyticalIntent.RANKING.value, AnalyticalIntent.AGGREGATION.value}:
        summary, table, evidence = _run_groupby_or_scalar(filtered_df, plan, input_rows)
    elif intent == AnalyticalIntent.COMPARISON.value:
        summary, table, evidence = _run_comparison(filtered_df, plan, input_rows)
    elif intent == AnalyticalIntent.TREND_ANALYSIS.value:
        summary, table, evidence = _run_trend(filtered_df, plan, input_rows)
    elif intent == AnalyticalIntent.CORRELATION.value:
        summary, table, evidence, corr_warnings = _run_correlation(filtered_df, plan, profile, input_rows)
        warnings.extend(corr_warnings)
    elif intent == AnalyticalIntent.DISTRIBUTION_ANALYSIS.value:
        summary, table, evidence = _run_distribution(filtered_df, plan, input_rows)
    elif intent == AnalyticalIntent.FILTERING.value:
        summary, table, evidence = _run_filtering(filtered_df, plan, input_rows)
    elif intent == AnalyticalIntent.MISSING_VALUE_ANALYSIS.value:
        summary, table, evidence = _run_missing_values(filtered_df, input_rows)
    elif intent == AnalyticalIntent.DATASET_SUMMARY.value:
        summary, table, evidence = _run_dataset_summary(filtered_df, profile, input_rows)
    elif intent == AnalyticalIntent.ANOMALY_ANALYSIS.value:
        if plan.metric_column:
            summary, table, evidence = _run_distribution(filtered_df, plan, input_rows)
            warnings.append("Statistical distribution computed; run Module 7 AnomalyDetector for Isolation Forest ML scoring.")
        else:
            summary, table, evidence = _run_dataset_summary(filtered_df, profile, input_rows)
    else:
        elapsed = round((time.perf_counter_ns() - start_ns) / 1_000_000.0, 3)
        return AnalysisResult(
            status="unsupported",
            intent=intent,
            dataset_name=resolved_name,
            summary_metrics={"reason": f"Unsupported analytical intent '{intent}'."},
            result_table=[],
            numerical_evidence=NumericalEvidence(
                columns_used=[],
                input_rows=input_rows,
                filtered_rows=len(filtered_df),
                valid_metric_rows=0,
                dropped_null_rows=0,
            ),
            warnings=[f"Intent '{intent}' is not supported by the deterministic analysis engine."],
            execution_time_ms=elapsed,
            plan_snapshot=plan.to_dict(),
        )

    elapsed_ms = round((time.perf_counter_ns() - start_ns) / 1_000_000.0, 3)
    logger.info(
        "Analysis completed (intent=%s, rows=%d->%d, time=%.2fms).",
        intent,
        input_rows,
        len(filtered_df),
        elapsed_ms,
    )

    return AnalysisResult(
        status="completed",
        intent=intent,
        dataset_name=resolved_name,
        summary_metrics=summary,
        result_table=table,
        numerical_evidence=evidence,
        warnings=warnings,
        execution_time_ms=elapsed_ms,
        plan_snapshot=plan.to_dict(),
    )


class AnalysisEngine:
    """Stateful offline service wrapper for Module 6 automated data analysis."""

    def analyze(
        self,
        df: pd.DataFrame,
        plan: AnalyticalTaskPlan,
        profile: DatasetProfileResult | None = None,
        dataset_name: str = "dataset",
    ) -> AnalysisResult:
        return execute_analysis(df=df, plan=plan, profile=profile, dataset_name=dataset_name)
