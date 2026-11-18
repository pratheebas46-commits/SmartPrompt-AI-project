"""Module 6: Automated EDA & Analytical Query Execution Engine for SmartPrompt AI.

Executes schema-verified AnalyticalTaskPlan objects (from Module 4) as well as
standalone automated Exploratory Data Analysis (EDA) on Pandas DataFrames.
Produces structured, JSON-safe statistical tables, correlation matrices,
time-series trends, distribution bins, Plotly-ready chart specifications,
and computation provenance metadata for downstream self-validation (Module 8).
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
# Custom Exceptions
# ---------------------------------------------------------------------------


class EDAExecutionError(Exception):
    """Base exception for automated EDA and query execution errors."""


class InvalidEDAPlanError(EDAExecutionError):
    """Raised when an analytical plan references missing columns or unsupported operations."""


class InsufficientDataForEDAError(EDAExecutionError):
    """Raised when the dataset or filtered subset has insufficient valid data to compute."""


# ---------------------------------------------------------------------------
# Structured Output Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChartSpec:
    """UI-agnostic, JSON-serializable visualization specification for Plotly."""

    chart_type: str  # "bar" | "horizontal_bar" | "line" | "histogram" | "heatmap" | "box" | "scatter"
    title: str
    x_label: str
    y_label: str
    x_values: list[Any] = field(default_factory=list)
    y_values: list[Any] = field(default_factory=list)
    series: list[dict[str, Any]] = field(default_factory=list)
    matrix: list[list[float | None]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EDAResult:
    """Structured analytical output + provenance metadata for validation & insight generation."""

    intent: str
    dataset_name: str
    input_row_count: int
    filtered_row_count: int
    valid_metric_row_count: int
    dropped_null_count: int
    execution_time_ms: float
    summary_metrics: dict[str, Any]
    result_table: list[dict[str, Any]]
    chart_specs: list[ChartSpec]
    warnings: list[str]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "dataset_name": self.dataset_name,
            "input_row_count": self.input_row_count,
            "filtered_row_count": self.filtered_row_count,
            "valid_metric_row_count": self.valid_metric_row_count,
            "dropped_null_count": self.dropped_null_count,
            "execution_time_ms": self.execution_time_ms,
            "summary_metrics": self.summary_metrics,
            "result_table": self.result_table,
            "chart_specs": [c.to_dict() for c in self.chart_specs],
            "warnings": self.warnings,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class AutomatedEDAReport:
    """Full standalone exploratory data analysis report across an entire dataset."""

    dataset_name: str
    row_count: int
    column_count: int
    numeric_summary: list[dict[str, Any]]
    categorical_breakdowns: dict[str, list[dict[str, Any]]]
    correlation_pearson: dict[str, dict[str, float | None]]
    correlation_spearman: dict[str, dict[str, float | None]]
    strong_correlations: list[dict[str, Any]]
    distributions: dict[str, dict[str, Any]]
    chart_specs: list[ChartSpec]
    warnings: list[str]
    execution_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "numeric_summary": self.numeric_summary,
            "categorical_breakdowns": self.categorical_breakdowns,
            "correlation_pearson": self.correlation_pearson,
            "correlation_spearman": self.correlation_spearman,
            "strong_correlations": self.strong_correlations,
            "distributions": self.distributions,
            "chart_specs": [c.to_dict() for c in self.chart_specs],
            "warnings": self.warnings,
            "execution_time_ms": self.execution_time_ms,
        }


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

_AGG_MAP: dict[str, str] = {
    "sum": "sum",
    "mean": "mean",
    "avg": "mean",
    "median": "median",
    "count": "count",
    "min": "min",
    "max": "max",
    "std": "std",
}

_GRANULARITY_FREQ: dict[str, str] = {
    "day": "D",
    "week": "W",
    "month": "ME",
    "quarter": "QE",
    "year": "YE",
}


def _clean_scalar(value: Any, decimals: int = 4) -> Any:
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


def _df_to_clean_records(df: pd.DataFrame, max_rows: int = 100) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in df.head(max_rows).to_dict(orient="records"):
        records.append({str(k): _clean_scalar(v) for k, v in row.items()})
    return records


def _apply_filters(df: pd.DataFrame, filters: list[dict[str, Any]]) -> tuple[pd.DataFrame, list[str]]:
    """Apply structured filter predicates safely and return filtered DataFrame + warnings."""
    if not filters:
        return df, []

    working = df.copy()
    warnings: list[str] = []

    for flt in filters:
        col = flt.get("column")
        op = flt.get("operator")
        val = flt.get("value")
        if not col or col not in working.columns:
            raise InvalidEDAPlanError(f"Filter references missing column '{col}'.")

        series = working[col]
        if op in {">", ">=", "<", "<="}:
            num_series = pd.to_numeric(series, errors="coerce")
            try:
                num_val = float(val)
            except (TypeError, ValueError) as exc:
                raise InvalidEDAPlanError(f"Numeric filter value expected for '{col} {op} {val}'.") from exc
            if op == ">":
                mask = num_series > num_val
            elif op == ">=":
                mask = num_series >= num_val
            elif op == "<":
                mask = num_series < num_val
            else:
                mask = num_series <= num_val
            working = working[mask.fillna(False)]
        elif op == "==":
            if pd.api.types.is_numeric_dtype(series) and isinstance(val, (int, float)):
                working = working[pd.to_numeric(series, errors="coerce") == val]
            else:
                mask = series.astype(str).str.strip().str.lower() == str(val).strip().lower()
                working = working[series.notna() & mask]
        elif op == "contains":
            mask = series.astype(str).str.lower().str.contains(str(val).strip().lower(), na=False, regex=False)
            working = working[series.notna() & mask]
        else:
            warnings.append(f"Unsupported filter operator '{op}' on '{col}' was skipped.")

    return working.reset_index(drop=True), warnings


def _safe_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# Intent-Specific Execution Handlers
# ---------------------------------------------------------------------------


def _exec_ranking_or_aggregation(
    df: pd.DataFrame,
    plan: AnalyticalTaskPlan,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[ChartSpec], int, int, list[str]]:
    warnings: list[str] = []
    metric = plan.metric_column
    group = plan.group_column
    agg_key = (plan.aggregation or "sum").lower()
    agg_fn = _AGG_MAP.get(agg_key, "sum")
    sort_desc = (plan.sort_order or "descending") == "descending"

    if not metric or metric not in df.columns:
        if agg_fn == "count" and group and group in df.columns:
            grouped = (
                df.dropna(subset=[group])
                .groupby(group, dropna=True)
                .size()
                .reset_index(name="count")
            )
            grouped = grouped.sort_values("count", ascending=not sort_desc).reset_index(drop=True)
            total_count = int(grouped["count"].sum())
            if total_count > 0:
                grouped["pct_of_total"] = (grouped["count"] / total_count * 100.0).round(2)
            limited = grouped.head(plan.limit) if plan.limit else grouped
            chart = ChartSpec(
                chart_type="horizontal_bar" if plan.intent == AnalyticalIntent.RANKING.value else "bar",
                title=f"Record Count by {group}",
                x_label="count",
                y_label=group,
                x_values=[_clean_scalar(v) for v in limited["count"].tolist()],
                y_values=[_clean_scalar(v) for v in limited[group].tolist()],
            )
            return (
                {"aggregation": "count", "group_column": group, "total_count": total_count, "group_count": len(grouped)},
                _df_to_clean_records(limited),
                [chart],
                len(df),
                0,
                warnings,
            )
        raise InvalidEDAPlanError(f"Metric column '{metric}' is missing from dataset.")

    numeric_vals = _safe_numeric_series(df, metric)
    valid_mask = numeric_vals.notna()
    valid_count = int(valid_mask.sum())
    dropped_nulls = int(len(df) - valid_count)

    if valid_count == 0:
        raise InsufficientDataForEDAError(f"Column '{metric}' has no valid numeric values.")

    work = df.loc[valid_mask].copy()
    work[metric] = numeric_vals.loc[valid_mask]

    if group and group in work.columns:
        non_null_group = work.dropna(subset=[group])
        if non_null_group.empty:
            raise InsufficientDataForEDAError(f"Grouping column '{group}' is completely null.")

        agg_series = non_null_group.groupby(group, dropna=True)[metric].agg(agg_fn)
        count_series = non_null_group.groupby(group, dropna=True)[metric].count()
        grouped = pd.DataFrame(
            {
                group: agg_series.index,
                f"{agg_key}_{metric}": agg_series.values,
                "record_count": count_series.values,
            }
        )
        val_col = f"{agg_key}_{metric}"
        grouped = grouped.sort_values(val_col, ascending=not sort_desc).reset_index(drop=True)

        total_metric_sum = float(non_null_group[metric].sum())
        if agg_fn == "sum" and total_metric_sum != 0:
            grouped["pct_of_total"] = (grouped[val_col] / total_metric_sum * 100.0).round(2)

        limited = grouped.head(plan.limit) if plan.limit else grouped
        top_row = grouped.iloc[0] if not grouped.empty else None

        summary = {
            "metric_column": metric,
            "group_column": group,
            "aggregation": agg_key,
            "group_count": int(len(grouped)),
            "grand_total_sum": _clean_scalar(total_metric_sum),
            "overall_mean": _clean_scalar(non_null_group[metric].mean()),
            "top_group": _clean_scalar(top_row[group]) if top_row is not None else None,
            "top_value": _clean_scalar(top_row[val_col]) if top_row is not None else None,
        }

        is_ranking = plan.intent == AnalyticalIntent.RANKING.value
        chart = ChartSpec(
            chart_type="horizontal_bar" if is_ranking else "bar",
            title=f"{agg_key.title()} of {metric} by {group}",
            x_label=val_col if is_ranking else group,
            y_label=group if is_ranking else val_col,
            x_values=[_clean_scalar(v) for v in (limited[val_col] if is_ranking else limited[group]).tolist()],
            y_values=[_clean_scalar(v) for v in (limited[group] if is_ranking else limited[val_col]).tolist()],
            metadata={"sort_order": "descending" if sort_desc else "ascending", "limit": plan.limit},
        )
        return summary, _df_to_clean_records(limited), [chart], valid_count, dropped_nulls, warnings

    # Ungrouped scalar aggregation
    scalar_val = getattr(work[metric], agg_fn)()
    summary = {
        "metric_column": metric,
        "aggregation": agg_key,
        "value": _clean_scalar(scalar_val),
        "grand_total_sum": _clean_scalar(work[metric].sum()),
        "mean": _clean_scalar(work[metric].mean()),
        "median": _clean_scalar(work[metric].median()),
        "min": _clean_scalar(work[metric].min()),
        "max": _clean_scalar(work[metric].max()),
        "count": valid_count,
    }
    table = [{"metric": metric, "aggregation": agg_key, "value": summary["value"], "count": valid_count}]
    return summary, table, [], valid_count, dropped_nulls, warnings


def _exec_comparison(
    df: pd.DataFrame,
    plan: AnalyticalTaskPlan,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[ChartSpec], int, int, list[str]]:
    warnings: list[str] = []
    metric = plan.metric_column
    group = plan.group_column

    if group and group in df.columns and metric and metric in df.columns:
        numeric_vals = _safe_numeric_series(df, metric)
        valid_mask = numeric_vals.notna() & df[group].notna()
        valid_count = int(valid_mask.sum())
        dropped_nulls = int(len(df) - valid_count)
        if valid_count == 0:
            raise InsufficientDataForEDAError(f"No valid pairs for '{metric}' and '{group}'.")

        work = df.loc[valid_mask].copy()
        work[metric] = numeric_vals.loc[valid_mask]
        comp = (
            work.groupby(group, dropna=True)[metric]
            .agg(["count", "sum", "mean", "median", "std", "min", "max"])
            .reset_index()
            .sort_values("mean", ascending=False)
            .reset_index(drop=True)
        )
        max_mean = float(comp["mean"].max())
        min_mean = float(comp["mean"].min())
        summary = {
            "metric_column": metric,
            "group_column": group,
            "groups_compared": int(len(comp)),
            "highest_mean_group": _clean_scalar(comp.iloc[0][group]),
            "lowest_mean_group": _clean_scalar(comp.iloc[-1][group]),
            "mean_spread": _clean_scalar(max_mean - min_mean),
            "grand_total_sum": _clean_scalar(work[metric].sum()),
        }
        chart = ChartSpec(
            chart_type="bar",
            title=f"Comparison of Mean {metric} across {group}",
            x_label=group,
            y_label=f"mean_{metric}",
            x_values=[_clean_scalar(v) for v in comp[group].tolist()],
            y_values=[_clean_scalar(v) for v in comp["mean"].tolist()],
        )
        return summary, _df_to_clean_records(comp), [chart], valid_count, dropped_nulls, warnings

    # Fallback: compare multiple numeric columns side-by-side
    cols = [c for c in plan.metric_columns if c in df.columns]
    if len(cols) >= 2:
        rows: list[dict[str, Any]] = []
        means: list[Any] = []
        for col in cols:
            s = _safe_numeric_series(df, col).dropna()
            rows.append(
                {
                    "column": col,
                    "count": int(len(s)),
                    "sum": _clean_scalar(s.sum()) if not s.empty else None,
                    "mean": _clean_scalar(s.mean()) if not s.empty else None,
                    "median": _clean_scalar(s.median()) if not s.empty else None,
                    "std": _clean_scalar(s.std(ddof=1)) if len(s) > 1 else 0.0,
                }
            )
            means.append(_clean_scalar(s.mean()) if not s.empty else None)
        chart = ChartSpec(
            chart_type="bar",
            title=f"Metric Comparison: {', '.join(cols)}",
            x_label="Metric",
            y_label="Mean",
            x_values=cols,
            y_values=means,
        )
        return {"compared_columns": cols}, rows, [chart], len(df), 0, warnings

    return _exec_ranking_or_aggregation(df, plan)


def _exec_trend_analysis(
    df: pd.DataFrame,
    plan: AnalyticalTaskPlan,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[ChartSpec], int, int, list[str]]:
    warnings: list[str] = []
    date_col = plan.date_column
    metric = plan.metric_column
    gran = (plan.time_granularity or "month").lower()
    agg_key = (plan.aggregation or "sum").lower()
    agg_fn = _AGG_MAP.get(agg_key, "sum")

    if not date_col or date_col not in df.columns:
        raise InvalidEDAPlanError(f"Date column '{date_col}' not found for trend analysis.")
    if not metric or metric not in df.columns:
        raise InvalidEDAPlanError(f"Metric column '{metric}' not found for trend analysis.")

    try:
        dt_series = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
    except TypeError:
        dt_series = pd.to_datetime(df[date_col], errors="coerce")
    num_series = _safe_numeric_series(df, metric)

    valid_mask = dt_series.notna() & num_series.notna()
    valid_count = int(valid_mask.sum())
    dropped_nulls = int(len(df) - valid_count)
    if valid_count == 0:
        raise InsufficientDataForEDAError(f"No valid (date, metric) pairs for '{date_col}' and '{metric}'.")

    work = pd.DataFrame({"_dt": dt_series.loc[valid_mask], metric: num_series.loc[valid_mask]})
    work = work.sort_values("_dt")

    freq = _GRANULARITY_FREQ.get(gran, "ME")
    try:
        resampled = work.set_index("_dt")[metric].resample(freq).agg([agg_fn, "count"])
    except ValueError:
        # Compatibility fallback for older pandas aliases ("M", "Q", "Y")
        legacy_freq = {"ME": "M", "QE": "Q", "YE": "Y"}.get(freq, freq)
        resampled = work.set_index("_dt")[metric].resample(legacy_freq).agg([agg_fn, "count"])

    resampled = resampled[resampled["count"] > 0].copy()
    resampled = resampled.rename(columns={agg_fn: "value", "count": "record_count"})
    resampled["period"] = resampled.index.strftime("%Y-%m-%d")
    resampled["delta"] = resampled["value"].diff()
    resampled["pct_change"] = resampled["value"].pct_change().replace([np.inf, -np.inf], np.nan) * 100.0

    table_df = resampled[["period", "value", "record_count", "delta", "pct_change"]].reset_index(drop=True)

    first_val = float(table_df["value"].iloc[0]) if not table_df.empty else 0.0
    last_val = float(table_df["value"].iloc[-1]) if not table_df.empty else 0.0
    overall_change_pct = ((last_val - first_val) / abs(first_val) * 100.0) if first_val != 0 else None

    summary = {
        "date_column": date_col,
        "metric_column": metric,
        "granularity": gran,
        "aggregation": agg_key,
        "period_count": int(len(table_df)),
        "grand_total_sum": _clean_scalar(work[metric].sum()),
        "start_period": _clean_scalar(table_df["period"].iloc[0]) if not table_df.empty else None,
        "end_period": _clean_scalar(table_df["period"].iloc[-1]) if not table_df.empty else None,
        "overall_change_pct": _clean_scalar(overall_change_pct, decimals=2),
    }

    chart = ChartSpec(
        chart_type="line",
        title=f"{gran.title()}ly {agg_key.title()} of {metric} over {date_col}",
        x_label="period",
        y_label=f"{agg_key}_{metric}",
        x_values=table_df["period"].tolist(),
        y_values=[_clean_scalar(v) for v in table_df["value"].tolist()],
    )
    return summary, _df_to_clean_records(table_df), [chart], valid_count, dropped_nulls, warnings


def _exec_correlation(
    df: pd.DataFrame,
    plan: AnalyticalTaskPlan,
    profile: DatasetProfileResult | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[ChartSpec], int, int, list[str]]:
    warnings: list[str] = []
    cols = [c for c in plan.metric_columns if c in df.columns]
    if len(cols) < 2 and profile is not None:
        cols = [c for c in profile.numerical_columns if c in df.columns]
    if len(cols) < 2:
        raise InsufficientDataForEDAError("Correlation analysis requires at least two numerical columns.")

    num_df = pd.DataFrame({c: _safe_numeric_series(df, c) for c in cols})
    valid_df = num_df.dropna(how="all")
    if len(valid_df) < 3:
        raise InsufficientDataForEDAError("At least 3 non-null rows are required to compute correlation.")

    # Exclude zero-variance columns cleanly from Pearson/Spearman
    active_cols = [c for c in cols if valid_df[c].dropna().nunique() > 1]
    if len(active_cols) < 2:
        raise InsufficientDataForEDAError("Fewer than 2 non-constant numerical columns available for correlation.")

    if len(active_cols) < len(cols):
        dropped_const = [c for c in cols if c not in active_cols]
        warnings.append(f"Excluded zero-variance column(s) from correlation: {dropped_const}.")

    pearson = valid_df[active_cols].corr(method="pearson")
    spearman = valid_df[active_cols].corr(method="spearman")

    pair_rows: list[dict[str, Any]] = []
    for i, c1 in enumerate(active_cols):
        for c2 in active_cols[i + 1 :]:
            pair_n = int(valid_df[[c1, c2]].dropna().shape[0])
            r_p = _clean_scalar(pearson.loc[c1, c2])
            r_s = _clean_scalar(spearman.loc[c1, c2])
            pair_rows.append(
                {
                    "column_a": c1,
                    "column_b": c2,
                    "pearson_r": r_p,
                    "spearman_rho": r_s,
                    "sample_pairs": pair_n,
                }
            )
    pair_rows.sort(key=lambda r: abs(r["pearson_r"] or 0.0), reverse=True)

    charts: list[ChartSpec] = []
    if len(active_cols) == 2:
        c1, c2 = active_cols[0], active_cols[1]
        scatter_df = valid_df[[c1, c2]].dropna().head(500)
        charts.append(
            ChartSpec(
                chart_type="scatter",
                title=f"{c1} vs {c2} (Pearson r = {pair_rows[0]['pearson_r']})",
                x_label=c1,
                y_label=c2,
                x_values=[_clean_scalar(v) for v in scatter_df[c1].tolist()],
                y_values=[_clean_scalar(v) for v in scatter_df[c2].tolist()],
            )
        )
    else:
        charts.append(
            ChartSpec(
                chart_type="heatmap",
                title="Pearson Correlation Matrix",
                x_label="Columns",
                y_label="Columns",
                x_values=active_cols,
                y_values=active_cols,
                matrix=[[_clean_scalar(pearson.loc[r, c]) for c in active_cols] for r in active_cols],
            )
        )

    strongest = pair_rows[0] if pair_rows else None
    summary = {
        "columns_analyzed": active_cols,
        "pair_count": len(pair_rows),
        "strongest_pair": strongest,
    }
    return summary, pair_rows, charts, int(len(valid_df)), int(len(df) - len(valid_df)), warnings


def _exec_distribution(
    df: pd.DataFrame,
    plan: AnalyticalTaskPlan,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[ChartSpec], int, int, list[str]]:
    metric = plan.metric_column
    if not metric or metric not in df.columns:
        raise InvalidEDAPlanError(f"Metric column '{metric}' not found for distribution analysis.")

    s = _safe_numeric_series(df, metric)
    valid = s.dropna()
    valid_count = int(len(valid))
    dropped_nulls = int(len(df) - valid_count)
    if valid_count == 0:
        raise InsufficientDataForEDAError(f"Column '{metric}' has no valid numerical values.")

    q25 = float(valid.quantile(0.25))
    q50 = float(valid.quantile(0.50))
    q75 = float(valid.quantile(0.75))
    iqr = q75 - q25
    std_val = float(valid.std(ddof=1)) if valid_count > 1 else 0.0
    skew_val = float(valid.skew()) if valid_count > 2 and std_val > 0 else 0.0
    kurt_val = float(valid.kurt()) if valid_count > 3 and std_val > 0 else 0.0

    bin_count = min(20, max(5, int(np.sqrt(valid_count))))
    counts, edges = np.histogram(valid.to_numpy(dtype=float), bins=bin_count)
    bin_labels = [f"{edges[i]:.2f}–{edges[i+1]:.2f}" for i in range(len(counts))]

    summary = {
        "column": metric,
        "count": valid_count,
        "mean": _clean_scalar(valid.mean()),
        "median": _clean_scalar(q50),
        "std": _clean_scalar(std_val),
        "min": _clean_scalar(valid.min()),
        "q25": _clean_scalar(q25),
        "q75": _clean_scalar(q75),
        "iqr": _clean_scalar(iqr),
        "max": _clean_scalar(valid.max()),
        "skewness": _clean_scalar(skew_val),
        "kurtosis": _clean_scalar(kurt_val),
    }

    bin_rows = [
        {"bin_range": bin_labels[i], "bin_start": _clean_scalar(edges[i]), "bin_end": _clean_scalar(edges[i + 1]), "count": int(counts[i])}
        for i in range(len(counts))
    ]
    chart = ChartSpec(
        chart_type="histogram",
        title=f"Distribution of {metric}",
        x_label=metric,
        y_label="Frequency",
        x_values=bin_labels,
        y_values=[int(c) for c in counts.tolist()],
        metadata={"bin_edges": [_clean_scalar(e) for e in edges.tolist()]},
    )
    return summary, bin_rows, [chart], valid_count, dropped_nulls, []


def _exec_missing_analysis(
    df: pd.DataFrame,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[ChartSpec], int, int, list[str]]:
    row_count = len(df)
    rows: list[dict[str, Any]] = []
    for col in df.columns:
        miss = int(df[col].isna().sum())
        pct = round((miss / row_count) * 100.0, 2) if row_count > 0 else 0.0
        rows.append(
            {
                "column": str(col),
                "missing_count": miss,
                "non_null_count": int(row_count - miss),
                "missing_pct": pct,
            }
        )
    rows.sort(key=lambda r: r["missing_count"], reverse=True)
    total_missing = sum(r["missing_count"] for r in rows)
    total_cells = row_count * df.shape[1]

    chart = ChartSpec(
        chart_type="bar",
        title="Missing Values (%) by Column",
        x_label="Column",
        y_label="Missing %",
        x_values=[r["column"] for r in rows],
        y_values=[r["missing_pct"] for r in rows],
    )
    summary = {
        "total_cells": int(total_cells),
        "total_missing_cells": int(total_missing),
        "overall_missing_pct": round((total_missing / total_cells) * 100.0, 2) if total_cells > 0 else 0.0,
        "columns_with_missing": sum(1 for r in rows if r["missing_count"] > 0),
    }
    return summary, rows, [chart], row_count, 0, []


def _exec_dataset_summary(
    df: pd.DataFrame,
    profile: DatasetProfileResult | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[ChartSpec], int, int, list[str]]:
    prof = profile or profile_dataframe(df, dataset_name="dataset")
    rows: list[dict[str, Any]] = []
    for col, cp in prof.column_profiles.items():
        rows.append(
            {
                "column": col,
                "type": cp.inferred_type,
                "non_null_count": cp.non_null_count,
                "missing_pct": cp.missing_pct,
                "unique_count": cp.unique_count,
                "mean_or_mode": cp.stats.get("mean") if cp.inferred_type == "numerical" else cp.stats.get("mode"),
            }
        )
    summary = {
        "row_count": prof.row_count,
        "column_count": prof.column_count,
        "numerical_columns": prof.numerical_columns,
        "categorical_columns": prof.categorical_columns,
        "datetime_columns": prof.datetime_columns,
        "boolean_columns": prof.boolean_columns,
        "quality_score": prof.quality_report.quality_score,
        "quality_grade": prof.quality_report.quality_grade,
        "overall_missing_pct": prof.missing_summary.overall_missing_pct,
        "duplicate_rows": prof.duplicate_summary.duplicate_row_count,
    }
    return summary, rows, [], prof.row_count, 0, []


def _exec_filtering(
    df: pd.DataFrame,
    original_row_count: int,
    plan: AnalyticalTaskPlan,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[ChartSpec], int, int, list[str]]:
    matched_count = len(df)
    match_pct = round((matched_count / original_row_count) * 100.0, 2) if original_row_count > 0 else 0.0
    summary: dict[str, Any] = {
        "matched_rows": matched_count,
        "original_rows": original_row_count,
        "matched_pct": match_pct,
        "filters_applied": plan.filters,
    }
    if plan.metric_column and plan.metric_column in df.columns and matched_count > 0:
        s = _safe_numeric_series(df, plan.metric_column).dropna()
        if not s.empty:
            summary["metric_sum"] = _clean_scalar(s.sum())
            summary["metric_mean"] = _clean_scalar(s.mean())
    return summary, _df_to_clean_records(df, max_rows=50), [], matched_count, 0, []


# ---------------------------------------------------------------------------
# Public API: Plan Execution, Standalone Automated EDA & Plotly Helper
# ---------------------------------------------------------------------------


def execute_analytical_plan(
    df: pd.DataFrame,
    plan: AnalyticalTaskPlan,
    profile: DatasetProfileResult | None = None,
    dataset_name: str = "dataset",
) -> EDAResult:
    """Execute a structured AnalyticalTaskPlan from Module 4 against a Pandas DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise EDAExecutionError("Input 'df' must be a pandas DataFrame.")
    if df.empty:
        raise InsufficientDataForEDAError("Cannot run EDA on an empty DataFrame (0 rows).")

    start_ns = time.perf_counter_ns()
    input_rows = int(len(df))

    filtered_df, filter_warnings = _apply_filters(df, plan.filters)
    filtered_rows = int(len(filtered_df))
    if filtered_rows == 0 and plan.intent != AnalyticalIntent.FILTERING.value:
        raise InsufficientDataForEDAError("Applied filters resulted in 0 matching rows.")

    intent = plan.intent
    if intent in {AnalyticalIntent.RANKING.value, AnalyticalIntent.AGGREGATION.value}:
        summary, table, charts, valid_count, dropped_nulls, exec_warnings = _exec_ranking_or_aggregation(
            filtered_df, plan
        )
    elif intent == AnalyticalIntent.COMPARISON.value:
        summary, table, charts, valid_count, dropped_nulls, exec_warnings = _exec_comparison(
            filtered_df, plan
        )
    elif intent == AnalyticalIntent.TREND_ANALYSIS.value:
        summary, table, charts, valid_count, dropped_nulls, exec_warnings = _exec_trend_analysis(
            filtered_df, plan
        )
    elif intent == AnalyticalIntent.CORRELATION.value:
        summary, table, charts, valid_count, dropped_nulls, exec_warnings = _exec_correlation(
            filtered_df, plan, profile
        )
    elif intent == AnalyticalIntent.DISTRIBUTION_ANALYSIS.value:
        summary, table, charts, valid_count, dropped_nulls, exec_warnings = _exec_distribution(
            filtered_df, plan
        )
    elif intent == AnalyticalIntent.MISSING_VALUE_ANALYSIS.value:
        summary, table, charts, valid_count, dropped_nulls, exec_warnings = _exec_missing_analysis(filtered_df)
    elif intent == AnalyticalIntent.FILTERING.value:
        summary, table, charts, valid_count, dropped_nulls, exec_warnings = _exec_filtering(
            filtered_df, input_rows, plan
        )
    elif intent == AnalyticalIntent.ANOMALY_ANALYSIS.value and plan.metric_column and plan.metric_column in filtered_df.columns:
        summary, table, charts, valid_count, dropped_nulls, exec_warnings = _exec_distribution(
            filtered_df, plan
        )
        summary["note"] = "Baseline distribution computed; run Module 7 AnomalyDetector for ML scoring."
    else:
        summary, table, charts, valid_count, dropped_nulls, exec_warnings = _exec_dataset_summary(
            filtered_df, profile
        )

    elapsed_ms = round((time.perf_counter_ns() - start_ns) / 1_000_000.0, 3)
    all_warnings = [*filter_warnings, *exec_warnings]

    provenance = {
        "target_columns": plan.target_columns,
        "metric_column": plan.metric_column,
        "metric_columns": plan.metric_columns,
        "group_column": plan.group_column,
        "date_column": plan.date_column,
        "aggregation": plan.aggregation,
        "filters": plan.filters,
        "limit": plan.limit,
    }

    logger.info(
        "Executed EDA plan (intent=%s, rows=%d->%d, valid=%d, time=%.2fms).",
        intent,
        input_rows,
        filtered_rows,
        valid_count,
        elapsed_ms,
    )

    return EDAResult(
        intent=intent,
        dataset_name=profile.dataset_name if profile else dataset_name,
        input_row_count=input_rows,
        filtered_row_count=filtered_rows,
        valid_metric_row_count=valid_count,
        dropped_null_count=dropped_nulls,
        execution_time_ms=elapsed_ms,
        summary_metrics=summary,
        result_table=table,
        chart_specs=charts,
        warnings=all_warnings,
        provenance=provenance,
    )


def run_automated_eda(
    df: pd.DataFrame,
    profile: DatasetProfileResult | None = None,
    dataset_name: str = "dataset",
) -> AutomatedEDAReport:
    """Perform comprehensive standalone automated EDA across all columns of a dataset."""
    if not isinstance(df, pd.DataFrame):
        raise EDAExecutionError("Input 'df' must be a pandas DataFrame.")
    if df.empty:
        raise InsufficientDataForEDAError("Cannot run automated EDA on an empty DataFrame.")

    start_ns = time.perf_counter_ns()
    prof = profile or profile_dataframe(df, dataset_name=dataset_name)
    warnings: list[str] = []
    charts: list[ChartSpec] = []

    # 1. Numeric summary & histograms
    numeric_summary: list[dict[str, Any]] = []
    distributions: dict[str, dict[str, Any]] = {}
    for col in prof.numerical_columns:
        s = _safe_numeric_series(df, col).dropna()
        if s.empty:
            continue
        q25 = float(s.quantile(0.25))
        q50 = float(s.quantile(0.50))
        q75 = float(s.quantile(0.75))
        std_val = float(s.std(ddof=1)) if len(s) > 1 else 0.0
        numeric_summary.append(
            {
                "column": col,
                "count": int(len(s)),
                "mean": _clean_scalar(s.mean()),
                "median": _clean_scalar(q50),
                "std": _clean_scalar(std_val),
                "min": _clean_scalar(s.min()),
                "q25": _clean_scalar(q25),
                "q75": _clean_scalar(q75),
                "iqr": _clean_scalar(q75 - q25),
                "max": _clean_scalar(s.max()),
                "skewness": _clean_scalar(float(s.skew())) if len(s) > 2 and std_val > 0 else 0.0,
            }
        )
        bins = min(15, max(5, int(np.sqrt(len(s)))))
        counts, edges = np.histogram(s.to_numpy(dtype=float), bins=bins)
        labels = [f"{edges[i]:.2f}–{edges[i+1]:.2f}" for i in range(len(counts))]
        distributions[col] = {"bin_labels": labels, "counts": [int(c) for c in counts.tolist()]}
        if len(charts) < 3:
            charts.append(
                ChartSpec(
                    chart_type="histogram",
                    title=f"Distribution of {col}",
                    x_label=col,
                    y_label="Count",
                    x_values=labels,
                    y_values=[int(c) for c in counts.tolist()],
                )
            )

    # 2. Categorical breakdowns
    categorical_breakdowns: dict[str, list[dict[str, Any]]] = {}
    for col in prof.categorical_columns[:8]:
        vc = df[col].dropna().astype(str).value_counts().head(10)
        total = int(df[col].notna().sum())
        categorical_breakdowns[col] = [
            {
                "category": str(k),
                "count": int(v),
                "percentage": round((int(v) / total) * 100.0, 2) if total > 0 else 0.0,
            }
            for k, v in vc.items()
        ]

    # 3. Correlation matrices
    corr_pearson: dict[str, dict[str, float | None]] = {}
    corr_spearman: dict[str, dict[str, float | None]] = {}
    strong_corrs: list[dict[str, Any]] = []
    active_num = [c for c in prof.numerical_columns if _safe_numeric_series(df, c).dropna().nunique() > 1]
    if len(active_num) >= 2:
        num_df = pd.DataFrame({c: _safe_numeric_series(df, c) for c in active_num})
        p_mat = num_df.corr(method="pearson")
        s_mat = num_df.corr(method="spearman")
        for r in active_num:
            corr_pearson[r] = {c: _clean_scalar(p_mat.loc[r, c]) for c in active_num}
            corr_spearman[r] = {c: _clean_scalar(s_mat.loc[r, c]) for c in active_num}
        for i, c1 in enumerate(active_num):
            for c2 in active_num[i + 1 :]:
                r_val = _clean_scalar(p_mat.loc[c1, c2])
                if r_val is not None:
                    strong_corrs.append({"column_a": c1, "column_b": c2, "pearson_r": r_val})
        strong_corrs.sort(key=lambda item: abs(item["pearson_r"]), reverse=True)
        charts.append(
            ChartSpec(
                chart_type="heatmap",
                title="Pearson Correlation Heatmap",
                x_label="Features",
                y_label="Features",
                x_values=active_num,
                y_values=active_num,
                matrix=[[corr_pearson[r][c] for c in active_num] for r in active_num],
            )
        )
    elif len(prof.numerical_columns) >= 2:
        warnings.append("Correlation matrix skipped because fewer than 2 numerical columns have non-zero variance.")

    elapsed_ms = round((time.perf_counter_ns() - start_ns) / 1_000_000.0, 3)
    return AutomatedEDAReport(
        dataset_name=prof.dataset_name,
        row_count=prof.row_count,
        column_count=prof.column_count,
        numeric_summary=numeric_summary,
        categorical_breakdowns=categorical_breakdowns,
        correlation_pearson=corr_pearson,
        correlation_spearman=corr_spearman,
        strong_correlations=strong_corrs[:10],
        distributions=distributions,
        chart_specs=charts,
        warnings=warnings,
        execution_time_ms=elapsed_ms,
    )


def build_plotly_figure(spec: ChartSpec) -> Any:
    """Convert a ChartSpec into an interactive Plotly Figure when Plotly is installed."""
    import plotly.graph_objects as go  # Lazy import so headless/API testing stays lightweight

    if spec.chart_type == "horizontal_bar":
        fig = go.Figure(go.Bar(x=spec.x_values, y=spec.y_values, orientation="h"))
        fig.update_layout(yaxis={"autorange": "reversed"})
    elif spec.chart_type in {"bar", "histogram"}:
        fig = go.Figure(go.Bar(x=spec.x_values, y=spec.y_values))
    elif spec.chart_type == "line":
        fig = go.Figure(go.Scatter(x=spec.x_values, y=spec.y_values, mode="lines+markers"))
    elif spec.chart_type == "scatter":
        fig = go.Figure(go.Scatter(x=spec.x_values, y=spec.y_values, mode="markers"))
    elif spec.chart_type == "heatmap":
        fig = go.Figure(go.Heatmap(z=spec.matrix, x=spec.x_values, y=spec.y_values, colorscale="RdBu", zmid=0))
    else:
        fig = go.Figure(go.Bar(x=spec.x_values, y=spec.y_values))

    fig.update_layout(title=spec.title, xaxis_title=spec.x_label, yaxis_title=spec.y_label, template="plotly_white")
    return fig


class EDAEngine:
    """Stateful service wrapper for Module 6 automated EDA and plan execution."""

    def execute_plan(
        self,
        df: pd.DataFrame,
        plan: AnalyticalTaskPlan,
        profile: DatasetProfileResult | None = None,
        dataset_name: str = "dataset",
    ) -> EDAResult:
        return execute_analytical_plan(df=df, plan=plan, profile=profile, dataset_name=dataset_name)

    def run_full_eda(
        self,
        df: pd.DataFrame,
        profile: DatasetProfileResult | None = None,
        dataset_name: str = "dataset",
    ) -> AutomatedEDAReport:
        return run_automated_eda(df=df, profile=profile, dataset_name=dataset_name)
