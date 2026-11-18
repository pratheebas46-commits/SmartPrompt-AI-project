"""Module 4: Dataset-Aware Smart Prompt Optimizer for SmartPrompt AI.

Transforms vague or underspecified natural-language questions into clear,
schema-grounded analytical prompts and structured execution plans using
Module 1 (DatasetProfileResult) and Module 3 (IntentDetector).
Works 100% offline by default and optionally supports server-side online
LLM rewriting when EXECUTION_MODE='online' with valid server credentials.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
import os
import re
from typing import Any

from config.logging_config import get_logger
from config.mode_manager import ModeManager
from config.settings import Settings, get_settings
from modules.dataset_profiler import DatasetProfileResult
from modules.intent_detector import AnalyticalIntent, AnalyticalTask, detect_intent

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------


class PromptOptimizationError(Exception):
    """Base exception for dataset-aware prompt optimization errors."""


class EmptyPromptError(PromptOptimizationError):
    """Raised when the user prompt is empty or invalid."""


# ---------------------------------------------------------------------------
# Structured Output Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalyticalStep:
    """Single deterministic step in the structured analytical task plan."""

    step_number: int
    operation: str
    description: str
    columns: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnalyticalTaskPlan:
    """Executable, schema-verified analytical plan derived from the optimized prompt."""

    intent: str
    can_execute: bool
    target_columns: list[str]
    metric_column: str | None
    metric_columns: list[str]
    group_column: str | None
    date_column: str | None
    time_granularity: str | None
    aggregation: str | None
    sort_order: str | None
    limit: int | None
    filters: list[dict[str, Any]]
    recommended_chart_type: str
    execution_steps: list[AnalyticalStep]
    quality_caveats: list[str]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["execution_steps"] = [step.to_dict() for step in self.execution_steps]
        return data


@dataclass(frozen=True)
class PromptOptimizationResult:
    """Complete output of Module 4 prompt optimization."""

    original_prompt: str
    improved_prompt: str
    detected_intent: str
    intent_confidence: float
    missing_context: list[str]
    improvement_reasons: list[str]
    unsupported_references: list[str]
    warnings: list[str]
    referenced_columns: list[str]
    rewritten_by: str  # "local_rule_engine" | "online_llm"
    task_plan: AnalyticalTaskPlan

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_prompt": self.original_prompt,
            "improved_prompt": self.improved_prompt,
            "detected_intent": self.detected_intent,
            "intent_confidence": self.intent_confidence,
            "missing_context": self.missing_context,
            "improvement_reasons": self.improvement_reasons,
            "unsupported_references": self.unsupported_references,
            "warnings": self.warnings,
            "referenced_columns": self.referenced_columns,
            "rewritten_by": self.rewritten_by,
            "task_plan": self.task_plan.to_dict(),
        }


# ---------------------------------------------------------------------------
# Vocabulary & Unsupported Reference Helpers
# ---------------------------------------------------------------------------

_ANALYTICAL_VOCABULARY: frozenset[str] = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "between", "by", "can", "do", "does",
        "for", "from", "get", "give", "has", "have", "how", "in", "is", "it", "me", "of",
        "on", "or", "per", "please", "show", "tell", "than", "that", "the", "their",
        "them", "there", "these", "this", "to", "us", "vs", "versus", "what", "when",
        "where", "which", "who", "why", "with", "would", "you",
        # Analytical verbs/nouns
        "across", "all", "among", "amount", "anomalies", "anomaly", "analyze", "analysis",
        "average", "avg", "below", "above", "best", "bottom", "break", "breakdown",
        "calculate", "change", "check", "column", "columns", "compare", "comparison",
        "compute", "contrast", "correlate", "correlation", "count", "daily", "data",
        "dataset", "date", "describe", "detect", "difference", "distribution", "each",
        "empty", "equals", "equal", "extreme", "fewest", "field", "fields", "filter",
        "filtered", "find", "first", "frequency", "generated", "greater", "group",
        "grouped", "growth", "having", "highest", "histogram", "impact", "incomplete",
        "insights", "irregular", "largest", "last", "leading", "least", "less", "limit",
        "list", "lower", "lowest", "many", "max", "maximum", "mean", "median", "metric",
        "metrics", "min", "minimum", "missing", "month", "monthly", "more", "most",
        "nan", "null", "nulls", "number", "numbers", "outlier", "outliers", "over",
        "overall", "overview", "pattern", "patterns", "peak", "percent", "percentile",
        "percentiles", "plot", "profile", "quarter", "quarterly", "quartile", "quartiles",
        "range", "rank", "ranked", "ranking", "ratio", "record", "records", "relate",
        "relationship", "report", "row", "rows", "scatter", "skew", "skewness",
        "smallest", "sort", "sorted", "spike", "spikes", "spread", "standard",
        "deviation", "stats", "statistics", "structure", "sum", "summarize", "summary",
        "suspicious", "table", "time", "series", "top", "total", "trajectory", "trend",
        "trends", "under", "unique", "unusual", "value", "values", "variance",
        "volatility", "week", "weekly", "worst", "year", "yearly", "annual", "annually",
    }
)

_CANDIDATE_METRIC_NOUNS = re.compile(
    r"\b(?:total|average|avg|mean|median|sum|min|max|highest|lowest|top|bottom|compare|trend\s+of|distribution\s+of|anomalies\s+in|outliers\s+in|correlation\s+between)\s+([a-z][a-z0-9_]{2,})"
    r"|\b(?:by|per|across|group\s+by)\s+([a-z][a-z0-9_]{2,})"
    r"|\b([a-z][a-z0-9_]{2,})\s*(?:>=|<=|>|<|==|=)\s*-?\d+",
    re.IGNORECASE,
)


def _detect_unsupported_references(
    question: str,
    task: AnalyticalTask,
    profile: DatasetProfileResult,
) -> list[str]:
    """Identify column-like or metric-like terms in the question absent from the dataset."""
    existing_lower = {c.lower() for c in profile.column_profiles}
    matched_tokens: set[str] = set()
    for col in task.relevant_columns:
        matched_tokens.add(col.lower())
        for part in re.split(r"[_\-\s]+", col.lower()):
            if part:
                matched_tokens.add(part)

    # Also collect known categorical sample/top values so filter literals aren't flagged as missing columns
    known_values: set[str] = set()
    for prof in profile.column_profiles.values():
        for sv in prof.sample_values:
            if sv is not None:
                known_values.add(str(sv).strip().lower())
        top_vals = prof.stats.get("top_values") if isinstance(prof.stats, dict) else None
        if isinstance(top_vals, dict):
            for k in top_vals.keys():
                known_values.add(str(k).strip().lower())

    unsupported: list[str] = []
    for match in _CANDIDATE_METRIC_NOUNS.finditer(question):
        candidate = next((g for g in match.groups() if g is not None), None)
        if not candidate:
            continue
        token = candidate.strip().lower()
        if (
            token in _ANALYTICAL_VOCABULARY
            or token in existing_lower
            or token in matched_tokens
            or token in known_values
        ):
            continue
        # Check singular form against existing column tokens
        singular = token[:-1] if token.endswith("s") and len(token) > 3 else token
        if singular in matched_tokens or singular in existing_lower or singular in _ANALYTICAL_VOCABULARY:
            continue
        # Fuzzy check against existing columns just in case it already matched softly
        if any(SequenceMatcher(None, token, col).ratio() >= 0.82 for col in existing_lower):
            continue
        if token not in unsupported:
            unsupported.append(candidate)

    return unsupported


def _collect_quality_caveats(
    profile: DatasetProfileResult,
    target_columns: list[str],
) -> list[str]:
    """Extract schema & data quality caveats relevant to the target columns."""
    caveats: list[str] = []
    if profile.row_count == 0:
        caveats.append("Dataset currently has 0 data rows.")
        return caveats

    for col in target_columns:
        col_prof = profile.column_profiles.get(col)
        if col_prof and col_prof.missing_count > 0:
            caveats.append(
                f"Column '{col}' has {col_prof.missing_count} missing value(s) ({col_prof.missing_pct}%); "
                "exclude or impute nulls before computing metrics."
            )
        out_info = profile.outlier_summary.get(col)
        if out_info and out_info.iqr_outlier_count > 0:
            caveats.append(
                f"Column '{col}' contains {out_info.iqr_outlier_count} IQR outlier(s) "
                f"({out_info.iqr_outlier_pct}%); verify whether outliers affect mean/sum aggregations."
            )

    if profile.duplicate_summary.duplicate_row_count > 0:
        caveats.append(
            f"Dataset contains {profile.duplicate_summary.duplicate_row_count} duplicate row(s) "
            f"({profile.duplicate_summary.duplicate_row_pct}%); verify deduplication preference."
        )

    for col in target_columns:
        if col in profile.quality_report.constant_columns:
            caveats.append(f"Column '{col}' has zero variance (constant value).")
        if col in profile.quality_report.all_null_columns:
            caveats.append(f"Column '{col}' is 100% null and cannot yield numerical aggregations.")

    return caveats


def _choose_chart_type(intent: str, has_group: bool, has_date: bool) -> str:
    if intent == AnalyticalIntent.TREND_ANALYSIS.value or (has_date and intent == AnalyticalIntent.AGGREGATION.value):
        return "line_chart"
    if intent == AnalyticalIntent.RANKING.value:
        return "horizontal_bar_chart"
    if intent == AnalyticalIntent.COMPARISON.value:
        return "grouped_bar_chart"
    if intent == AnalyticalIntent.CORRELATION.value:
        return "scatter_plot_or_heatmap"
    if intent == AnalyticalIntent.DISTRIBUTION_ANALYSIS.value:
        return "histogram_and_box_plot"
    if intent == AnalyticalIntent.ANOMALY_ANALYSIS.value:
        return "anomaly_scatter_and_box_plot"
    if intent == AnalyticalIntent.MISSING_VALUE_ANALYSIS.value:
        return "missingness_bar_chart"
    if intent == AnalyticalIntent.AGGREGATION.value and has_group:
        return "bar_chart"
    return "summary_table"


def _build_execution_steps(
    intent: str,
    metric_col: str | None,
    metric_cols: list[str],
    group_col: str | None,
    date_col: str | None,
    granularity: str | None,
    aggregation: str | None,
    sort_order: str | None,
    limit: int | None,
    filters: list[dict[str, Any]],
    profile: DatasetProfileResult,
) -> list[AnalyticalStep]:
    """Construct an ordered, deterministic execution plan referencing only verified columns."""
    steps: list[AnalyticalStep] = []
    step_idx = 1

    relevant = [c for c in ([metric_col] + metric_cols + [group_col, date_col]) if c]
    dedup_cols = list(dict.fromkeys(relevant))

    steps.append(
        AnalyticalStep(
            step_number=step_idx,
            operation="schema_and_null_check",
            description="Verify column types and handle missing values prior to analytical computation.",
            columns=dedup_cols,
            parameters={"row_count": profile.row_count},
        )
    )
    step_idx += 1

    if filters:
        filter_cols = list(dict.fromkeys(f["column"] for f in filters))
        steps.append(
            AnalyticalStep(
                step_number=step_idx,
                operation="apply_filters",
                description="Filter dataset rows using extracted predicate conditions.",
                columns=filter_cols,
                parameters={"filters": filters},
            )
        )
        step_idx += 1

    if intent == AnalyticalIntent.DATASET_SUMMARY.value:
        steps.append(
            AnalyticalStep(
                step_number=step_idx,
                operation="compute_dataset_overview",
                description="Summarize row/column counts, column types, descriptive statistics, and quality score.",
                columns=list(profile.column_profiles.keys()),
            )
        )
    elif intent == AnalyticalIntent.MISSING_VALUE_ANALYSIS.value:
        steps.append(
            AnalyticalStep(
                step_number=step_idx,
                operation="audit_missing_values",
                description="Compute per-column missing counts, percentages, and row-level completeness.",
                columns=list(profile.missing_summary.columns_with_missing.keys()) or list(profile.column_profiles.keys()),
            )
        )
    elif intent == AnalyticalIntent.CORRELATION.value:
        corr_cols = metric_cols if len(metric_cols) >= 2 else profile.numerical_columns[:4]
        steps.append(
            AnalyticalStep(
                step_number=step_idx,
                operation="compute_correlation_matrix",
                description=f"Compute Pearson and Spearman correlation coefficients across {corr_cols}.",
                columns=corr_cols,
                parameters={"methods": ["pearson", "spearman"]},
            )
        )
    elif intent == AnalyticalIntent.TREND_ANALYSIS.value:
        gran = granularity or "month"
        agg = aggregation or "sum"
        cols = [c for c in [date_col, metric_col] if c]
        steps.append(
            AnalyticalStep(
                step_number=step_idx,
                operation="time_series_aggregation",
                description=f"Resample/group '{metric_col}' by '{date_col}' at '{gran}' granularity using '{agg}'.",
                columns=cols,
                parameters={"date_column": date_col, "granularity": gran, "aggregation": agg},
            )
        )
    elif intent in {AnalyticalIntent.RANKING.value, AnalyticalIntent.COMPARISON.value, AnalyticalIntent.AGGREGATION.value}:
        agg = aggregation or "sum"
        cols = [c for c in [group_col, metric_col] if c]
        params: dict[str, Any] = {"aggregation": agg}
        if sort_order:
            params["sort_order"] = sort_order
        if limit:
            params["limit"] = limit
        desc = f"Aggregate '{metric_col or 'records'}' using '{agg}'"
        if group_col:
            desc += f" grouped by '{group_col}'"
        if sort_order:
            desc += f", sorted {sort_order}"
        if limit:
            desc += f" (limit {limit})"
        steps.append(
            AnalyticalStep(
                step_number=step_idx,
                operation="group_aggregate_and_sort",
                description=desc + ".",
                columns=cols,
                parameters=params,
            )
        )
    elif intent == AnalyticalIntent.DISTRIBUTION_ANALYSIS.value:
        cols = [metric_col] if metric_col else profile.numerical_columns[:1]
        steps.append(
            AnalyticalStep(
                step_number=step_idx,
                operation="compute_distribution_stats",
                description=f"Compute quantiles, skewness, histogram bins, and summary spread for {cols}.",
                columns=cols,
            )
        )
    elif intent == AnalyticalIntent.ANOMALY_ANALYSIS.value:
        cols = metric_cols or ([metric_col] if metric_col else profile.numerical_columns)
        steps.append(
            AnalyticalStep(
                step_number=step_idx,
                operation="run_anomaly_detection",
                description=f"Apply IQR/Z-score diagnostics and Isolation Forest anomaly scoring on {cols}.",
                columns=cols,
            )
        )
    elif intent == AnalyticalIntent.FILTERING.value:
        steps.append(
            AnalyticalStep(
                step_number=step_idx,
                operation="return_filtered_subset",
                description="Return matching records count, summary metrics, and preview rows.",
                columns=dedup_cols,
            )
        )

    step_idx += 1
    steps.append(
        AnalyticalStep(
            step_number=step_idx,
            operation="self_validate_results",
            description="Cross-check computed totals, row conservation, sample sizes, and null impact.",
            columns=dedup_cols,
        )
    )
    return steps


def _resolve_plan_fields(
    task: AnalyticalTask,
    profile: DatasetProfileResult,
    unsupported_refs: list[str],
) -> tuple[
    dict[str, Any],
    list[str],
    list[str],
    list[str],
]:
    """Identify missing analytical context and ground missing slots safely to existing schema columns."""
    existing_cols = set(profile.column_profiles.keys())
    missing_context: list[str] = []
    improvement_reasons: list[str] = []
    warnings: list[str] = []

    if unsupported_refs:
        warnings.append(
            f"Unsupported or unavailable column/metric reference(s) detected and excluded: {unsupported_refs}."
        )
        improvement_reasons.append(
            f"Flagged unavailable reference(s) ({', '.join(unsupported_refs)}) instead of inventing missing columns."
        )

    intent = task.intent
    metric_col = task.metric_column if task.metric_column in existing_cols else None
    metric_cols = [c for c in task.metric_columns if c in existing_cols]
    group_col = task.group_column if task.group_column in existing_cols else None
    date_col = task.date_column if task.date_column in existing_cols else None
    granularity = task.time_granularity
    aggregation = task.aggregation
    sort_order = task.sort_order
    limit = task.limit
    filters = [f.to_dict() for f in task.filters if f.column in existing_cols]

    # Bind missing slots safely only from existing profile columns
    if intent in {
        AnalyticalIntent.RANKING.value,
        AnalyticalIntent.COMPARISON.value,
        AnalyticalIntent.TREND_ANALYSIS.value,
        AnalyticalIntent.AGGREGATION.value,
        AnalyticalIntent.DISTRIBUTION_ANALYSIS.value,
        AnalyticalIntent.ANOMALY_ANALYSIS.value,
    } and metric_col is None:
        missing_context.append("Target numerical metric column was not specified.")
        if profile.numerical_columns:
            metric_col = profile.numerical_columns[0]
            metric_cols = [metric_col]
            improvement_reasons.append(
                f"Grounded missing metric to available numerical column '{metric_col}'."
            )
        else:
            warnings.append("Dataset contains no numerical columns for quantitative metric calculation.")

    if intent == AnalyticalIntent.CORRELATION.value and len(metric_cols) < 2:
        missing_context.append("At least two numerical columns are required for correlation analysis.")
        if len(profile.numerical_columns) >= 2:
            for cand in profile.numerical_columns:
                if cand not in metric_cols:
                    metric_cols.append(cand)
                if len(metric_cols) >= 2:
                    break
            metric_col = metric_cols[0]
            improvement_reasons.append(
                f"Expanded correlation scope to available numerical columns: {metric_cols[:2]}."
            )
        else:
            warnings.append("Dataset has fewer than 2 numerical columns; correlation cannot be computed.")

    if intent in {AnalyticalIntent.RANKING.value, AnalyticalIntent.COMPARISON.value} and group_col is None:
        missing_context.append("Grouping categorical dimension was not specified.")
        if profile.categorical_columns:
            group_col = profile.categorical_columns[0]
            improvement_reasons.append(
                f"Added grouping dimension '{group_col}' from available categorical columns."
            )

    if intent == AnalyticalIntent.TREND_ANALYSIS.value:
        if date_col is None:
            missing_context.append("Datetime column was not specified.")
            if profile.datetime_columns:
                date_col = profile.datetime_columns[0]
                improvement_reasons.append(f"Bound time dimension to '{date_col}'.")
            else:
                warnings.append("No datetime column exists in the dataset for time-series trend analysis.")
        if granularity is None and date_col is not None:
            missing_context.append("Time aggregation granularity (day/week/month/quarter/year) was omitted.")
            granularity = "month"
            improvement_reasons.append("Specified default 'month' granularity for clear time-series aggregation.")

    if intent in {
        AnalyticalIntent.RANKING.value,
        AnalyticalIntent.COMPARISON.value,
        AnalyticalIntent.TREND_ANALYSIS.value,
        AnalyticalIntent.AGGREGATION.value,
    } and aggregation is None:
        missing_context.append("Aggregation operation (sum/mean/count/median) was omitted.")
        aggregation = "mean" if intent == AnalyticalIntent.COMPARISON.value else "sum"
        improvement_reasons.append(f"Added explicit '{aggregation}' aggregation to remove calculation ambiguity.")

    if intent == AnalyticalIntent.RANKING.value:
        if sort_order is None:
            missing_context.append("Ranking sort direction (ascending/descending) was omitted.")
            sort_order = "descending"
            improvement_reasons.append("Added explicit descending sort order for ranking.")
        if limit is None:
            missing_context.append("Result limit (e.g., Top N) was omitted.")
            limit = 5
            improvement_reasons.append("Added Top-5 limit to focus ranking output on leading items.")

    if intent == AnalyticalIntent.UNKNOWN.value:
        missing_context.append("Analytical intent and target dataset columns could not be identified.")
        warnings.append("Original prompt did not map to a supported analytical intent or existing column.")
        if profile.numerical_columns and profile.categorical_columns:
            metric_col = profile.numerical_columns[0]
            metric_cols = [metric_col]
            group_col = profile.categorical_columns[0]
            aggregation = "sum"
            improvement_reasons.append(
                f"Converted unrecognizable prompt into a schema-grounded starter analysis of '{metric_col}' by '{group_col}'."
            )
        else:
            improvement_reasons.append(
                "Reframed prompt into a structured dataset summary and quality overview using verified columns."
            )

    if not improvement_reasons:
        improvement_reasons.append(
            "Formalized the prompt with explicit dataset schema bounds, null-handling guidance, and validation checks."
        )

    resolved = {
        "metric_col": metric_col,
        "metric_cols": metric_cols,
        "group_col": group_col,
        "date_col": date_col,
        "granularity": granularity,
        "aggregation": aggregation,
        "sort_order": sort_order,
        "limit": limit,
        "filters": filters,
    }
    return resolved, missing_context, improvement_reasons, warnings


def _build_local_improved_prompt(
    original_prompt: str,
    intent: str,
    resolved: dict[str, Any],
    profile: DatasetProfileResult,
    quality_caveats: list[str],
    unsupported_refs: list[str],
) -> str:
    """Construct a precise, schema-grounded analytical prompt deterministically offline."""
    metric_col = resolved["metric_col"]
    metric_cols = resolved["metric_cols"]
    group_col = resolved["group_col"]
    date_col = resolved["date_col"]
    granularity = resolved["granularity"]
    aggregation = resolved["aggregation"]
    sort_order = resolved["sort_order"]
    limit = resolved["limit"]
    filters = resolved["filters"]

    filter_clause = ""
    if filters:
        parts = [f"{f['column']} {f['operator']} {repr(f['value'])}" for f in filters]
        filter_clause = f" after filtering records where {' AND '.join(parts)}"

    if intent == AnalyticalIntent.DATASET_SUMMARY.value:
        core = (
            f"Generate a comprehensive analytical overview of dataset '{profile.dataset_name}' "
            f"({profile.row_count} rows, {profile.column_count} columns), summarizing numerical columns "
            f"({', '.join(profile.numerical_columns) or 'none'}), categorical dimensions "
            f"({', '.join(profile.categorical_columns) or 'none'}), and overall data quality."
        )
    elif intent == AnalyticalIntent.MISSING_VALUE_ANALYSIS.value:
        core = (
            f"Audit missing values across all {profile.column_count} columns of '{profile.dataset_name}', "
            "reporting per-column null counts, missing percentages, and affected row counts."
        )
    elif intent == AnalyticalIntent.CORRELATION.value and len(metric_cols) >= 2:
        core = (
            f"Compute Pearson and Spearman correlation coefficients between {', '.join(metric_cols)} "
            f"in '{profile.dataset_name}'{filter_clause}, reporting correlation strength and sample size."
        )
    elif intent == AnalyticalIntent.TREND_ANALYSIS.value and date_col and metric_col:
        gran_adj = {
            "day": "daily",
            "week": "weekly",
            "month": "monthly",
            "quarter": "quarterly",
            "year": "yearly",
        }.get(granularity or "month", "monthly")
        core = (
            f"Analyze the {gran_adj} trend of {aggregation or 'sum'}({metric_col}) "
            f"ordered chronologically by '{date_col}' in '{profile.dataset_name}'{filter_clause}, "
            "highlighting period-over-period changes."
        )
    elif intent == AnalyticalIntent.RANKING.value and metric_col:
        order_label = "highest" if sort_order == "descending" else "lowest"
        group_part = f" grouped by '{group_col}'" if group_col else ""
        core = (
            f"Rank the top {limit or 5} {order_label} values of {aggregation or 'sum'}({metric_col})"
            f"{group_part} in '{profile.dataset_name}'{filter_clause}, sorted {sort_order or 'descending'}."
        )
    elif intent == AnalyticalIntent.COMPARISON.value and metric_col:
        group_part = f" across '{group_col}'" if group_col else ""
        core = (
            f"Compare {aggregation or 'mean'}({metric_col}){group_part} in '{profile.dataset_name}'"
            f"{filter_clause}, reporting group counts, dispersion, and relative differences."
        )
    elif intent == AnalyticalIntent.AGGREGATION.value and metric_col:
        group_part = f" grouped by '{group_col}'" if group_col else ""
        core = (
            f"Compute {aggregation or 'sum'}({metric_col}){group_part} in '{profile.dataset_name}'"
            f"{filter_clause}, including record counts per group."
        )
    elif intent == AnalyticalIntent.DISTRIBUTION_ANALYSIS.value and metric_col:
        core = (
            f"Analyze the statistical distribution of '{metric_col}' in '{profile.dataset_name}'"
            f"{filter_clause}, reporting mean, median, standard deviation, quartiles (Q1/Q3), IQR, and skewness."
        )
    elif intent == AnalyticalIntent.ANOMALY_ANALYSIS.value and metric_col:
        core = (
            f"Detect statistical outliers (IQR and Z-score) and ML-based anomalies in '{metric_col}' "
            f"within '{profile.dataset_name}'{filter_clause}, listing anomalous records and severity scores."
        )
    elif intent == AnalyticalIntent.FILTERING.value and filters:
        cols_preview = ", ".join(dict.fromkeys([*(metric_cols or []), *(group_col and [group_col] or [])]))
        col_note = f" focusing on {cols_preview}" if cols_preview else ""
        core = (
            f"Filter '{profile.dataset_name}'{filter_clause}{col_note}, reporting matching row count, "
            "percentage of total records, and summary metrics."
        )
    else:
        if metric_col and group_col:
            core = (
                f"Perform an exploratory analysis of '{profile.dataset_name}' by computing "
                f"{aggregation or 'sum'}({metric_col}) grouped by '{group_col}' alongside a data quality summary."
            )
        else:
            core = (
                f"Summarize '{profile.dataset_name}' across its verified columns "
                f"({', '.join(list(profile.column_profiles.keys())[:6])}) and report key statistical findings."
            )

    notes: list[str] = []
    if quality_caveats:
        notes.append(f"Data quality guardrails: {' '.join(quality_caveats[:2])}")
    if unsupported_refs:
        notes.append(
            f"Note: Requested term(s) {unsupported_refs} are not present in the dataset schema and were excluded."
        )
    notes.append("Validate computed results for row-count consistency and sample sufficiency.")

    return f"{core} {' '.join(notes)}".strip()


def _try_online_llm_rewrite(
    local_prompt: str,
    original_prompt: str,
    profile: DatasetProfileResult,
    settings: Settings,
) -> str | None:
    """Optional server-side online LLM rewrite, strictly verified against existing schema columns."""
    api_key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY", "")
    if not api_key or api_key == "MY_GEMINI_API_KEY":
        return None

    try:
        from google import genai  # type: ignore
    except ImportError:
        logger.warning("google-genai package unavailable; using local prompt optimizer.")
        return None

    existing_columns = list(profile.column_profiles.keys())
    system_instruction = (
        "Rewrite the analytical prompt for maximum clarity and precision. "
        f"You MUST ONLY reference columns from this exact list: {existing_columns}. "
        "Do not invent column names or numerical results. Return only the rewritten prompt text."
    )
    user_payload = (
        f"Original prompt: {original_prompt}\n"
        f"Baseline structured prompt: {local_prompt}\n"
        f"Schema summary: {profile.schema_context_summary()}"
    )
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=settings.online_ai_model,
            contents=f"{system_instruction}\n\n{user_payload}",
        )
        candidate_text = (getattr(response, "text", None) or "").strip()
        if not candidate_text or len(candidate_text) < 15:
            return None
        return candidate_text
    except Exception as exc:
        logger.warning("Online LLM rewrite failed (%s); falling back to local optimizer.", exc)
        return None


# ---------------------------------------------------------------------------
# Public API & Service Class
# ---------------------------------------------------------------------------


def optimize_prompt(
    user_prompt: str,
    profile: DatasetProfileResult,
    detected_task: AnalyticalTask | None = None,
    use_online_llm: bool = False,
    settings: Settings | None = None,
) -> PromptOptimizationResult:
    """Transform a user prompt into a dataset-aware improved prompt and execution plan."""
    if not isinstance(user_prompt, str) or not user_prompt.strip():
        raise EmptyPromptError("User prompt cannot be empty.")

    raw_prompt = user_prompt.strip()
    task = detected_task or detect_intent(raw_prompt, profile)
    unsupported_refs = _detect_unsupported_references(raw_prompt, task, profile)

    resolved, missing_context, improvement_reasons, warnings = _resolve_plan_fields(
        task=task,
        profile=profile,
        unsupported_refs=unsupported_refs,
    )

    # Strictly enforce referencing only existing columns
    existing_cols = set(profile.column_profiles.keys())
    referenced_cols: list[str] = []
    for col in [
        resolved["metric_col"],
        *resolved["metric_cols"],
        resolved["group_col"],
        resolved["date_col"],
        *[f["column"] for f in resolved["filters"]],
    ]:
        if col and col in existing_cols and col not in referenced_cols:
            referenced_cols.append(col)

    quality_caveats = _collect_quality_caveats(profile, referenced_cols)
    if quality_caveats:
        improvement_reasons.append("Added data-quality guardrails based on dataset profiling diagnostics.")

    local_improved = _build_local_improved_prompt(
        original_prompt=raw_prompt,
        intent=task.intent,
        resolved=resolved,
        profile=profile,
        quality_caveats=quality_caveats,
        unsupported_refs=unsupported_refs,
    )

    improved_prompt = local_improved
    rewritten_by = "local_rule_engine"

    if use_online_llm:
        cfg = settings or get_settings()
        caps = ModeManager(cfg).resolve(strict_online=False)
        if caps.use_external_ai:
            online_text = _try_online_llm_rewrite(
                local_prompt=local_improved,
                original_prompt=raw_prompt,
                profile=profile,
                settings=cfg,
            )
            if online_text:
                online_unsupported = _detect_unsupported_references(online_text, task, profile)
                if not online_unsupported:
                    improved_prompt = online_text
                    rewritten_by = "online_llm"
                else:
                    logger.warning(
                        "Online LLM rewrite referenced unavailable terms %s; keeping verified local prompt.",
                        online_unsupported,
                    )

    can_execute = profile.row_count > 0 and (
        task.intent in {AnalyticalIntent.DATASET_SUMMARY.value, AnalyticalIntent.MISSING_VALUE_ANALYSIS.value}
        or (task.intent == AnalyticalIntent.CORRELATION.value and len(resolved["metric_cols"]) >= 2)
        or (task.intent == AnalyticalIntent.TREND_ANALYSIS.value and resolved["date_col"] is not None and resolved["metric_col"] is not None)
        or (task.intent == AnalyticalIntent.FILTERING.value and len(resolved["filters"]) > 0)
        or (
            task.intent
            in {
                AnalyticalIntent.RANKING.value,
                AnalyticalIntent.COMPARISON.value,
                AnalyticalIntent.AGGREGATION.value,
                AnalyticalIntent.DISTRIBUTION_ANALYSIS.value,
                AnalyticalIntent.ANOMALY_ANALYSIS.value,
            }
            and resolved["metric_col"] is not None
        )
        or (task.intent == AnalyticalIntent.UNKNOWN.value and len(existing_cols) > 0)
    )

    steps = _build_execution_steps(
        intent=task.intent,
        metric_col=resolved["metric_col"],
        metric_cols=resolved["metric_cols"],
        group_col=resolved["group_col"],
        date_col=resolved["date_col"],
        granularity=resolved["granularity"],
        aggregation=resolved["aggregation"],
        sort_order=resolved["sort_order"],
        limit=resolved["limit"],
        filters=resolved["filters"],
        profile=profile,
    )

    plan = AnalyticalTaskPlan(
        intent=task.intent,
        can_execute=can_execute,
        target_columns=referenced_cols,
        metric_column=resolved["metric_col"],
        metric_columns=resolved["metric_cols"],
        group_column=resolved["group_col"],
        date_column=resolved["date_col"],
        time_granularity=resolved["granularity"],
        aggregation=resolved["aggregation"],
        sort_order=resolved["sort_order"],
        limit=resolved["limit"],
        filters=resolved["filters"],
        recommended_chart_type=_choose_chart_type(
            intent=task.intent,
            has_group=resolved["group_col"] is not None,
            has_date=resolved["date_col"] is not None,
        ),
        execution_steps=steps,
        quality_caveats=quality_caveats,
    )

    logger.info(
        "Optimized prompt (intent=%s, mode=%s, columns=%s, unsupported=%s).",
        task.intent,
        rewritten_by,
        referenced_cols,
        unsupported_refs,
    )

    return PromptOptimizationResult(
        original_prompt=raw_prompt,
        improved_prompt=improved_prompt,
        detected_intent=task.intent,
        intent_confidence=task.confidence,
        missing_context=missing_context,
        improvement_reasons=improvement_reasons,
        unsupported_references=unsupported_refs,
        warnings=warnings,
        referenced_columns=referenced_cols,
        rewritten_by=rewritten_by,
        task_plan=plan,
    )


class PromptOptimizer:
    """Stateful service wrapper for Module 4 dataset-aware prompt optimization."""

    def __init__(self, use_online_llm: bool = False, settings: Settings | None = None) -> None:
        self.use_online_llm = use_online_llm
        self.settings = settings or get_settings()

    def optimize(
        self,
        user_prompt: str,
        profile: DatasetProfileResult,
        detected_task: AnalyticalTask | None = None,
    ) -> PromptOptimizationResult:
        return optimize_prompt(
            user_prompt=user_prompt,
            profile=profile,
            detected_task=detected_task,
            use_online_llm=self.use_online_llm,
            settings=self.settings,
        )
