"""Module 3: Natural Language Analytical Intent Detection for SmartPrompt AI.

Parses natural-language analytical questions against a Module 1 DatasetProfileResult
using rule-based & NLP lexical/fuzzy matching to extract intent, relevant columns,
metrics, grouping dimensions, aggregations, date granularity, filters, ambiguity flags,
and safe fallbacks for unknown queries.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from enum import Enum
import re
from typing import Any

from config.logging_config import get_logger
from modules.dataset_profiler import DatasetProfileResult

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Custom Exceptions & Enums
# ---------------------------------------------------------------------------


class IntentDetectionError(Exception):
    """Base exception for natural-language intent detection errors."""


class InvalidQuestionError(IntentDetectionError):
    """Raised when the input question is missing or not a string."""


class AnalyticalIntent(str, Enum):
    """All supported analytical intents plus explicit unknown fallback."""

    DATASET_SUMMARY = "dataset_summary"
    RANKING = "ranking"
    COMPARISON = "comparison"
    TREND_ANALYSIS = "trend_analysis"
    AGGREGATION = "aggregation"
    CORRELATION = "correlation"
    DISTRIBUTION_ANALYSIS = "distribution_analysis"
    ANOMALY_ANALYSIS = "anomaly_analysis"
    FILTERING = "filtering"
    MISSING_VALUE_ANALYSIS = "missing_value_analysis"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Structured Output Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FilterCondition:
    """Structured filter predicate extracted from natural language."""

    column: str
    operator: str  # "==" | ">" | ">=" | "<" | "<=" | "contains"
    value: Any

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnalyticalTask:
    """Structured analytical task returned by the intent detector."""

    raw_question: str
    normalized_question: str
    intent: str
    confidence: float
    relevant_columns: list[str] = field(default_factory=list)
    metric_column: str | None = None
    metric_columns: list[str] = field(default_factory=list)
    group_column: str | None = None
    group_columns: list[str] = field(default_factory=list)
    date_column: str | None = None
    time_granularity: str | None = None  # "day" | "week" | "month" | "quarter" | "year"
    aggregation: str | None = None  # "sum" | "mean" | "count" | "median" | "min" | "max" | "std"
    sort_order: str | None = None  # "ascending" | "descending"
    limit: int | None = None
    filters: list[FilterCondition] = field(default_factory=list)
    is_ambiguous: bool = False
    ambiguity_reasons: list[str] = field(default_factory=list)
    clarification_questions: list[str] = field(default_factory=list)
    is_fallback: bool = False
    fallback_reason: str | None = None
    suggested_questions: list[str] = field(default_factory=list)
    candidate_intents: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["filters"] = [f.to_dict() for f in self.filters]
        return data


# ---------------------------------------------------------------------------
# Lexical & Rule Tables
# ---------------------------------------------------------------------------

_INTENT_RULES: dict[AnalyticalIntent, list[tuple[str, float]]] = {
    AnalyticalIntent.MISSING_VALUE_ANALYSIS: [
        (r"\bmissing(\s+values?)?\b", 4.2),
        (r"\bnull(\s+values?)?\b", 4.2),
        (r"\bnan\b", 3.8),
        (r"\bempty(\s+cells?|\s+values?)?\b", 3.5),
        (r"\bincomplete\b", 3.0),
        (r"\bblank(\s+values?)?\b", 3.0),
        (r"\bdata\s+completeness\b", 3.6),
    ],
    AnalyticalIntent.ANOMALY_ANALYSIS: [
        (r"\banomal(y|ies)\b", 4.0),
        (r"\boutliers?\b", 3.8),
        (r"\bunusual\b", 3.0),
        (r"\babnormal\b", 3.2),
        (r"\bspikes?\b", 2.6),
        (r"\bsuspicious\b", 2.8),
        (r"\birregular(ities)?\b", 2.8),
        (r"\bextreme\s+values?\b", 2.8),
    ],
    AnalyticalIntent.CORRELATION: [
        (r"\bcorrelat(e|ion|ions|ed)?\b", 4.0),
        (r"\brelationship\s+between\b", 3.6),
        (r"\bassociation\s+between\b", 3.4),
        (r"\bimpact\s+of\b.*\bon\b", 2.8),
        (r"\bhow\s+does\b.*\baffect\b", 2.8),
        (r"\bdepend(s|ence)?\s+on\b", 2.4),
        (r"\bscatter\b", 2.5),
    ],
    AnalyticalIntent.TREND_ANALYSIS: [
        (r"\btrend(s)?\b", 3.6),
        (r"\bover\s+time\b", 3.8),
        (r"\btime\s+series\b", 3.8),
        (r"\b(daily|weekly|monthly|quarterly|yearly|annual(ly)?)\b", 3.0),
        (r"\bgrowth\b", 2.8),
        (r"\btrajectory\b", 2.8),
        (r"\bchange\s+over\b", 2.8),
        (r"\bby\s+(day|week|month|quarter|year)\b", 3.2),
    ],
    AnalyticalIntent.RANKING: [
        (r"\btop\s+\d+\b", 3.8),
        (r"\bbottom\s+\d+\b", 3.8),
        (r"\btop\b", 2.4),
        (r"\bbottom\b", 2.4),
        (r"\bhighest\b", 3.2),
        (r"\blowest\b", 3.2),
        (r"\blargest\b", 3.0),
        (r"\bsmallest\b", 3.0),
        (r"\bbest\b", 2.6),
        (r"\bworst\b", 2.6),
        (r"\bmost\b", 2.3),
        (r"\bleast\b", 2.3),
        (r"\brank(ing|ed)?\b", 3.5),
        (r"\bleading\b", 2.6),
    ],
    AnalyticalIntent.COMPARISON: [
        (r"\bcompar(e|ing|ison)\b", 3.6),
        (r"\bversus\b|\bvs\.?\b", 3.8),
        (r"\bdifference\s+between\b", 3.5),
        (r"\bcontrast\b", 3.0),
        (r"\bacross\s+(different\s+)?\w+\b", 2.2),
        (r"\bbetween\s+\w+\s+and\s+\w+\b", 2.4),
    ],
    AnalyticalIntent.DISTRIBUTION_ANALYSIS: [
        (r"\bdistribution\b", 3.8),
        (r"\bhistogram\b", 3.6),
        (r"\bspread\b", 3.0),
        (r"\bskew(ness)?\b", 3.2),
        (r"\bquartiles?\b|\bpercentiles?\b", 3.2),
        (r"\bfrequency\b", 2.6),
        (r"\brange\s+of\b", 2.4),
        (r"\bvariance\b|\bstandard\s+deviation\b", 2.6),
    ],
    AnalyticalIntent.FILTERING: [
        (r"\bwhere\b", 3.0),
        (r"\bfilter(ed|ing)?\b", 3.6),
        (r"\bonly\s+(show|include|rows|records)\b", 3.0),
        (r"\b(greater|more|higher)\s+than\b", 3.2),
        (r"\b(less|fewer|lower|smaller)\s+than\b", 3.2),
        (r"\babove\s+\d+\b|\bbelow\s+\d+\b", 3.2),
        (r"\b(equal\s+to|equals)\b", 3.0),
        (r"[><=]=?\s*-?\d+(?:\.\d+)?", 3.2),
        (r"\brows\s+(with|having)\b", 2.8),
    ],
    AnalyticalIntent.AGGREGATION: [
        (r"\btotal\b", 2.8),
        (r"\bsum\s+of\b|\bsum\b", 3.0),
        (r"\baverage\b|\bmean\b|\bavg\b", 3.0),
        (r"\bmedian\b", 3.0),
        (r"\bhow\s+many\b|\bcount(\s+of)?\b|\bnumber\s+of\b", 2.8),
        (r"\bby\s+[a-z0-9_ ]+\b", 1.8),
        (r"\bper\s+[a-z0-9_ ]+\b", 1.8),
        (r"\bgrouped\s+by\b", 3.0),
    ],
    AnalyticalIntent.DATASET_SUMMARY: [
        (r"\bsummar(y|ize|ise)\b", 3.6),
        (r"\boverview\b", 3.6),
        (r"\bdescribe(\s+the)?(\s+dataset|\s+data)?\b", 3.4),
        (r"\bdataset\s+(profile|info|details|structure)\b", 3.8),
        (r"\bwhat\s+(columns|fields|features)\b", 3.2),
        (r"\bdata\s+quality\b", 3.2),
        (r"\bhow\s+big\s+is\s+(the\s+)?dataset\b", 3.4),
    ],
}

_AGGREGATION_PATTERNS: list[tuple[str, str]] = [
    (r"\b(average|avg|mean)\b", "mean"),
    (r"\b(median)\b", "median"),
    (r"\b(how\s+many|count|number\s+of|frequency)\b", "count"),
    (r"\b(standard\s+deviation|std\s+dev|volatility)\b", "std"),
    (r"\b(total|sum|generated|overall|combined)\b", "sum"),
    (r"\b(minimum|min|smallest|lowest|least)\b", "min"),
    (r"\b(maximum|max|largest|highest|most|peak)\b", "max"),
]

_GRANULARITY_PATTERNS: list[tuple[str, str]] = [
    (r"\b(daily|per\s+day|by\s+day)\b", "day"),
    (r"\b(weekly|per\s+week|by\s+week)\b", "week"),
    (r"\b(monthly|per\s+month|by\s+month)\b", "month"),
    (r"\b(quarterly|per\s+quarter|by\s+quarter)\b", "quarter"),
    (r"\b(yearly|annual|annually|per\s+year|by\s+year)\b", "year"),
]

_DESCENDING_TOKENS = re.compile(r"\b(top|highest|largest|biggest|best|most|maximum|max|greatest|leading)\b")
_ASCENDING_TOKENS = re.compile(r"\b(bottom|lowest|smallest|worst|least|minimum|min|fewest)\b")

_VAGUE_QUESTIONS = {
    "analyze",
    "analyze data",
    "analyze dataset",
    "analyze this",
    "show data",
    "check this",
    "help",
    "insights",
    "tell me something",
    "what does this say",
}

_MATCH_STOPWORDS = frozenset(
    {
        "is",
        "are",
        "was",
        "were",
        "be",
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "to",
        "for",
        "by",
        "with",
        "and",
        "or",
        "at",
        "from",
        "as",
        "has",
        "have",
        "had",
        "it",
        "this",
        "that",
    }
)


# ---------------------------------------------------------------------------
# Column & Entity Matching Utilities
# ---------------------------------------------------------------------------


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9_.\s><=]", " ", text.lower())).strip()


def _canonical_col_name(col: str) -> str:
    return re.sub(r"[_\-\s]+", " ", col.lower()).strip()


def _singularize(word: str) -> str:
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _match_columns_from_question(
    raw_question: str,
    profile: DatasetProfileResult,
) -> list[tuple[str, float, int]]:
    """Return matched columns as (column_name, match_score, char_position) sorted by score & position."""
    q_lower = raw_question.lower()
    q_words = [w for w in re.findall(r"[a-z0-9_]+", q_lower) if len(w) >= 2]
    q_singular_words = {_singularize(w) for w in q_words}

    matches: list[tuple[str, float, int]] = []
    all_columns = list(profile.column_profiles.keys())

    for col in all_columns:
        col_lower = col.lower()
        col_canonical = _canonical_col_name(col)
        col_tokens = [t for t in col_canonical.split() if t]
        col_singular_tokens = {_singularize(t) for t in col_tokens}

        best_score = 0.0
        pos = len(q_lower) + 1

        # 1. Exact substring boundary match
        exact_pat = rf"\b{re.escape(col_lower)}\b"
        m = re.search(exact_pat, q_lower)
        if m:
            best_score = 1.0
            pos = m.start()
        else:
            # 2. Canonical (underscore -> space) match
            canon_pat = rf"\b{re.escape(col_canonical)}\b"
            m_canon = re.search(canon_pat, q_lower)
            if m_canon:
                best_score = 0.96
                pos = m_canon.start()

        # 3. Singular/plural & meaningful token overlap (ignoring stopwords)
        if best_score < 0.9:
            meaningful_col_tokens = {t for t in col_singular_tokens if t not in _MATCH_STOPWORDS}
            meaningful_q_words = {w for w in q_singular_words if w not in _MATCH_STOPWORDS}

            if col_singular_tokens and col_singular_tokens.issubset(q_singular_words):
                best_score = max(best_score, 0.9)
                for tok in col_tokens:
                    idx = q_lower.find(tok)
                    if idx != -1:
                        pos = min(pos, idx)
            elif meaningful_col_tokens and meaningful_col_tokens.issubset(meaningful_q_words):
                best_score = max(best_score, 0.85)
                for tok in col_tokens:
                    if tok not in _MATCH_STOPWORDS:
                        idx = q_lower.find(tok)
                        if idx != -1:
                            pos = min(pos, idx)
            elif len(meaningful_col_tokens) > 1:
                overlap = len(meaningful_col_tokens & meaningful_q_words) / len(meaningful_col_tokens)
                if overlap >= 0.6:
                    best_score = max(best_score, 0.65 + 0.2 * overlap)

        # 4. Fuzzy similarity against single words or adjacent bi-grams
        if best_score < 0.78:
            ngrams = list(q_words) + [f"{q_words[i]} {q_words[i+1]}" for i in range(len(q_words) - 1)]
            for ng in ngrams:
                if len(ng) < 3:
                    continue
                ratio = SequenceMatcher(None, col_canonical, ng).ratio()
                if ratio >= 0.82:
                    best_score = max(best_score, ratio * 0.88)
                    idx = q_lower.find(ng)
                    if idx != -1:
                        pos = min(pos, idx)

        if best_score >= 0.65:
            matches.append((col, round(best_score, 4), pos))

    matches.sort(key=lambda item: (-item[1], item[2]))
    return matches


def _extract_limit(q_lower: str) -> int | None:
    m = re.search(r"\b(?:top|bottom|first|last)\s+(\d+)\b", q_lower)
    if m:
        return int(m.group(1))
    return None


def _extract_sort_order(q_lower: str) -> str | None:
    if _DESCENDING_TOKENS.search(q_lower):
        return "descending"
    if _ASCENDING_TOKENS.search(q_lower):
        return "ascending"
    return None


def _extract_aggregation(q_lower: str, intent: AnalyticalIntent) -> str | None:
    for pattern, agg in _AGGREGATION_PATTERNS:
        if re.search(pattern, q_lower):
            # In ranking queries like "Which product generated the highest revenue?",
            # "highest" denotes ranking order while "revenue"/"generated" implies total ("sum").
            if intent == AnalyticalIntent.RANKING and agg in {"max", "min"}:
                if not re.search(r"\b(maximum|max|minimum|min)\b", q_lower):
                    return "sum"
            return agg

    if intent == AnalyticalIntent.RANKING:
        return "sum"
    if intent == AnalyticalIntent.AGGREGATION:
        return "sum"
    if intent == AnalyticalIntent.TREND_ANALYSIS:
        return "sum"
    if intent == AnalyticalIntent.COMPARISON:
        return "mean"
    return None


def _extract_time_granularity(q_lower: str) -> str | None:
    for pattern, gran in _GRANULARITY_PATTERNS:
        if re.search(pattern, q_lower):
            return gran
    return None


def _extract_filters(
    raw_question: str,
    profile: DatasetProfileResult,
    matched_cols: list[tuple[str, float, int]],
) -> list[FilterCondition]:
    """Extract numeric comparison filters and categorical value-match filters."""
    filters: list[FilterCondition] = []
    q_lower = raw_question.lower()
    matched_names = [m[0] for m in matched_cols]

    # 1. Numeric operator filters: e.g., "revenue > 1000", "amount greater than 500"
    numeric_patterns = [
        (r"([a-z0-9_ ]+?)\s*(>=|<=|>|<|==|=)\s*(-?\d+(?:\.\d+)?)", None),
        (r"([a-z0-9_ ]+?)\s+(?:is\s+)?(?:greater|more|higher|above)\s+(?:than\s+)?(-?\d+(?:\.\d+)?)", ">"),
        (r"([a-z0-9_ ]+?)\s+(?:is\s+)?(?:less|fewer|lower|below|under)\s+(?:than\s+)?(-?\d+(?:\.\d+)?)", "<"),
        (r"([a-z0-9_ ]+?)\s+(?:is\s+)?(?:equal\s+to|equals)\s+(-?\d+(?:\.\d+)?)", "=="),
    ]

    seen_filters: set[tuple[str, str, str]] = set()
    for pat, fixed_op in numeric_patterns:
        for match in re.finditer(pat, q_lower):
            if fixed_op is None:
                lhs, op, val_str = match.group(1), match.group(2), match.group(3)
                if op == "=":
                    op = "=="
            else:
                lhs, val_str = match.group(1), match.group(2)
                op = fixed_op

            # Match LHS text to a numerical column
            lhs_matches = _match_columns_from_question(lhs, profile)
            num_candidates = [c for c, _, _ in lhs_matches if c in profile.numerical_columns]
            target_col = num_candidates[0] if num_candidates else None
            if not target_col and len(matched_names) == 1 and matched_names[0] in profile.numerical_columns:
                target_col = matched_names[0]

            if target_col:
                num_val: int | float = float(val_str) if "." in val_str else int(val_str)
                key = (target_col, op, str(num_val))
                if key not in seen_filters:
                    seen_filters.add(key)
                    filters.append(FilterCondition(column=target_col, operator=op, value=num_val))

    # 2. Categorical sample/top value matching: e.g., "for North region" or "in Electronics"
    for col in profile.categorical_columns:
        col_prof = profile.column_profiles[col]
        known_values: list[str] = []
        top_vals = col_prof.stats.get("top_values") if isinstance(col_prof.stats, dict) else None
        if isinstance(top_vals, dict):
            known_values.extend(str(k) for k in top_vals.keys())
        for sv in col_prof.sample_values:
            if sv is not None and str(sv) not in known_values:
                known_values.append(str(sv))

        for val in known_values:
            val_clean = val.strip()
            if len(val_clean) < 2:
                continue
            if re.search(rf"\b{re.escape(val_clean.lower())}\b", q_lower):
                # Avoid false match if the value is identical to the column name itself
                if val_clean.lower() == col.lower():
                    continue
                key = (col, "==", val_clean)
                if key not in seen_filters:
                    seen_filters.add(key)
                    filters.append(FilterCondition(column=col, operator="==", value=val_clean))

    return filters


def _build_suggested_questions(profile: DatasetProfileResult) -> list[str]:
    """Generate schema-grounded fallback questions tailored to the active dataset."""
    suggestions: list[str] = ["Give me a summary overview of the dataset."]
    if profile.numerical_columns and profile.categorical_columns:
        num = profile.numerical_columns[0]
        cat = profile.categorical_columns[0]
        suggestions.append(f"Which {cat} has the highest total {num}?")
        suggestions.append(f"Compare average {num} across {cat}.")
    if profile.datetime_columns and profile.numerical_columns:
        dt = profile.datetime_columns[0]
        num = profile.numerical_columns[0]
        suggestions.append(f"Show the trend of {num} over {dt}.")
    if len(profile.numerical_columns) >= 2:
        suggestions.append(
            f"What is the correlation between {profile.numerical_columns[0]} and {profile.numerical_columns[1]}?"
        )
    if profile.numerical_columns:
        suggestions.append(f"Detect anomalies in {profile.numerical_columns[0]}.")
    return suggestions[:5]


# ---------------------------------------------------------------------------
# Public Intent Detection Function & Service Class
# ---------------------------------------------------------------------------


def detect_intent(
    question: str,
    profile: DatasetProfileResult,
) -> AnalyticalTask:
    """Detect analytical intent, columns, operations, and ambiguity from a natural-language query."""
    if not isinstance(question, str):
        raise InvalidQuestionError("Question must be a string.")

    raw_question = question.strip()
    if not raw_question:
        raise InvalidQuestionError("Question cannot be empty.")

    normalized_question = _normalize_text(raw_question)
    q_lower = raw_question.lower()

    # 1. Match columns and filter conditions from dataset schema
    matched_cols = _match_columns_from_question(raw_question, profile)
    filters = _extract_filters(raw_question, profile, matched_cols)

    relevant_columns: list[str] = []
    for col, _, _ in matched_cols:
        if col not in relevant_columns:
            relevant_columns.append(col)
    for flt in filters:
        if flt.column not in relevant_columns:
            relevant_columns.append(flt.column)

    # 2. Score all supported intents using rule-based regex signals + schema bonuses
    scores: dict[AnalyticalIntent, float] = {intent: 0.0 for intent in _INTENT_RULES}
    for intent, patterns in _INTENT_RULES.items():
        for pattern, weight in patterns:
            if re.search(pattern, q_lower):
                scores[intent] += weight

    # Schema & structural disambiguation boosts
    matched_num = [c for c in relevant_columns if c in profile.numerical_columns]
    matched_cat = [c for c in relevant_columns if c in (profile.categorical_columns + profile.boolean_columns)]
    matched_dt = [c for c in relevant_columns if c in profile.datetime_columns]

    if len(matched_num) >= 2 and scores[AnalyticalIntent.CORRELATION] > 0:
        scores[AnalyticalIntent.CORRELATION] += 1.0
    if (matched_dt or profile.datetime_columns) and scores[AnalyticalIntent.TREND_ANALYSIS] > 0:
        scores[AnalyticalIntent.TREND_ANALYSIS] += 0.8
    if filters and any(f.operator in {">", ">=", "<", "<="} for f in filters):
        scores[AnalyticalIntent.FILTERING] += 1.5
    elif filters and scores[AnalyticalIntent.FILTERING] > 0:
        scores[AnalyticalIntent.FILTERING] += 1.0

    # If user says "which <category> ... highest/top <metric>", boost ranking over aggregation
    if re.search(r"\bwhich\b", q_lower) and _extract_sort_order(q_lower) and matched_cat:
        scores[AnalyticalIntent.RANKING] += 1.5

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_intent, top_score = ranked[0]
    second_intent, second_score = ranked[1]

    candidate_intents = [
        {"intent": intent.value, "score": round(score, 2)}
        for intent, score in ranked[:3]
        if score > 0
    ]

    # 3. Fallback handling for unknown / non-analytical questions
    if top_score <= 0.0 and not relevant_columns:
        logger.info("No intent or columns matched question '%s'; returning fallback task.", raw_question)
        return AnalyticalTask(
            raw_question=raw_question,
            normalized_question=normalized_question,
            intent=AnalyticalIntent.UNKNOWN.value,
            confidence=0.0,
            relevant_columns=[],
            is_ambiguous=True,
            ambiguity_reasons=["Unable to identify a supported analytical operation or dataset column."],
            clarification_questions=[
                "Which metric or column would you like to analyze, and what operation should be performed?"
            ],
            is_fallback=True,
            fallback_reason="No recognizable analytical intent or dataset column matched the question.",
            suggested_questions=_build_suggested_questions(profile),
            candidate_intents=[],
        )

    # If columns matched (e.g., "Revenue by Region") without explicit verb, infer aggregation/summary
    if top_score <= 0.0 and relevant_columns:
        if matched_num and matched_cat:
            top_intent = AnalyticalIntent.AGGREGATION
            top_score = 2.0
        else:
            top_intent = AnalyticalIntent.DATASET_SUMMARY
            top_score = 1.5

    confidence = round(min(1.0, top_score / 6.0), 2)

    # 4. Extract role assignments (metrics, groups, dates, aggregation, sort order, limit)
    metric_columns = list(matched_num)
    metric_column = metric_columns[0] if metric_columns else None

    group_columns = [c for c in matched_cat if not any(f.column == c and f.operator == "==" for f in filters)]
    # Keep categorical column as group if explicitly referenced alongside a filter or ranking
    if not group_columns and matched_cat:
        group_columns = list(matched_cat)
    group_column = group_columns[0] if group_columns else None

    date_column = matched_dt[0] if matched_dt else (
        profile.datetime_columns[0]
        if top_intent == AnalyticalIntent.TREND_ANALYSIS and profile.datetime_columns
        else None
    )
    time_granularity = _extract_time_granularity(q_lower)
    aggregation = _extract_aggregation(q_lower, top_intent)
    sort_order = _extract_sort_order(q_lower)
    limit = _extract_limit(q_lower)

    # Default metric fallback when ranking/aggregation/trend/distribution omits explicit column?
    # Only auto-assign if there is a single numerical column; otherwise flag ambiguity when needed!
    ambiguity_reasons: list[str] = []
    clarification_questions: list[str] = []

    if normalized_question in _VAGUE_QUESTIONS or (len(normalized_question.split()) <= 2 and top_score < 3.0):
        ambiguity_reasons.append("Question is too brief or underspecified.")
        clarification_questions.append("Please specify the target metric, grouping column, or comparison criteria.")

    if second_score > 0 and (top_score - second_score) < 0.8 and top_score < 4.0:
        ambiguity_reasons.append(
            f"Close intent ambiguity between '{top_intent.value}' and '{second_intent.value}'."
        )
        clarification_questions.append(
            f"Did you want a {top_intent.value.replace('_', ' ')} or a {second_intent.value.replace('_', ' ')}?"
        )

    if top_intent in {
        AnalyticalIntent.RANKING,
        AnalyticalIntent.DISTRIBUTION_ANALYSIS,
        AnalyticalIntent.AGGREGATION,
    } and metric_column is None:
        if len(profile.numerical_columns) == 1:
            metric_column = profile.numerical_columns[0]
            metric_columns = [metric_column]
        elif len(profile.numerical_columns) > 1 and aggregation != "count":
            ambiguity_reasons.append(
                f"No numerical metric specified for '{top_intent.value}' among available columns: {profile.numerical_columns}."
            )
            clarification_questions.append(
                f"Which numerical column should be analyzed ({', '.join(profile.numerical_columns[:4])})?"
            )

    if top_intent == AnalyticalIntent.CORRELATION and len(metric_columns) < 2:
        if len(profile.numerical_columns) < 2:
            ambiguity_reasons.append("Correlation requires at least two numerical columns in the dataset.")
        elif len(metric_columns) == 1:
            ambiguity_reasons.append(
                f"Only one numerical column ('{metric_columns[0]}') was specified for correlation."
            )
            clarification_questions.append(
                f"Which second numerical column should '{metric_columns[0]}' be correlated against?"
            )
        else:
            ambiguity_reasons.append("No numerical columns specified for correlation analysis.")
            clarification_questions.append("Which two numerical columns would you like to correlate?")

    if top_intent == AnalyticalIntent.TREND_ANALYSIS and date_column is None:
        ambiguity_reasons.append("Trend analysis requested, but no datetime column was detected in the dataset.")
        clarification_questions.append("Which column represents the time or sequence dimension?")

    is_ambiguous = len(ambiguity_reasons) > 0

    logger.info(
        "Detected intent='%s' (conf=%.2f, metric=%s, group=%s, agg=%s, ambiguous=%s) for question='%s'.",
        top_intent.value,
        confidence,
        metric_column,
        group_column,
        aggregation,
        is_ambiguous,
        raw_question,
    )

    return AnalyticalTask(
        raw_question=raw_question,
        normalized_question=normalized_question,
        intent=top_intent.value,
        confidence=confidence,
        relevant_columns=relevant_columns,
        metric_column=metric_column,
        metric_columns=metric_columns,
        group_column=group_column,
        group_columns=group_columns,
        date_column=date_column,
        time_granularity=time_granularity,
        aggregation=aggregation,
        sort_order=sort_order,
        limit=limit,
        filters=filters,
        is_ambiguous=is_ambiguous,
        ambiguity_reasons=ambiguity_reasons,
        clarification_questions=clarification_questions,
        is_fallback=False,
        fallback_reason=None,
        suggested_questions=_build_suggested_questions(profile) if is_ambiguous else [],
        candidate_intents=candidate_intents,
    )


class IntentDetector:
    """Stateful service wrapper for Module 3 natural-language intent detection."""

    def detect(self, question: str, profile: DatasetProfileResult) -> AnalyticalTask:
        return detect_intent(question=question, profile=profile)
