"""Module 5: ML-Based Prompt Quality Evaluator for SmartPrompt AI.

Uses Scikit-learn (TF-IDF + engineered analytical prompt features + LogisticRegression)
to classify analytical prompts into Low, Medium, and High quality tiers,
preventing data leakage via strict train/test splitting prior to fitting,
evaluating real held-out metrics, persisting via Joblib, and generating
actionable prompt improvement recommendations in offline mode.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import re
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.logging_config import get_logger
from config.settings import get_settings
from modules.dataset_profiler import DatasetProfileResult

logger = get_logger(__name__)

DEFAULT_MODEL_FILENAME = "prompt_quality_model.joblib"
ENGINEERED_FEATURE_COLUMNS = [
    "specificity_score",
    "context_score",
    "intent_clarity_score",
    "column_relevance_score",
]


# ---------------------------------------------------------------------------
# Enums & Exceptions
# ---------------------------------------------------------------------------


class PromptQualityClass(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class PromptQualityModelError(Exception):
    """Base exception for prompt quality model errors."""


class ModelNotFoundError(PromptQualityModelError):
    """Raised when a saved prompt quality model artifact cannot be loaded."""


class InvalidPromptInputError(PromptQualityModelError):
    """Raised when an empty or non-string prompt is passed for evaluation."""


# ---------------------------------------------------------------------------
# Structured Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabeledPromptSample:
    """Single labeled training example paired with its target dataset schema columns."""

    prompt_text: str
    quality_label: str  # "Low" | "Medium" | "High"
    schema_columns: tuple[str, ...]


@dataclass(frozen=True)
class PromptFeatures:
    """Engineered numerical features quantifying analytical prompt quality."""

    specificity_score: float
    context_score: float
    intent_clarity_score: float
    column_relevance_score: float
    matched_columns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingEvaluationReport:
    """Real Scikit-learn evaluation metrics measured on a held-out test split."""

    accuracy: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    classes: list[str]
    per_class_metrics: dict[str, dict[str, float]]
    confusion_matrix: list[list[int]]
    train_size: int
    test_size: int
    trained_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PromptQualityModelBundle:
    """Serializable container pairing the fitted Scikit-learn Pipeline with evaluation metadata."""

    pipeline: Pipeline
    evaluation_report: TrainingEvaluationReport
    feature_columns: list[str] = field(default_factory=lambda: list(ENGINEERED_FEATURE_COLUMNS))
    model_version: str = "1.0.0"


@dataclass(frozen=True)
class PromptQualityPrediction:
    """Inference result with class probabilities, quality score, and recommendations."""

    prompt: str
    predicted_class: str
    quality_score: float  # 0 to 100 probability-weighted score
    class_probabilities: dict[str, float]
    extracted_features: PromptFeatures
    recommendations: list[str]
    model_accuracy_on_test: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "predicted_class": self.predicted_class,
            "quality_score": self.quality_score,
            "class_probabilities": self.class_probabilities,
            "extracted_features": self.extracted_features.to_dict(),
            "recommendations": self.recommendations,
            "model_accuracy_on_test": self.model_accuracy_on_test,
        }


# ---------------------------------------------------------------------------
# Feature Extraction Helpers (Deterministic & Leakage-Free)
# ---------------------------------------------------------------------------

_AGGREGATION_RE = re.compile(
    r"\b(sum|total|average|avg|mean|median|count|min|minimum|max|maximum|std|standard\s+deviation|variance|percentile|quartile)\b",
    re.IGNORECASE,
)
_OPERATOR_LIMIT_RE = re.compile(
    r"\b(top\s+\d+|bottom\s+\d+|highest|lowest|rank|sort|ascending|descending|greater\s+than|less\s+than|above|below|between)\b|[><=]=?\s*-?\d+",
    re.IGNORECASE,
)
_TIME_SCOPE_RE = re.compile(
    r"\b(daily|weekly|monthly|quarterly|yearly|annual|annually|over\s+time|by\s+(day|week|month|quarter|year)|last\s+\d+\s+(days|weeks|months|quarters|years)|20\d{2})\b",
    re.IGNORECASE,
)
_GROUPING_CONTEXT_RE = re.compile(
    r"\b(by\s+[a-z0-9_]+|per\s+[a-z0-9_]+|grouped\s+by|across\s+[a-z0-9_]+|for\s+each|between\s+[a-z0-9_]+\s+and\s+[a-z0-9_]+|where\s+[a-z0-9_]+)\b",
    re.IGNORECASE,
)
_INTENT_ACTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(rank|top\s+\d+|bottom\s+\d+|highest|lowest)\b", re.IGNORECASE),
    re.compile(r"\b(compare|comparison|versus|vs\.?|difference\s+between)\b", re.IGNORECASE),
    re.compile(r"\b(trend|over\s+time|time\s+series|growth|trajectory)\b", re.IGNORECASE),
    re.compile(r"\b(correlat\w+|relationship\s+between|association)\b", re.IGNORECASE),
    re.compile(r"\b(distribution|histogram|skewness|quartiles?|spread)\b", re.IGNORECASE),
    re.compile(r"\b(anomal\w+|outliers?|abnormal|spikes?)\b", re.IGNORECASE),
    re.compile(r"\b(missing|null|incomplete|empty\s+cells?)\b", re.IGNORECASE),
    re.compile(r"\b(filter|where|greater\s+than|less\s+than)\b", re.IGNORECASE),
    re.compile(r"\b(total|sum|average|mean|median|count|aggregate)\b", re.IGNORECASE),
    re.compile(r"\b(summary|summarize|overview|describe|profile)\b", re.IGNORECASE),
]
_VAGUE_FILLER_RE = re.compile(
    r"\b(something|stuff|things|anything|whatever|good|bad|nice|check\s+this|analyze\s+this)\b",
    re.IGNORECASE,
)


def _match_available_columns(prompt: str, available_columns: list[str] | tuple[str, ...] | None) -> list[str]:
    if not available_columns:
        return []
    q_lower = prompt.lower()
    q_tokens = set(re.findall(r"[a-z0-9_]+", q_lower))
    matched: list[str] = []
    for col in available_columns:
        col_lower = col.lower()
        col_space = re.sub(r"[_\-]+", " ", col_lower).strip()
        if re.search(rf"\b{re.escape(col_lower)}\b", q_lower) or re.search(rf"\b{re.escape(col_space)}\b", q_lower):
            matched.append(col)
            continue
        # Singular/plural match on single-word column names
        if col_lower.endswith("s") and col_lower[:-1] in q_tokens:
            matched.append(col)
        elif f"{col_lower}s" in q_tokens:
            matched.append(col)
    return matched


def extract_prompt_features(
    prompt: str,
    available_columns: list[str] | tuple[str, ...] | None = None,
    profile: DatasetProfileResult | None = None,
) -> PromptFeatures:
    """Extract the 4 analytical quality features from a prompt and optional dataset schema."""
    text = prompt.strip()
    words = re.findall(r"[a-z0-9_]+", text.lower())
    word_count = len(words)

    cols = list(profile.column_profiles.keys()) if profile is not None else list(available_columns or [])
    matched_cols = _match_available_columns(text, cols)

    # 1. Prompt specificity (0..1): length sufficiency, aggregations, operators/limits, time precision
    spec_score = 0.0
    if word_count >= 12:
        spec_score += 0.30
    elif word_count >= 7:
        spec_score += 0.20
    elif word_count >= 4:
        spec_score += 0.10

    if _AGGREGATION_RE.search(text):
        spec_score += 0.25
    if _OPERATOR_LIMIT_RE.search(text):
        spec_score += 0.25
    if _TIME_SCOPE_RE.search(text):
        spec_score += 0.20
    specificity_score = round(min(1.0, spec_score), 4)

    # 2. Context presence (0..1): grouping/filtering clauses, time scope, clear analytical framing
    ctx_score = 0.0
    if _GROUPING_CONTEXT_RE.search(text):
        ctx_score += 0.45
    if _TIME_SCOPE_RE.search(text):
        ctx_score += 0.30
    if len(matched_cols) >= 2:
        ctx_score += 0.25
    elif len(matched_cols) == 1:
        ctx_score += 0.10
    context_score = round(min(1.0, ctx_score), 4)

    # 3. Analytical intent clarity (0..1): clear analytical intent signals minus vague filler penalty
    intent_hits = sum(1 for pat in _INTENT_ACTION_PATTERNS if pat.search(text))
    vague_hits = len(_VAGUE_FILLER_RE.findall(text))
    if intent_hits == 0:
        clarity = 0.10 if word_count >= 4 else 0.0
    elif intent_hits == 1:
        clarity = 0.80
    elif intent_hits == 2:
        clarity = 0.95
    else:
        clarity = 0.75  # Slightly lower if overloaded with 3+ competing action verbs
    clarity -= min(0.4, vague_hits * 0.25)
    intent_clarity_score = round(max(0.0, min(1.0, clarity)), 4)

    # 4. Dataset column relevance (0..1): column alignment against dataset schema
    if cols:
        if len(matched_cols) >= 2:
            col_rel = 1.0
        elif len(matched_cols) == 1:
            col_rel = 0.65
        else:
            col_rel = 0.0
    else:
        col_rel = 0.0

    return PromptFeatures(
        specificity_score=specificity_score,
        context_score=context_score,
        intent_clarity_score=intent_clarity_score,
        column_relevance_score=round(col_rel, 4),
        matched_columns=matched_cols,
    )


# ---------------------------------------------------------------------------
# Curated Labeled Dataset of Analytical Prompts (Diverse Domains)
# ---------------------------------------------------------------------------


def build_labeled_prompt_dataset() -> list[LabeledPromptSample]:
    """Return a balanced, domain-diverse labeled dataset across Low, Medium, and High quality tiers."""
    sales_cols = ("order_date", "product", "region", "revenue", "units_sold")
    hr_cols = ("hire_date", "department", "role", "salary", "attrition")
    health_cols = ("visit_date", "diagnosis", "ward", "billing_amount", "length_of_stay")
    finance_cols = ("txn_date", "merchant_category", "account_tier", "amount", "is_fraud")
    ops_cols = ("ship_date", "carrier", "warehouse", "delivery_days", "shipping_cost")

    low_samples = [
        ("analyze data", sales_cols),
        ("show something", sales_cols),
        ("check this file", hr_cols),
        ("give insights", finance_cols),
        ("what is happening", ops_cols),
        ("tell me about stuff", health_cols),
        ("look at numbers", sales_cols),
        ("do analysis", hr_cols),
        ("anything interesting here", finance_cols),
        ("check good and bad things", ops_cols),
        ("help me with data", health_cols),
        ("show report", sales_cols),
        ("find something useful", hr_cols),
        ("data please", finance_cols),
        ("examine records", ops_cols),
        ("what does this say", health_cols),
        ("check performance", sales_cols),
        ("show trends maybe", finance_cols),
        ("is anything wrong", ops_cols),
        ("run analytics", hr_cols),
        ("explore", sales_cols),
        ("summarize stuff", health_cols),
        ("numbers overview", finance_cols),
        ("check columns", hr_cols),
        ("show details", ops_cols),
        ("what happened", sales_cols),
        ("tell me more", health_cols),
        ("quick look", finance_cols),
        ("check values", ops_cols),
        ("find patterns", hr_cols),
    ]

    medium_samples = [
        ("Show revenue", sales_cols),
        ("Compare salary by department", hr_cols),
        ("What is the trend of billing_amount?", health_cols),
        ("Detect anomalies in amount", finance_cols),
        ("Show distribution of delivery_days", ops_cols),
        ("Which product has high revenue?", sales_cols),
        ("Count attrition", hr_cols),
        ("Check missing values in dataset", health_cols),
        ("Correlation between amount and is_fraud", finance_cols),
        ("Filter shipping_cost above 100", ops_cols),
        ("Total units_sold", sales_cols),
        ("Average salary", hr_cols),
        ("Rank diagnosis", health_cols),
        ("Show monthly amount", finance_cols),
        ("Compare carrier delivery_days", ops_cols),
        ("Revenue by region", sales_cols),
        ("Check outliers in salary", hr_cols),
        ("Summarize ward data", health_cols),
        ("Find top merchant_category", finance_cols),
        ("Delivery_days distribution", ops_cols),
        ("Show missing cells", sales_cols),
        ("Compare revenue and units_sold", sales_cols),
        ("Trend over hire_date", hr_cols),
        ("Show fraud transactions", finance_cols),
        ("Average shipping_cost per warehouse", ops_cols),
        ("Which region is best?", sales_cols),
        ("Check correlation in dataset", hr_cols),
        ("Filter billing_amount greater than 5000", health_cols),
        ("Dataset summary", finance_cols),
        ("Show top warehouses", ops_cols),
    ]

    high_samples = [
        ("Rank the top 5 product categories by total sum of revenue sorted descending across all regions.", sales_cols),
        ("Calculate the monthly trend of total revenue over order_date and highlight period-over-period growth.", sales_cols),
        ("Compare average salary and median salary grouped by department for employees hired after 2022.", hr_cols),
        ("Compute the Pearson correlation between billing_amount and length_of_stay across each ward.", health_cols),
        ("Detect statistical outliers and anomalies in transaction amount grouped by merchant_category.", finance_cols),
        ("Show the distribution of delivery_days including quartiles and skewness for each carrier.", ops_cols),
        ("Filter records where revenue > 1000 in the North region and compute total units_sold per product.", sales_cols),
        ("Audit missing value counts and percentages across salary, role, and hire_date in the HR dataset.", hr_cols),
        ("Rank the bottom 3 warehouses by average shipping_cost sorted ascending for shipments over 5 days.", ops_cols),
        ("Compare total transaction amount across account_tier where is_fraud equals true vs false.", finance_cols),
        ("Analyze the quarterly trend of average length_of_stay by visit_date across each diagnosis group.", health_cols),
        ("Which top 3 region segments generated the highest total revenue and units_sold over order_date?", sales_cols),
        ("Compute the correlation between delivery_days and shipping_cost across every warehouse.", ops_cols),
        ("Summarize row counts, missing percentages, and descriptive statistics for salary grouped by role.", hr_cols),
        ("Identify anomalous spikes in billing_amount by ward using IQR thresholds and z-scores.", health_cols),
        ("Filter transactions where amount >= 500 and rank the top 5 merchant_category groups by total amount.", finance_cols),
        ("Analyze the weekly trend of total units_sold over order_date for the Electronics product line.", sales_cols),
        ("Compare mean and standard deviation of delivery_days across carrier and warehouse combinations.", ops_cols),
        ("Show the statistical distribution and outlier bounds of salary across each department.", hr_cols),
        ("Generate a complete dataset quality summary checking missing values in billing_amount and diagnosis.", health_cols),
        ("Calculate total revenue and average units_sold per region sorted descending for the last 6 months.", sales_cols),
        ("Detect anomalies in delivery_days where shipping_cost > 250 grouped by carrier.", ops_cols),
        ("Compare attrition count and average salary per department sorted from highest to lowest.", hr_cols),
        ("Compute correlation between revenue and units_sold after filtering rows where revenue > 0.", sales_cols),
        ("Track yearly total transaction amount over txn_date grouped by account_tier.", finance_cols),
        ("Identify the top 5 diagnosis categories by average billing_amount sorted descending.", health_cols),
        ("Audit missingness and duplicate percentages across txn_date, merchant_category, and amount.", finance_cols),
        ("Filter records where delivery_days > 7 and compare average shipping_cost across warehouse.", ops_cols),
        ("Analyze monthly attrition count by hire_date grouped by department.", hr_cols),
        ("Rank the top 10 products by average revenue per order in the East region.", sales_cols),
    ]

    samples: list[LabeledPromptSample] = []
    for text, cols in low_samples:
        samples.append(LabeledPromptSample(text, PromptQualityClass.LOW.value, cols))
    for text, cols in medium_samples:
        samples.append(LabeledPromptSample(text, PromptQualityClass.MEDIUM.value, cols))
    for text, cols in high_samples:
        samples.append(LabeledPromptSample(text, PromptQualityClass.HIGH.value, cols))
    return samples


def _samples_to_dataframe(samples: list[LabeledPromptSample]) -> tuple[pd.DataFrame, pd.Series]:
    rows: list[dict[str, Any]] = []
    labels: list[str] = []
    for sample in samples:
        feats = extract_prompt_features(sample.prompt_text, available_columns=sample.schema_columns)
        rows.append(
            {
                "prompt_text": sample.prompt_text,
                "specificity_score": feats.specificity_score,
                "context_score": feats.context_score,
                "intent_clarity_score": feats.intent_clarity_score,
                "column_relevance_score": feats.column_relevance_score,
            }
        )
        labels.append(sample.quality_label)
    return pd.DataFrame(rows), pd.Series(labels, name="quality_label")


# ---------------------------------------------------------------------------
# Training, Evaluation, Persistence & Inference
# ---------------------------------------------------------------------------


def train_prompt_quality_model(
    samples: list[LabeledPromptSample] | None = None,
    test_size: float = 0.25,
    random_state: int = 42,
    save_path: str | Path | None = None,
) -> PromptQualityModelBundle:
    """Train a leakage-free TF-IDF + LogisticRegression prompt quality classifier."""
    dataset = samples or build_labeled_prompt_dataset()
    X_df, y = _samples_to_dataframe(dataset)

    # Split BEFORE any TF-IDF or scaler fitting to strictly prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X_df,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "tfidf",
                TfidfVectorizer(ngram_range=(1, 2), lowercase=True, max_features=600, min_df=1),
                "prompt_text",
            ),
            (
                "numeric",
                StandardScaler(),
                ENGINEERED_FEATURE_COLUMNS,
            ),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=random_state, class_weight="balanced"),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    ordered_classes = [
        PromptQualityClass.LOW.value,
        PromptQualityClass.MEDIUM.value,
        PromptQualityClass.HIGH.value,
    ]
    present_classes = [c for c in ordered_classes if c in set(y.unique())]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    cls_report = classification_report(
        y_test,
        y_pred,
        labels=present_classes,
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        label: {
            "precision": round(float(cls_report[label]["precision"]), 4),
            "recall": round(float(cls_report[label]["recall"]), 4),
            "f1_score": round(float(cls_report[label]["f1-score"]), 4),
            "support": int(cls_report[label]["support"]),
        }
        for label in present_classes
        if label in cls_report
    }
    cm = confusion_matrix(y_test, y_pred, labels=present_classes).tolist()

    eval_report = TrainingEvaluationReport(
        accuracy=round(acc, 4),
        precision_weighted=round(prec, 4),
        recall_weighted=round(rec, 4),
        f1_weighted=round(f1, 4),
        classes=present_classes,
        per_class_metrics=per_class,
        confusion_matrix=cm,
        train_size=int(len(X_train)),
        test_size=int(len(X_test)),
        trained_at_utc=datetime.now(timezone.utc).isoformat(),
    )

    bundle = PromptQualityModelBundle(pipeline=pipeline, evaluation_report=eval_report)

    if save_path is not None:
        save_prompt_quality_model(bundle, model_path=save_path)

    logger.info(
        "Trained PromptQualityModel (train=%d, test=%d, accuracy=%.4f, f1=%.4f).",
        eval_report.train_size,
        eval_report.test_size,
        eval_report.accuracy,
        eval_report.f1_weighted,
    )
    return bundle


def get_default_model_path() -> Path:
    """Return default Joblib model path inside configured ml_models directory."""
    settings = get_settings()
    settings.ensure_directories()
    return settings.models_dir / DEFAULT_MODEL_FILENAME


def save_prompt_quality_model(
    bundle: PromptQualityModelBundle,
    model_path: str | Path | None = None,
) -> Path:
    """Persist trained Pipeline and evaluation metadata to disk using Joblib."""
    target_path = Path(model_path) if model_path is not None else get_default_model_path()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, target_path)
    logger.info("Saved prompt quality model bundle to '%s'.", target_path)
    return target_path


def load_prompt_quality_model(
    model_path: str | Path | None = None,
    auto_train_if_missing: bool = False,
) -> PromptQualityModelBundle:
    """Load a persisted prompt quality model bundle using Joblib."""
    target_path = Path(model_path) if model_path is not None else get_default_model_path()
    if not target_path.is_file():
        if auto_train_if_missing:
            logger.info("Model file '%s' not found; training and persisting offline model.", target_path)
            return train_prompt_quality_model(save_path=target_path)
        raise ModelNotFoundError(f"Saved prompt quality model not found at '{target_path}'.")

    loaded = joblib.load(target_path)
    if not isinstance(loaded, PromptQualityModelBundle):
        raise PromptQualityModelError(f"Invalid model artifact loaded from '{target_path}'.")
    return loaded


def _generate_recommendations(
    predicted_class: str,
    features: PromptFeatures,
    profile: DatasetProfileResult | None,
) -> list[str]:
    recs: list[str] = []

    if features.column_relevance_score < 0.5:
        if profile is not None:
            sample_cols = list(profile.column_profiles.keys())[:4]
            recs.append(f"Reference explicit dataset columns such as {sample_cols}.")
        else:
            recs.append("Reference exact dataset column names for metrics and grouping dimensions.")

    if features.specificity_score < 0.5:
        recs.append(
            "Specify an explicit aggregation (sum, mean, median, count), ranking limit (e.g., Top 5), or filter threshold."
        )

    if features.context_score < 0.5:
        if profile is not None and profile.datetime_columns and profile.categorical_columns:
            recs.append(
                f"Add analytical context such as grouping by '{profile.categorical_columns[0]}' "
                f"or specifying a time window on '{profile.datetime_columns[0]}'."
            )
        else:
            recs.append("Include a grouping dimension (e.g., 'by category') or time granularity (e.g., 'monthly').")

    if features.intent_clarity_score < 0.6:
        recs.append(
            "Use a clear analytical action verb (e.g., rank, compare, trend, correlate, or detect anomalies)."
        )

    if not recs and predicted_class == PromptQualityClass.HIGH.value:
        recs.append("Prompt is clear, schema-relevant, and ready for deterministic analytical execution.")

    return recs


def evaluate_prompt_quality(
    prompt: str,
    profile: DatasetProfileResult | None = None,
    available_columns: list[str] | tuple[str, ...] | None = None,
    model_bundle: PromptQualityModelBundle | None = None,
    model_path: str | Path | None = None,
) -> PromptQualityPrediction:
    """Predict prompt quality (Low/Medium/High) and generate improvement recommendations."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise InvalidPromptInputError("Prompt must be a non-empty string.")

    bundle = model_bundle or load_prompt_quality_model(
        model_path=model_path,
        auto_train_if_missing=True,
    )

    clean_prompt = prompt.strip()
    features = extract_prompt_features(
        clean_prompt,
        available_columns=available_columns,
        profile=profile,
    )

    inference_df = pd.DataFrame(
        [
            {
                "prompt_text": clean_prompt,
                "specificity_score": features.specificity_score,
                "context_score": features.context_score,
                "intent_clarity_score": features.intent_clarity_score,
                "column_relevance_score": features.column_relevance_score,
            }
        ]
    )

    predicted_class = str(bundle.pipeline.predict(inference_df)[0])
    proba_values = bundle.pipeline.predict_proba(inference_df)[0]
    class_names = [str(c) for c in bundle.pipeline.classes_]
    probabilities = {
        cls_name: round(float(prob), 4)
        for cls_name, prob in zip(class_names, proba_values)
    }

    # Probability-weighted continuous quality index on [0, 100]
    tier_anchors = {
        PromptQualityClass.LOW.value: 20.0,
        PromptQualityClass.MEDIUM.value: 60.0,
        PromptQualityClass.HIGH.value: 95.0,
    }
    quality_score = round(
        sum(probabilities.get(cls_name, 0.0) * anchor for cls_name, anchor in tier_anchors.items()),
        2,
    )

    recommendations = _generate_recommendations(predicted_class, features, profile)

    return PromptQualityPrediction(
        prompt=clean_prompt,
        predicted_class=predicted_class,
        quality_score=quality_score,
        class_probabilities=probabilities,
        extracted_features=features,
        recommendations=recommendations,
        model_accuracy_on_test=bundle.evaluation_report.accuracy,
    )


class PromptQualityEvaluator:
    """Stateful offline-capable service wrapper for Module 5."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path) if model_path is not None else get_default_model_path()
        self._bundle: PromptQualityModelBundle | None = None

    def train(self, save: bool = True, test_size: float = 0.25) -> TrainingEvaluationReport:
        self._bundle = train_prompt_quality_model(
            test_size=test_size,
            save_path=self.model_path if save else None,
        )
        return self._bundle.evaluation_report

    def load(self, auto_train_if_missing: bool = True) -> PromptQualityModelBundle:
        self._bundle = load_prompt_quality_model(
            model_path=self.model_path,
            auto_train_if_missing=auto_train_if_missing,
        )
        return self._bundle

    def predict(
        self,
        prompt: str,
        profile: DatasetProfileResult | None = None,
        available_columns: list[str] | tuple[str, ...] | None = None,
    ) -> PromptQualityPrediction:
        bundle = self._bundle or self.load(auto_train_if_missing=True)
        return evaluate_prompt_quality(
            prompt=prompt,
            profile=profile,
            available_columns=available_columns,
            model_bundle=bundle,
        )
