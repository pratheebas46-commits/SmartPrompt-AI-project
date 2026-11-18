"""Module 2: Data Cleaning and Preprocessing for SmartPrompt AI.

Provides non-destructive-by-default, user-configurable data cleaning and
preprocessing designed to work seamlessly with Module 1 (dataset_profiler.py).
Preserves the original DataFrame, supports missing-value imputation/removal,
opt-in duplicate removal, numerical scaling/outlier clipping, categorical
normalization/encoding, safe type & datetime conversion, and detailed audit reporting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import re
from typing import Any

import numpy as np
import pandas as pd

from config.logging_config import get_logger
from modules.dataset_profiler import DatasetProfileResult, profile_dataframe

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------


class DataCleaningError(Exception):
    """Base exception for data cleaning and preprocessing errors."""


class InvalidCleaningStrategyError(DataCleaningError):
    """Raised when an unsupported cleaning or preprocessing strategy is requested."""


class ColumnNotFoundError(DataCleaningError):
    """Raised when a configured column does not exist in the DataFrame."""


# ---------------------------------------------------------------------------
# Strategy Enums & Configuration
# ---------------------------------------------------------------------------


class MissingValueStrategy(str, Enum):
    NONE = "none"
    DROP_ROWS = "drop_rows"
    DROP_COLUMNS = "drop_columns"
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    CONSTANT = "constant"
    FORWARD_FILL = "ffill"
    BACKWARD_FILL = "bfill"


class NumericScalingStrategy(str, Enum):
    NONE = "none"
    STANDARDIZE = "standardize"
    MIN_MAX = "min_max"
    CLIP_OUTLIERS_IQR = "clip_outliers_iqr"


class CategoricalCaseMode(str, Enum):
    NONE = "none"
    LOWER = "lower"
    UPPER = "upper"
    TITLE = "title"


class CategoricalEncodingStrategy(str, Enum):
    NONE = "none"
    LABEL = "label"
    ONE_HOT = "one_hot"


class TargetDataType(str, Enum):
    NUMERICAL = "numerical"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"


@dataclass
class CleaningConfig:
    """User-selectable cleaning strategies. Defaults are strictly non-destructive."""

    # Missing value handling
    numeric_missing_strategy: MissingValueStrategy | str = MissingValueStrategy.NONE
    categorical_missing_strategy: MissingValueStrategy | str = MissingValueStrategy.NONE
    column_missing_strategies: dict[str, MissingValueStrategy | str] = field(default_factory=dict)
    constant_fill_value: Any = "Unknown"
    numeric_constant_fill_value: float | int = 0
    drop_column_missing_threshold: float = 0.5  # Drop column if missing ratio >= threshold

    # Duplicate handling (opt-in only to prevent automatic destructive changes)
    remove_duplicates: bool = False
    duplicate_subset: list[str] | None = None
    duplicate_keep: str | bool = "first"  # "first" | "last" | False

    # Type coercion & datetime conversion
    type_conversions: dict[str, TargetDataType | str] = field(default_factory=dict)
    datetime_columns: list[str] = field(default_factory=list)
    datetime_format: str | None = None
    strip_numeric_symbols: bool = True
    replace_inf_with_nan: bool = True

    # Numerical preprocessing
    numeric_scaling: NumericScalingStrategy | str = NumericScalingStrategy.NONE
    numeric_columns_to_preprocess: list[str] | None = None
    iqr_clip_multiplier: float = 1.5

    # Categorical preprocessing
    strip_categorical_whitespace: bool = False
    categorical_case: CategoricalCaseMode | str = CategoricalCaseMode.NONE
    rare_category_min_freq: float | None = None  # Fraction in (0, 1)
    categorical_encoding: CategoricalEncodingStrategy | str = CategoricalEncodingStrategy.NONE
    categorical_columns_to_preprocess: list[str] | None = None

    # Optional re-profiling via Module 1 after cleaning
    reprofile_after_cleaning: bool = True

    def validate(self, available_columns: list[str]) -> None:
        """Validate strategy enums, thresholds, and referenced columns."""
        col_set = set(available_columns)

        _parse_enum(self.numeric_missing_strategy, MissingValueStrategy, "numeric_missing_strategy")
        _parse_enum(self.categorical_missing_strategy, MissingValueStrategy, "categorical_missing_strategy")
        _parse_enum(self.numeric_scaling, NumericScalingStrategy, "numeric_scaling")
        _parse_enum(self.categorical_case, CategoricalCaseMode, "categorical_case")
        _parse_enum(self.categorical_encoding, CategoricalEncodingStrategy, "categorical_encoding")

        if not (0.0 < self.drop_column_missing_threshold <= 1.0):
            raise InvalidCleaningStrategyError(
                "drop_column_missing_threshold must be in (0.0, 1.0]."
            )

        if self.duplicate_keep not in ("first", "last", False):
            raise InvalidCleaningStrategyError("duplicate_keep must be 'first', 'last', or False.")

        if self.duplicate_subset:
            missing_cols = [c for c in self.duplicate_subset if c not in col_set]
            if missing_cols:
                raise ColumnNotFoundError(f"duplicate_subset columns not found: {missing_cols}")

        for col, strategy in self.column_missing_strategies.items():
            if col not in col_set:
                raise ColumnNotFoundError(f"Column '{col}' in column_missing_strategies not found.")
            _parse_enum(strategy, MissingValueStrategy, f"column_missing_strategies[{col}]")

        for col, target_type in self.type_conversions.items():
            if col not in col_set:
                raise ColumnNotFoundError(f"Column '{col}' in type_conversions not found.")
            _parse_enum(target_type, TargetDataType, f"type_conversions[{col}]")

        for col in self.datetime_columns:
            if col not in col_set:
                raise ColumnNotFoundError(f"Column '{col}' in datetime_columns not found.")

        if self.numeric_columns_to_preprocess is not None:
            missing = [c for c in self.numeric_columns_to_preprocess if c not in col_set]
            if missing:
                raise ColumnNotFoundError(f"numeric_columns_to_preprocess not found: {missing}")

        if self.categorical_columns_to_preprocess is not None:
            missing = [c for c in self.categorical_columns_to_preprocess if c not in col_set]
            if missing:
                raise ColumnNotFoundError(f"categorical_columns_to_preprocess not found: {missing}")

        if self.rare_category_min_freq is not None and not (0.0 < self.rare_category_min_freq < 1.0):
            raise InvalidCleaningStrategyError("rare_category_min_freq must be between 0 and 1.")

        if self.iqr_clip_multiplier <= 0:
            raise InvalidCleaningStrategyError("iqr_clip_multiplier must be positive.")


# ---------------------------------------------------------------------------
# Structured Output Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TypeConversionDetail:
    """Audit record of a column type or datetime conversion."""

    column: str
    target_type: str
    non_null_before: int
    valid_after: int
    invalid_coerced_to_null: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CleaningReport:
    """Comprehensive summary of all cleaning operations and dataset deltas."""

    original_shape: tuple[int, int]
    cleaned_shape: tuple[int, int]
    rows_removed: int
    columns_removed: list[str]
    columns_added: list[str]
    missing_values_before: int
    missing_values_after: int
    missing_cells_filled: int
    columns_imputed: dict[str, str]
    duplicates_removed: int
    type_conversions: list[TypeConversionDetail]
    numerical_preprocessing_applied: list[str]
    categorical_preprocessing_applied: list[str]
    actions_taken: list[str]
    warnings: list[str]
    was_modified: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_shape": list(self.original_shape),
            "cleaned_shape": list(self.cleaned_shape),
            "rows_removed": self.rows_removed,
            "columns_removed": self.columns_removed,
            "columns_added": self.columns_added,
            "missing_values_before": self.missing_values_before,
            "missing_values_after": self.missing_values_after,
            "missing_cells_filled": self.missing_cells_filled,
            "columns_imputed": self.columns_imputed,
            "duplicates_removed": self.duplicates_removed,
            "type_conversions": [item.to_dict() for item in self.type_conversions],
            "numerical_preprocessing_applied": self.numerical_preprocessing_applied,
            "categorical_preprocessing_applied": self.categorical_preprocessing_applied,
            "actions_taken": self.actions_taken,
            "warnings": self.warnings,
            "was_modified": self.was_modified,
        }


@dataclass
class CleaningResult:
    """Output bundle containing the cleaned DataFrame, CleaningReport, and optional new profile."""

    cleaned_df: pd.DataFrame
    report: CleaningReport
    updated_profile: DatasetProfileResult | None = None


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def _parse_enum(value: Any, enum_cls: type[Enum], field_name: str) -> Any:
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        for member in enum_cls:
            if member.value == normalized:
                return member
    valid = [m.value for m in enum_cls]
    raise InvalidCleaningStrategyError(
        f"Invalid value '{value}' for '{field_name}'. Expected one of: {valid}."
    )


def _safe_convert_numeric(series: pd.Series, strip_symbols: bool = True) -> tuple[pd.Series, int, int, int]:
    """Safely convert a series to numeric, coercing unparseable tokens to NaN."""
    non_null_before = int(series.notna().sum())
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        converted = pd.to_numeric(series, errors="coerce")
    else:
        working = series.copy()
        if strip_symbols:
            mask = working.notna()
            cleaned_str = (
                working[mask]
                .astype(str)
                .str.strip()
                .str.replace(r"[$€£¥,\s]", "", regex=True)
            )
            working = working.astype(object)
            working.loc[mask] = cleaned_str
        converted = pd.to_numeric(working, errors="coerce")

    valid_after = int(converted.notna().sum())
    invalid_count = max(0, non_null_before - valid_after)
    return converted, non_null_before, valid_after, invalid_count


def _safe_convert_boolean(series: pd.Series) -> tuple[pd.Series, int, int, int]:
    """Safely convert common boolean representations into nullable boolean dtype."""
    non_null_before = int(series.notna().sum())
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean"), non_null_before, non_null_before, 0

    true_tokens = {"true", "1", "yes", "y", "t"}
    false_tokens = {"false", "0", "no", "n", "f"}

    def _map_val(val: Any) -> Any:
        if pd.isna(val):
            return pd.NA
        if isinstance(val, (bool, np.bool_)):
            return bool(val)
        if isinstance(val, (int, float, np.integer, np.floating)) and not np.isnan(float(val)):
            if float(val) == 1.0:
                return True
            if float(val) == 0.0:
                return False
            return pd.NA
        token = str(val).strip().lower()
        if token in true_tokens:
            return True
        if token in false_tokens:
            return False
        return pd.NA

    converted = series.map(_map_val).astype("boolean")
    valid_after = int(converted.notna().sum())
    invalid_count = max(0, non_null_before - valid_after)
    return converted, non_null_before, valid_after, invalid_count


def safe_convert_datetime(
    series: pd.Series,
    datetime_format: str | None = None,
) -> tuple[pd.Series, int, int, int]:
    """Safely convert a series to pandas datetime64, coercing invalid strings to NaT."""
    non_null_before = int(series.notna().sum())
    if pd.api.types.is_datetime64_any_dtype(series):
        valid_after = int(series.notna().sum())
        return series, non_null_before, valid_after, 0

    if datetime_format:
        converted = pd.to_datetime(series, format=datetime_format, errors="coerce")
    else:
        try:
            converted = pd.to_datetime(series, errors="coerce", format="mixed")
        except TypeError:
            converted = pd.to_datetime(series, errors="coerce")

    valid_after = int(converted.notna().sum())
    invalid_count = max(0, non_null_before - valid_after)
    return converted, non_null_before, valid_after, invalid_count


def _detect_column_groups(
    df: pd.DataFrame,
    profile: DatasetProfileResult | None = None,
) -> tuple[list[str], list[str]]:
    """Determine numerical and categorical columns from profile or current DataFrame dtypes."""
    current_cols = set(df.columns)
    if profile is not None:
        num_cols = [c for c in profile.numerical_columns if c in current_cols and pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [
            c
            for c in (profile.categorical_columns + profile.boolean_columns)
            if c in current_cols and c not in num_cols
        ]
        # Catch any newly coerced numerical columns
        for col in df.columns:
            if col not in num_cols and col not in cat_cols:
                if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
                    num_cols.append(col)
                elif not pd.api.types.is_datetime64_any_dtype(df[col]):
                    cat_cols.append(col)
        return num_cols, cat_cols

    num_cols = [
        str(col)
        for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col])
    ]
    cat_cols = [
        str(col)
        for col in df.columns
        if col not in num_cols and not pd.api.types.is_datetime64_any_dtype(df[col])
    ]
    return num_cols, cat_cols


def recommend_cleaning_config(profile: DatasetProfileResult) -> CleaningConfig:
    """Generate a non-destructive recommended CleaningConfig from a Module 1 profile.

    Does NOT modify any data; returns a candidate configuration for user approval.
    """
    column_strategies: dict[str, MissingValueStrategy] = {}
    for col, info in profile.missing_summary.columns_with_missing.items():
        pct = float(info["percentage"])
        if pct >= 60.0:
            column_strategies[col] = MissingValueStrategy.DROP_COLUMNS
        elif col in profile.numerical_columns:
            col_stats = profile.column_profiles[col].stats
            skewness = abs(float(col_stats.get("skewness") or 0.0))
            column_strategies[col] = (
                MissingValueStrategy.MEDIAN if skewness > 1.0 else MissingValueStrategy.MEAN
            )
        elif col in profile.categorical_columns or col in profile.boolean_columns:
            column_strategies[col] = MissingValueStrategy.MODE

    return CleaningConfig(
        numeric_missing_strategy=MissingValueStrategy.NONE,
        categorical_missing_strategy=MissingValueStrategy.NONE,
        column_missing_strategies=column_strategies,
        remove_duplicates=profile.duplicate_summary.duplicate_row_count > 0,
        datetime_columns=list(profile.datetime_columns),
        strip_categorical_whitespace=len(profile.categorical_columns) > 0,
    )


# ---------------------------------------------------------------------------
# Core Cleaning Pipeline
# ---------------------------------------------------------------------------


def clean_dataset(
    df: pd.DataFrame,
    config: CleaningConfig | None = None,
    profile: DatasetProfileResult | None = None,
    dataset_name: str = "cleaned_dataset",
) -> CleaningResult:
    """Clean and preprocess a DataFrame according to user-selected strategies.

    Always operates on a deep copy so the original DataFrame is never mutated.
    When `config` is omitted, defaults are completely non-destructive.
    """
    if not isinstance(df, pd.DataFrame):
        raise DataCleaningError("Input 'df' must be a pandas DataFrame.")

    cfg = config or CleaningConfig()
    cfg.validate(list(df.columns))

    working_df = df.copy(deep=True)
    original_shape = (int(df.shape[0]), int(df.shape[1]))
    missing_before = int(working_df.isna().sum().sum())

    actions: list[str] = []
    warnings: list[str] = []
    columns_removed: list[str] = []
    columns_added: list[str] = []
    columns_imputed: dict[str, str] = {}
    missing_cells_filled = 0
    type_conversions: list[TypeConversionDetail] = []
    numerical_applied: list[str] = []
    categorical_applied: list[str] = []
    duplicates_removed = 0

    if working_df.empty:
        warnings.append("Input DataFrame has 0 rows; cleaning steps skipped safely.")
        report = CleaningReport(
            original_shape=original_shape,
            cleaned_shape=(int(working_df.shape[0]), int(working_df.shape[1])),
            rows_removed=0,
            columns_removed=[],
            columns_added=[],
            missing_values_before=missing_before,
            missing_values_after=missing_before,
            missing_cells_filled=0,
            columns_imputed={},
            duplicates_removed=0,
            type_conversions=[],
            numerical_preprocessing_applied=[],
            categorical_preprocessing_applied=[],
            actions_taken=[],
            warnings=warnings,
            was_modified=False,
        )
        updated_prof = (
            profile_dataframe(working_df, dataset_name=dataset_name)
            if cfg.reprofile_after_cleaning
            else None
        )
        return CleaningResult(cleaned_df=working_df, report=report, updated_profile=updated_prof)

    # 1. Replace +/-inf with NaN when configured
    if cfg.replace_inf_with_nan:
        num_like_cols = [
            c
            for c in working_df.columns
            if pd.api.types.is_numeric_dtype(working_df[c]) and not pd.api.types.is_bool_dtype(working_df[c])
        ]
        if num_like_cols:
            inf_count = int(np.isinf(working_df[num_like_cols].to_numpy(dtype=float, na_value=np.nan)).sum())
            if inf_count > 0:
                working_df[num_like_cols] = working_df[num_like_cols].replace([np.inf, -np.inf], np.nan)
                actions.append(f"Replaced {inf_count} infinite value(s) with NaN.")

    # 2. Explicit type conversions & safe datetime conversion
    converted_cols: set[str] = set()
    for col, raw_target in cfg.type_conversions.items():
        target = _parse_enum(raw_target, TargetDataType, f"type_conversions[{col}]")
        if target == TargetDataType.NUMERICAL:
            series, nb, va, inv = _safe_convert_numeric(working_df[col], strip_symbols=cfg.strip_numeric_symbols)
        elif target == TargetDataType.DATETIME:
            series, nb, va, inv = safe_convert_datetime(working_df[col], datetime_format=cfg.datetime_format)
        elif target == TargetDataType.BOOLEAN:
            series, nb, va, inv = _safe_convert_boolean(working_df[col])
        else:
            nb = int(working_df[col].notna().sum())
            series = working_df[col].astype("string")
            va = int(series.notna().sum())
            inv = 0

        working_df[col] = series
        converted_cols.add(col)
        type_conversions.append(
            TypeConversionDetail(
                column=col,
                target_type=target.value,
                non_null_before=nb,
                valid_after=va,
                invalid_coerced_to_null=inv,
            )
        )
        actions.append(
            f"Converted column '{col}' to '{target.value}' (valid={va}, invalid_coerced={inv})."
        )
        if inv > 0:
            warnings.append(f"Column '{col}' had {inv} invalid value(s) coerced to null during {target.value} conversion.")

    for col in cfg.datetime_columns:
        if col in converted_cols:
            continue
        series, nb, va, inv = safe_convert_datetime(working_df[col], datetime_format=cfg.datetime_format)
        working_df[col] = series
        type_conversions.append(
            TypeConversionDetail(
                column=col,
                target_type=TargetDataType.DATETIME.value,
                non_null_before=nb,
                valid_after=va,
                invalid_coerced_to_null=inv,
            )
        )
        actions.append(f"Converted column '{col}' to datetime (valid={va}, invalid_coerced={inv}).")
        if inv > 0:
            warnings.append(f"Datetime column '{col}' had {inv} unparseable value(s) coerced to NaT.")

    # Refresh missing_before if type conversion introduced NaN/NaT values?
    # Keep missing_before as original missing count, and compute filled accurately.

    # 3. Duplicate removal (opt-in only)
    if cfg.remove_duplicates:
        before_rows = len(working_df)
        working_df = working_df.drop_duplicates(subset=cfg.duplicate_subset, keep=cfg.duplicate_keep).reset_index(
            drop=True
        )
        duplicates_removed = before_rows - len(working_df)
        if duplicates_removed > 0:
            actions.append(f"Removed {duplicates_removed} duplicate row(s).")

    # 4. Missing-value handling
    num_strategy = _parse_enum(cfg.numeric_missing_strategy, MissingValueStrategy, "numeric_missing_strategy")
    cat_strategy = _parse_enum(
        cfg.categorical_missing_strategy, MissingValueStrategy, "categorical_missing_strategy"
    )
    num_cols, cat_cols = _detect_column_groups(working_df, profile=profile)

    # 4a. Per-column overrides first
    drop_rows_cols: list[str] = []
    cols_to_drop: list[str] = []
    for col, raw_strat in cfg.column_missing_strategies.items():
        if col not in working_df.columns:
            continue
        strat = _parse_enum(raw_strat, MissingValueStrategy, f"column_missing_strategies[{col}]")
        if strat == MissingValueStrategy.DROP_COLUMNS:
            missing_ratio = float(working_df[col].isna().mean())
            if missing_ratio >= cfg.drop_column_missing_threshold or col in cfg.column_missing_strategies:
                cols_to_drop.append(col)
        elif strat == MissingValueStrategy.DROP_ROWS:
            drop_rows_cols.append(col)
        elif strat != MissingValueStrategy.NONE:
            filled_count = _apply_column_imputation(
                working_df,
                col=col,
                strategy=strat,
                constant_val=(
                    cfg.numeric_constant_fill_value if col in num_cols else cfg.constant_fill_value
                ),
                warnings=warnings,
            )
            if filled_count > 0:
                missing_cells_filled += filled_count
                columns_imputed[col] = strat.value
                actions.append(f"Imputed {filled_count} missing value(s) in '{col}' using '{strat.value}'.")

    # 4b. Global numeric / categorical missing strategies (skipping columns already handled)
    handled_cols = set(cfg.column_missing_strategies.keys())
    for col in list(working_df.columns):
        if col in handled_cols or col in cols_to_drop or working_df[col].isna().sum() == 0:
            continue

        if col in num_cols:
            strat = num_strategy
            const_val = cfg.numeric_constant_fill_value
        elif col in cat_cols:
            strat = cat_strategy
            const_val = cfg.constant_fill_value
        else:
            # Datetime columns only respond to explicit column strategies or drop_rows/ffill/bfill
            continue

        if strat == MissingValueStrategy.NONE:
            continue
        if strat == MissingValueStrategy.DROP_COLUMNS:
            if float(working_df[col].isna().mean()) >= cfg.drop_column_missing_threshold:
                cols_to_drop.append(col)
            continue
        if strat == MissingValueStrategy.DROP_ROWS:
            drop_rows_cols.append(col)
            continue

        filled_count = _apply_column_imputation(
            working_df,
            col=col,
            strategy=strat,
            constant_val=const_val,
            warnings=warnings,
        )
        if filled_count > 0:
            missing_cells_filled += filled_count
            columns_imputed[col] = strat.value
            actions.append(f"Imputed {filled_count} missing value(s) in '{col}' using '{strat.value}'.")

    if cols_to_drop:
        unique_drop = [c for c in dict.fromkeys(cols_to_drop) if c in working_df.columns]
        working_df = working_df.drop(columns=unique_drop)
        columns_removed.extend(unique_drop)
        actions.append(f"Dropped {len(unique_drop)} high-missingness column(s): {unique_drop}.")

    if drop_rows_cols:
        valid_subset = [c for c in dict.fromkeys(drop_rows_cols) if c in working_df.columns]
        if valid_subset:
            before_drop = len(working_df)
            working_df = working_df.dropna(subset=valid_subset).reset_index(drop=True)
            dropped = before_drop - len(working_df)
            if dropped > 0:
                actions.append(f"Dropped {dropped} row(s) with missing values in {valid_subset}.")

    # Refresh column groups after any drops
    num_cols, cat_cols = _detect_column_groups(working_df, profile=None)

    # 5. Numerical preprocessing
    scaling_strategy = _parse_enum(cfg.numeric_scaling, NumericScalingStrategy, "numeric_scaling")
    target_num_cols = (
        [c for c in cfg.numeric_columns_to_preprocess if c in num_cols]
        if cfg.numeric_columns_to_preprocess is not None
        else num_cols
    )
    if scaling_strategy != NumericScalingStrategy.NONE and target_num_cols:
        for col in target_num_cols:
            series = pd.to_numeric(working_df[col], errors="coerce")
            non_null = series.dropna()
            if non_null.empty:
                warnings.append(f"Skipped '{scaling_strategy.value}' on all-null column '{col}'.")
                continue

            if scaling_strategy == NumericScalingStrategy.CLIP_OUTLIERS_IQR:
                q1 = float(non_null.quantile(0.25))
                q3 = float(non_null.quantile(0.75))
                iqr = q3 - q1
                if iqr > 0:
                    lower = q1 - cfg.iqr_clip_multiplier * iqr
                    upper = q3 + cfg.iqr_clip_multiplier * iqr
                    working_df[col] = series.clip(lower=lower, upper=upper)
                    numerical_applied.append(f"{col}:{scaling_strategy.value}")
                    actions.append(
                        f"Clipped outliers in '{col}' to IQR bounds [{lower:.4f}, {upper:.4f}]."
                    )
                else:
                    warnings.append(f"Skipped IQR clipping on '{col}' because IQR is 0.")
            elif scaling_strategy == NumericScalingStrategy.STANDARDIZE:
                std_val = float(non_null.std(ddof=0))
                if std_val > 0:
                    mean_val = float(non_null.mean())
                    working_df[col] = (series - mean_val) / std_val
                    numerical_applied.append(f"{col}:{scaling_strategy.value}")
                    actions.append(f"Standardized '{col}' (z-score).")
                else:
                    warnings.append(f"Skipped standardization on zero-variance column '{col}'.")
            elif scaling_strategy == NumericScalingStrategy.MIN_MAX:
                min_val = float(non_null.min())
                max_val = float(non_null.max())
                span = max_val - min_val
                if span > 0:
                    working_df[col] = (series - min_val) / span
                    numerical_applied.append(f"{col}:{scaling_strategy.value}")
                    actions.append(f"Min-max scaled '{col}' to [0, 1].")
                else:
                    warnings.append(f"Skipped min-max scaling on constant column '{col}'.")

    # 6. Categorical preprocessing
    case_mode = _parse_enum(cfg.categorical_case, CategoricalCaseMode, "categorical_case")
    encode_strategy = _parse_enum(
        cfg.categorical_encoding, CategoricalEncodingStrategy, "categorical_encoding"
    )
    target_cat_cols = (
        [c for c in cfg.categorical_columns_to_preprocess if c in cat_cols]
        if cfg.categorical_columns_to_preprocess is not None
        else [c for c in cat_cols if not pd.api.types.is_bool_dtype(working_df[c])]
    )

    for col in target_cat_cols:
        if cfg.strip_categorical_whitespace or case_mode != CategoricalCaseMode.NONE:
            mask = working_df[col].notna()
            s = working_df.loc[mask, col].astype(str)
            if cfg.strip_categorical_whitespace:
                s = s.str.strip()
            if case_mode == CategoricalCaseMode.LOWER:
                s = s.str.lower()
            elif case_mode == CategoricalCaseMode.UPPER:
                s = s.str.upper()
            elif case_mode == CategoricalCaseMode.TITLE:
                s = s.str.title()
            working_df.loc[mask, col] = s
            categorical_applied.append(f"{col}:text_normalized")
            actions.append(
                f"Normalized text in '{col}' (strip={cfg.strip_categorical_whitespace}, case={case_mode.value})."
            )

        if cfg.rare_category_min_freq is not None and not working_df[col].dropna().empty:
            freqs = working_df[col].value_counts(normalize=True)
            rare_labels = set(freqs[freqs < cfg.rare_category_min_freq].index)
            if rare_labels:
                working_df[col] = working_df[col].apply(lambda v: "Other" if v in rare_labels else v)
                categorical_applied.append(f"{col}:rare_grouped")
                actions.append(
                    f"Grouped {len(rare_labels)} rare category(ies) (<{cfg.rare_category_min_freq:.1%}) "
                    f"in '{col}' into 'Other'."
                )

    if encode_strategy == CategoricalEncodingStrategy.LABEL and target_cat_cols:
        for col in target_cat_cols:
            encoded_col = f"{col}_encoded"
            codes, _ = pd.factorize(working_df[col], sort=True)
            working_df[encoded_col] = pd.Series(codes, index=working_df.index).replace(-1, pd.NA)
            columns_added.append(encoded_col)
            categorical_applied.append(f"{col}:label_encoded")
            actions.append(f"Added label-encoded column '{encoded_col}'.")
    elif encode_strategy == CategoricalEncodingStrategy.ONE_HOT and target_cat_cols:
        before_cols = set(working_df.columns)
        working_df = pd.get_dummies(working_df, columns=target_cat_cols, dummy_na=False, dtype=int)
        after_cols = list(working_df.columns)
        new_cols = [c for c in after_cols if c not in before_cols]
        removed_by_ohe = [c for c in target_cat_cols if c not in after_cols]
        columns_added.extend(new_cols)
        columns_removed.extend(removed_by_ohe)
        categorical_applied.append(f"one_hot:{len(target_cat_cols)}_cols")
        actions.append(f"One-hot encoded {target_cat_cols} into {len(new_cols)} indicator column(s).")

    cleaned_shape = (int(working_df.shape[0]), int(working_df.shape[1]))
    rows_removed = max(0, original_shape[0] - cleaned_shape[0])
    missing_after = int(working_df.isna().sum().sum())

    was_modified = not working_df.equals(df)

    report = CleaningReport(
        original_shape=original_shape,
        cleaned_shape=cleaned_shape,
        rows_removed=rows_removed,
        columns_removed=columns_removed,
        columns_added=columns_added,
        missing_values_before=missing_before,
        missing_values_after=missing_after,
        missing_cells_filled=missing_cells_filled,
        columns_imputed=columns_imputed,
        duplicates_removed=duplicates_removed,
        type_conversions=type_conversions,
        numerical_preprocessing_applied=numerical_applied,
        categorical_preprocessing_applied=categorical_applied,
        actions_taken=actions,
        warnings=warnings,
        was_modified=was_modified,
    )

    logger.info(
        "Completed cleaning for '%s': %s -> %s, actions=%d, modified=%s.",
        dataset_name,
        original_shape,
        cleaned_shape,
        len(actions),
        was_modified,
    )

    updated_prof = (
        profile_dataframe(working_df, dataset_name=dataset_name)
        if cfg.reprofile_after_cleaning
        else None
    )
    return CleaningResult(cleaned_df=working_df, report=report, updated_profile=updated_prof)


def _apply_column_imputation(
    df: pd.DataFrame,
    col: str,
    strategy: MissingValueStrategy,
    constant_val: Any,
    warnings: list[str],
) -> int:
    """Apply an in-place imputation strategy on a single column and return cells filled."""
    nulls_before = int(df[col].isna().sum())
    if nulls_before == 0:
        return 0

    is_num = pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col])

    if strategy == MissingValueStrategy.MEAN:
        if not is_num:
            warnings.append(f"Cannot apply 'mean' imputation to non-numerical column '{col}'; skipped.")
            return 0
        mean_val = df[col].mean()
        if pd.isna(mean_val):
            warnings.append(f"Column '{col}' has no non-null values for 'mean' imputation; skipped.")
            return 0
        df[col] = df[col].fillna(mean_val)
    elif strategy == MissingValueStrategy.MEDIAN:
        if not is_num:
            warnings.append(f"Cannot apply 'median' imputation to non-numerical column '{col}'; skipped.")
            return 0
        median_val = df[col].median()
        if pd.isna(median_val):
            warnings.append(f"Column '{col}' has no non-null values for 'median' imputation; skipped.")
            return 0
        df[col] = df[col].fillna(median_val)
    elif strategy == MissingValueStrategy.MODE:
        modes = df[col].dropna().mode()
        if modes.empty:
            warnings.append(f"Column '{col}' has no non-null values for 'mode' imputation; skipped.")
            return 0
        df[col] = df[col].fillna(modes.iloc[0])
    elif strategy == MissingValueStrategy.CONSTANT:
        if pd.api.types.is_bool_dtype(df[col]) and not isinstance(constant_val, (bool, np.bool_)):
            df[col] = df[col].astype(object).fillna(constant_val)
        else:
            df[col] = df[col].fillna(constant_val)
    elif strategy == MissingValueStrategy.FORWARD_FILL:
        df[col] = df[col].ffill()
    elif strategy == MissingValueStrategy.BACKWARD_FILL:
        df[col] = df[col].bfill()
    else:
        return 0

    nulls_after = int(df[col].isna().sum())
    return max(0, nulls_before - nulls_after)


class DataCleaner:
    """Stateful service wrapper for Module 2 data cleaning and preprocessing."""

    def recommend(self, profile: DatasetProfileResult) -> CleaningConfig:
        return recommend_cleaning_config(profile)

    def clean(
        self,
        df: pd.DataFrame,
        config: CleaningConfig | None = None,
        profile: DatasetProfileResult | None = None,
        dataset_name: str = "cleaned_dataset",
    ) -> CleaningResult:
        return clean_dataset(df=df, config=config, profile=profile, dataset_name=dataset_name)
