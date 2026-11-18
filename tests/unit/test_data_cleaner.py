"""Unit tests for Module 2: Data Cleaning and Preprocessing (modules/data_cleaner.py)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from modules.data_cleaner import (
    CategoricalCaseMode,
    CategoricalEncodingStrategy,
    CleaningConfig,
    ColumnNotFoundError,
    DataCleaner,
    InvalidCleaningStrategyError,
    MissingValueStrategy,
    NumericScalingStrategy,
    TargetDataType,
    clean_dataset,
    recommend_cleaning_config,
)
from modules.dataset_profiler import profile_dataframe


@pytest.fixture()
def messy_dataframe() -> pd.DataFrame:
    """Create a messy DataFrame with missing values, duplicates, invalid strings, and dates."""
    return pd.DataFrame(
        {
            "order_date": ["2026-01-01", "2026-01-02", "not-a-date", "2026-01-04", "2026-01-04"],
            "revenue": ["$1,000.00", "2000.50", "invalid_num", None, None],
            "units": [10.0, 20.0, np.nan, 40.0, 40.0],
            "region": [" north ", "SOUTH", None, "East ", "East "],
            "is_active": ["true", "false", "maybe", "yes", "yes"],
        }
    )


def test_default_config_is_non_destructive_and_preserves_original(messy_dataframe: pd.DataFrame) -> None:
    snapshot = messy_dataframe.copy(deep=True)
    result = clean_dataset(messy_dataframe)

    # Original DataFrame must remain completely unmodified
    pd.testing.assert_frame_equal(messy_dataframe, snapshot)
    # Default config makes zero destructive modifications
    pd.testing.assert_frame_equal(result.cleaned_df, snapshot)
    assert result.report.was_modified is False
    assert result.report.rows_removed == 0
    assert result.report.duplicates_removed == 0
    assert result.report.missing_values_before == result.report.missing_values_after


def test_missing_value_imputation_and_drop_strategies() -> None:
    df = pd.DataFrame(
        {
            "score": [10.0, 20.0, np.nan, 30.0],
            "category": ["A", "B", None, "A"],
            "mostly_empty": [np.nan, np.nan, np.nan, 1.0],
        }
    )
    original = df.copy(deep=True)

    config = CleaningConfig(
        numeric_missing_strategy=MissingValueStrategy.MEDIAN,
        categorical_missing_strategy=MissingValueStrategy.MODE,
        column_missing_strategies={"mostly_empty": MissingValueStrategy.DROP_COLUMNS},
        drop_column_missing_threshold=0.5,
    )
    result = clean_dataset(df, config=config)

    pd.testing.assert_frame_equal(df, original)
    assert "mostly_empty" not in result.cleaned_df.columns
    assert "mostly_empty" in result.report.columns_removed
    assert result.cleaned_df["score"].isna().sum() == 0
    assert result.cleaned_df["score"].iloc[2] == 20.0
    assert result.cleaned_df["category"].iloc[2] == "A"
    assert result.report.missing_cells_filled == 2
    assert result.report.missing_values_after == 0


def test_opt_in_duplicate_removal(messy_dataframe: pd.DataFrame) -> None:
    df = pd.DataFrame(
        {
            "id": [1, 2, 2, 3],
            "value": [10, 20, 20, 30],
        }
    )
    # Without opt-in, duplicates stay
    no_dup_result = clean_dataset(df, CleaningConfig(remove_duplicates=False))
    assert len(no_dup_result.cleaned_df) == 4
    assert no_dup_result.report.duplicates_removed == 0

    # With opt-in, duplicate is removed
    dup_result = clean_dataset(df, CleaningConfig(remove_duplicates=True))
    assert len(dup_result.cleaned_df) == 3
    assert dup_result.report.duplicates_removed == 1
    assert dup_result.report.rows_removed == 1


def test_safe_type_and_datetime_conversion_with_invalid_data(messy_dataframe: pd.DataFrame) -> None:
    config = CleaningConfig(
        type_conversions={
            "revenue": TargetDataType.NUMERICAL,
            "order_date": TargetDataType.DATETIME,
            "is_active": TargetDataType.BOOLEAN,
        },
        strip_numeric_symbols=True,
    )
    result = clean_dataset(messy_dataframe, config=config)

    cleaned = result.cleaned_df
    assert pd.api.types.is_numeric_dtype(cleaned["revenue"])
    assert cleaned["revenue"].iloc[0] == 1000.0
    assert pd.isna(cleaned["revenue"].iloc[2])

    assert pd.api.types.is_datetime64_any_dtype(cleaned["order_date"])
    assert pd.isna(cleaned["order_date"].iloc[2])

    assert pd.api.types.is_bool_dtype(cleaned["is_active"])
    assert bool(cleaned["is_active"].iloc[0]) is True
    assert bool(cleaned["is_active"].iloc[1]) is False
    assert pd.isna(cleaned["is_active"].iloc[2])

    # Audit trail records coerced invalid values and warnings
    conv_map = {item.column: item for item in result.report.type_conversions}
    assert conv_map["revenue"].invalid_coerced_to_null == 1
    assert conv_map["order_date"].invalid_coerced_to_null == 1
    assert conv_map["is_active"].invalid_coerced_to_null == 1
    assert len(result.report.warnings) >= 3


def test_numerical_and_categorical_preprocessing_and_report_json() -> None:
    df = pd.DataFrame(
        {
            "amount": [10.0, 12.0, 11.0, 13.0, 1000.0],
            "segment": ["  retail ", "RETAIL", "enterprise", "smb", "rare_vip"],
        }
    )
    config = CleaningConfig(
        numeric_scaling=NumericScalingStrategy.CLIP_OUTLIERS_IQR,
        strip_categorical_whitespace=True,
        categorical_case=CategoricalCaseMode.LOWER,
        categorical_encoding=CategoricalEncodingStrategy.ONE_HOT,
    )
    result = clean_dataset(df, config=config)

    # Outlier 1000.0 should be clipped down to IQR upper bound (< 1000)
    assert result.cleaned_df["amount"].max() < 1000.0
    assert "segment" in result.report.columns_removed
    assert len(result.report.columns_added) >= 3
    assert result.updated_profile is not None

    serialized = json.dumps(result.report.to_dict(), allow_nan=False)
    assert "amount:clip_outliers_iqr" in serialized


def test_recommend_cleaning_config_from_module1_profile(messy_dataframe: pd.DataFrame) -> None:
    profile = profile_dataframe(messy_dataframe, dataset_name="messy.csv")
    cleaner = DataCleaner()
    rec_config = recommend_cleaning_config(profile)

    assert isinstance(rec_config, CleaningConfig)
    assert rec_config.remove_duplicates is True
    result = cleaner.clean(messy_dataframe, config=rec_config, profile=profile)
    assert result.report.duplicates_removed == 1


def test_invalid_strategy_or_missing_column_raises_clean_errors(messy_dataframe: pd.DataFrame) -> None:
    with pytest.raises(InvalidCleaningStrategyError):
        clean_dataset(messy_dataframe, CleaningConfig(numeric_missing_strategy="magic_fill"))

    with pytest.raises(ColumnNotFoundError):
        clean_dataset(
            messy_dataframe,
            CleaningConfig(type_conversions={"non_existent_col": TargetDataType.NUMERICAL}),
        )

    # Empty DataFrame handled safely
    empty_df = pd.DataFrame({"a": pd.Series(dtype="float64"), "b": pd.Series(dtype="object")})
    empty_res = clean_dataset(empty_df, CleaningConfig(numeric_missing_strategy="mean"))
    assert empty_res.cleaned_df.empty
    assert empty_res.report.was_modified is False
