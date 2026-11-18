"""Unit tests for Module 1: Dataset Upload and Profiling (modules/dataset_profiler.py)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import pytest

from modules.dataset_profiler import (
    DatasetLoadError,
    DatasetProfileResult,
    DatasetProfiler,
    EmptyDatasetError,
    FileSizeLimitError,
    UnsupportedFileFormatError,
    load_and_profile_dataset,
    load_dataset,
    profile_dataframe,
)


@pytest.fixture()
def sample_csv_path(tmp_path: Path) -> Path:
    """Create a realistic CSV dataset containing mixed types, missing values, duplicates, and an outlier."""
    csv_content = (
        "order_id,order_date,revenue,region,is_returned\n"
        "1,2026-01-01,100.0,North,false\n"
        "2,2026-01-02,110.0,South,false\n"
        "3,2026-01-03,105.0,East,true\n"
        "4,2026-01-04,115.0,North,false\n"
        "5,2026-01-05,120.0,West,false\n"
        "6,2026-01-06,98.0,East,false\n"
        "7,2026-01-07,102.0,South,true\n"
        "8,2026-01-08,108.0,North,false\n"
        "9,2026-01-09,112.0,West,false\n"
        "10,2026-01-10,104.0,East,false\n"
        "11,2026-01-11,109.0,North,false\n"
        "12,2026-01-12,107.0,South,false\n"
        "13,2026-01-13,5000.0,West,true\n"
        "13,2026-01-13,5000.0,West,true\n"
        "14,2026-01-14,,North,false\n"
    )
    file_path = tmp_path / "sales_sample.csv"
    file_path.write_text(csv_content, encoding="utf-8")
    return file_path


@pytest.fixture()
def sample_excel_path(tmp_path: Path) -> Path:
    """Create a multi-sheet Excel (.xlsx) workbook for testing."""
    file_path = tmp_path / "multi_sheet.xlsx"
    df_orders = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "amount": [10.5, 20.0, 30.5, 40.0],
            "active": [True, False, True, True],
        }
    )
    df_customers = pd.DataFrame(
        {
            "customer_id": ["C1", "C2"],
            "segment": ["Enterprise", "SMB"],
        }
    )
    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        df_orders.to_excel(writer, sheet_name="Orders", index=False)
        df_customers.to_excel(writer, sheet_name="Customers", index=False)
    return file_path


def test_load_and_profile_csv_complete(sample_csv_path: Path) -> None:
    df, profile = load_and_profile_dataset(sample_csv_path)

    assert isinstance(df, pd.DataFrame)
    assert isinstance(profile, DatasetProfileResult)
    assert profile.dataset_name == "sales_sample.csv"
    assert profile.file_format == "csv"
    assert profile.row_count == 15
    assert profile.column_count == 5

    # Column type detection
    assert set(profile.numerical_columns) == {"order_id", "revenue"}
    assert profile.datetime_columns == ["order_date"]
    assert profile.categorical_columns == ["region"]
    assert profile.boolean_columns == ["is_returned"]

    # Missing values detection
    assert profile.missing_summary.total_missing_cells == 1
    assert profile.missing_summary.rows_with_missing_count == 1
    assert "revenue" in profile.missing_summary.columns_with_missing
    assert profile.missing_summary.columns_with_missing["revenue"]["count"] == 1

    # Duplicate detection (row 13 duplicated once)
    assert profile.duplicate_summary.duplicate_row_count == 1
    assert profile.duplicate_summary.sample_duplicate_indices == [13]

    # Outlier detection (5000.0 vs ~107.0)
    rev_outliers = profile.outlier_summary["revenue"]
    assert rev_outliers.iqr_outlier_count >= 2
    assert rev_outliers.zscore_outlier_count >= 1

    # Column summaries
    rev_stats = profile.column_profiles["revenue"].stats
    assert rev_stats["min"] == 98.0
    assert rev_stats["max"] == 5000.0
    assert rev_stats["mean"] is not None

    bool_stats = profile.column_profiles["is_returned"].stats
    assert bool_stats["true_count"] == 4
    assert bool_stats["false_count"] == 11

    # Quality report and JSON serialization
    assert 0.0 <= profile.quality_report.quality_score <= 100.0
    assert profile.quality_report.has_missing_values is True
    assert profile.quality_report.has_duplicates is True
    assert profile.quality_report.has_outliers is True

    serialized = json.dumps(profile.to_dict(), allow_nan=False)
    assert "sales_sample.csv" in serialized
    schema_ctx = profile.schema_context_summary()
    assert schema_ctx["row_count"] == 15
    assert "revenue" in schema_ctx["columns"]


def test_load_excel_sheets(sample_excel_path: Path) -> None:
    # Default first sheet
    result_default = load_dataset(sample_excel_path)
    assert result_default.file_format == "excel"
    assert result_default.sheet_name == "Orders"
    assert result_default.available_sheets == ["Orders", "Customers"]
    assert list(result_default.dataframe.columns) == ["id", "amount", "active"]

    # Explicit second sheet
    df_cust, prof_cust = load_and_profile_dataset(sample_excel_path, sheet_name="Customers")
    assert prof_cust.sheet_name == "Customers"
    assert prof_cust.row_count == 2
    assert list(df_cust.columns) == ["customer_id", "segment"]

    # Invalid sheet name
    with pytest.raises(DatasetLoadError, match="not found"):
        load_dataset(sample_excel_path, sheet_name="NonExistent")


def test_load_from_binary_stream() -> None:
    stream = io.BytesIO(b"metric,category\n42,Alpha\n84,Beta\n")
    df, profile = load_and_profile_dataset(stream, filename="stream_upload.csv")

    assert df.shape == (2, 2)
    assert profile.dataset_name == "stream_upload.csv"
    assert profile.numerical_columns == ["metric"]
    assert profile.categorical_columns == ["category"]


def test_unsupported_extension_rejected(tmp_path: Path) -> None:
    bad_file = tmp_path / "payload.json"
    bad_file.write_text('{"a": 1}', encoding="utf-8")

    with pytest.raises(UnsupportedFileFormatError):
        load_dataset(bad_file)


def test_empty_and_oversized_files_rejected(tmp_path: Path) -> None:
    zero_byte = tmp_path / "empty.csv"
    zero_byte.write_bytes(b"")
    with pytest.raises(FileSizeLimitError, match="0 bytes"):
        load_dataset(zero_byte)

    oversized = tmp_path / "large.csv"
    oversized.write_bytes(b"a,b\n" + b"1,2\n" * 400_000)
    with pytest.raises(FileSizeLimitError, match="exceeds maximum"):
        load_dataset(oversized, max_file_size_mb=1)


def test_empty_rows_and_corrupt_files_handled_safely(tmp_path: Path) -> None:
    headers_only = tmp_path / "headers_only.csv"
    headers_only.write_text("col_a,col_b\n", encoding="utf-8")

    with pytest.raises(EmptyDatasetError, match="0 data rows"):
        load_dataset(headers_only)

    # Explicit allow_empty_rows=True returns valid 0-row profile without division errors
    df_empty, prof_empty = load_and_profile_dataset(headers_only, allow_empty_rows=True)
    assert df_empty.shape == (0, 2)
    assert prof_empty.row_count == 0
    assert prof_empty.quality_report.quality_score == 0.0
    assert "zero rows" in prof_empty.quality_report.issues[0].lower()

    # Corrupt xlsx bytes
    corrupt_xlsx = tmp_path / "corrupt.xlsx"
    corrupt_xlsx.write_bytes(b"not-a-valid-excel-archive")
    with pytest.raises(DatasetLoadError):
        load_dataset(corrupt_xlsx)

    # In-memory completely empty dataframe
    profiler = DatasetProfiler()
    empty_profile = profiler.profile(pd.DataFrame(), dataset_name="empty_df")
    assert empty_profile.row_count == 0
    assert empty_profile.column_count == 0
    assert empty_profile.quality_report.quality_grade == "Needs Attention"
