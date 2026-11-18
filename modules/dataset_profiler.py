"""Module 1: Dataset Upload and Profiling for SmartPrompt AI.

Provides validated CSV/Excel ingestion, semantic column type detection,
column-wise statistical summaries, missing-value & duplicate diagnostics,
IQR/Z-score outlier detection, and structured dataset quality reporting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import io
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pandas as pd

from config.logging_config import get_logger
from config.settings import get_settings

logger = get_logger(__name__)

SUPPORTED_CSV_EXTENSIONS: frozenset[str] = frozenset({".csv"})
SUPPORTED_EXCEL_EXTENSIONS: frozenset[str] = frozenset({".xlsx", ".xls"})
SUPPORTED_EXTENSIONS: frozenset[str] = SUPPORTED_CSV_EXTENSIONS | SUPPORTED_EXCEL_EXTENSIONS

_BOOLEAN_STRING_TOKENS: frozenset[str] = frozenset({"true", "false", "yes", "no"})


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------


class DatasetProfilerError(Exception):
    """Base exception for dataset upload and profiling errors."""


class UnsupportedFileFormatError(DatasetProfilerError):
    """Raised when the uploaded or provided file extension is unsupported."""


class FileSizeLimitError(DatasetProfilerError):
    """Raised when a file is empty (0 bytes) or exceeds the configured size limit."""


class EmptyDatasetError(DatasetProfilerError):
    """Raised when the parsed dataset contains no rows or no columns."""


class DatasetLoadError(DatasetProfilerError):
    """Raised when Pandas/OpenPyXL fails to parse a file safely."""


# ---------------------------------------------------------------------------
# Structured Output Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnProfile:
    """Column-level metadata, inferred semantic type, and summary statistics."""

    name: str
    inferred_type: str  # "numerical" | "categorical" | "datetime" | "boolean"
    pandas_dtype: str
    non_null_count: int
    missing_count: int
    missing_pct: float
    unique_count: int
    sample_values: list[Any]
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MissingValueSummary:
    """Dataset-wide missing value diagnostics."""

    total_cells: int
    total_missing_cells: int
    overall_missing_pct: float
    rows_with_missing_count: int
    columns_with_missing: dict[str, dict[str, float | int]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DuplicateSummary:
    """Dataset-wide duplicate record diagnostics."""

    duplicate_row_count: int
    duplicate_row_pct: float
    sample_duplicate_indices: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OutlierSummary:
    """Statistical outlier diagnostics for a single numerical column."""

    column: str
    iqr_outlier_count: int
    iqr_outlier_pct: float
    iqr_lower_bound: float | None
    iqr_upper_bound: float | None
    zscore_outlier_count: int
    zscore_outlier_pct: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetQualityReport:
    """Overall dataset quality score, flags, and actionable findings."""

    quality_score: float
    quality_grade: str  # "Excellent" | "Good" | "Fair" | "Needs Attention"
    has_missing_values: bool
    has_duplicates: bool
    has_outliers: bool
    constant_columns: list[str]
    all_null_columns: list[str]
    issues: list[str]
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetProfileResult:
    """Complete structured dataset profile consumable by downstream modules."""

    dataset_name: str
    file_format: str
    file_size_bytes: int
    dataset_fingerprint: str
    row_count: int
    column_count: int
    memory_usage_mb: float
    sheet_name: str | None
    available_sheets: list[str]
    numerical_columns: list[str]
    categorical_columns: list[str]
    datetime_columns: list[str]
    boolean_columns: list[str]
    column_profiles: dict[str, ColumnProfile]
    missing_summary: MissingValueSummary
    duplicate_summary: DuplicateSummary
    outlier_summary: dict[str, OutlierSummary]
    quality_report: DatasetQualityReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "file_format": self.file_format,
            "file_size_bytes": self.file_size_bytes,
            "dataset_fingerprint": self.dataset_fingerprint,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "memory_usage_mb": self.memory_usage_mb,
            "sheet_name": self.sheet_name,
            "available_sheets": self.available_sheets,
            "numerical_columns": self.numerical_columns,
            "categorical_columns": self.categorical_columns,
            "datetime_columns": self.datetime_columns,
            "boolean_columns": self.boolean_columns,
            "column_profiles": {col: prof.to_dict() for col, prof in self.column_profiles.items()},
            "missing_summary": self.missing_summary.to_dict(),
            "duplicate_summary": self.duplicate_summary.to_dict(),
            "outlier_summary": {col: out.to_dict() for col, out in self.outlier_summary.items()},
            "quality_report": self.quality_report.to_dict(),
        }

    def schema_context_summary(self) -> dict[str, Any]:
        """Return a compact schema dictionary tailored for prompt optimization & NLP modules."""
        return {
            "dataset_name": self.dataset_name,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "numerical_columns": self.numerical_columns,
            "categorical_columns": self.categorical_columns,
            "datetime_columns": self.datetime_columns,
            "boolean_columns": self.boolean_columns,
            "columns": {
                name: {
                    "type": prof.inferred_type,
                    "missing_pct": prof.missing_pct,
                    "unique_count": prof.unique_count,
                    "sample_values": prof.sample_values[:3],
                }
                for name, prof in self.column_profiles.items()
            },
        }


@dataclass
class DatasetLoadResult:
    """Container pairing a loaded DataFrame with its source file metadata."""

    dataframe: pd.DataFrame
    dataset_name: str
    file_format: str
    file_size_bytes: int
    dataset_fingerprint: str
    sheet_name: str | None = None
    available_sheets: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper Utilities
# ---------------------------------------------------------------------------


def _to_python_scalar(value: Any) -> Any:
    """Convert NumPy/Pandas scalars and NaN/NaT values into JSON-safe Python primitives."""
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp,)):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        val = float(value)
        return None if (np.isnan(val) or np.isinf(val)) else round(val, 6)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _safe_float(value: Any, decimals: int = 4) -> float | None:
    """Safely convert numeric values to rounded floats, returning None for NaN/Inf."""
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(num) or np.isinf(num):
        return None
    return round(num, decimals)


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and deduplicate/sanitize column names cleanly."""
    cleaned: list[str] = []
    seen: dict[str, int] = {}
    for idx, raw_col in enumerate(df.columns):
        name = str(raw_col).strip()
        if not name or name.lower().startswith("unnamed:"):
            name = f"column_{idx + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        cleaned.append(name)

    normalized = df.copy()
    normalized.columns = cleaned
    return normalized


def _read_source_bytes(
    source: str | Path | BinaryIO,
    filename: str | None,
    max_file_size_mb: int,
) -> tuple[bytes, str, str]:
    """Validate file path/stream, extension, and size; return (raw_bytes, name, ext)."""
    max_bytes = max_file_size_mb * 1024 * 1024

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists() or not path.is_file():
            raise DatasetLoadError(f"Dataset file not found: '{path}'.")
        resolved_name = filename or path.name
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            supported_list = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise UnsupportedFileFormatError(
                f"Unsupported file format '{ext or 'none'}'. Supported formats: {supported_list}."
            )
        size_bytes = path.stat().st_size
        if size_bytes == 0:
            raise FileSizeLimitError(f"Dataset file '{resolved_name}' is empty (0 bytes).")
        if size_bytes > max_bytes:
            raise FileSizeLimitError(
                f"File '{resolved_name}' ({size_bytes / (1024 * 1024):.2f} MB) exceeds "
                f"maximum allowed size of {max_file_size_mb} MB."
            )
        raw_bytes = path.read_bytes()
        return raw_bytes, resolved_name, ext

    # File-like object / BinaryIO (e.g., Streamlit UploadedFile, FastAPI UploadFile.file, BytesIO)
    resolved_name = filename or getattr(source, "name", "")
    if not resolved_name:
        raise DatasetLoadError("Filename is required when loading from a binary stream.")
    ext = Path(resolved_name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        supported_list = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedFileFormatError(
            f"Unsupported file format '{ext or 'none'}'. Supported formats: {supported_list}."
        )

    if hasattr(source, "seek"):
        source.seek(0)
    raw_bytes = source.read()
    if not isinstance(raw_bytes, (bytes, bytearray)):
        raise DatasetLoadError("Stream source must return bytes.")
    raw_bytes = bytes(raw_bytes)

    size_bytes = len(raw_bytes)
    if size_bytes == 0:
        raise FileSizeLimitError(f"Uploaded file '{resolved_name}' is empty (0 bytes).")
    if size_bytes > max_bytes:
        raise FileSizeLimitError(
            f"Uploaded file '{resolved_name}' ({size_bytes / (1024 * 1024):.2f} MB) exceeds "
            f"maximum allowed size of {max_file_size_mb} MB."
        )
    return raw_bytes, resolved_name, ext


def _load_csv_bytes(raw_bytes: bytes) -> pd.DataFrame:
    """Decode and parse CSV bytes with automatic encoding fallback."""
    last_error: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            text_buffer = io.BytesIO(raw_bytes)
            df = pd.read_csv(text_buffer, encoding=encoding, sep=None, engine="python")
            return df
        except pd.errors.EmptyDataError as exc:
            raise EmptyDatasetError("CSV file contains no parseable columns or data.") from exc
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except Exception as exc:
            last_error = exc
            # Retry once with explicit comma separator if python sniffer choked on edge-case CSV
            try:
                df = pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding)
                return df
            except pd.errors.EmptyDataError as empty_exc:
                raise EmptyDatasetError("CSV file contains no parseable columns or data.") from empty_exc
            except Exception as inner_exc:
                last_error = inner_exc

    raise DatasetLoadError(f"Failed to parse CSV file: {last_error}") from last_error


def _load_excel_bytes(
    raw_bytes: bytes,
    ext: str,
    sheet_name: str | int | None = None,
) -> tuple[pd.DataFrame, str | None, list[str]]:
    """Load an Excel worksheet and return (dataframe, selected_sheet, available_sheets)."""
    engine = "openpyxl" if ext == ".xlsx" else None
    try:
        excel_file = pd.ExcelFile(io.BytesIO(raw_bytes), engine=engine)
    except Exception as exc:
        raise DatasetLoadError(f"Failed to open Excel workbook: {exc}") from exc

    sheets = [str(s) for s in excel_file.sheet_names]
    if not sheets:
        raise EmptyDatasetError("Excel workbook contains no worksheets.")

    if sheet_name is None:
        target_sheet: str | int = sheets[0]
    elif isinstance(sheet_name, int):
        if sheet_name < 0 or sheet_name >= len(sheets):
            raise DatasetLoadError(
                f"Requested sheet index {sheet_name} out of range (available: {len(sheets)})."
            )
        target_sheet = sheets[sheet_name]
    else:
        if sheet_name not in sheets:
            raise DatasetLoadError(
                f"Worksheet '{sheet_name}' not found. Available sheets: {sheets}."
            )
        target_sheet = sheet_name

    try:
        df = excel_file.parse(sheet_name=target_sheet)
    except Exception as exc:
        raise DatasetLoadError(f"Failed to parse Excel worksheet '{target_sheet}': {exc}") from exc

    return df, str(target_sheet), sheets


# ---------------------------------------------------------------------------
# Column Inference & Analytical Profiling Functions
# ---------------------------------------------------------------------------


def infer_column_type(series: pd.Series) -> str:
    """Infer semantic column type: numerical, boolean, datetime, or categorical."""
    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    if pd.api.types.is_numeric_dtype(series):
        return "numerical"

    non_null = series.dropna()
    if non_null.empty:
        return "categorical"

    str_values = non_null.astype(str).str.strip()
    lower_unique = set(str_values.str.lower().unique())
    if lower_unique and lower_unique.issubset(_BOOLEAN_STRING_TOKENS):
        return "boolean"

    # Attempt datetime detection only on string-like values that are not purely numeric
    sample = str_values.head(200)
    pure_numeric_ratio = float(sample.str.fullmatch(r"[+-]?\d+(\.\d+)?").mean())
    if pure_numeric_ratio < 0.5:
        try:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        except TypeError:  # Older pandas fallback without format="mixed"
            parsed = pd.to_datetime(sample, errors="coerce")
        valid_ratio = float(parsed.notna().mean())
        if valid_ratio >= 0.8:
            return "datetime"

    return "categorical"


def compute_column_summary(series: pd.Series, inferred_type: str) -> ColumnProfile:
    """Compute column-wise statistics tailored to the column's inferred semantic type."""
    total_rows = len(series)
    missing_count = int(series.isna().sum())
    non_null_count = int(total_rows - missing_count)
    missing_pct = round((missing_count / total_rows) * 100.0, 2) if total_rows > 0 else 0.0

    non_null = series.dropna()
    unique_count = int(non_null.nunique())
    sample_values = [_to_python_scalar(v) for v in non_null.head(5).tolist()]

    stats: dict[str, Any] = {}
    if inferred_type == "numerical" and not non_null.empty:
        numeric_series = (
            pd.to_numeric(non_null, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if not numeric_series.empty:
            q25 = float(numeric_series.quantile(0.25))
            q75 = float(numeric_series.quantile(0.75))
            iqr = q75 - q25
            std_val = float(numeric_series.std(ddof=1)) if len(numeric_series) > 1 else 0.0
            skew_val = float(numeric_series.skew()) if len(numeric_series) > 2 and std_val > 0 else 0.0
            stats = {
                "min": _safe_float(numeric_series.min()),
                "max": _safe_float(numeric_series.max()),
                "mean": _safe_float(numeric_series.mean()),
                "median": _safe_float(numeric_series.median()),
                "std": _safe_float(std_val),
                "q25": _safe_float(q25),
                "q75": _safe_float(q75),
                "iqr": _safe_float(iqr),
                "skewness": _safe_float(skew_val),
                "zero_count": int((numeric_series == 0).sum()),
                "negative_count": int((numeric_series < 0).sum()),
            }
    elif inferred_type == "boolean" and not non_null.empty:
        if pd.api.types.is_bool_dtype(non_null):
            true_count = int(non_null.sum())
        else:
            normalized = non_null.astype(str).str.strip().str.lower()
            true_count = int(normalized.isin({"true", "yes"}).sum())
        false_count = int(len(non_null) - true_count)
        stats = {
            "true_count": true_count,
            "false_count": false_count,
            "true_pct": round((true_count / len(non_null)) * 100.0, 2),
        }
    elif inferred_type == "datetime" and not non_null.empty:
        try:
            dt_series = pd.to_datetime(non_null, errors="coerce", format="mixed").dropna()
        except TypeError:
            dt_series = pd.to_datetime(non_null, errors="coerce").dropna()
        if not dt_series.empty:
            min_dt = dt_series.min()
            max_dt = dt_series.max()
            stats = {
                "min_date": min_dt.isoformat(),
                "max_date": max_dt.isoformat(),
                "span_days": int((max_dt - min_dt).days),
            }
    elif inferred_type == "categorical" and not non_null.empty:
        str_series = non_null.astype(str)
        value_counts = str_series.value_counts().head(5)
        top_values = {str(k): int(v) for k, v in value_counts.items()}
        mode_val = next(iter(top_values.keys()), None)
        stats = {
            "mode": mode_val,
            "top_values": top_values,
            "avg_length": _safe_float(str_series.str.len().mean(), decimals=2),
        }

    return ColumnProfile(
        name=str(series.name),
        inferred_type=inferred_type,
        pandas_dtype=str(series.dtype),
        non_null_count=non_null_count,
        missing_count=missing_count,
        missing_pct=missing_pct,
        unique_count=unique_count,
        sample_values=sample_values,
        stats=stats,
    )


def detect_missing_values(df: pd.DataFrame) -> MissingValueSummary:
    """Compute dataset-wide and column-level missing value statistics."""
    row_count, col_count = df.shape
    total_cells = int(row_count * col_count)
    if total_cells == 0:
        return MissingValueSummary(
            total_cells=0,
            total_missing_cells=0,
            overall_missing_pct=0.0,
            rows_with_missing_count=0,
            columns_with_missing={},
        )

    per_col_missing = df.isna().sum()
    total_missing = int(per_col_missing.sum())
    overall_pct = round((total_missing / total_cells) * 100.0, 2)
    rows_with_missing = int(df.isna().any(axis=1).sum())

    cols_detail: dict[str, dict[str, float | int]] = {}
    for col, count in per_col_missing.items():
        cnt = int(count)
        if cnt > 0:
            cols_detail[str(col)] = {
                "count": cnt,
                "percentage": round((cnt / row_count) * 100.0, 2) if row_count > 0 else 0.0,
            }

    return MissingValueSummary(
        total_cells=total_cells,
        total_missing_cells=total_missing,
        overall_missing_pct=overall_pct,
        rows_with_missing_count=rows_with_missing,
        columns_with_missing=cols_detail,
    )


def detect_duplicates(df: pd.DataFrame, max_sample_indices: int = 20) -> DuplicateSummary:
    """Detect exact duplicate rows and return count, percentage, and sample indices."""
    row_count = len(df)
    if row_count == 0 or df.shape[1] == 0:
        return DuplicateSummary(duplicate_row_count=0, duplicate_row_pct=0.0, sample_duplicate_indices=[])

    dup_mask = df.duplicated(keep="first")
    dup_count = int(dup_mask.sum())
    dup_pct = round((dup_count / row_count) * 100.0, 2)
    sample_indices = [int(idx) for idx in df.index[dup_mask][:max_sample_indices].tolist()]

    return DuplicateSummary(
        duplicate_row_count=dup_count,
        duplicate_row_pct=dup_pct,
        sample_duplicate_indices=sample_indices,
    )


def detect_statistical_outliers(
    df: pd.DataFrame,
    numerical_columns: list[str],
    iqr_multiplier: float = 1.5,
    zscore_threshold: float = 3.0,
) -> dict[str, OutlierSummary]:
    """Identify potential outliers per numerical column using IQR and Z-score methods."""
    results: dict[str, OutlierSummary] = {}
    for col in numerical_columns:
        series = (
            pd.to_numeric(df[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        n = len(series)
        if n < 4:
            results[col] = OutlierSummary(
                column=col,
                iqr_outlier_count=0,
                iqr_outlier_pct=0.0,
                iqr_lower_bound=None,
                iqr_upper_bound=None,
                zscore_outlier_count=0,
                zscore_outlier_pct=0.0,
            )
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1

        if iqr > 0:
            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr
            iqr_mask = (series < lower) | (series > upper)
            iqr_count = int(iqr_mask.sum())
            lower_rounded = _safe_float(lower)
            upper_rounded = _safe_float(upper)
        else:
            iqr_count = 0
            lower_rounded = _safe_float(q1)
            upper_rounded = _safe_float(q3)

        std_val = float(series.std(ddof=1))
        if std_val > 0:
            mean_val = float(series.mean())
            zscores = (series - mean_val).abs() / std_val
            z_count = int((zscores > zscore_threshold).sum())
        else:
            z_count = 0

        results[col] = OutlierSummary(
            column=col,
            iqr_outlier_count=iqr_count,
            iqr_outlier_pct=round((iqr_count / n) * 100.0, 2),
            iqr_lower_bound=lower_rounded,
            iqr_upper_bound=upper_rounded,
            zscore_outlier_count=z_count,
            zscore_outlier_pct=round((z_count / n) * 100.0, 2),
        )

    return results


def build_quality_report(
    row_count: int,
    column_count: int,
    column_profiles: dict[str, ColumnProfile],
    missing_summary: MissingValueSummary,
    duplicate_summary: DuplicateSummary,
    outlier_summary: dict[str, OutlierSummary],
) -> DatasetQualityReport:
    """Synthesize a deterministic quality score, flags, issues, and recommendations."""
    if row_count == 0 or column_count == 0:
        return DatasetQualityReport(
            quality_score=0.0,
            quality_grade="Needs Attention",
            has_missing_values=False,
            has_duplicates=False,
            has_outliers=False,
            constant_columns=[],
            all_null_columns=[],
            issues=["Dataset contains zero rows or columns."],
            recommendations=["Upload a non-empty dataset with at least one data row and column."],
        )

    all_null_cols = [name for name, prof in column_profiles.items() if prof.non_null_count == 0]
    constant_cols = [
        name for name, prof in column_profiles.items() if prof.non_null_count > 0 and prof.unique_count <= 1
    ]
    outlier_cols = [
        col for col, out in outlier_summary.items() if out.iqr_outlier_count > 0 or out.zscore_outlier_count > 0
    ]

    has_missing = missing_summary.total_missing_cells > 0
    has_duplicates = duplicate_summary.duplicate_row_count > 0
    has_outliers = len(outlier_cols) > 0

    # Transparent weighted quality scoring (starts at 100, bounded to [0, 100])
    score = 100.0
    score -= min(40.0, missing_summary.overall_missing_pct * 0.8)
    score -= min(25.0, duplicate_summary.duplicate_row_pct * 0.6)
    score -= min(20.0, (len(all_null_cols) / column_count) * 30.0)
    score -= min(10.0, (len(constant_cols) / column_count) * 15.0)
    if outlier_summary:
        high_outlier_cols = sum(1 for out in outlier_summary.values() if out.iqr_outlier_pct >= 5.0)
        score -= min(10.0, (high_outlier_cols / len(outlier_summary)) * 10.0)
    score = round(max(0.0, min(100.0, score)), 2)

    if score >= 90.0:
        grade = "Excellent"
    elif score >= 75.0:
        grade = "Good"
    elif score >= 55.0:
        grade = "Fair"
    else:
        grade = "Needs Attention"

    issues: list[str] = []
    recommendations: list[str] = []

    if has_missing:
        affected = ", ".join(list(missing_summary.columns_with_missing.keys())[:5])
        issues.append(
            f"Missing values detected: {missing_summary.total_missing_cells} cells "
            f"({missing_summary.overall_missing_pct}%) across {len(missing_summary.columns_with_missing)} column(s) "
            f"[{affected}]."
        )
        recommendations.append("Consider imputing or filtering missing values before model training.")

    if has_duplicates:
        issues.append(
            f"Duplicate records detected: {duplicate_summary.duplicate_row_count} row(s) "
            f"({duplicate_summary.duplicate_row_pct}%)."
        )
        recommendations.append("Review and deduplicate repeated rows to avoid skewed aggregations.")

    if all_null_cols:
        issues.append(f"All-null columns detected: {', '.join(all_null_cols)}.")
        recommendations.append("Drop completely empty columns before running analytical pipelines.")

    if constant_cols:
        issues.append(f"Constant (zero-variance) columns detected: {', '.join(constant_cols)}.")
        recommendations.append("Exclude zero-variance columns from correlation and ML anomaly models.")

    if has_outliers:
        issues.append(f"Statistical outliers detected in numerical column(s): {', '.join(outlier_cols[:5])}.")
        recommendations.append("Inspect flagged outlier values via the ML Anomaly Detection module.")

    if not issues:
        recommendations.append("Dataset quality is clean and ready for prompt-guided EDA and anomaly detection.")

    return DatasetQualityReport(
        quality_score=score,
        quality_grade=grade,
        has_missing_values=has_missing,
        has_duplicates=has_duplicates,
        has_outliers=has_outliers,
        constant_columns=constant_cols,
        all_null_columns=all_null_cols,
        issues=issues,
        recommendations=recommendations,
    )


# ---------------------------------------------------------------------------
# Public Module API
# ---------------------------------------------------------------------------


def load_dataset(
    source: str | Path | BinaryIO,
    filename: str | None = None,
    sheet_name: str | int | None = None,
    max_file_size_mb: int | None = None,
    allow_empty_rows: bool = False,
) -> DatasetLoadResult:
    """Validate and load a CSV or Excel dataset into a normalized Pandas DataFrame."""
    settings = get_settings()
    size_limit = max_file_size_mb if max_file_size_mb is not None else settings.max_upload_size_mb

    raw_bytes, resolved_name, ext = _read_source_bytes(source, filename=filename, max_file_size_mb=size_limit)
    file_hash = hashlib.sha256(raw_bytes).hexdigest()

    selected_sheet: str | None = None
    available_sheets: list[str] = []

    if ext in SUPPORTED_CSV_EXTENSIONS:
        df = _load_csv_bytes(raw_bytes)
        file_format = "csv"
    else:
        df, selected_sheet, available_sheets = _load_excel_bytes(raw_bytes, ext=ext, sheet_name=sheet_name)
        file_format = "excel"

    if df.shape[1] == 0:
        raise EmptyDatasetError(f"Dataset '{resolved_name}' contains no columns.")
    if df.shape[0] == 0 and not allow_empty_rows:
        raise EmptyDatasetError(f"Dataset '{resolved_name}' contains headers only and 0 data rows.")

    normalized_df = _normalize_column_names(df)
    logger.info(
        "Loaded dataset '%s' (%s, rows=%d, cols=%d, size=%d bytes).",
        resolved_name,
        file_format,
        normalized_df.shape[0],
        normalized_df.shape[1],
        len(raw_bytes),
    )

    return DatasetLoadResult(
        dataframe=normalized_df,
        dataset_name=resolved_name,
        file_format=file_format,
        file_size_bytes=len(raw_bytes),
        dataset_fingerprint=file_hash,
        sheet_name=selected_sheet,
        available_sheets=available_sheets,
    )


def profile_dataframe(
    df: pd.DataFrame,
    dataset_name: str = "in_memory_dataset",
    file_format: str = "dataframe",
    file_size_bytes: int = 0,
    dataset_fingerprint: str | None = None,
    sheet_name: str | None = None,
    available_sheets: list[str] | None = None,
) -> DatasetProfileResult:
    """Profile an in-memory Pandas DataFrame and return structured diagnostics."""
    normalized_df = _normalize_column_names(df) if df.shape[1] > 0 else df.copy()
    row_count, column_count = normalized_df.shape
    memory_mb = round(float(normalized_df.memory_usage(deep=True).sum()) / (1024 * 1024), 4)

    fingerprint = dataset_fingerprint
    if not fingerprint:
        payload = f"{dataset_name}:{row_count}:{column_count}:{list(normalized_df.columns)}".encode("utf-8")
        fingerprint = hashlib.sha256(payload).hexdigest()

    numerical_cols: list[str] = []
    categorical_cols: list[str] = []
    datetime_cols: list[str] = []
    boolean_cols: list[str] = []
    column_profiles: dict[str, ColumnProfile] = {}

    for col in normalized_df.columns:
        series = normalized_df[col]
        inferred_type = infer_column_type(series)
        profile = compute_column_summary(series, inferred_type=inferred_type)
        column_profiles[str(col)] = profile

        if inferred_type == "numerical":
            numerical_cols.append(str(col))
        elif inferred_type == "boolean":
            boolean_cols.append(str(col))
        elif inferred_type == "datetime":
            datetime_cols.append(str(col))
        else:
            categorical_cols.append(str(col))

    missing_summary = detect_missing_values(normalized_df)
    duplicate_summary = detect_duplicates(normalized_df)
    outlier_summary = detect_statistical_outliers(normalized_df, numerical_columns=numerical_cols)
    quality_report = build_quality_report(
        row_count=row_count,
        column_count=column_count,
        column_profiles=column_profiles,
        missing_summary=missing_summary,
        duplicate_summary=duplicate_summary,
        outlier_summary=outlier_summary,
    )

    logger.info(
        "Profiled dataset '%s': rows=%d, cols=%d, quality=%.2f (%s).",
        dataset_name,
        row_count,
        column_count,
        quality_report.quality_score,
        quality_report.quality_grade,
    )

    return DatasetProfileResult(
        dataset_name=dataset_name,
        file_format=file_format,
        file_size_bytes=file_size_bytes,
        dataset_fingerprint=fingerprint,
        row_count=int(row_count),
        column_count=int(column_count),
        memory_usage_mb=memory_mb,
        sheet_name=sheet_name,
        available_sheets=available_sheets or [],
        numerical_columns=numerical_cols,
        categorical_columns=categorical_cols,
        datetime_columns=datetime_cols,
        boolean_columns=boolean_cols,
        column_profiles=column_profiles,
        missing_summary=missing_summary,
        duplicate_summary=duplicate_summary,
        outlier_summary=outlier_summary,
        quality_report=quality_report,
    )


def load_and_profile_dataset(
    source: str | Path | BinaryIO,
    filename: str | None = None,
    sheet_name: str | int | None = None,
    max_file_size_mb: int | None = None,
    allow_empty_rows: bool = False,
) -> tuple[pd.DataFrame, DatasetProfileResult]:
    """Load a CSV/Excel dataset and immediately generate its structured profile."""
    loaded = load_dataset(
        source=source,
        filename=filename,
        sheet_name=sheet_name,
        max_file_size_mb=max_file_size_mb,
        allow_empty_rows=allow_empty_rows,
    )
    profile = profile_dataframe(
        loaded.dataframe,
        dataset_name=loaded.dataset_name,
        file_format=loaded.file_format,
        file_size_bytes=loaded.file_size_bytes,
        dataset_fingerprint=loaded.dataset_fingerprint,
        sheet_name=loaded.sheet_name,
        available_sheets=loaded.available_sheets,
    )
    return loaded.dataframe, profile


class DatasetProfiler:
    """Stateful service wrapper for loading and profiling datasets."""

    def __init__(self, max_file_size_mb: int | None = None) -> None:
        settings = get_settings()
        self.max_file_size_mb = max_file_size_mb if max_file_size_mb is not None else settings.max_upload_size_mb

    def load(
        self,
        source: str | Path | BinaryIO,
        filename: str | None = None,
        sheet_name: str | int | None = None,
        allow_empty_rows: bool = False,
    ) -> DatasetLoadResult:
        return load_dataset(
            source=source,
            filename=filename,
            sheet_name=sheet_name,
            max_file_size_mb=self.max_file_size_mb,
            allow_empty_rows=allow_empty_rows,
        )

    def profile(
        self,
        df: pd.DataFrame,
        dataset_name: str = "in_memory_dataset",
    ) -> DatasetProfileResult:
        return profile_dataframe(df, dataset_name=dataset_name)

    def load_and_profile(
        self,
        source: str | Path | BinaryIO,
        filename: str | None = None,
        sheet_name: str | int | None = None,
        allow_empty_rows: bool = False,
    ) -> tuple[pd.DataFrame, DatasetProfileResult]:
        return load_and_profile_dataset(
            source=source,
            filename=filename,
            sheet_name=sheet_name,
            max_file_size_mb=self.max_file_size_mb,
            allow_empty_rows=allow_empty_rows,
        )
