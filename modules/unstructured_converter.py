"""Independent Module: Unstructured Data to Structured Data Converter.

Extracts text and tables from PDF (.pdf), Word (.docx), and Text (.txt) files,
organizes extracted content into structured rows and columns without inventing
missing values, and generates genuine Excel (.xlsx) workbooks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import io
from pathlib import Path
import re
from typing import Any
import xml.etree.ElementTree as ET
import zipfile

SUPPORTED_UNSTRUCTURED_EXTENSIONS: tuple[str, ...] = (".pdf", ".docx", ".txt")


class UnstructuredConverterError(Exception):
    """Base error for unstructured file extraction and conversion."""


class UnsupportedDocumentFormatError(UnstructuredConverterError):
    """Raised when the uploaded document format is not .pdf, .docx, or .txt."""


class EmptyOrCorruptedDocumentError(UnstructuredConverterError):
    """Raised when the uploaded document is empty, password-protected, or unreadable."""


@dataclass(frozen=True)
class ConvertedDocumentResult:
    """Structured tabular dataset converted from an unstructured document."""

    source_file_name: str
    source_file_type: str
    file_size_bytes: int
    extraction_strategy: str
    columns: list[str]
    rows: list[dict[str, str]]
    raw_text_preview: str
    raw_character_count: int
    missing_cell_count: int
    warnings: list[str]
    extracted_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_header(raw: str, idx: int) -> str:
    cleaned = re.sub(r"[^\w\s-]", " ", raw.replace("\r", " ").replace("\n", " "))
    cleaned = re.sub(r"\s+", "_", cleaned.strip())
    return cleaned or f"Column_{idx + 1}"


def _deduplicate_headers(raw_headers: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    out: list[str] = []
    for idx, raw in enumerate(raw_headers):
        base = _normalize_header(raw, idx)
        n = counts.get(base, 0)
        counts[base] = n + 1
        out.append(base if n == 0 else f"{base}_{n + 1}")
    return out


def _clean_cell(val: Any) -> str:
    if val is None:
        return ""
    return re.sub(r"\s+", " ", str(val)).strip()


def _extract_txt(raw_bytes: bytes) -> tuple[str, list[list[list[str]]], list[str]]:
    if not raw_bytes:
        raise EmptyOrCorruptedDocumentError("The uploaded text file is empty.")
    has_utf16 = raw_bytes[:2] in (b"\xff\xfe", b"\xfe\xff")
    if not has_utf16 and raw_bytes[:4096].count(b"\x00") > 2:
        raise EmptyOrCorruptedDocumentError("The uploaded .txt file contains binary or corrupted data.")

    text: str | None = None
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            text = raw_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue

    if not text or not text.strip():
        raise EmptyOrCorruptedDocumentError("The uploaded text file contains no readable text.")
    return text, [], []


def _extract_docx(raw_bytes: bytes) -> tuple[str, list[list[list[str]]], list[str]]:
    # Try python-docx if installed, else clean stdlib zipfile XML extraction
    try:
        import docx  # type: ignore

        try:
            document = docx.Document(io.BytesIO(raw_bytes))
        except Exception as exc:
            raise EmptyOrCorruptedDocumentError(
                f"Unable to read this .docx document (it may be corrupted or encrypted): {exc}"
            ) from exc

        tables: list[list[list[str]]] = []
        for tbl in document.tables:
            mat: list[list[str]] = []
            for row in tbl.rows:
                cells = [_clean_cell(c.text) for c in row.cells]
                if any(cells):
                    mat.append(cells)
            if len(mat) >= 2:
                tables.append(mat)

        paragraphs = [_clean_cell(p.text) for p in document.paragraphs if _clean_cell(p.text)]
        table_lines = ["\t".join(r) for t in tables for r in t]
        combined = "\n".join(paragraphs + table_lines).strip()
        if not combined:
            raise EmptyOrCorruptedDocumentError("The Word (.docx) document contains no readable text or tables.")
        return combined, tables, []
    except ImportError:
        pass

    try:
        zf = zipfile.ZipFile(io.BytesIO(raw_bytes))
    except zipfile.BadZipFile as exc:
        raise EmptyOrCorruptedDocumentError("Unable to read this .docx archive; it appears corrupted.") from exc

    with zf:
        names = set(zf.namelist())
        if "EncryptedPackage" in names or "EncryptionInfo" in names:
            raise EmptyOrCorruptedDocumentError("This Word (.docx) file is password-protected and cannot be read.")
        if "word/document.xml" not in names:
            raise EmptyOrCorruptedDocumentError("Invalid .docx structure: missing word/document.xml.")
        xml_bytes = zf.read("word/document.xml")

    root = ET.fromstring(xml_bytes)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    tables = []
    for tbl_el in root.findall(".//w:tbl", ns):
        mat = []
        for tr_el in tbl_el.findall("./w:tr", ns):
            row_cells = []
            for tc_el in tr_el.findall("./w:tc", ns):
                texts = [t.text or "" for t in tc_el.findall(".//w:t", ns)]
                row_cells.append(_clean_cell("".join(texts)))
            if any(row_cells):
                mat.append(row_cells)
        if len(mat) >= 2:
            tables.append(mat)

    paragraphs = []
    for p_el in root.findall("./w:body/w:p", ns):
        texts = [t.text or "" for t in p_el.findall(".//w:t", ns)]
        txt = _clean_cell("".join(texts))
        if txt:
            paragraphs.append(txt)

    table_lines = ["\t".join(r) for t in tables for r in t]
    combined = "\n".join(paragraphs + table_lines).strip()
    if not combined:
        raise EmptyOrCorruptedDocumentError("The Word (.docx) document contains no readable text or tables.")
    return combined, tables, []


def _extract_pdf(raw_bytes: bytes) -> tuple[str, list[list[list[str]]], list[str]]:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:
        raise UnstructuredConverterError(
            "PDF extraction library 'pypdf' is not installed in the Python environment."
        ) from exc

    try:
        reader = PdfReader(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise EmptyOrCorruptedDocumentError(f"Unable to read PDF document (it may be corrupted): {exc}") from exc

    if reader.is_encrypted:
        raise EmptyOrCorruptedDocumentError(
            "This PDF document is password-protected. Please unlock it before uploading."
        )

    lines: list[str] = []
    warnings: list[str] = []
    max_pages = min(len(reader.pages), 200)
    if len(reader.pages) > max_pages:
        warnings.append(f"Processed first {max_pages} of {len(reader.pages)} pages.")

    for page in reader.pages[:max_pages]:
        txt = page.extract_text() or ""
        for line in txt.splitlines():
            if line.strip():
                lines.append(line.strip())

    combined = "\n".join(lines).strip()
    if not combined:
        raise EmptyOrCorruptedDocumentError(
            "No readable text could be extracted from this PDF (it may contain scanned images only)."
        )
    return combined, [], warnings


def _build_table_rows(matrix: list[list[str]], strategy: str) -> tuple[list[str], list[dict[str, str]], str]:
    max_cols = max(len(r) for r in matrix)
    raw_hdr = list(matrix[0]) + [""] * (max_cols - len(matrix[0]))
    headers = _deduplicate_headers(raw_hdr)
    rows: list[dict[str, str]] = []
    for raw_row in matrix[1:]:
        if all(_clean_cell(c) == "" for c in raw_row):
            continue
        row_dict: dict[str, str] = {}
        for idx, h in enumerate(headers):
            row_dict[h] = _clean_cell(raw_row[idx]) if idx < len(raw_row) else ""
        rows.append(row_dict)
    return headers, rows, strategy


def _structure_text_or_tables(
    raw_text: str,
    tables: list[list[list[str]]],
    warnings: list[str],
) -> tuple[list[str], list[dict[str, str]], str, list[str]]:
    if tables:
        best = max(tables, key=lambda t: len(t) * max(len(r) for r in t))
        headers, rows, strategy = _build_table_rows(best, "Extracted Native Document Table")
        if rows:
            if len(tables) > 1:
                warnings.append(f"Found {len(tables)} tables; converted the largest table ({len(rows)} rows).")
            return headers, rows, strategy, warnings

    non_empty = [l.rstrip() for l in raw_text.splitlines() if l.strip() and not re.match(r"^[-=_|+ ]+$", l.strip())]

    # Pipe or tab delimited tables
    pipe_lines = [l for l in non_empty if "|" in l]
    if len(pipe_lines) >= 3 and len(pipe_lines) >= len(non_empty) * 0.5:
        mat = [[_clean_cell(c) for c in ln.strip("|").split("|")] for ln in pipe_lines]
        mat = [r for r in mat if len(r) >= 2]
        if len(mat) >= 2:
            h, r, s = _build_table_rows(mat, "Detected Pipe-Delimited Table")
            if r:
                return h, r, s, warnings

    tab_lines = [l for l in non_empty if "\t" in l]
    if len(tab_lines) >= 3 and len(tab_lines) >= len(non_empty) * 0.5:
        mat = [[_clean_cell(c) for c in ln.split("\t")] for ln in tab_lines]
        mat = [r for r in mat if len(r) >= 2]
        if len(mat) >= 2:
            h, r, s = _build_table_rows(mat, "Detected Tab-Delimited Table")
            if r:
                return h, r, s, warnings

    # Key-value records
    kv_re = re.compile(r"^([A-Za-z][A-Za-z0-9\s/_#().-]{1,42})\s*[:=]\s*(.*)$")
    blocks: list[list[tuple[str, str]]] = []
    cur: list[tuple[str, str]] = []
    for ln in raw_text.splitlines():
        tr = ln.strip()
        if not tr or re.match(r"^[-=_*]{3,}$", tr):
            if cur:
                blocks.append(cur)
                cur = []
            continue
        m = kv_re.match(tr)
        if m:
            k, v = m.group(1).strip(), _clean_cell(m.group(2))
            if any(ek.lower() == k.lower() for ek, _ in cur):
                blocks.append(cur)
                cur = []
            cur.append((k, v))
    if cur:
        blocks.append(cur)

    total_pairs = sum(len(b) for b in blocks)
    if blocks and total_pairs >= 3 and total_pairs >= len(non_empty) * 0.3:
        if len(blocks) >= 2:
            key_order: list[str] = []
            seen_lower: set[str] = set()
            for b in blocks:
                for k, _ in b:
                    if k.lower() not in seen_lower:
                        seen_lower.add(k.lower())
                        key_order.append(k)
            headers = _deduplicate_headers(key_order)
            h_map = {k.lower(): headers[i] for i, k in enumerate(key_order)}
            rows = []
            for b in blocks:
                obj = {h: "" for h in headers}
                for k, v in b:
                    obj[h_map[k.lower()]] = v
                rows.append(obj)
            return headers, rows, "Structured Key-Value Records", warnings

        headers = ["Field", "Value", "Numeric_Value"]
        rows = []
        for k, v in blocks[0]:
            num_m = re.search(r"(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)", v)
            rows.append(
                {
                    "Field": k,
                    "Value": v,
                    "Numeric_Value": num_m.group(1).replace(",", "") if num_m else "",
                }
            )
        return headers, rows, "Single-Document Key-Value Attributes", warnings

    # Prose entity & metric extraction
    headers = ["Row_Number", "Section", "Date_Mentioned", "Numeric_Value", "Extracted_Text"]
    rows = []
    section = ""
    idx = 0
    for ln in non_empty:
        tr = ln.strip()
        if (
            len(tr) <= 60
            and not tr.endswith((".", "!", "?", ";"))
            and not re.search(r"[$€£₹%]", tr)
            and not re.search(r"\b\d{2,}\b", tr)
            and len(tr.split()) <= 6
            and not tr.startswith(("-", "*", "•"))
        ):
            section = tr.rstrip(":")
            continue
        idx += 1
        date_m = re.search(r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", tr)
        num_m = re.search(r"(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)", tr)
        rows.append(
            {
                "Row_Number": str(idx),
                "Section": section,
                "Date_Mentioned": date_m.group(1) if date_m else "",
                "Numeric_Value": num_m.group(1).replace(",", "") if num_m else "",
                "Extracted_Text": tr,
            }
        )

    if not rows:
        raise EmptyOrCorruptedDocumentError("Unable to extract meaningful structured rows from this document.")
    return headers, rows, "Entity & Fact Extraction from Unstructured Text", warnings


def convert_unstructured_bytes(
    raw_bytes: bytes,
    filename: str,
    max_upload_mb: int = 15,
) -> ConvertedDocumentResult:
    """Extract and structure data from .pdf, .docx, or .txt bytes without inventing missing cells."""
    safe_name = Path(filename or "").name
    ext = Path(safe_name).suffix.lower()
    if ext not in SUPPORTED_UNSTRUCTURED_EXTENSIONS:
        raise UnsupportedDocumentFormatError(
            f"Unsupported format '{ext or 'none'}'. Supported: {list(SUPPORTED_UNSTRUCTURED_EXTENSIONS)}."
        )
    if not raw_bytes:
        raise EmptyOrCorruptedDocumentError(f"Uploaded file '{safe_name}' is empty.")
    if len(raw_bytes) > max_upload_mb * 1024 * 1024:
        raise UnstructuredConverterError(f"File '{safe_name}' exceeds maximum size of {max_upload_mb} MB.")

    if ext == ".txt":
        raw_text, tables, warnings = _extract_txt(raw_bytes)
        file_type = "TXT"
    elif ext == ".docx":
        raw_text, tables, warnings = _extract_docx(raw_bytes)
        file_type = "DOCX"
    else:
        raw_text, tables, warnings = _extract_pdf(raw_bytes)
        file_type = "PDF"

    columns, rows, strategy, warnings = _structure_text_or_tables(raw_text, tables, warnings)
    missing_count = sum(1 for r in rows for c in columns if not r.get(c, "").strip())

    return ConvertedDocumentResult(
        source_file_name=safe_name,
        source_file_type=file_type,
        file_size_bytes=len(raw_bytes),
        extraction_strategy=strategy,
        columns=columns,
        rows=rows,
        raw_text_preview=raw_text[:4000],
        raw_character_count=len(raw_text),
        missing_cell_count=missing_count,
        warnings=warnings,
        extracted_at=datetime.now(timezone.utc).isoformat(),
    )


def export_converted_to_excel_bytes(result: ConvertedDocumentResult) -> bytes:
    """Export ConvertedDocumentResult to a genuine .xlsx workbook preserving blank missing cells."""
    import openpyxl  # type: ignore

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Structured_Data"

    ws.append(result.columns)
    for row in result.rows:
        out_row: list[Any] = []
        for col in result.columns:
            raw = (row.get(col) or "").strip()
            if raw == "":
                out_row.append(None)
            elif re.fullmatch(r"-?\d+(?:\.\d+)?", raw):
                out_row.append(float(raw) if "." in raw else int(raw))
            else:
                out_row.append(raw)
        ws.append(out_row)

    summary_ws = wb.create_sheet(title="Conversion_Summary")
    summary_ws.append(["Property", "Value"])
    summary_ws.append(["Source File", result.source_file_name])
    summary_ws.append(["Source Type", result.source_file_type])
    summary_ws.append(["Extraction Strategy", result.extraction_strategy])
    summary_ws.append(["Extracted Rows", len(result.rows)])
    summary_ws.append(["Extracted Columns", len(result.columns)])
    summary_ws.append(["Blank Cells Preserved", result.missing_cell_count])
    summary_ws.append(["Extracted At", result.extracted_at])

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
