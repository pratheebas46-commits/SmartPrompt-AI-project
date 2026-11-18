"""Unit tests for Unstructured Data to Structured Data Converter module."""

from __future__ import annotations

import io
import zipfile

import pytest

from modules.unstructured_converter import (
    EmptyOrCorruptedDocumentError,
    UnsupportedDocumentFormatError,
    convert_unstructured_bytes,
    export_converted_to_excel_bytes,
)


def _build_minimal_docx_bytes(xml_body: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "<Types></Types>")
        zf.writestr(
            "word/document.xml",
            f'<?xml version="1.0" encoding="UTF-8"?>'
            f'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f"<w:body>{xml_body}</w:body></w:document>",
        )
    return buf.getvalue()


def test_txt_key_value_conversion_preserves_missing_cells() -> None:
    sample_txt = (
        "Order_ID: ORD-101\n"
        "Region: North\n"
        "Revenue: 1450\n\n"
        "Order_ID: ORD-102\n"
        "Region: South\n"
    ).encode("utf-8")

    res = convert_unstructured_bytes(sample_txt, "orders_log.txt")
    assert res.source_file_type == "TXT"
    assert res.columns == ["Order_ID", "Region", "Revenue"]
    assert len(res.rows) == 2
    assert res.rows[0]["Revenue"] == "1450"
    assert res.rows[1]["Revenue"] == ""  # missing cell preserved, never invented
    assert res.missing_cell_count == 1


def test_docx_table_conversion_and_excel_roundtrip() -> None:
    xml_body = (
        "<w:tbl>"
        "<w:tr><w:tc><w:p><w:r><w:t>Product</w:t></w:r></w:p></w:tc>"
        "<w:tc><w:p><w:r><w:t>Sales</w:t></w:r></w:p></w:tc></w:tr>"
        "<w:tr><w:tc><w:p><w:r><w:t>Laptop</w:t></w:r></w:p></w:tc>"
        "<w:tc><w:p><w:r><w:t>2400</w:t></w:r></w:p></w:tc></w:tr>"
        "<w:tr><w:tc><w:p><w:r><w:t>Monitor</w:t></w:r></w:p></w:tc>"
        "<w:tc><w:p><w:r><w:t></w:t></w:r></w:p></w:tc></w:tr>"
        "</w:tbl>"
    )
    docx_bytes = _build_minimal_docx_bytes(xml_body)
    res = convert_unstructured_bytes(docx_bytes, "products.docx")
    assert res.source_file_type == "DOCX"
    assert res.columns == ["Product", "Sales"]
    assert res.rows[0]["Sales"] == "2400"
    assert res.rows[1]["Sales"] == ""

    xlsx_bytes = export_converted_to_excel_bytes(res)
    assert xlsx_bytes[:2] == b"PK"  # genuine zip-based .xlsx file


def test_rejects_unsupported_empty_and_corrupted_files() -> None:
    with pytest.raises(UnsupportedDocumentFormatError):
        convert_unstructured_bytes(b"abc", "image.png")

    with pytest.raises(EmptyOrCorruptedDocumentError):
        convert_unstructured_bytes(b"", "empty.txt")

    with pytest.raises(EmptyOrCorruptedDocumentError):
        convert_unstructured_bytes(b"not-a-valid-zip", "corrupt.docx")
