"""Module 11: Report Generation for SmartPrompt AI.

Builds evidence-grounded analytical reports and exports them to CSV and
professional ReportLab PDF formats. Includes dataset summary, original and
optimized prompts, key results, self-validation status, anomaly summary, and
verified statistical insights while excluding unsupported claims and handling
missing or corrupt chart images gracefully. Independent of Streamlit.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import io
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape as xml_escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from config.logging_config import get_logger
from modules.analysis_engine import AnalysisResult
from modules.anomaly_detector import AnomalyReport, INTERPRETATION_DISCLAIMER
from modules.dataset_profiler import DatasetProfileResult
from modules.prompt_optimizer import PromptOptimizationResult
from modules.validator import ValidationReport, ValidationStatus

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Exceptions & Structured Payloads
# ---------------------------------------------------------------------------


class ReportGenerationError(Exception):
    """Base exception for analytical report generation and export errors."""


@dataclass(frozen=True)
class AnalyticalReportPayload:
    """Normalized, Streamlit-independent payload consumed by CSV and PDF exporters."""

    report_title: str
    generated_at: str
    dataset_summary: dict[str, Any]
    original_prompt: str
    optimized_prompt: str
    detected_intent: str
    rewritten_by: str
    key_results: dict[str, Any]
    result_table: list[dict[str, Any]]
    validation_status: str
    validation_summary: dict[str, Any]
    excluded_unsupported_claims: list[str]
    statistical_insights: list[str]
    anomaly_summary: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReportBundle:
    """Container holding generated CSV and/or PDF report bytes and optional file paths."""

    payload: AnalyticalReportPayload
    csv_bytes: bytes | None = None
    pdf_bytes: bytes | None = None
    csv_path: str | None = None
    pdf_path: str | None = None


# ---------------------------------------------------------------------------
# Evidence-Only Statistical Insight Extraction (Suppresses Unsupported Claims)
# ---------------------------------------------------------------------------


def _extract_verified_insights(
    analysis_result: AnalysisResult | None,
    validation_report: ValidationReport | None,
    anomaly_report: AnomalyReport | None,
    additional_insights: list[str] | None,
) -> tuple[list[str], list[str]]:
    """Return (verified_insights, excluded_unsupported_descriptions)."""
    excluded: list[str] = []
    unsupported_tokens: list[str] = []

    if validation_report is not None:
        for chk in validation_report.checks:
            if chk.status == ValidationStatus.UNSUPPORTED.value:
                excluded.append(f"[{chk.check_name}] {chk.message}")
                if chk.column:
                    unsupported_tokens.append(chk.column.lower())
                if chk.reported_value is not None:
                    unsupported_tokens.append(str(chk.reported_value).lower())

    insights: list[str] = []

    if analysis_result is not None:
        if analysis_result.status != "completed":
            excluded.append(
                f"Analytical execution ended with status '{analysis_result.status}'; no analytical insights promoted."
            )
        else:
            ev = analysis_result.numerical_evidence
            summary = analysis_result.summary_metrics

            insights.append(
                f"Evaluated {ev.filtered_rows} record(s) out of {ev.input_rows} input row(s) "
                f"for '{analysis_result.intent}' analysis."
            )

            group_col = summary.get("group_column")
            metric_col = summary.get("metric_column")
            top_group = summary.get("top_group")
            top_value = summary.get("top_value")
            agg = summary.get("aggregation")
            if group_col and top_group is not None and top_value is not None:
                # Promote only if ranking check was not flagged as Unsupported
                ranking_unsupported = any(
                    chk.check_name.startswith("ranking:") and chk.status == ValidationStatus.UNSUPPORTED.value
                    for chk in (validation_report.checks if validation_report else [])
                )
                if not ranking_unsupported:
                    metric_desc = f"{agg}({metric_col})" if metric_col else (agg or "value")
                    insights.append(
                        f"Top {group_col} by {metric_desc} is '{top_group}' with value {top_value}."
                    )

            for col, total_val in ev.totals.items():
                col_unsupported = any(
                    chk.check_name == f"total:{col}" and chk.status == ValidationStatus.UNSUPPORTED.value
                    for chk in (validation_report.checks if validation_report else [])
                )
                if not col_unsupported:
                    avg_val = ev.averages.get(col)
                    med_val = ev.medians.get(col)
                    mm = ev.min_max.get(col, {})
                    parts = [f"total={total_val}"]
                    if avg_val is not None:
                        parts.append(f"mean={avg_val}")
                    if med_val is not None:
                        parts.append(f"median={med_val}")
                    if mm:
                        parts.append(f"range=[{mm.get('min')}, {mm.get('max')}]")
                    insights.append(f"Column '{col}': {', '.join(parts)}.")

            if ev.correlations:
                top_corr = ev.correlations[0]
                a, b, r_val = top_corr.get("column_a"), top_corr.get("column_b"), top_corr.get("pearson_r")
                corr_unsupported = any(
                    chk.check_name == f"correlation:{a}:{b}" and chk.status == ValidationStatus.UNSUPPORTED.value
                    for chk in (validation_report.checks if validation_report else [])
                )
                if a and b and r_val is not None and not corr_unsupported:
                    insights.append(
                        f"Strongest linear relationship observed between '{a}' and '{b}' (Pearson r = {r_val}, n = {top_corr.get('sample_pairs')})."
                    )

            if "overall_change_pct" in summary and summary["overall_change_pct"] is not None:
                insights.append(
                    f"Across {summary.get('period_count', 0)} period(s) ({summary.get('start_period')} to {summary.get('end_period')}), "
                    f"'{metric_col}' changed by {summary['overall_change_pct']}%."
                )

    if anomaly_report is not None:
        insights.append(
            f"Isolation Forest flagged {anomaly_report.anomaly_count} of {anomaly_report.rows_analyzed} analyzed row(s) "
            f"({anomaly_report.anomaly_percentage}%) across features {anomaly_report.features_used}."
        )

    # Filter caller-supplied insights carefully against unsupported flags or unverified prose
    if additional_insights:
        overall_unsupported = (
            validation_report is not None
            and validation_report.overall_status == ValidationStatus.UNSUPPORTED.value
        )
        for item in additional_insights:
            text = str(item).strip()
            if not text:
                continue
            low = text.lower()
            if "unsupported" in low or any(tok and tok in low for tok in unsupported_tokens):
                excluded.append(f"[filtered_insight] {text}")
                continue
            if overall_unsupported and validation_report and validation_report.verified_count == 0:
                excluded.append(f"[unverified_under_unsupported_status] {text}")
                continue
            insights.append(text)

    if not insights:
        insights.append("No verified statistical insights are available for this run.")

    return insights, excluded


# ---------------------------------------------------------------------------
# Payload Builder
# ---------------------------------------------------------------------------


def build_report_payload(
    profile: DatasetProfileResult | None = None,
    prompt_result: PromptOptimizationResult | None = None,
    analysis_result: AnalysisResult | None = None,
    validation_report: ValidationReport | None = None,
    anomaly_report: AnomalyReport | None = None,
    additional_insights: list[str] | None = None,
    report_title: str = "SmartPrompt AI — Analytical Report",
    original_prompt: str | None = None,
    optimized_prompt: str | None = None,
    detected_intent: str | None = None,
) -> AnalyticalReportPayload:
    """Assemble a unified, evidence-verified report payload from pipeline module outputs."""
    generated_at = datetime.now(timezone.utc).isoformat()

    if profile is not None:
        dataset_summary = {
            "dataset_name": profile.dataset_name,
            "file_format": profile.file_format,
            "row_count": profile.row_count,
            "column_count": profile.column_count,
            "quality_score": profile.quality_report.quality_score,
            "quality_grade": profile.quality_report.quality_grade,
            "numerical_columns": profile.numerical_columns,
            "categorical_columns": profile.categorical_columns,
            "datetime_columns": profile.datetime_columns,
            "boolean_columns": profile.boolean_columns,
            "overall_missing_pct": profile.missing_summary.overall_missing_pct,
            "duplicate_row_count": profile.duplicate_summary.duplicate_row_count,
        }
    else:
        ds_name = analysis_result.dataset_name if analysis_result else "dataset"
        row_cnt = analysis_result.numerical_evidence.input_rows if analysis_result else 0
        col_cnt = len(analysis_result.numerical_evidence.columns_used) if analysis_result else 0
        dataset_summary = {
            "dataset_name": ds_name,
            "file_format": "unknown",
            "row_count": row_cnt,
            "column_count": col_cnt,
            "quality_score": None,
            "quality_grade": "N/A",
        }

    resolved_orig_prompt = (
        original_prompt
        or (prompt_result.original_prompt if prompt_result else None)
        or "Not provided"
    )
    resolved_opt_prompt = (
        optimized_prompt
        or (prompt_result.improved_prompt if prompt_result else None)
        or "Not provided"
    )
    resolved_intent = (
        detected_intent
        or (prompt_result.detected_intent if prompt_result else None)
        or (analysis_result.intent if analysis_result else "unknown")
    )
    rewritten_by = prompt_result.rewritten_by if prompt_result else "local_rule_engine"

    if analysis_result is not None:
        key_results: dict[str, Any] = {
            "status": analysis_result.status,
            "intent": analysis_result.intent,
            "execution_time_ms": analysis_result.execution_time_ms,
            "summary_metrics": analysis_result.summary_metrics,
        }
        result_table = list(analysis_result.result_table)
    else:
        key_results = {"status": "not_run", "intent": resolved_intent, "summary_metrics": {}}
        result_table = []

    if validation_report is not None:
        val_status = validation_report.overall_status
        val_summary = {
            "overall_status": validation_report.overall_status,
            "total_checks": validation_report.total_checks,
            "verified_count": validation_report.verified_count,
            "warning_count": validation_report.warning_count,
            "unsupported_count": validation_report.unsupported_count,
            "limitations_note": validation_report.limitations_note,
            "checks": [c.to_dict() for c in validation_report.checks],
        }
    else:
        val_status = "Not Validated"
        val_summary = {
            "overall_status": val_status,
            "total_checks": 0,
            "verified_count": 0,
            "warning_count": 0,
            "unsupported_count": 0,
            "limitations_note": "Validation was not executed for this report.",
            "checks": [],
        }

    anom_summary: dict[str, Any] | None = None
    if anomaly_report is not None:
        anom_summary = {
            "algorithm": anomaly_report.algorithm,
            "contamination_setting": anomaly_report.contamination_setting,
            "features_used": anomaly_report.features_used,
            "rows_analyzed": anomaly_report.rows_analyzed,
            "anomaly_count": anomaly_report.anomaly_count,
            "anomaly_percentage": anomaly_report.anomaly_percentage,
            "anomalous_row_indices": anomaly_report.anomalous_row_indices[:25],
            "interpretation_note": anomaly_report.interpretation_note or INTERPRETATION_DISCLAIMER,
        }

    verified_insights, excluded_claims = _extract_verified_insights(
        analysis_result=analysis_result,
        validation_report=validation_report,
        anomaly_report=anomaly_report,
        additional_insights=additional_insights,
    )

    warnings: list[str] = []
    if prompt_result:
        warnings.extend(prompt_result.warnings)
    if analysis_result:
        warnings.extend(analysis_result.warnings)
    if anomaly_report:
        warnings.extend(anomaly_report.warnings)

    return AnalyticalReportPayload(
        report_title=report_title,
        generated_at=generated_at,
        dataset_summary=dataset_summary,
        original_prompt=resolved_orig_prompt,
        optimized_prompt=resolved_opt_prompt,
        detected_intent=resolved_intent,
        rewritten_by=rewritten_by,
        key_results=key_results,
        result_table=result_table,
        validation_status=val_status,
        validation_summary=val_summary,
        excluded_unsupported_claims=excluded_claims,
        statistical_insights=verified_insights,
        anomaly_summary=anom_summary,
        warnings=list(dict.fromkeys(warnings)),
    )


# ---------------------------------------------------------------------------
# CSV Export
# ---------------------------------------------------------------------------


def export_report_to_csv(
    payload: AnalyticalReportPayload,
    output_path: str | Path | None = None,
) -> bytes:
    """Export analytical summary, verified insights, validation status, and results table to CSV."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)

    writer.writerow(["Section", "Field", "Value"])
    writer.writerow(["Metadata", "Report Title", payload.report_title])
    writer.writerow(["Metadata", "Generated At (UTC)", payload.generated_at])

    for key, val in payload.dataset_summary.items():
        writer.writerow(["Dataset Summary", str(key), str(val)])

    writer.writerow(["Prompt Context", "Original Prompt", payload.original_prompt])
    writer.writerow(["Prompt Context", "Optimized Prompt", payload.optimized_prompt])
    writer.writerow(["Prompt Context", "Detected Intent", payload.detected_intent])
    writer.writerow(["Prompt Context", "Rewritten By", payload.rewritten_by])

    writer.writerow(["Validation", "Overall Status", payload.validation_status])
    writer.writerow(["Validation", "Total Checks", payload.validation_summary.get("total_checks", 0)])
    writer.writerow(["Validation", "Verified Checks", payload.validation_summary.get("verified_count", 0)])
    writer.writerow(["Validation", "Warning Checks", payload.validation_summary.get("warning_count", 0)])
    writer.writerow(["Validation", "Unsupported Checks", payload.validation_summary.get("unsupported_count", 0)])

    for key, val in (payload.key_results.get("summary_metrics") or {}).items():
        writer.writerow(["Key Results", str(key), str(val)])

    for idx, insight in enumerate(payload.statistical_insights, start=1):
        writer.writerow(["Statistical Insights", f"Insight {idx}", insight])

    if payload.excluded_unsupported_claims:
        for idx, exc in enumerate(payload.excluded_unsupported_claims, start=1):
            writer.writerow(["Excluded Unsupported Claims", f"Excluded {idx}", exc])

    if payload.anomaly_summary is not None:
        for key, val in payload.anomaly_summary.items():
            writer.writerow(["Anomaly Summary", str(key), str(val)])

    if payload.result_table:
        writer.writerow([])
        writer.writerow(["--- Result Table ---"])
        columns = list(payload.result_table[0].keys())
        writer.writerow(columns)
        for row in payload.result_table:
            writer.writerow([row.get(col, "") for col in columns])

    csv_bytes = buffer.getvalue().encode("utf-8")
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(csv_bytes)
        logger.info("Exported CSV report to '%s' (%d bytes).", target, len(csv_bytes))

    return csv_bytes


# ---------------------------------------------------------------------------
# PDF Export (ReportLab)
# ---------------------------------------------------------------------------


def _safe_load_chart_flowable(
    item: bytes | str | Path | dict[str, Any],
    caption_style: ParagraphStyle,
    note_style: ParagraphStyle,
    max_width: float = 6.2 * inch,
    max_height: float = 3.2 * inch,
) -> list[Any]:
    """Decode a chart image safely, falling back to a clear note if missing or corrupt."""
    caption: str | None = None
    source: Any = item
    if isinstance(item, dict):
        source = item.get("image")
        caption = item.get("caption")

    img_bytes: bytes | None = None
    label = caption or "Analytical Visualization"

    try:
        if isinstance(source, (bytes, bytearray)):
            img_bytes = bytes(source)
        elif isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists() or not path.is_file():
                return [Paragraph(f"<i>Chart unavailable ({xml_escape(str(label))}): file '{xml_escape(path.name)}' not found.</i>", note_style)]
            img_bytes = path.read_bytes()
        else:
            return [Paragraph(f"<i>Chart unavailable ({xml_escape(str(label))}): unsupported image source type.</i>", note_style)]

        if not img_bytes:
            return [Paragraph(f"<i>Chart unavailable ({xml_escape(str(label))}): empty image payload.</i>", note_style)]

        reader = ImageReader(io.BytesIO(img_bytes))
        iw, ih = reader.getSize()
        if iw <= 0 or ih <= 0:
            raise ValueError("Invalid image dimensions")

        scale = min(max_width / float(iw), max_height / float(ih), 1.0)
        flowables: list[Any] = [
            RLImage(io.BytesIO(img_bytes), width=iw * scale, height=ih * scale)
        ]
        if caption:
            flowables.append(Paragraph(xml_escape(str(caption)), caption_style))
        return flowables
    except Exception as exc:
        logger.warning("Skipping unreadable chart image (%s): %s", label, exc)
        return [Paragraph(f"<i>Chart image could not be embedded ({xml_escape(str(label))}); omitted safely.</i>", note_style)]


def _styled_kv_table(rows: list[tuple[str, str]], styles: Any) -> Table:
    cell_style = ParagraphStyle(
        "KVCell",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#1E293B"),
    )
    key_style = ParagraphStyle(
        "KVKey",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#0F172A"),
    )
    table_data = [[Paragraph(xml_escape(str(k)), key_style), Paragraph(xml_escape(str(v)), cell_style)] for k, v in rows]
    tbl = Table(table_data, colWidths=[2.0 * inch, 4.7 * inch])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1F5F9")),
                ("ROWBACKGROUNDS", (1, 0), (1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return tbl


def export_report_to_pdf(
    payload: AnalyticalReportPayload,
    chart_images: list[bytes | str | Path | dict[str, Any]] | None = None,
    output_path: str | Path | None = None,
    max_table_rows: int = 20,
) -> bytes:
    """Render a professional multi-section analytical PDF report via ReportLab."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title=payload.report_title,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=4,
        alignment=0,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#475569"),
        spaceAfter=14,
    )
    h2_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#1D4ED8"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=5,
    )
    caption_style = ParagraphStyle(
        "ChartCaption",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#475569"),
        spaceBefore=3,
        spaceAfter=8,
    )
    note_style = ParagraphStyle(
        "NoteText",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#64748B"),
        spaceAfter=6,
    )

    story: list[Any] = []

    # Header
    story.append(Paragraph(xml_escape(payload.report_title), title_style))
    story.append(
        Paragraph(
            f"Generated (UTC): {xml_escape(payload.generated_at)} | Validation Status: <b>{xml_escape(payload.validation_status)}</b>",
            subtitle_style,
        )
    )

    # 1. Dataset Summary
    story.append(Paragraph("1. Dataset Summary", h2_style))
    ds = payload.dataset_summary
    ds_rows = [
        ("Dataset Name", str(ds.get("dataset_name", "N/A"))),
        ("Dimensions", f"{ds.get('row_count', 0)} rows x {ds.get('column_count', 0)} columns"),
        ("Quality Score", f"{ds.get('quality_score', 'N/A')} ({ds.get('quality_grade', 'N/A')})"),
        ("Missing / Duplicates", f"{ds.get('overall_missing_pct', 0)}% missing | {ds.get('duplicate_row_count', 0)} duplicate rows"),
        ("Numerical Columns", ", ".join(ds.get("numerical_columns", [])) or "None"),
        ("Categorical Columns", ", ".join(ds.get("categorical_columns", [])) or "None"),
    ]
    story.append(_styled_kv_table(ds_rows, styles))

    # 2. Prompt Optimization Context
    story.append(Paragraph("2. Prompt Context & Optimization", h2_style))
    prompt_rows = [
        ("Original Prompt", payload.original_prompt),
        ("Optimized Prompt", payload.optimized_prompt),
        ("Detected Intent", f"{payload.detected_intent} (engine: {payload.rewritten_by})"),
    ]
    story.append(_styled_kv_table(prompt_rows, styles))

    # 3. Key Analytical Results
    story.append(Paragraph("3. Key Analytical Results", h2_style))
    summary_metrics = payload.key_results.get("summary_metrics") or {}
    if summary_metrics:
        metric_rows = [(str(k), str(v)) for k, v in list(summary_metrics.items())[:12]]
        story.append(_styled_kv_table(metric_rows, styles))
    else:
        story.append(Paragraph("No scalar summary metrics recorded for this analysis.", note_style))

    if payload.result_table:
        story.append(Spacer(1, 0.1 * inch))
        cols = list(payload.result_table[0].keys())[:6]
        tbl_header = [Paragraph(f"<b>{xml_escape(str(c))}</b>", note_style) for c in cols]
        tbl_rows = [tbl_header]
        for row in payload.result_table[:max_table_rows]:
            tbl_rows.append([Paragraph(xml_escape(str(row.get(c, ""))), note_style) for c in cols])
        col_width = (6.7 * inch) / max(len(cols), 1)
        data_table = Table(tbl_rows, colWidths=[col_width] * len(cols))
        data_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DBEAFE")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        story.append(data_table)

    # 4. Self-Validation Status
    story.append(Paragraph("4. Evidence-Based Validation Status", h2_style))
    vs = payload.validation_summary
    val_rows = [
        ("Overall Status", payload.validation_status),
        (
            "Check Breakdown",
            f"Total={vs.get('total_checks', 0)} | Verified={vs.get('verified_count', 0)} | "
            f"Warning={vs.get('warning_count', 0)} | Unsupported={vs.get('unsupported_count', 0)}",
        ),
        ("Methodology Note", str(vs.get("limitations_note", ""))),
    ]
    story.append(_styled_kv_table(val_rows, styles))

    if payload.excluded_unsupported_claims:
        story.append(Spacer(1, 0.06 * inch))
        story.append(
            Paragraph(
                f"<b>Excluded Unsupported Claims ({len(payload.excluded_unsupported_claims)}):</b> "
                "The following discrepancies or unverified items were excluded from statistical insights:",
                note_style,
            )
        )
        for item in payload.excluded_unsupported_claims[:10]:
            story.append(Paragraph(f"• {xml_escape(str(item))}", note_style))

    # 5. Anomaly Detection Summary
    story.append(Paragraph("5. ML Anomaly Detection Summary", h2_style))
    if payload.anomaly_summary is not None:
        am = payload.anomaly_summary
        anom_rows = [
            ("Algorithm", f"{am.get('algorithm', 'IsolationForest')} (contamination={am.get('contamination_setting')})"),
            ("Features Analyzed", ", ".join(am.get("features_used", [])) or "N/A"),
            ("Anomalies Detected", f"{am.get('anomaly_count', 0)} / {am.get('rows_analyzed', 0)} rows ({am.get('anomaly_percentage', 0)}%)"),
            ("Interpretation Note", str(am.get("interpretation_note", INTERPRETATION_DISCLAIMER))),
        ]
        story.append(_styled_kv_table(anom_rows, styles))
    else:
        story.append(Paragraph("ML anomaly detection was not executed in this workflow.", note_style))

    # 6. Verified Statistical Insights
    story.append(Paragraph("6. Verified Statistical Insights", h2_style))
    for insight in payload.statistical_insights:
        story.append(Paragraph(f"• {xml_escape(str(insight))}", body_style))

    # 7. Charts (Graceful missing/corrupt fallback)
    if chart_images:
        story.append(Paragraph("7. Analytical Visualizations", h2_style))
        for item in chart_images:
            story.extend(_safe_load_chart_flowable(item, caption_style=caption_style, note_style=note_style))
            story.append(Spacer(1, 0.08 * inch))

    try:
        doc.build(story)
    except Exception as exc:
        logger.error("ReportLab PDF generation failed: %s", exc)
        raise ReportGenerationError(f"Failed to build PDF report: {exc}") from exc

    pdf_bytes = buffer.getvalue()
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(pdf_bytes)
        logger.info("Exported PDF report to '%s' (%d bytes).", target, len(pdf_bytes))

    return pdf_bytes


# ---------------------------------------------------------------------------
# High-Level Service & Convenience Wrapper
# ---------------------------------------------------------------------------


def generate_report_bundle(
    profile: DatasetProfileResult | None = None,
    prompt_result: PromptOptimizationResult | None = None,
    analysis_result: AnalysisResult | None = None,
    validation_report: ValidationReport | None = None,
    anomaly_report: AnomalyReport | None = None,
    additional_insights: list[str] | None = None,
    chart_images: list[bytes | str | Path | dict[str, Any]] | None = None,
    export_csv: bool = True,
    export_pdf: bool = True,
    csv_output_path: str | Path | None = None,
    pdf_output_path: str | Path | None = None,
    report_title: str = "SmartPrompt AI — Analytical Report",
) -> ReportBundle:
    """Build payload and generate CSV and/or PDF report bytes in one call."""
    payload = build_report_payload(
        profile=profile,
        prompt_result=prompt_result,
        analysis_result=analysis_result,
        validation_report=validation_report,
        anomaly_report=anomaly_report,
        additional_insights=additional_insights,
        report_title=report_title,
    )
    csv_bytes = export_report_to_csv(payload, output_path=csv_output_path) if export_csv else None
    pdf_bytes = (
        export_report_to_pdf(payload, chart_images=chart_images, output_path=pdf_output_path)
        if export_pdf
        else None
    )
    return ReportBundle(
        payload=payload,
        csv_bytes=csv_bytes,
        pdf_bytes=pdf_bytes,
        csv_path=str(csv_output_path) if csv_output_path else None,
        pdf_path=str(pdf_output_path) if pdf_output_path else None,
    )


class ReportGenerator:
    """Stateful Streamlit-independent service wrapper for Module 11."""

    def build_payload(self, **kwargs: Any) -> AnalyticalReportPayload:
        return build_report_payload(**kwargs)

    def to_csv(self, payload: AnalyticalReportPayload, output_path: str | Path | None = None) -> bytes:
        return export_report_to_csv(payload=payload, output_path=output_path)

    def to_pdf(
        self,
        payload: AnalyticalReportPayload,
        chart_images: list[bytes | str | Path | dict[str, Any]] | None = None,
        output_path: str | Path | None = None,
    ) -> bytes:
        return export_report_to_pdf(payload=payload, chart_images=chart_images, output_path=output_path)

    def generate_bundle(self, **kwargs: Any) -> ReportBundle:
        return generate_report_bundle(**kwargs)
