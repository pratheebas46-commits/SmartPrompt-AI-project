"""Module 13: SmartPrompt AI Streamlit Frontend and Full Backend Integration.

Connects strictly to the FastAPI backend (`backend/main.py`) via HTTP using
`SmartPromptAPIClient` without reimplementing any analytical, ML, validation,
database, or reporting logic locally in the frontend.
"""

from __future__ import annotations

import html
import json
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend.api_client import (
    BackendAPIError,
    BackendConnectionError,
    SmartPromptAPIClient,
    get_default_api_base_url,
)

SECTIONS: list[str] = [
    "Home Dashboard",
    "Dataset Upload",
    "Unstructured Data Converter",
    "Dataset Profiling",
    "Smart Prompt Optimizer",
    "Prompt Quality Score",
    "AI Data Analysis",
    "Anomaly Detection",
    "Self-Validation Results",
    "Interactive Visualizations",
    "Analysis History",
    "Report Download",
    "Settings",
]


# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------


def _configure_page() -> None:
    st.set_page_config(
        page_title="SmartPrompt AI — Intelligent Data Analyst",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
            max-width: 1320px;
        }
        .sp-kpi-card {
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 0.9rem 1.05rem;
            margin-bottom: 0.5rem;
        }
        .sp-kpi-label {
            font-size: 0.78rem;
            font-weight: 600;
            color: #475569;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.2rem;
        }
        .sp-kpi-value {
            font-size: 1.45rem;
            font-weight: 700;
            color: #0F172A;
            line-height: 1.2;
        }
        .sp-kpi-sub {
            font-size: 0.78rem;
            color: #64748B;
            margin-top: 0.2rem;
        }
        .sp-prompt-box {
            background: #FFFFFF;
            border: 1px solid #CBD5E1;
            border-radius: 8px;
            padding: 0.9rem 1rem;
            font-size: 0.93rem;
            color: #0F172A;
            min-height: 92px;
        }
        .sp-prompt-box-opt {
            background: #F0F9FF;
            border: 1px solid #7DD3FC;
            border-radius: 8px;
            padding: 0.9rem 1rem;
            font-size: 0.93rem;
            color: #0C4A6E;
            min-height: 92px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_session_state() -> None:
    defaults: dict[str, Any] = {
        "api_base_url": get_default_api_base_url(),
        "auth_page": "Home",
        "auth_token": None,
        "current_user": None,
        "auth_notice": None,
        "nav_section": "Home Dashboard",
        "dataset_id": None,
        "profile": None,
        "preview_rows": [],
        "columns": [],
        "cleaning_report": None,
        "user_prompt": "",
        "prompt_optimization": None,
        "prompt_score_orig": None,
        "prompt_score_opt": None,
        "analysis_run": None,
        "analysis_id": None,
        "anomaly_result": None,
        "validation_override": None,
        "report_csv_bytes": None,
        "report_csv_name": "smartprompt_report.csv",
        "report_pdf_bytes": None,
        "report_pdf_name": "smartprompt_report.pdf",
        "unstructured_result": None,
        "unstructured_xlsx_bytes": None,
        "unstructured_xlsx_name": "converted_structured.xlsx",
        "pref_default_export": "Excel (.xlsx)",
        "pref_notify_analysis": True,
        "pref_notify_report": True,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _get_client() -> SmartPromptAPIClient:
    return SmartPromptAPIClient(
        base_url=st.session_state["api_base_url"],
        auth_token=st.session_state.get("auth_token"),
    )


def _kpi_card(label: str, value: Any, subtitle: str = "") -> None:
    safe_label = html.escape(str(label))
    safe_value = html.escape(str(value))
    sub_html = f'<div class="sp-kpi-sub">{html.escape(str(subtitle))}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="sp-kpi-card">
            <div class="sp-kpi-label">{safe_label}</div>
            <div class="sp-kpi-value">{safe_value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_api_error(exc: Exception) -> None:
    if isinstance(exc, BackendConnectionError):
        st.error(f"Backend Connection Error: {exc}")
        st.info("Verify the FastAPI server is running (`python run_api.py`) or update the API Base URL under Settings.")
    elif isinstance(exc, BackendAPIError):
        st.error(f"API Error ({exc.error} — HTTP {exc.status_code}): {exc.detail}")
    else:
        st.error(f"Unexpected Error: {exc}")


def _require_dataset() -> bool:
    if not st.session_state["dataset_id"] or not st.session_state["profile"]:
        st.warning("No dataset is currently loaded. Go to **Dataset Upload** to upload a CSV or Excel file first.")
        return False
    return True


def _render_plotly_viz(viz: dict[str, Any] | None) -> None:
    if not viz:
        st.info("No visualization payload available yet.")
        return
    if not viz.get("is_renderable") or not viz.get("figure"):
        st.warning(f"Chart not rendered: {viz.get('reason') or 'Data is unsuitable for this chart type.'}")
        return
    try:
        fig = go.Figure(viz["figure"])
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.error(f"Failed to render Plotly figure: {exc}")


def _render_validation_panel(
    validation_report: dict[str, Any],
    verified_insights: list[str],
    excluded_claims: list[str],
) -> None:
    status_val = validation_report.get("overall_status", "Unknown")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _kpi_card("Validation Status", status_val)
    with c2:
        _kpi_card("Verified Checks", validation_report.get("verified_count", 0))
    with c3:
        _kpi_card("Warnings", validation_report.get("warning_count", 0))
    with c4:
        _kpi_card("Unsupported Claims", validation_report.get("unsupported_count", 0))

    if status_val == "Verified":
        st.success("All deterministic validation checks matched raw dataset calculations.")
    elif status_val == "Warning":
        st.warning("Numerical calculations matched, with sample-size or edge-case warnings.")
    else:
        st.error("One or more claims or checks were flagged as Unsupported.")

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Verified Insights")
        if verified_insights:
            for item in verified_insights:
                st.markdown(f"- {item}")
        else:
            st.caption("No verified insights generated.")
    with col_right:
        st.subheader("Excluded / Unsupported Claims")
        if excluded_claims:
            for item in excluded_claims:
                st.error(item)
        else:
            st.caption("No unsupported claims detected.")

    checks = validation_report.get("checks", [])
    if checks:
        st.subheader("Deterministic Validation Audit Table")
        st.dataframe(pd.DataFrame(checks), use_container_width=True, hide_index=True)

    note = validation_report.get("limitations_note")
    if note:
        st.caption(note)


# ---------------------------------------------------------------------------
# Sidebar Navigation & Health Probe
# ---------------------------------------------------------------------------


def _render_sidebar(client: SmartPromptAPIClient) -> tuple[str, dict[str, Any] | None]:
    with st.sidebar:
        st.title("SmartPrompt AI")
        st.caption("Dataset-Aware, Self-Validating Analyst")

        health_data: dict[str, Any] | None = None
        try:
            health_data = client.health()
            mode_label = str(health_data.get("execution_mode", "offline")).upper()
            db_label = str(health_data.get("database_backend", "sqlite"))
            st.success(f"Backend Connected ({mode_label} | {db_label})")
        except Exception as exc:
            st.error("Backend Disconnected")
            st.caption(str(exc))

        st.divider()
        section = st.radio(
            "Navigation",
            options=SECTIONS,
            index=SECTIONS.index(st.session_state["nav_section"]),
            key="sidebar_nav_radio",
        )
        st.session_state["nav_section"] = section

        st.divider()
        if st.session_state["profile"]:
            prof = st.session_state["profile"]
            st.markdown("**Active Dataset**")
            st.caption(
                f"{prof.get('dataset_name')} — {prof.get('row_count')} rows x "
                f"{prof.get('column_count')} cols (Quality: {prof.get('quality_report', {}).get('quality_score')})"
            )
        else:
            st.caption("No dataset loaded.")

        user = st.session_state.get("current_user")
        if user:
            st.divider()
            st.caption(f"Signed in as {user.get('full_name')} (@{user.get('username')})")
            if st.button("Logout", key="sidebar_logout_btn", use_container_width=True):
                try:
                    client.logout_user()
                except Exception:
                    pass
                st.session_state["auth_token"] = None
                st.session_state["current_user"] = None
                st.session_state["auth_page"] = "Home"
                st.rerun()

    return section, health_data


# ---------------------------------------------------------------------------
# 12 Frontend Sections
# ---------------------------------------------------------------------------


def _page_home(client: SmartPromptAPIClient, health_data: dict[str, Any] | None) -> None:
    st.header("Home Dashboard")
    st.caption("Dataset-aware prompt engineering, deterministic data analysis, ML anomaly detection, and self-validation.")

    prof = st.session_state["profile"]
    run = st.session_state["analysis_run"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _kpi_card(
            "Backend Status",
            "Online" if health_data else "Disconnected",
            f"API: {st.session_state['api_base_url']}",
        )
    with c2:
        _kpi_card(
            "Active Dataset",
            prof.get("dataset_name", "None") if prof else "None",
            f"{prof.get('row_count', 0)} rows" if prof else "Upload CSV/Excel to begin",
        )
    with c3:
        q_score = prof.get("quality_report", {}).get("quality_score", "—") if prof else "—"
        q_grade = prof.get("quality_report", {}).get("quality_grade", "") if prof else ""
        _kpi_card("Data Quality Score", q_score, q_grade)
    with c4:
        val_status = run["validation_report"].get("overall_status", "—") if run else "—"
        _kpi_card("Last Validation", val_status, f"Intent: {run['analysis_result'].get('intent', '—')}" if run else "No analysis run yet")

    st.subheader("Workflow Progress")
    steps = [
        ("1. Upload & Profile Dataset", st.session_state["profile"] is not None),
        ("2. Optimize & Score Prompt", st.session_state["prompt_optimization"] is not None),
        ("3. Execute & Validate Analysis", st.session_state["analysis_run"] is not None),
        ("4. Anomaly Scan (Optional)", st.session_state["anomaly_result"] is not None),
    ]
    cols = st.columns(4)
    for col, (label, done) in zip(cols, steps):
        with col:
            if done:
                st.success(label)
            else:
                st.info(label)

    if health_data:
        try:
            hist = client.list_history(limit=5)
            items = hist.get("items", [])
            st.subheader("Recent Analysis History")
            if items:
                summary_rows = [
                    {
                        "Dataset": it["dataset_name"],
                        "Intent": it["detected_intent"],
                        "Original Prompt": it["original_prompt"],
                        "Validation": it["validation_status"],
                        "Timestamp": it["created_at"],
                    }
                    for it in items
                ]
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
            else:
                st.caption("No saved analyses yet.")
        except Exception as exc:
            _render_api_error(exc)


def _page_upload(client: SmartPromptAPIClient) -> None:
    st.header("Dataset Upload & Preprocessing")

    with st.form("dataset_upload_form"):
        uploaded = st.file_uploader("Upload CSV or Excel Dataset", type=["csv", "xlsx", "xls"])
        sheet_name = st.text_input("Excel Sheet Name (optional)", value="")
        submitted = st.form_submit_button("Upload & Profile Dataset", type="primary")

    if submitted:
        if uploaded is None:
            st.warning("Please select a CSV or Excel file before submitting.")
        else:
            with st.spinner("Uploading and profiling dataset on FastAPI backend..."):
                try:
                    res = client.profile_dataset(
                        file_bytes=uploaded.getvalue(),
                        filename=uploaded.name,
                        sheet_name=sheet_name.strip() or None,
                    )
                    st.session_state["dataset_id"] = res["dataset_id"]
                    st.session_state["profile"] = res["profile"]
                    st.session_state["preview_rows"] = res["preview_rows"]
                    st.session_state["columns"] = res["columns"]
                    st.session_state["cleaning_report"] = None
                    st.session_state["prompt_optimization"] = None
                    st.session_state["analysis_run"] = None
                    st.session_state["anomaly_result"] = None
                    st.success(f"Loaded and profiled '{uploaded.name}' (session ID: {res['dataset_id']}).")
                except Exception as exc:
                    _render_api_error(exc)

    if st.session_state["preview_rows"]:
        st.subheader("Dataset Preview (Top 25 Rows)")
        st.dataframe(pd.DataFrame(st.session_state["preview_rows"]), use_container_width=True)

    if st.session_state["dataset_id"]:
        st.divider()
        st.subheader("Optional Data Cleaning & Preprocessing")
        with st.form("dataset_clean_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                num_strategy = st.selectbox(
                    "Numerical Missing Strategy",
                    options=["none", "median", "mean", "constant", "drop_rows"],
                )
            with c2:
                cat_strategy = st.selectbox(
                    "Categorical Missing Strategy",
                    options=["none", "mode", "constant", "drop_rows"],
                )
            with c3:
                remove_dups = st.checkbox("Remove Duplicate Rows", value=False)
                strip_ws = st.checkbox("Trim Categorical Whitespace", value=False)

            clean_btn = st.form_submit_button("Apply Cleaning Pipeline")

        if clean_btn:
            with st.spinner("Cleaning dataset via backend..."):
                try:
                    clean_res = client.clean_dataset(
                        {
                            "dataset_id": st.session_state["dataset_id"],
                            "numeric_missing_strategy": num_strategy,
                            "categorical_missing_strategy": cat_strategy,
                            "remove_duplicates": remove_dups,
                            "strip_categorical_whitespace": strip_ws,
                            "update_active_dataset": True,
                        }
                    )
                    st.session_state["cleaning_report"] = clean_res["cleaning_report"]
                    if clean_res.get("updated_profile"):
                        st.session_state["profile"] = clean_res["updated_profile"]
                    st.session_state["preview_rows"] = clean_res["preview_rows"]
                    st.session_state["columns"] = clean_res["columns"]
                    st.success("Dataset cleaned and active profile updated.")
                except Exception as exc:
                    _render_api_error(exc)

        if st.session_state["cleaning_report"]:
            rep = st.session_state["cleaning_report"]
            st.markdown("**Cleaning Summary**")
            st.json(
                {
                    "original_shape": rep.get("original_shape"),
                    "cleaned_shape": rep.get("cleaned_shape"),
                    "rows_removed": rep.get("rows_removed"),
                    "duplicates_removed": rep.get("duplicates_removed"),
                    "missing_cells_filled": rep.get("missing_cells_filled"),
                    "actions_taken": rep.get("actions_taken"),
                }
            )


def _page_profiling() -> None:
    st.header("Dataset Profiling & Quality Report")
    if not _require_dataset():
        return

    prof = st.session_state["profile"]
    qr = prof.get("quality_report", {})
    ms = prof.get("missing_summary", {})
    ds = prof.get("duplicate_summary", {})

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _kpi_card("Rows", prof.get("row_count", 0))
    with c2:
        _kpi_card("Columns", prof.get("column_count", 0))
    with c3:
        _kpi_card("Missing Cells %", f"{ms.get('overall_missing_pct', 0)}%", f"{ms.get('total_missing_cells', 0)} cells")
    with c4:
        _kpi_card("Quality Score", f"{qr.get('quality_score', 0)} / 100", qr.get("quality_grade", ""))

    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown(f"**Numerical ({len(prof.get('numerical_columns', []))})**")
        st.write(prof.get("numerical_columns", []))
    with t2:
        st.markdown(f"**Categorical ({len(prof.get('categorical_columns', []))})**")
        st.write(prof.get("categorical_columns", []))
    with t3:
        st.markdown(f"**Datetime / Boolean**")
        st.write(prof.get("datetime_columns", []) + prof.get("boolean_columns", []))

    st.subheader("Column-Wise Profiles")
    col_rows = []
    for name, cp in prof.get("column_profiles", {}).items():
        col_rows.append(
            {
                "Column": name,
                "Type": cp.get("inferred_type"),
                "Pandas Dtype": cp.get("pandas_dtype"),
                "Non-Null": cp.get("non_null_count"),
                "Missing": cp.get("missing_count"),
                "Missing %": cp.get("missing_pct"),
                "Unique": cp.get("unique_count"),
                "Sample Values": str(cp.get("sample_values", [])[:3]),
            }
        )
    st.dataframe(pd.DataFrame(col_rows), use_container_width=True, hide_index=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Quality Issues & Duplicates")
        st.write(f"Duplicate Rows: {ds.get('duplicate_row_count', 0)} ({ds.get('duplicate_row_pct', 0)}%)")
        for issue in qr.get("issues", []):
            st.warning(issue)
        if not qr.get("issues"):
            st.success("No major structural quality issues detected.")
    with col_b:
        st.subheader("Statistical Outliers (IQR / Z-Score)")
        outliers = prof.get("outlier_summary", {})
        if outliers:
            st.dataframe(pd.DataFrame(list(outliers.values())), use_container_width=True, hide_index=True)
        else:
            st.caption("No numerical columns for statistical outlier checks.")


def _page_prompt_optimizer(client: SmartPromptAPIClient, health_data: dict[str, Any] | None) -> None:
    st.header("Dataset-Aware Smart Prompt Optimizer")
    if not _require_dataset():
        return

    online_ai = bool(health_data and health_data.get("online_ai_enabled"))
    prompt_input = st.text_area(
        "Enter your analytical question",
        value=st.session_state["user_prompt"],
        placeholder="e.g., Top products? or Show monthly revenue trend",
        height=100,
    )
    use_online = st.checkbox(
        "Enhance rewrite with Online LLM (when server key is active)",
        value=False,
        disabled=not online_ai,
    )

    if st.button("Optimize Prompt", type="primary"):
        if not prompt_input.strip():
            st.warning("Please enter an analytical question.")
        else:
            st.session_state["user_prompt"] = prompt_input.strip()
            with st.spinner("Detecting intent and optimizing prompt against dataset schema..."):
                try:
                    res = client.optimize_prompt(
                        dataset_id=st.session_state["dataset_id"],
                        prompt=prompt_input.strip(),
                        use_online_llm=use_online,
                    )
                    st.session_state["prompt_optimization"] = res["optimization"]
                except Exception as exc:
                    _render_api_error(exc)

    opt = st.session_state["prompt_optimization"]
    if opt:
        c1, c2, c3 = st.columns(3)
        with c1:
            _kpi_card("Detected Intent", opt.get("detected_intent", "unknown"))
        with c2:
            _kpi_card("Intent Confidence", f"{round(float(opt.get('intent_confidence', 0)) * 100, 1)}%")
        with c3:
            _kpi_card("Optimizer Engine", opt.get("rewritten_by", "local_rule_engine"))

        st.subheader("Original vs. Optimized Prompt Comparison")
        comp_l, comp_r = st.columns(2)
        with comp_l:
            st.markdown("**Original Prompt**")
            st.markdown(
                f'<div class="sp-prompt-box">{html.escape(str(opt.get("original_prompt", "")))}</div>',
                unsafe_allow_html=True,
            )
        with comp_r:
            st.markdown("**Optimized Dataset-Aware Prompt**")
            st.markdown(
                f'<div class="sp-prompt-box-opt">{html.escape(str(opt.get("improved_prompt", "")))}</div>',
                unsafe_allow_html=True,
            )

        if opt.get("unsupported_references"):
            st.error(f"Unavailable / Unsupported References Flagged: {opt['unsupported_references']}")
        for w in opt.get("warnings", []):
            st.warning(w)

        exp_l, exp_r = st.columns(2)
        with exp_l:
            st.subheader("Why the Prompt Was Improved")
            for reason in opt.get("improvement_reasons", []):
                st.markdown(f"- {reason}")
        with exp_r:
            st.subheader("Missing Context Identified")
            missing = opt.get("missing_context", [])
            if missing:
                for m in missing:
                    st.markdown(f"- {m}")
            else:
                st.success("All required analytical slots were specified.")

        with st.expander("Structured Analytical Task Plan", expanded=False):
            st.json(opt.get("task_plan", {}))


def _page_prompt_score(client: SmartPromptAPIClient) -> None:
    st.header("ML-Based Prompt Quality Score")
    st.caption("Evaluated via Scikit-learn TF-IDF + Logistic Regression trained on labeled prompt quality data.")

    default_text = st.session_state["user_prompt"] or (
        st.session_state["prompt_optimization"].get("original_prompt", "")
        if st.session_state["prompt_optimization"]
        else ""
    )
    custom_prompt = st.text_area("Prompt to Evaluate", value=default_text, height=90)

    col_btn1, col_btn2 = st.columns(2)
    eval_single = col_btn1.button("Score Prompt Quality", type="primary")
    compare_opt = col_btn2.button(
        "Compare Original vs. Optimized Prompt",
        disabled=st.session_state["prompt_optimization"] is None,
    )

    if eval_single:
        if not custom_prompt.strip():
            st.warning("Please enter a prompt to score.")
        else:
            with st.spinner("Scoring prompt via ML model..."):
                try:
                    res = client.score_prompt(
                        prompt=custom_prompt.strip(),
                        dataset_id=st.session_state["dataset_id"],
                    )
                    st.session_state["prompt_score_orig"] = res["prediction"]
                    st.session_state["prompt_score_opt"] = None
                except Exception as exc:
                    _render_api_error(exc)

    if compare_opt and st.session_state["prompt_optimization"]:
        opt = st.session_state["prompt_optimization"]
        with st.spinner("Scoring both original and optimized prompts..."):
            try:
                r1 = client.score_prompt(prompt=opt["original_prompt"], dataset_id=st.session_state["dataset_id"])
                r2 = client.score_prompt(prompt=opt["improved_prompt"], dataset_id=st.session_state["dataset_id"])
                st.session_state["prompt_score_orig"] = r1["prediction"]
                st.session_state["prompt_score_opt"] = r2["prediction"]
            except Exception as exc:
                _render_api_error(exc)

    def _display_pred(title: str, pred: dict[str, Any]) -> None:
        st.subheader(title)
        k1, k2, k3 = st.columns(3)
        with k1:
            _kpi_card("Predicted Quality", pred.get("predicted_class"))
        with k2:
            _kpi_card("Quality Score", f"{pred.get('quality_score')} / 100")
        with k3:
            _kpi_card("Held-Out Test Acc.", f"{round(float(pred.get('model_accuracy_on_test', 0)) * 100, 1)}%")

        st.progress(min(max(float(pred.get("quality_score", 0)) / 100.0, 0.0), 1.0))
        st.markdown("**Engineered Features & Class Probabilities**")
        st.json(
            {
                "class_probabilities": pred.get("class_probabilities"),
                "extracted_features": pred.get("extracted_features"),
            }
        )
        st.markdown("**Actionable Recommendations**")
        for rec in pred.get("recommendations", []):
            st.markdown(f"- {rec}")

    if st.session_state["prompt_score_orig"] and st.session_state["prompt_score_opt"]:
        cl, cr = st.columns(2)
        with cl:
            _display_pred("Original Prompt Score", st.session_state["prompt_score_orig"])
        with cr:
            _display_pred("Optimized Prompt Score", st.session_state["prompt_score_opt"])
    elif st.session_state["prompt_score_orig"]:
        _display_pred("Prompt Quality Evaluation", st.session_state["prompt_score_orig"])


def _page_analysis(client: SmartPromptAPIClient, health_data: dict[str, Any] | None) -> None:
    st.header("AI Data Analysis Execution")
    if not _require_dataset():
        return

    online_ai = bool(health_data and health_data.get("online_ai_enabled"))
    default_q = st.session_state["user_prompt"] or (
        st.session_state["prompt_optimization"].get("original_prompt", "")
        if st.session_state["prompt_optimization"]
        else ""
    )

    with st.form("analysis_run_form"):
        question = st.text_input("Analytical Question", value=default_q, placeholder="e.g., Rank top 5 products by total revenue")
        c1, c2, c3, c4 = st.columns(4)
        chart_override = c1.selectbox(
            "Chart Type",
            options=["auto", "bar", "horizontal_bar", "line", "pie", "histogram", "scatter", "heatmap"],
        )
        prefer_pie = c2.checkbox("Prefer Pie (when suitable)", value=False)
        include_anom = c3.checkbox("Include Anomaly Scan", value=False)
        save_hist = c4.checkbox("Save to History", value=True)
        use_online = st.checkbox("Use Online LLM Rewrite", value=False, disabled=not online_ai)
        submitted = st.form_submit_button("Run Analysis & Self-Validation", type="primary")

    if submitted:
        if not question.strip():
            st.warning("Please enter an analytical question.")
        else:
            st.session_state["user_prompt"] = question.strip()
            with st.spinner("Executing deterministic analysis, validation, and chart generation..."):
                try:
                    res = client.run_analysis(
                        {
                            "dataset_id": st.session_state["dataset_id"],
                            "prompt": question.strip(),
                            "use_online_llm": use_online,
                            "chart_type_override": None if chart_override == "auto" else chart_override,
                            "prefer_pie": prefer_pie,
                            "save_to_history": save_hist,
                            "run_anomaly_detection": include_anom,
                        }
                    )
                    st.session_state["analysis_run"] = res
                    st.session_state["analysis_id"] = res.get("analysis_id")
                    st.session_state["prompt_optimization"] = res.get("prompt_optimization")
                    if res.get("anomaly_report"):
                        st.session_state["anomaly_result"] = {"anomaly_report": res["anomaly_report"], "visualization": None}
                    st.success("Analysis completed and self-validated.")
                except Exception as exc:
                    _render_api_error(exc)

    run = st.session_state["analysis_run"]
    if not run:
        return

    ares = run["analysis_result"]
    vrep = run["validation_report"]

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        _kpi_card("Status", ares.get("status"))
    with k2:
        _kpi_card("Intent", ares.get("intent"))
    with k3:
        _kpi_card("Validation", vrep.get("overall_status"))
    with k4:
        _kpi_card("Execution Time", f"{ares.get('execution_time_ms', 0)} ms")

    for w in ares.get("warnings", []):
        st.warning(w)

    st.subheader("Original vs. Optimized Prompt")
    po = run["prompt_optimization"]
    p1, p2 = st.columns(2)
    with p1:
        st.caption("Original")
        st.info(po.get("original_prompt"))
    with p2:
        st.caption("Optimized")
        st.success(po.get("improved_prompt"))

    st.subheader("Verified Insights vs. Unsupported Claims")
    ins_l, ins_r = st.columns(2)
    with ins_l:
        st.markdown("**Verified Statistical Insights**")
        for ins in run.get("verified_insights", []):
            st.markdown(f"- {ins}")
    with ins_r:
        st.markdown("**Excluded / Unsupported Claims**")
        excl = run.get("excluded_unsupported_claims", [])
        if excl:
            for item in excl:
                st.error(item)
        else:
            st.caption("None flagged.")

    st.subheader("Computed Summary & Result Table")
    st.json(ares.get("summary_metrics", {}))
    table = ares.get("result_table", [])
    if table:
        st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

    st.subheader("Interactive Visualization")
    _render_plotly_viz(run.get("visualization"))


def _page_anomaly(client: SmartPromptAPIClient) -> None:
    st.header("ML-Based Anomaly Detection (Isolation Forest)")
    if not _require_dataset():
        return

    prof = st.session_state["profile"]
    num_cols = prof.get("numerical_columns", [])
    if not num_cols:
        st.warning("Dataset has no numerical columns for Isolation Forest anomaly detection.")
        return

    with st.form("anomaly_form"):
        features = st.multiselect("Numerical Features", options=num_cols, default=num_cols[: min(5, len(num_cols))])
        c1, c2, c3 = st.columns(3)
        contamination = c1.slider("Contamination", min_value=0.01, max_value=0.35, value=0.05, step=0.01)
        missing_strategy = c2.selectbox("Missing Strategy", options=["median", "mean", "drop"])
        n_estimators = c3.slider("Trees (n_estimators)", min_value=50, max_value=300, value=100, step=25)
        run_anom = st.form_submit_button("Run Isolation Forest", type="primary")

    if run_anom:
        with st.spinner("Running Scikit-learn Isolation Forest..."):
            try:
                res = client.detect_anomalies(
                    {
                        "dataset_id": st.session_state["dataset_id"],
                        "feature_columns": features or None,
                        "contamination": contamination,
                        "missing_strategy": missing_strategy,
                        "n_estimators": n_estimators,
                    }
                )
                st.session_state["anomaly_result"] = res
            except Exception as exc:
                _render_api_error(exc)

    anom_wrap = st.session_state["anomaly_result"]
    if not anom_wrap:
        return

    rep = anom_wrap["anomaly_report"]
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        _kpi_card("Rows Analyzed", rep.get("rows_analyzed", 0))
    with k2:
        _kpi_card("Anomalies Flagged", rep.get("anomaly_count", 0))
    with k3:
        _kpi_card("Anomaly %", f"{rep.get('anomaly_percentage', 0)}%")
    with k4:
        _kpi_card("Normal Rows", rep.get("normal_count", 0))

    st.info(rep.get("interpretation_note", ""))

    if anom_wrap.get("visualization"):
        st.subheader("Anomaly Scatter Visualization")
        _render_plotly_viz(anom_wrap["visualization"])

    records = rep.get("anomalous_records", [])
    if records:
        st.subheader("Flagged Anomalous Records")
        flat_rows = [
            {
                "Row Index": r["row_index"],
                "Anomaly Score": r["anomaly_score"],
                "Decision Function": r["decision_function_score"],
                "Feature Values": json.dumps(r["feature_values"]),
                "Top Deviating Features": ", ".join(
                    f"{d['feature']} ({d['std_deviations_from_median']}σ)" for d in r.get("top_deviating_features", [])
                ),
            }
            for r in records
        ]
        st.dataframe(pd.DataFrame(flat_rows), use_container_width=True, hide_index=True)

    with st.expander("Unsupervised Score Diagnostics", expanded=False):
        st.json(rep.get("diagnostics", {}))


def _page_validation(client: SmartPromptAPIClient) -> None:
    st.header("Evidence-Based Self-Validation Panel")
    if not _require_dataset():
        return

    run = st.session_state["analysis_run"]
    if run and not st.session_state["validation_override"]:
        _render_validation_panel(
            run["validation_report"],
            run.get("verified_insights", []),
            run.get("excluded_unsupported_claims", []),
        )
    elif st.session_state["validation_override"]:
        vo = st.session_state["validation_override"]
        _render_validation_panel(
            vo["validation_report"],
            vo.get("verified_insights", []),
            vo.get("excluded_unsupported_claims", []),
        )
    else:
        st.info("Run an analysis in **AI Data Analysis** or test a specific claim below.")

    st.divider()
    st.subheader("Verify a Quantitative Claim Against Raw Dataset")
    num_cols = st.session_state["profile"].get("numerical_columns", [])
    with st.form("claim_validate_form"):
        c1, c2, c3 = st.columns(3)
        claim_type = c1.selectbox("Claim Type", options=["total", "mean", "median", "count", "text"])
        column = c2.selectbox("Column (for numeric claim)", options=num_cols or st.session_state["columns"])
        claimed_val = c3.number_input("Claimed Numerical Value", value=0.0, format="%.4f")
        statement_text = st.text_input(
            "Or Natural-Language Statement (when Claim Type = 'text')",
            placeholder="e.g., total Revenue is 15000",
        )
        verify_btn = st.form_submit_button("Validate Claim")

    if verify_btn:
        claim_payload: dict[str, Any] = {
            "claim_type": claim_type,
            "description": f"User claim verification ({claim_type})",
            "column": column if claim_type != "text" else None,
            "claimed_value": claimed_val if claim_type != "text" else None,
            "statement_text": statement_text.strip() if claim_type == "text" else None,
        }
        with st.spinner("Cross-checking claim against raw dataset..."):
            try:
                res = client.validate_analysis(
                    {"dataset_id": st.session_state["dataset_id"], "claims": [claim_payload]}
                )
                st.session_state["validation_override"] = res
                st.rerun()
            except Exception as exc:
                _render_api_error(exc)


def _page_visualizations(client: SmartPromptAPIClient) -> None:
    st.header("Interactive Visualizations")
    if not _require_dataset():
        return

    run = st.session_state["analysis_run"]
    if run:
        col_sel, col_btn = st.columns([3, 1])
        chosen = col_sel.selectbox(
            "Switch Chart Type (re-renders via backend)",
            options=["auto", "bar", "horizontal_bar", "line", "pie", "histogram", "scatter", "heatmap"],
        )
        if col_btn.button("Update Chart"):
            prompt_text = run["prompt_optimization"].get("original_prompt") or st.session_state["user_prompt"]
            if prompt_text:
                with st.spinner("Regenerating Plotly figure on backend..."):
                    try:
                        updated = client.run_analysis(
                            {
                                "dataset_id": st.session_state["dataset_id"],
                                "prompt": prompt_text,
                                "chart_type_override": None if chosen == "auto" else chosen,
                                "prefer_pie": chosen == "pie",
                                "save_to_history": False,
                            }
                        )
                        st.session_state["analysis_run"] = updated
                        run = updated
                    except Exception as exc:
                        _render_api_error(exc)

        st.subheader("Primary Analytical Chart")
        _render_plotly_viz(run.get("visualization"))
    else:
        st.info("Execute an analysis in **AI Data Analysis** to generate your primary chart.")

    if st.session_state["anomaly_result"] and st.session_state["anomaly_result"].get("visualization"):
        st.subheader("Anomaly Detection Scatter Plot")
        _render_plotly_viz(st.session_state["anomaly_result"]["visualization"])


def _page_history(client: SmartPromptAPIClient) -> None:
    st.header("Analysis History")

    c1, c2, c3 = st.columns([2, 2, 1])
    intent_filter = c1.selectbox(
        "Filter by Intent",
        options=[
            "All", "ranking", "comparison", "trend_analysis", "aggregation",
            "correlation", "distribution_analysis", "anomaly_analysis",
            "filtering", "missing_value_analysis", "dataset_summary",
        ],
    )
    val_filter = c2.selectbox("Filter by Validation", options=["All", "Verified", "Warning", "Unsupported"])
    if c3.button("Refresh History"):
        pass

    try:
        hist = client.list_history(limit=100, detected_intent=intent_filter, validation_status=val_filter)
    except Exception as exc:
        _render_api_error(exc)
        return

    items = hist.get("items", [])
    st.caption(f"Total records in database: {hist.get('total_count', 0)}")
    if not items:
        st.info("No analysis history records match the selected filters.")
        return

    table_rows = [
        {
            "Analysis ID": it["analysis_id"],
            "Dataset": it["dataset_name"],
            "Intent": it["detected_intent"],
            "Original Prompt": it["original_prompt"],
            "Validation": it["validation_status"],
            "Created At": it["created_at"],
        }
        for it in items
    ]
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    id_options = [it["analysis_id"] for it in items]
    selected_id = st.selectbox("Inspect or Delete Record", options=id_options)
    b1, b2 = st.columns(2)
    if b1.button("Inspect Details"):
        try:
            detail = client.get_history_item(selected_id)["item"]
            st.json(detail)
        except Exception as exc:
            _render_api_error(exc)
    if b2.button("Delete Selected Record"):
        try:
            client.delete_history_item(selected_id)
            st.success(f"Deleted record '{selected_id}'.")
            st.rerun()
        except Exception as exc:
            _render_api_error(exc)


def _page_report_download(client: SmartPromptAPIClient) -> None:
    st.header("Report Generation & Download")
    if not _require_dataset():
        return

    report_title = st.text_input("Report Title", value="SmartPrompt AI — Analytical Report")
    col1, col2 = st.columns(2)

    if col1.button("Generate PDF Report", type="primary"):
        with st.spinner("Building ReportLab PDF report on backend..."):
            try:
                content, fname = client.export_report(
                    dataset_id=st.session_state["dataset_id"],
                    export_format="pdf",
                    analysis_id=st.session_state["analysis_id"],
                    report_title=report_title,
                )
                st.session_state["report_pdf_bytes"] = content
                st.session_state["report_pdf_name"] = fname
            except Exception as exc:
                _render_api_error(exc)

    if col2.button("Generate CSV Report"):
        with st.spinner("Building CSV report on backend..."):
            try:
                content, fname = client.export_report(
                    dataset_id=st.session_state["dataset_id"],
                    export_format="csv",
                    analysis_id=st.session_state["analysis_id"],
                    report_title=report_title,
                )
                st.session_state["report_csv_bytes"] = content
                st.session_state["report_csv_name"] = fname
            except Exception as exc:
                _render_api_error(exc)

    d1, d2 = st.columns(2)
    with d1:
        if st.session_state["report_pdf_bytes"]:
            st.download_button(
                label="Download PDF Report",
                data=st.session_state["report_pdf_bytes"],
                file_name=st.session_state["report_pdf_name"],
                mime="application/pdf",
            )
    with d2:
        if st.session_state["report_csv_bytes"]:
            st.download_button(
                label="Download CSV Report",
                data=st.session_state["report_csv_bytes"],
                file_name=st.session_state["report_csv_name"],
                mime="text/csv",
            )


def _page_settings(client: SmartPromptAPIClient) -> None:
    st.header("Settings")
    st.caption("Manage your profile, preferences, notifications, and account security.")

    user = st.session_state.get("current_user") or {}

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Profile Settings")
        st.text_input("Full Name", value=user.get("full_name", ""), disabled=True)
        st.text_input("Username", value=user.get("username", ""), disabled=True)
        st.text_input("Email Address", value=user.get("email", ""), disabled=True)

        st.subheader("Notifications")
        notify_analysis = st.checkbox(
            "Analysis completion notifications",
            value=st.session_state.get("pref_notify_analysis", True),
        )
        notify_report = st.checkbox(
            "Report export notifications",
            value=st.session_state.get("pref_notify_report", True),
        )
        st.session_state["pref_notify_analysis"] = notify_analysis
        st.session_state["pref_notify_report"] = notify_report

    with col_right:
        st.subheader("Data Preferences")
        default_fmt = st.selectbox(
            "Default Export Format",
            options=["Excel (.xlsx)", "CSV (.csv)"],
            index=0 if st.session_state.get("pref_default_export") == "Excel (.xlsx)" else 1,
        )
        st.session_state["pref_default_export"] = default_fmt

        st.subheader("Appearance")
        st.caption("Theme appearance follows your Streamlit menu theme preference (Light / Dark).")

        st.subheader("Privacy & Security")
        if st.button("Sign Out of Current Session", key="settings_logout_btn"):
            try:
                client.logout_user()
            except Exception:
                pass
            st.session_state["auth_token"] = None
            st.session_state["current_user"] = None
            st.session_state["auth_page"] = "Home"
            st.rerun()

    st.divider()
    st.subheader("About SmartPrompt AI")
    st.write("SmartPrompt AI v1.0.0 — Dataset-aware, self-validating intelligent data analysis platform.")
    st.caption("Support: support@smartprompt.ai")


# ---------------------------------------------------------------------------
# Public Home, Login & Register Views
# ---------------------------------------------------------------------------


def _render_auth_portal(client: SmartPromptAPIClient) -> None:
    nav_cols = st.columns([3, 1, 1, 1])
    with nav_cols[0]:
        st.markdown("### SmartPrompt AI")
    with nav_cols[1]:
        if st.button("Home", use_container_width=True):
            st.session_state["auth_page"] = "Home"
            st.rerun()
    with nav_cols[2]:
        if st.button("Login", use_container_width=True):
            st.session_state["auth_page"] = "Login"
            st.rerun()
    with nav_cols[3]:
        if st.button("Register", use_container_width=True):
            st.session_state["auth_page"] = "Register"
            st.rerun()

    st.divider()
    notice = st.session_state.get("auth_notice")
    if notice:
        st.success(notice)
        st.session_state["auth_notice"] = None

    active_page = st.session_state.get("auth_page", "Home")
    if active_page == "Home":
        st.title("SmartPrompt AI – Intelligent Data Analysis Made Simple")
        st.write(
            "Upload structured CSV or Excel datasets, automate data cleaning, optimize analytical prompts, "
            "and generate verified insights."
        )
        if st.button("Get Started", type="primary"):
            st.session_state["auth_page"] = "Login"
            st.rerun()

        st.divider()
        st.caption("© 2025 SmartPrompt AI. All rights reserved.")

    elif active_page == "Login":
        st.header("Sign in to SmartPrompt AI")
        with st.form("streamlit_login_form"):
            identifier = st.text_input("Username or Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", type="primary")

        if submitted:
            if not identifier.strip() or not password:
                st.error("Please enter both username/email and password.")
            else:
                try:
                    resp = client.login_user(identifier=identifier, password=password)
                    st.session_state["auth_token"] = resp.get("access_token")
                    st.session_state["current_user"] = resp.get("user")
                    st.session_state["nav_section"] = "Home Dashboard"
                    st.rerun()
                except Exception as exc:
                    _render_api_error(exc)

        if st.button("Need an account? Register"):
            st.session_state["auth_page"] = "Register"
            st.rerun()

    elif active_page == "Register":
        st.header("Create a SmartPrompt AI Account")
        with st.form("streamlit_register_form"):
            full_name = st.text_input("Full Name")
            username = st.text_input("Username")
            email = st.text_input("Email Address")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            reg_submitted = st.form_submit_button("Register", type="primary")

        if reg_submitted:
            if not all([full_name.strip(), username.strip(), email.strip(), password, confirm_password]):
                st.error("All fields are required.")
            elif password != confirm_password:
                st.error("Password and confirm password must match.")
            else:
                try:
                    resp = client.register_user(
                        full_name=full_name,
                        username=username,
                        email=email,
                        password=password,
                        confirm_password=confirm_password,
                    )
                    st.session_state["auth_notice"] = resp.get("message") or "Registration successful. Please log in."
                    st.session_state["auth_page"] = "Login"
                    st.rerun()
                except Exception as exc:
                    _render_api_error(exc)

        if st.button("Already registered? Login"):
            st.session_state["auth_page"] = "Login"
            st.rerun()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def _page_unstructured_converter(client: SmartPromptAPIClient) -> None:
    st.header("Unstructured Data to Structured Data Converter")
    st.caption("Upload PDF (.pdf), Word (.docx), or Text (.txt) documents, preview extracted tabular data, and download as Excel (.xlsx).")

    uploaded = st.file_uploader(
        "Upload unstructured document (.pdf, .docx, .txt)",
        type=["pdf", "docx", "txt"],
        key="unstructured_file_uploader",
    )

    if st.button("Extract and Convert", type="primary", disabled=uploaded is None):
        if uploaded is not None:
            try:
                res = client.extract_unstructured(uploaded.getvalue(), uploaded.name)
                st.session_state["unstructured_result"] = res
                xlsx_bytes, out_name = client.export_converted_excel(res)
                st.session_state["unstructured_xlsx_bytes"] = xlsx_bytes
                st.session_state["unstructured_xlsx_name"] = out_name
                st.success(f"Extracted {len(res.get('rows', []))} rows across {len(res.get('columns', []))} columns.")
            except Exception as exc:
                _render_api_error(exc)

    res = st.session_state.get("unstructured_result")
    if res:
        st.subheader("Structured Data Preview")
        rows = res.get("rows", [])
        cols = res.get("columns", [])
        if rows:
            st.dataframe(pd.DataFrame(rows, columns=cols), use_container_width=True, hide_index=True)
        if st.session_state.get("unstructured_xlsx_bytes"):
            st.download_button(
                label="Download as Excel (.xlsx)",
                data=st.session_state["unstructured_xlsx_bytes"],
                file_name=st.session_state.get("unstructured_xlsx_name", "converted_structured.xlsx"),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


def main() -> None:
    _configure_page()
    _init_session_state()
    client = _get_client()

    if st.session_state.get("auth_token") and not st.session_state.get("current_user"):
        try:
            st.session_state["current_user"] = client.get_current_user()
        except Exception:
            st.session_state["auth_token"] = None
            st.session_state["current_user"] = None

    if not st.session_state.get("current_user"):
        _render_auth_portal(client)
        return

    section, health_data = _render_sidebar(client)

    if section == "Home Dashboard":
        _page_home(client, health_data)
    elif section == "Dataset Upload":
        _page_upload(client)
    elif section == "Unstructured Data Converter":
        _page_unstructured_converter(client)
    elif section == "Dataset Profiling":
        _page_profiling()
    elif section == "Smart Prompt Optimizer":
        _page_prompt_optimizer(client, health_data)
    elif section == "Prompt Quality Score":
        _page_prompt_score(client)
    elif section == "AI Data Analysis":
        _page_analysis(client, health_data)
    elif section == "Anomaly Detection":
        _page_anomaly(client)
    elif section == "Self-Validation Results":
        _page_validation(client)
    elif section == "Interactive Visualizations":
        _page_visualizations(client)
    elif section == "Analysis History":
        _page_history(client)
    elif section == "Report Download":
        _page_report_download(client)
    elif section == "Settings":
        _page_settings(client)


if __name__ == "__main__":
    main()
