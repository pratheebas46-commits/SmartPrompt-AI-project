"""Application service orchestrating Modules 1-11 for the FastAPI backend."""

from __future__ import annotations

import io
import json
from typing import Any
import uuid

import pandas as pd

from backend.schemas.requests import (
    AnalysisRunRequest,
    AnalysisValidateRequest,
    AnomalyDetectRequest,
    CleanDatasetRequest,
    PromptOptimizeRequest,
    PromptScoreRequest,
    ReportExportRequest,
)
from backend.services.session_store import DatasetSession, DatasetSessionStore
from config.logging_config import get_logger
from config.mode_manager import ModeManager
from config.settings import Settings, get_settings
from database.db_manager import DatabaseManager, RecordNotFoundError
from modules.analysis_engine import AnalysisResult, _records_from_df, execute_analysis
from modules.anomaly_detector import AnomalyConfig, AnomalyReport, detect_anomalies
from modules.data_cleaner import CleaningConfig, clean_dataset
from modules.dataset_profiler import (
    DatasetProfileResult,
    EmptyDatasetError,
    load_and_profile_dataset,
    profile_dataframe,
)
from modules.prompt_optimizer import PromptOptimizationResult, optimize_prompt
from modules.prompt_quality_model import PromptQualityEvaluator
from modules.report_generator import (
    build_report_payload,
    export_report_to_csv,
    export_report_to_pdf,
)
from modules.validator import (
    AnalyticalClaim,
    ValidationReport,
    validate_analysis_result,
    validate_claims,
)
from modules.visualization import generate_anomaly_chart, generate_chart_for_analysis

logger = get_logger(__name__)


class AnalyticsService:
    """Facade coordinating profiling, cleaning, prompt ML, analysis, validation, charts, DB, and reports."""

    def __init__(
        self,
        settings: Settings | None = None,
        db_manager: DatabaseManager | None = None,
        session_store: DatasetSessionStore | None = None,
        quality_evaluator: PromptQualityEvaluator | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_directories()
        self.db_manager = db_manager or DatabaseManager(settings=self.settings)
        self.session_store = session_store or DatasetSessionStore(
            max_sessions=self.settings.max_active_sessions,
            ttl_seconds=self.settings.session_ttl_seconds,
        )
        self.quality_evaluator = quality_evaluator or PromptQualityEvaluator(
            model_path=self.settings.models_dir / "prompt_quality_model.joblib"
        )

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _resolve_dataframe_and_profile(
        self,
        dataset_id: str | None,
        records: list[dict[str, Any]] | None,
        dataset_name: str = "inline_dataset",
    ) -> tuple[pd.DataFrame, DatasetProfileResult, DatasetSession | None]:
        if dataset_id:
            session = self.session_store.get_session(dataset_id)
            return session.dataframe, session.profile, session
        if records is not None:
            if not records:
                raise EmptyDatasetError("Provided inline 'records' list is empty.")
            df = pd.DataFrame(records)
            profile = profile_dataframe(df, dataset_name=dataset_name)
            return df, profile, None
        raise EmptyDatasetError("Either 'dataset_id' or non-empty 'records' must be provided.")

    @staticmethod
    def _serialize_viz(viz_result: Any) -> dict[str, Any]:
        figure_dict: dict[str, Any] | None = None
        if viz_result.figure is not None:
            figure_dict = json.loads(viz_result.figure.to_json())
        return {
            "is_renderable": viz_result.is_renderable,
            "chart_type": viz_result.chart_type,
            "title": viz_result.title,
            "reason": viz_result.reason,
            "metadata": viz_result.metadata,
            "figure": figure_dict,
        }

    # ------------------------------------------------------------------
    # Health & Settings
    # ------------------------------------------------------------------

    def get_health_status(self) -> dict[str, Any]:
        caps = ModeManager(self.settings).resolve(strict_online=False)
        db_connected = False
        db_backend = caps.database_backend
        warnings = list(caps.warnings)
        try:
            health = self.db_manager.health_check()
            db_connected = bool(health.get("connected", False))
            db_backend = str(health.get("backend", db_backend))
        except Exception as exc:
            warnings.append(f"Database check warning: {exc}")

        return {
            "status": "healthy" if db_connected else "degraded",
            "app_name": self.settings.app_name,
            "execution_mode": caps.mode.value,
            "database_connected": db_connected,
            "database_backend": db_backend,
            "online_ai_enabled": caps.use_external_ai,
            "active_dataset_sessions": self.session_store.count_sessions(),
            "warnings": warnings,
        }

    # ------------------------------------------------------------------
    # Dataset Upload, Profiling & Cleaning (Modules 1 & 2)
    # ------------------------------------------------------------------

    def profile_uploaded_file(
        self,
        file_bytes: bytes,
        filename: str,
        sheet_name: str | None = None,
    ) -> dict[str, Any]:
        df, profile = load_and_profile_dataset(
            source=io.BytesIO(file_bytes),
            filename=filename,
            sheet_name=sheet_name,
            max_file_size_mb=self.settings.max_upload_size_mb,
        )
        session = self.session_store.create_session(df=df, profile=profile)
        return {
            "dataset_id": session.dataset_id,
            "profile": profile.to_dict(),
            "preview_rows": _records_from_df(df, max_rows=25),
            "columns": [str(c) for c in df.columns],
        }

    def get_dataset_preview(self, dataset_id: str, max_rows: int = 50) -> dict[str, Any]:
        session = self.session_store.get_session(dataset_id)
        return {
            "dataset_id": session.dataset_id,
            "profile": session.profile.to_dict(),
            "preview_rows": _records_from_df(session.dataframe, max_rows=max_rows),
            "columns": [str(c) for c in session.dataframe.columns],
        }

    def delete_dataset_session(self, dataset_id: str) -> bool:
        return self.session_store.delete_session(dataset_id)

    def clean_session_dataset(self, req: CleanDatasetRequest) -> dict[str, Any]:
        df, profile, session = self._resolve_dataframe_and_profile(
            req.dataset_id, req.records, req.dataset_name
        )
        cfg = CleaningConfig(
            numeric_missing_strategy=req.numeric_missing_strategy,
            categorical_missing_strategy=req.categorical_missing_strategy,
            column_missing_strategies=req.column_missing_strategies,
            constant_fill_value=req.constant_fill_value,
            numeric_constant_fill_value=req.numeric_constant_fill_value,
            drop_column_missing_threshold=req.drop_column_missing_threshold,
            remove_duplicates=req.remove_duplicates,
            duplicate_subset=req.duplicate_subset,
            duplicate_keep=req.duplicate_keep,
            type_conversions=req.type_conversions,
            datetime_columns=req.datetime_columns,
            datetime_format=req.datetime_format,
            numeric_scaling=req.numeric_scaling,
            numeric_columns_to_preprocess=req.numeric_columns_to_preprocess,
            strip_categorical_whitespace=req.strip_categorical_whitespace,
            categorical_case=req.categorical_case,
            categorical_encoding=req.categorical_encoding,
            categorical_columns_to_preprocess=req.categorical_columns_to_preprocess,
            reprofile_after_cleaning=req.reprofile_after_cleaning,
        )
        result = clean_dataset(df, config=cfg, profile=profile, dataset_name=profile.dataset_name)
        updated_profile = result.updated_profile or profile_dataframe(
            result.cleaned_df, dataset_name=profile.dataset_name
        )

        resolved_id = req.dataset_id
        if session is not None and req.update_active_dataset:
            self.session_store.update_dataframe(session.dataset_id, result.cleaned_df, updated_profile)
        elif session is None:
            new_session = self.session_store.create_session(result.cleaned_df, updated_profile)
            resolved_id = new_session.dataset_id

        return {
            "dataset_id": resolved_id or "inline",
            "cleaning_report": result.report.to_dict(),
            "updated_profile": updated_profile.to_dict(),
            "preview_rows": _records_from_df(result.cleaned_df, max_rows=25),
            "columns": [str(c) for c in result.cleaned_df.columns],
        }

    # ------------------------------------------------------------------
    # Prompt Optimization & Prompt Quality Scoring (Modules 3, 4, 5)
    # ------------------------------------------------------------------

    def optimize_user_prompt(self, req: PromptOptimizeRequest) -> dict[str, Any]:
        _, profile, session = self._resolve_dataframe_and_profile(
            req.dataset_id, req.records, req.dataset_name
        )
        opt_result = optimize_prompt(
            user_prompt=req.prompt,
            profile=profile,
            use_online_llm=req.use_online_llm,
            settings=self.settings,
        )
        if session is not None:
            self.session_store.record_run_artifacts(
                dataset_id=session.dataset_id,
                analysis_id=None,
                prompt_result=opt_result,
            )
        return {
            "dataset_id": session.dataset_id if session else req.dataset_id,
            "optimization": opt_result.to_dict(),
        }

    def score_user_prompt(self, req: PromptScoreRequest) -> dict[str, Any]:
        profile: DatasetProfileResult | None = None
        if req.dataset_id:
            session = self.session_store.get_session(req.dataset_id)
            profile = session.profile
        prediction = self.quality_evaluator.predict(
            prompt=req.prompt,
            profile=profile,
            available_columns=req.available_columns,
        )
        return {"prediction": prediction.to_dict()}

    # ------------------------------------------------------------------
    # End-to-End Analysis, Anomaly Detection & Self-Validation (Modules 6-10)
    # ------------------------------------------------------------------

    def run_full_analysis(self, req: AnalysisRunRequest) -> dict[str, Any]:
        df, profile, session = self._resolve_dataframe_and_profile(
            req.dataset_id, req.records, req.dataset_name
        )
        prompt_res = optimize_prompt(
            user_prompt=req.prompt,
            profile=profile,
            use_online_llm=req.use_online_llm,
            settings=self.settings,
        )
        analysis_res = execute_analysis(
            df=df,
            plan=prompt_res.task_plan,
            profile=profile,
            dataset_name=profile.dataset_name,
        )
        validation_rep = validate_analysis_result(df=df, analysis_result=analysis_res, profile=profile)
        viz_res = generate_chart_for_analysis(
            analysis_result=analysis_res,
            plan=prompt_res.task_plan,
            df=df,
            chart_type_override=req.chart_type_override,
            prefer_pie=req.prefer_pie,
        )

        anomaly_rep: AnomalyReport | None = None
        if req.run_anomaly_detection and profile.numerical_columns:
            try:
                anomaly_rep = detect_anomalies(
                    df=df,
                    config=AnomalyConfig(contamination=req.anomaly_contamination),
                    profile=profile,
                )
            except Exception as exc:
                logger.warning("Optional anomaly detection skipped during analysis run: %s", exc)

        report_payload = build_report_payload(
            profile=profile,
            prompt_result=prompt_res,
            analysis_result=analysis_res,
            validation_report=validation_rep,
            anomaly_report=anomaly_rep,
        )

        analysis_id: str | None = None
        if req.save_to_history:
            saved = self.db_manager.save_analysis(
                analysis_id=str(uuid.uuid4()),
                dataset_name=profile.dataset_name,
                dataset_fingerprint=profile.dataset_fingerprint,
                original_prompt=prompt_res.original_prompt,
                optimized_prompt=prompt_res.improved_prompt,
                detected_intent=prompt_res.detected_intent,
                analysis_summary={
                    "status": analysis_res.status,
                    "summary_metrics": analysis_res.summary_metrics,
                    "execution_time_ms": analysis_res.execution_time_ms,
                    "verified_insights": report_payload.statistical_insights,
                },
                validation_status=validation_rep.overall_status,
                dataset_metadata=profile.schema_context_summary(),
                dataset_row_count=profile.row_count,
                dataset_column_count=profile.column_count,
                anomaly_summary=report_payload.anomaly_summary,
            )
            analysis_id = saved.analysis_id

        if session is not None:
            self.session_store.record_run_artifacts(
                dataset_id=session.dataset_id,
                analysis_id=analysis_id,
                prompt_result=prompt_res,
                analysis_result=analysis_res,
                validation_report=validation_rep,
                anomaly_report=anomaly_rep,
            )

        return {
            "dataset_id": session.dataset_id if session else req.dataset_id,
            "analysis_id": analysis_id,
            "prompt_optimization": prompt_res.to_dict(),
            "analysis_result": analysis_res.to_dict(),
            "validation_report": validation_rep.to_dict(),
            "visualization": self._serialize_viz(viz_res),
            "anomaly_report": anomaly_rep.to_dict() if anomaly_rep else None,
            "verified_insights": report_payload.statistical_insights,
            "excluded_unsupported_claims": report_payload.excluded_unsupported_claims,
        }

    def run_anomaly_detection(self, req: AnomalyDetectRequest) -> dict[str, Any]:
        df, profile, session = self._resolve_dataframe_and_profile(
            req.dataset_id, req.records, req.dataset_name
        )
        cfg = AnomalyConfig(
            feature_columns=req.feature_columns,
            contamination=req.contamination,
            missing_strategy=req.missing_strategy,
            n_estimators=req.n_estimators,
        )
        anomaly_rep = detect_anomalies(df=df, config=cfg, profile=profile)
        viz_res = generate_anomaly_chart(
            anomaly_report=anomaly_rep,
            df=df,
            x_feature=req.x_feature,
            y_feature=req.y_feature,
        )
        if session is not None:
            self.session_store.record_run_artifacts(
                dataset_id=session.dataset_id,
                analysis_id=None,
                anomaly_report=anomaly_rep,
            )
        return {
            "dataset_id": session.dataset_id if session else req.dataset_id,
            "anomaly_report": anomaly_rep.to_dict(),
            "visualization": self._serialize_viz(viz_res),
        }

    def validate_results_or_claims(self, req: AnalysisValidateRequest) -> dict[str, Any]:
        df, profile, session = self._resolve_dataframe_and_profile(
            req.dataset_id, req.records, req.dataset_name
        )
        report: ValidationReport
        analysis_res: AnalysisResult | None = None

        if req.claims:
            domain_claims = [
                AnalyticalClaim(
                    claim_type=c.claim_type,
                    description=c.description or c.claim_type,
                    column=c.column,
                    secondary_column=c.secondary_column,
                    group_column=c.group_column,
                    group_value=c.group_value,
                    claimed_value=c.claimed_value,
                    claimed_entity=c.claimed_entity,
                    sort_order=c.sort_order,
                    aggregation=c.aggregation,
                    filters=c.filters,
                    statement_text=c.statement_text,
                )
                for c in req.claims
            ]
            report = validate_claims(df=df, claims=domain_claims, abs_tol=req.abs_tol, rel_tol=req.rel_tol)
        else:
            if session is not None and req.analysis_id and req.analysis_id in session.analysis_artifacts:
                analysis_res = session.analysis_artifacts[req.analysis_id].get("analysis_result")
            elif session is not None and session.latest_analysis_result is not None and not req.prompt:
                analysis_res = session.latest_analysis_result
            elif req.prompt:
                opt = optimize_prompt(req.prompt, profile, settings=self.settings)
                analysis_res = execute_analysis(df, opt.task_plan, profile=profile)
            else:
                raise EmptyDatasetError(
                    "Provide 'claims', a valid 'analysis_id', or a 'prompt' (or run analysis first) to validate."
                )

            report = validate_analysis_result(
                df=df,
                analysis_result=analysis_res,
                profile=profile,
                abs_tol=req.abs_tol,
                rel_tol=req.rel_tol,
            )

        if session is not None:
            self.session_store.record_run_artifacts(
                dataset_id=session.dataset_id,
                analysis_id=req.analysis_id,
                validation_report=report,
            )

        payload = build_report_payload(
            profile=profile,
            analysis_result=analysis_res,
            validation_report=report,
        )
        return {
            "dataset_id": session.dataset_id if session else req.dataset_id,
            "validation_report": report.to_dict(),
            "verified_insights": payload.statistical_insights,
            "excluded_unsupported_claims": payload.excluded_unsupported_claims,
        }

    # ------------------------------------------------------------------
    # Report Export (Module 11)
    # ------------------------------------------------------------------

    def export_report(self, req: ReportExportRequest) -> tuple[bytes, str, str]:
        _, profile, session = self._resolve_dataframe_and_profile(
            req.dataset_id, req.records, req.dataset_name
        )
        prompt_res: PromptOptimizationResult | None = None
        analysis_res: AnalysisResult | None = None
        val_rep: ValidationReport | None = None
        anom_rep: AnomalyReport | None = None

        if session is not None:
            if req.analysis_id and req.analysis_id in session.analysis_artifacts:
                arts = session.analysis_artifacts[req.analysis_id]
                prompt_res = arts.get("prompt_result")
                analysis_res = arts.get("analysis_result")
                val_rep = arts.get("validation_report")
                anom_rep = arts.get("anomaly_report")
            else:
                prompt_res = session.latest_prompt_result
                analysis_res = session.latest_analysis_result
                val_rep = session.latest_validation_report
                anom_rep = session.latest_anomaly_report

        payload = build_report_payload(
            profile=profile,
            prompt_result=prompt_res,
            analysis_result=analysis_res,
            validation_report=val_rep,
            anomaly_report=anom_rep,
            additional_insights=req.additional_insights,
            report_title=req.report_title,
        )

        fmt = req.export_format.strip().lower()
        safe_stem = profile.dataset_name.rsplit(".", 1)[0].replace(" ", "_")
        if fmt == "csv":
            return export_report_to_csv(payload), "text/csv; charset=utf-8", f"{safe_stem}_report.csv"
        if fmt == "pdf":
            return export_report_to_pdf(payload), "application/pdf", f"{safe_stem}_report.pdf"
        raise ValueError(f"Unsupported export_format '{req.export_format}'. Use 'pdf' or 'csv'.")

    # ------------------------------------------------------------------
    # History CRUD (Module 10)
    # ------------------------------------------------------------------

    def list_history(
        self,
        limit: int = 50,
        offset: int = 0,
        dataset_fingerprint: str | None = None,
        detected_intent: str | None = None,
        validation_status: str | None = None,
    ) -> dict[str, Any]:
        records = self.db_manager.list_analyses(
            limit=limit,
            offset=offset,
            dataset_fingerprint=dataset_fingerprint,
            detected_intent=detected_intent,
            validation_status=validation_status,
        )
        total = self.db_manager.count_analyses(dataset_fingerprint=dataset_fingerprint)
        return {
            "total_count": total,
            "items": [r.to_dict() for r in records],
        }

    def get_history_item(self, analysis_id: str) -> dict[str, Any]:
        rec = self.db_manager.get_analysis(analysis_id)
        if rec is None:
            raise RecordNotFoundError(f"Analysis record '{analysis_id}' not found.")
        return {"item": rec.to_dict()}

    def delete_history_item(self, analysis_id: str) -> dict[str, Any]:
        deleted = self.db_manager.delete_analysis(analysis_id)
        if not deleted:
            raise RecordNotFoundError(f"Analysis record '{analysis_id}' not found.")
        return {"deleted": True, "analysis_id": analysis_id}
