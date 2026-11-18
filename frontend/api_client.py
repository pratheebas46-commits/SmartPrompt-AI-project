"""HTTP client connecting the Streamlit frontend strictly to the FastAPI backend."""

from __future__ import annotations

import os
from typing import Any

import httpx


class BackendConnectionError(RuntimeError):
    """Raised when the FastAPI backend is unreachable or times out."""


class BackendAPIError(RuntimeError):
    """Raised when the FastAPI backend returns a 4xx/5xx structured error."""

    def __init__(self, status_code: int, error: str, detail: str) -> None:
        self.status_code = status_code
        self.error = error
        self.detail = detail
        super().__init__(f"[{status_code}] {error}: {detail}")


def get_default_api_base_url() -> str:
    """Resolve configurable API base URL from environment variables."""
    env_url = os.getenv("API_BASE_URL", "").strip()
    if env_url:
        return env_url.rstrip("/")
    host = os.getenv("API_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = os.getenv("API_PORT", "8000").strip() or "8000"
    return f"http://{host}:{port}"


class SmartPromptAPIClient:
    """Stateless HTTP client delegating all analytics and persistence to FastAPI."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout_sec: float = 45.0,
        auth_token: str | None = None,
    ) -> None:
        self.base_url = (base_url or get_default_api_base_url()).rstrip("/")
        self.timeout_sec = timeout_sec
        self.auth_token = auth_token

    def _request(
        self,
        method: str,
        path: str,
        json_payload: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        expect_bytes: bool = False,
    ) -> Any:
        url = f"{self.base_url}{path}"
        headers: dict[str, str] = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        try:
            with httpx.Client(timeout=self.timeout_sec) as client:
                response = client.request(
                    method=method,
                    url=url,
                    json=json_payload,
                    params=params,
                    files=files,
                    data=data,
                    headers=headers or None,
                )
        except httpx.ConnectError as exc:
            raise BackendConnectionError(
                f"Cannot reach SmartPrompt AI backend at '{self.base_url}'. "
                "Ensure FastAPI is running (python run_api.py)."
            ) from exc
        except httpx.TimeoutException as exc:
            raise BackendConnectionError(
                f"Request to '{url}' timed out after {self.timeout_sec:.0f}s."
            ) from exc
        except httpx.HTTPError as exc:
            raise BackendConnectionError(f"Network error contacting backend: {exc}") from exc

        if response.status_code >= 400:
            error_name = "APIError"
            detail = response.text
            try:
                err_json = response.json()
                error_name = str(err_json.get("error", error_name))
                detail = str(err_json.get("detail", detail))
            except Exception:
                pass
            raise BackendAPIError(response.status_code, error_name, detail)

        if expect_bytes:
            return response.content, response.headers
        return response.json()

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def register_user(
        self,
        full_name: str,
        username: str,
        email: str,
        password: str,
        confirm_password: str,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/auth/register",
            json_payload={
                "full_name": full_name,
                "username": username,
                "email": email,
                "password": password,
                "confirm_password": confirm_password,
            },
        )

    def login_user(self, identifier: str, password: str) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/auth/login",
            json_payload={"identifier": identifier, "password": password},
        )

    def get_current_user(self) -> dict[str, Any]:
        return self._request("GET", "/api/auth/me")

    def logout_user(self) -> dict[str, Any]:
        return self._request("POST", "/api/auth/logout")

    def get_settings_summary(self) -> dict[str, Any]:
        return self._request("GET", "/api/settings")

    def profile_dataset(
        self,
        file_bytes: bytes,
        filename: str,
        sheet_name: str | None = None,
    ) -> dict[str, Any]:
        form_data = {"sheet_name": sheet_name} if sheet_name else None
        return self._request(
            "POST",
            "/api/dataset/profile",
            files={"file": (filename, file_bytes)},
            data=form_data,
        )

    def preview_dataset(self, dataset_id: str, max_rows: int = 50) -> dict[str, Any]:
        return self._request("GET", f"/api/dataset/{dataset_id}/preview", params={"max_rows": max_rows})

    def clean_dataset(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/dataset/clean", json_payload=payload)

    def optimize_prompt(
        self,
        dataset_id: str,
        prompt: str,
        use_online_llm: bool = False,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/prompt/optimize",
            json_payload={
                "dataset_id": dataset_id,
                "prompt": prompt,
                "use_online_llm": use_online_llm,
            },
        )

    def score_prompt(
        self,
        prompt: str,
        dataset_id: str | None = None,
        available_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/prompt/score",
            json_payload={
                "prompt": prompt,
                "dataset_id": dataset_id,
                "available_columns": available_columns,
            },
        )

    def run_analysis(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/analysis/run", json_payload=payload)

    def detect_anomalies(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/anomaly/detect", json_payload=payload)

    def validate_analysis(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/analysis/validate", json_payload=payload)

    def list_history(
        self,
        limit: int = 50,
        detected_intent: str | None = None,
        validation_status: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if detected_intent and detected_intent != "All":
            params["detected_intent"] = detected_intent
        if validation_status and validation_status != "All":
            params["validation_status"] = validation_status
        return self._request("GET", "/api/history", params=params)

    def get_history_item(self, analysis_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/history/{analysis_id}")

    def delete_history_item(self, analysis_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/api/history/{analysis_id}")

    def export_report(
        self,
        dataset_id: str,
        export_format: str,
        analysis_id: str | None = None,
        report_title: str = "SmartPrompt AI — Analytical Report",
    ) -> tuple[bytes, str]:
        content, headers = self._request(
            "POST",
            "/api/report/export",
            json_payload={
                "dataset_id": dataset_id,
                "analysis_id": analysis_id,
                "export_format": export_format,
                "report_title": report_title,
            },
            expect_bytes=True,
        )
        disposition = headers.get("content-disposition", "")
        filename = f"smartprompt_report.{export_format}"
        if 'filename="' in disposition:
            filename = disposition.split('filename="', 1)[1].split('"', 1)[0]
        return content, filename

    def extract_unstructured(self, file_bytes: bytes, filename: str) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/converter/extract",
            files={"file": (filename, file_bytes)},
        )

    def export_converted_excel(self, conversion_payload: dict[str, Any]) -> tuple[bytes, str]:
        content, headers = self._request(
            "POST",
            "/api/converter/export-excel",
            json_payload=conversion_payload,
            expect_bytes=True,
        )
        disposition = headers.get("content-disposition", "")
        filename = "converted_structured.xlsx"
        if 'filename="' in disposition:
            filename = disposition.split('filename="', 1)[1].split('"', 1)[0]
        return content, filename
