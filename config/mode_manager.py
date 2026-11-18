"""Offline Local Mode & Online Demo Mode configuration resolver and validator."""

from __future__ import annotations

from dataclasses import dataclass

from config.settings import ExecutionMode, Settings, get_settings


class ModeConfigurationError(RuntimeError):
    """Raised when required configuration for the selected mode is invalid."""


@dataclass(frozen=True)
class ModeCapabilities:
    """Resolved runtime capabilities derived from active execution mode."""

    mode: ExecutionMode
    database_backend: str
    database_url: str
    use_external_ai: bool
    ai_provider: str
    allow_local_fallback: bool
    persist_models_locally: bool
    requires_local_storage: bool
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "database_backend": self.database_backend,
            "use_external_ai": self.use_external_ai,
            "ai_provider": self.ai_provider,
            "allow_local_fallback": self.allow_local_fallback,
            "persist_models_locally": self.persist_models_locally,
            "requires_local_storage": self.requires_local_storage,
            "warnings": list(self.warnings),
        }


class ModeManager:
    """Validates and resolves runtime behavior for Offline and Online modes."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def resolve(self, strict_online: bool | None = None) -> ModeCapabilities:
        """Resolve runtime capabilities and optionally enforce strict online prerequisites.

        In Offline Local Mode:
        - Never uses external AI APIs or internet connectivity.
        - Always uses local SQLite and local Scikit-learn/Joblib models.

        In Online Demo Mode:
        - Uses configured PostgreSQL DATABASE_URL or falls back safely to SQLite with a warning.
        - Does not require local repo file paths for database persistence when DATABASE_URL is set.
        - Enables external AI enrichment only when a valid server-side GEMINI_API_KEY is present;
          otherwise falls back cleanly to local deterministic processing unless strict_online=True.
        """
        enforce_strict = self.settings.strict_online_mode if strict_online is None else strict_online
        self.settings.ensure_directories()
        warnings: list[str] = []

        if self.settings.is_offline:
            return ModeCapabilities(
                mode=ExecutionMode.OFFLINE,
                database_backend="sqlite",
                database_url=self.settings.effective_database_url,
                use_external_ai=False,
                ai_provider="local",
                allow_local_fallback=True,
                persist_models_locally=True,
                requires_local_storage=True,
                warnings=(),
            )

        # Online Demo Mode resolution
        db_url = self.settings.effective_database_url
        db_backend = db_url.split("://", 1)[0]
        has_remote_db = bool(self.settings.database_url.strip())
        if not has_remote_db:
            msg = "ONLINE mode active without DATABASE_URL; falling back to SQLite database."
            if enforce_strict:
                raise ModeConfigurationError(msg)
            warnings.append(msg)

        has_ai_key = self.settings.has_valid_online_ai_key
        use_external_ai = has_ai_key and self.settings.ai_provider != "local"
        effective_ai_provider = self.settings.ai_provider if use_external_ai else "local"

        if not has_ai_key:
            msg = (
                "ONLINE mode active without a valid server-side GEMINI_API_KEY; "
                "using local deterministic NLP/analytics fallback."
            )
            if enforce_strict:
                raise ModeConfigurationError(msg)
            warnings.append(msg)

        return ModeCapabilities(
            mode=ExecutionMode.ONLINE,
            database_backend=db_backend,
            database_url=db_url,
            use_external_ai=use_external_ai,
            ai_provider=effective_ai_provider,
            allow_local_fallback=not enforce_strict,
            persist_models_locally=True,
            requires_local_storage=not has_remote_db,
            warnings=tuple(warnings),
        )


def resolve_runtime_capabilities(
    settings: Settings | None = None,
    strict_online: bool | None = None,
) -> ModeCapabilities:
    """Convenience helper to resolve runtime mode capabilities."""
    return ModeManager(settings=settings).resolve(strict_online=strict_online)
