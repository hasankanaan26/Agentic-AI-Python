"""Typed configuration loaded from the environment.

Engineering standard: all configuration goes through ONE Settings class
built on `pydantic-settings`. Required keys are validated at startup —
if `LLM_PROVIDER=openai` but `OPENAI_API_KEY` is empty, the app refuses
to boot. No `os.getenv` scattered across modules.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Two ``parents`` because this file lives at ``<root>/base-app/app/settings.py``
# and we want ``<root>`` so default data paths resolve regardless of cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Strongly-typed application settings, sourced from env / ``.env``.

    Subclassing ``BaseSettings`` gives us automatic env-var loading,
    type coercion, and a single validation pass at instantiation. Every
    field has a default so ``Settings()`` works in tests, but the
    ``_require_provider_credentials`` validator below ensures production
    deployments fail loudly if the chosen provider is missing its keys.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Ignore unknown env vars so local shells full of unrelated
        # exports don't break startup.
        extra="ignore",
        case_sensitive=False,
    )

    # --- provider ----------------------------------------------------
    llm_provider: Literal["gemini", "openai", "azure"] = "gemini"

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "text-embedding-004"

    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_deployment: str = ""
    azure_openai_embedding_deployment: str = ""

    # --- runtime -----------------------------------------------------
    max_agent_steps: int = 10
    agent_timeout_seconds: int = 60
    llm_request_timeout_seconds: int = 30

    # --- safety ------------------------------------------------------
    enable_injection_detection: bool = False
    require_approval_for_writes: bool = False

    # --- observability ----------------------------------------------
    trace_store_max: int = 100
    log_level: str = "INFO"
    log_json: bool = True

    # --- caching -----------------------------------------------------
    tool_cache_ttl_seconds: int = 300
    tool_cache_maxsize: int = 512

    # --- storage paths ----------------------------------------------
    knowledge_data_path: Path = Field(
        default=PROJECT_ROOT / "data" / "acme-knowledge.json",
        description="JSON file backing the knowledge-base / RAG tool.",
    )
    tasks_data_path: Path = Field(
        default=PROJECT_ROOT / "data" / "sample-tasks.json",
        description="JSON file backing the task-manager tool's persistent state.",
    )
    chroma_path: Path = Field(
        default=PROJECT_ROOT / ".chroma",
        description="On-disk directory used by the Chroma vector store.",
    )

    # --- LangSmith (optional) ---------------------------------------
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "project-4-agents"

    # ----------------------------------------------------------------
    @model_validator(mode="after")
    def _require_provider_credentials(self) -> Settings:
        """Fail fast if the selected provider isn't actually configured."""
        # Each branch raises a specific, actionable error message so the
        # operator immediately knows which env var to set.
        if self.llm_provider == "gemini" and not self.gemini_api_key:
            raise ValueError("LLM_PROVIDER=gemini but GEMINI_API_KEY is empty.")
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("LLM_PROVIDER=openai but OPENAI_API_KEY is empty.")
        if self.llm_provider == "azure" and not (
            self.azure_openai_api_key
            and self.azure_openai_endpoint
            and self.azure_openai_deployment
        ):
            raise ValueError(
                "LLM_PROVIDER=azure requires AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, "
                "and AZURE_OPENAI_DEPLOYMENT."
            )
        return self

    def model_name(self) -> str:
        """Return the model identifier for the active provider.

        Different providers name their model field differently
        (``gemini_model`` vs ``openai_model`` vs ``azure_openai_deployment``),
        so callers use this helper instead of branching on ``llm_provider``
        every time they need to log or pass the model name.
        """
        return {
            "gemini": self.gemini_model,
            "openai": self.openai_model,
            "azure": self.azure_openai_deployment,
        }[self.llm_provider]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached so every Depends(get_settings) hands back the same instance.

    Tests override this with `app.dependency_overrides[get_settings] = ...`.
    """
    # ``type: ignore`` because pydantic-settings populates fields from the
    # environment, which mypy can't see from the constructor signature.
    return Settings()  # type: ignore[call-arg]
