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

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Typed configuration sourced from environment variables and ``.env``.

    Use :func:`get_settings` (lru-cached) to obtain the shared instance.
    Tests override via ``app.dependency_overrides[get_settings] = ...``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",          # Ignore unknown env vars instead of raising.
        case_sensitive=False,    # ``LLM_PROVIDER`` and ``llm_provider`` both work.
    )

    # --- provider ----------------------------------------------------
    llm_provider: Literal["gemini", "openai", "azure"] = Field(
        default="gemini", description="Which provider stack the app uses."
    )

    gemini_api_key: str = Field(default="", description="API key for Google Gemini.")
    gemini_model: str = Field(default="gemini-2.5-flash", description="Gemini chat model id.")
    gemini_embedding_model: str = Field(
        default="text-embedding-004", description="Gemini embedding model id."
    )

    openai_api_key: str = Field(default="", description="API key for OpenAI.")
    openai_model: str = Field(default="gpt-4o", description="OpenAI chat model id.")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", description="OpenAI embedding model id."
    )

    azure_openai_api_key: str = Field(default="", description="Azure OpenAI key.")
    azure_openai_endpoint: str = Field(default="", description="Azure OpenAI endpoint URL.")
    azure_openai_api_version: str = Field(
        default="2024-10-21", description="Azure OpenAI API version pin."
    )
    azure_openai_deployment: str = Field(
        default="", description="Azure deployment name for chat completions."
    )
    azure_openai_embedding_deployment: str = Field(
        default="", description="Azure deployment name for embeddings."
    )

    # --- runtime -----------------------------------------------------
    max_agent_steps: int = Field(default=10, description="Hard cap on agent loop iterations.")
    agent_timeout_seconds: int = Field(
        default=60, description="(Reserved) end-to-end agent timeout in seconds."
    )
    llm_request_timeout_seconds: int = Field(
        default=30, description="Per-request HTTP timeout for LLM/embedding calls."
    )

    # --- safety ------------------------------------------------------
    enable_injection_detection: bool = Field(
        default=False,
        description="Run heuristic injection check before agent/orchestrate runs.",
    )
    require_approval_for_writes: bool = Field(
        default=False,
        description="(Reserved) default approval flag for write tools when not provided.",
    )

    # --- observability ----------------------------------------------
    trace_store_max: int = Field(default=100, description="Max traces kept in memory.")
    log_level: str = Field(default="INFO", description="Root log level.")
    log_json: bool = Field(default=True, description="Emit JSON logs (true) or pretty (false).")

    # --- caching -----------------------------------------------------
    tool_cache_ttl_seconds: int = Field(default=300, description="TTL cache lifetime for tools.")
    tool_cache_maxsize: int = Field(default=512, description="TTL cache capacity for tools.")

    # --- storage paths ----------------------------------------------
    knowledge_data_path: Path = Field(
        default=PROJECT_ROOT / "data" / "acme-knowledge.json",
        description="Path to the JSON knowledge base for RAG ingest.",
    )
    tasks_data_path: Path = Field(
        default=PROJECT_ROOT / "data" / "sample-tasks.json",
        description="Path to the seed tasks JSON used by task_manager.",
    )
    chroma_path: Path = Field(
        default=PROJECT_ROOT / ".chroma",
        description="Persistent ChromaDB directory for the vector store.",
    )

    # --- LangSmith (optional) ---------------------------------------
    langchain_tracing_v2: bool = Field(
        default=False,
        description="Enable LangSmith auto-tracing (also requires LANGCHAIN_API_KEY).",
    )
    langchain_api_key: str = Field(default="", description="LangSmith API key.")
    langchain_project: str = Field(
        default="project-4-agents", description="LangSmith project name."
    )

    # ----------------------------------------------------------------
    @model_validator(mode="after")
    def _require_provider_credentials(self) -> Settings:
        """Fail fast if the selected provider isn't actually configured.

        Runs after pydantic populates the model so we can cross-reference
        ``llm_provider`` and the per-provider credentials in one place.
        """
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
        """Return the provider-specific model identifier currently in use."""
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
    return Settings()  # type: ignore[call-arg]
