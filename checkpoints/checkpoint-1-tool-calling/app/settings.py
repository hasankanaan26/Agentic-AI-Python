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

# Repo root — `parents[4]` walks up:
#   settings.py -> app -> checkpoint-1-tool-calling -> checkpoints -> <repo root>
# We anchor data file defaults here so the app works regardless of where
# uvicorn happens to be launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[4]


class Settings(BaseSettings):
    """Strongly-typed view of the application's environment configuration.

    All settings are read from environment variables (or a `.env` file) and
    validated by pydantic at startup. A single `Settings` instance is built
    via `get_settings()` and shared across the process; tests swap it out
    by overriding the FastAPI dependency.
    """

    # `pydantic-settings` will also pull values from a local `.env` file if
    # present. `extra="ignore"` keeps unrelated keys from blowing up boot.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- provider ----------------------------------------------------
    # Which LLM backend the app should use. Each provider has its own
    # block of credentials below; only the selected one is required.
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
    # Bounds on agent execution. `max_agent_steps` caps tool-call loops in
    # later checkpoints; `*_timeout_seconds` keep external calls from
    # hanging an HTTP request indefinitely.
    max_agent_steps: int = 10
    agent_timeout_seconds: int = 60
    llm_request_timeout_seconds: int = 30

    # --- safety ------------------------------------------------------
    # Toggles for guardrails added in later checkpoints. Off by default
    # in CP1 since the surface area is intentionally tiny.
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
    # Defaults are derived from PROJECT_ROOT so the app finds its data
    # files no matter what the current working directory is.
    knowledge_data_path: Path = Field(
        default=PROJECT_ROOT / "data" / "acme-knowledge.json",
        description="Path to the seed knowledge-base JSON used by RAG in later checkpoints.",
    )
    tasks_data_path: Path = Field(
        default=PROJECT_ROOT / "data" / "sample-tasks.json",
        description="Path to the sample tasks JSON consumed by task-management tools.",
    )
    chroma_path: Path = Field(
        default=PROJECT_ROOT / ".chroma",
        description="On-disk location for the Chroma vector store (persists between runs).",
    )

    # --- LangSmith (optional) ---------------------------------------
    # If `langchain_tracing_v2` is true and an API key is set, LangChain
    # will ship traces to LangSmith. Disabled by default for offline use.
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "project-4-agents"

    # ----------------------------------------------------------------
    @model_validator(mode="after")
    def _require_provider_credentials(self) -> Settings:
        """Fail fast if the selected provider isn't actually configured."""
        # Each branch only checks the credentials relevant to the chosen
        # provider; the other blocks may legitimately be empty.
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
        """Return the provider-specific model identifier currently in use.

        Returns:
            The model name for OpenAI/Gemini, or the deployment name for
            Azure (Azure routes by deployment, not model id).
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
    # `lru_cache(maxsize=1)` turns this into a lazy singleton: the first
    # call constructs and validates the settings; subsequent calls return
    # the same object without re-reading the environment.
    return Settings()  # type: ignore[call-arg]
