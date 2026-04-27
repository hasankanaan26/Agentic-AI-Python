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
    """Strongly-typed configuration loaded from environment / ``.env`` file.

    Pydantic-settings turns environment variables into typed Python values and
    raises on unknown / malformed inputs. Every field below has a sensible
    default so the app can boot in dev mode by setting only the provider key.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- provider ----------------------------------------------------
    # Which LLM provider drives chat + embeddings. The validator below
    # ensures the matching credential block is also populated.
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
    # Hard ceilings used by both agent runners and the LLM HTTP client.
    max_agent_steps: int = 10
    agent_timeout_seconds: int = 60
    llm_request_timeout_seconds: int = 30

    # --- safety ------------------------------------------------------
    # Off by default so dev iteration is fast. Production should enable both.
    enable_injection_detection: bool = False
    require_approval_for_writes: bool = False

    # --- observability ----------------------------------------------
    trace_store_max: int = 100
    log_level: str = "INFO"
    log_json: bool = True

    # --- caching -----------------------------------------------------
    # Per-tool TTL caches (employee_lookup, knowledge_search) read these.
    tool_cache_ttl_seconds: int = 300
    tool_cache_maxsize: int = 512

    # --- storage paths ----------------------------------------------
    knowledge_data_path: Path = Field(
        default=PROJECT_ROOT / "data" / "acme-knowledge.json",
        description="Path to the JSON knowledge base ingested into Chroma.",
    )
    tasks_data_path: Path = Field(
        default=PROJECT_ROOT / "data" / "sample-tasks.json",
        description="Path to the seed JSON used by the task_manager tool.",
    )
    chroma_path: Path = Field(
        default=PROJECT_ROOT / ".chroma",
        description="On-disk location of the persistent Chroma collection.",
    )

    # --- LangSmith (optional) ---------------------------------------
    # When ``langchain_tracing_v2`` is true and a key is set, LangChain
    # automatically streams runs to LangSmith — no app-side wiring needed.
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "project-4-agents"

    # ----------------------------------------------------------------
    @model_validator(mode="after")
    def _require_provider_credentials(self) -> Settings:
        """Fail fast if the selected provider isn't actually configured."""
        # Crashing at boot is intentional: a half-configured app would
        # otherwise produce confusing 500s deep inside the LLM service.
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
        """Return the human-readable model identifier for the active provider.

        For Azure this is the deployment name (which functions as the model
        identifier in the OpenAI-compatible Azure API).
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
    return Settings()  # type: ignore[call-arg]
