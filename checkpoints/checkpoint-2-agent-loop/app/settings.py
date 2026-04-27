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

# Anchor for default file paths: walk up three directories from this file
# (app/ -> checkpoint-2-agent-loop/ -> checkpoints/ -> repo root).
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """All runtime configuration, loaded from environment variables and ``.env``.

    Pydantic-settings handles type coercion and validation. Required keys
    for the selected provider are checked in
    :meth:`_require_provider_credentials` so misconfiguration fails at boot
    rather than mid-request.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",       # tolerate unrelated keys in the environment
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
    max_agent_steps: int = 10              # hard ceiling on think->act->observe loops
    agent_timeout_seconds: int = 60        # wall-clock budget per agent run
    llm_request_timeout_seconds: int = 30  # per-call HTTP timeout for LLM SDKs

    # --- safety ------------------------------------------------------
    enable_injection_detection: bool = False  # toggles the prompt-injection guard
    require_approval_for_writes: bool = False  # human-in-the-loop for write tools

    # --- observability ----------------------------------------------
    trace_store_max: int = 100  # max traces retained in memory for the UI
    log_level: str = "INFO"
    log_json: bool = True

    # --- caching -----------------------------------------------------
    tool_cache_ttl_seconds: int = 300
    tool_cache_maxsize: int = 512

    # --- storage paths ----------------------------------------------
    knowledge_data_path: Path = Field(
        default=PROJECT_ROOT / "data" / "acme-knowledge.json",
        description="JSON corpus loaded into the RAG index in later checkpoints.",
    )
    tasks_data_path: Path = Field(
        default=PROJECT_ROOT / "data" / "sample-tasks.json",
        description="Seed file backing the task_manager tool.",
    )
    chroma_path: Path = Field(
        default=PROJECT_ROOT / ".chroma",
        description="On-disk persistence directory for the Chroma vector store.",
    )

    # --- LangSmith (optional) ---------------------------------------
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "project-4-agents"

    # ----------------------------------------------------------------
    @model_validator(mode="after")
    def _require_provider_credentials(self) -> Settings:
        """Fail fast if the selected provider isn't actually configured."""
        # Each branch checks just enough to know the chosen provider can boot.
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
        """Return the model/deployment identifier for the selected provider."""
        return {
            "gemini": self.gemini_model,
            "openai": self.openai_model,
            # On Azure the "model" the SDK takes is actually a deployment name.
            "azure": self.azure_openai_deployment,
        }[self.llm_provider]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached :class:`Settings` instance.

    The ``lru_cache`` ensures every ``Depends(get_settings)`` resolution and
    every internal call returns the *same* object — important because some
    callers stash references to it.

    Tests override this with ``app.dependency_overrides[get_settings] = ...``.
    """
    return Settings()  # type: ignore[call-arg]
