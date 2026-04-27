"""Pydantic models for the ``/safety`` endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PromptCheckRequest(BaseModel):
    """Body for ``POST /safety/check-prompt``."""

    text: str = Field(..., min_length=1, description="Text to scan for prompt-injection patterns.")


class SafetyCheckResult(BaseModel):
    """Outcome of a prompt-injection scan."""

    flagged: bool = Field(description="True when one or more patterns matched the input.")
    findings: list[dict] = Field(
        description="One entry per matched pattern (``{pattern, description}``)."
    )
    risk_level: str = Field(description="'none', 'medium', or 'high' (>=2 matches).")
