"""Safety check request/response models.

These cover the heuristic prompt-injection inspection exposed at
``POST /safety/check-prompt``. Detection logic itself lives in
:mod:`app.services.safety`; this module is just the wire contract.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PromptCheckRequest(BaseModel):
    """Body for ``POST /safety/check-prompt``."""

    text: str = Field(
        ...,
        min_length=1,
        description="The user-supplied text to inspect for injection patterns.",
    )


class SafetyCheckResult(BaseModel):
    """Outcome of a heuristic prompt-injection check.

    ``risk_level`` mirrors the count buckets in
    :func:`app.services.safety.check_prompt_injection`:
    ``"none"`` -> 0 findings, ``"medium"`` -> 1 finding, ``"high"`` -> 2+.
    """

    flagged: bool
    findings: list[dict]
    risk_level: str
