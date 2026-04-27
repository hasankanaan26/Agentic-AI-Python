"""Heuristic prompt-injection detection. Pure function, no I/O.

Deliberately a regex sweep -- not a model call. It catches the
low-effort attacks (the "ignore previous instructions" family) at zero
cost, which is the right cheap layer to put before any expensive guard
model. Users keen on stronger defences should plug in something like
LLM-Guard or Lakera here.
"""

from __future__ import annotations

import re

# Each pattern targets a class of common jailbreak phrasing.
# ``(?i)`` makes them case-insensitive.
INJECTION_PATTERNS = [
    r"(?i)ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)",
    r"(?i)disregard\s+(all\s+)?(previous|prior|above)",
    r"(?i)you\s+are\s+now\s+(a|an)\s+",
    r"(?i)^system\s*:",
    r"(?i)forget\s+(everything|all|your)\s+(you|instructions|rules)",
    r"(?i)override\s+(your|all|the)\s+(instructions|rules|constraints)",
    r"(?i)act\s+as\s+if\s+you\s+(have\s+)?no\s+(restrictions|rules|limits)",
    r"(?i)pretend\s+(you\s+are|to\s+be)\s+",
]

# Substring lookup -> human description; first key found in the pattern wins.
_DESCRIPTIONS = {
    "ignore": "Attempts to override previous instructions",
    "disregard": "Attempts to disregard instructions",
    "you\\s+are\\s+now": "Attempts to change assistant identity",
    "system": "Attempts to inject system-level instructions",
    "forget": "Attempts to erase instruction memory",
    "override": "Attempts to override constraints",
    "act\\s+as": "Attempts to remove restrictions",
    "pretend": "Attempts to change assistant behavior",
}


def check_prompt_injection(text: str) -> dict:
    """Run the regex sweep against ``text`` and bucket the results.

    Args:
        text: User input to inspect.

    Returns:
        ``{"flagged": bool, "findings": [...], "risk_level": "none"|"medium"|"high"}``.
        ``risk_level`` is bumped to ``"high"`` when 2+ patterns trigger,
        which strongly suggests an intentional injection attempt.
    """
    findings = [
        {"pattern": p, "description": _describe(p)}
        for p in INJECTION_PATTERNS
        if re.search(p, text)
    ]
    return {
        "flagged": bool(findings),
        "findings": findings,
        "risk_level": "high" if len(findings) >= 2 else "medium" if findings else "none",
    }


def _describe(pattern: str) -> str:
    """Return a human-readable label for a matched pattern."""
    for key, desc in _DESCRIPTIONS.items():
        if key in pattern:
            return desc
    return "Suspicious pattern detected"
