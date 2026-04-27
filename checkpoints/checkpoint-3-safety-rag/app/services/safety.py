"""Heuristic prompt-injection detection. Pure function, no I/O.

This is a defense-in-depth layer, not a guarantee — sophisticated
injections will slip through the regex list. The point is to catch the
"low-effort obvious" attacks (``ignore previous instructions``, ``you are
now``, etc.) cheaply, before any LLM call is made.
"""

from __future__ import annotations

import re

# Regex patterns covering the most common prompt-injection idioms. All are
# case-insensitive and match anywhere in the input.
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

# Map a fragment of each pattern to a human-readable description used in
# the API response. Keys are matched as substrings of the regex source.
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
    """Scan ``text`` for known prompt-injection patterns.

    Args:
        text: The user-supplied content to inspect.

    Returns:
        A dict with keys ``flagged`` (bool), ``findings`` (list of
        ``{pattern, description}``), and ``risk_level`` (``"none"``,
        ``"medium"``, or ``"high"``). ``high`` requires two or more matches.
    """
    findings = [
        {"pattern": p, "description": _describe(p)}
        for p in INJECTION_PATTERNS
        if re.search(p, text)
    ]
    return {
        "flagged": bool(findings),
        "findings": findings,
        # Multiple matches is much stronger evidence than a single keyword.
        "risk_level": "high" if len(findings) >= 2 else "medium" if findings else "none",
    }


def _describe(pattern: str) -> str:
    """Return a human-readable description for a matched pattern."""
    for key, desc in _DESCRIPTIONS.items():
        if key in pattern:
            return desc
    return "Suspicious pattern detected"
