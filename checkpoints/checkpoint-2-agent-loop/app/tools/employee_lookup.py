"""Acme Corp employee directory — read-only tool with TTL caching.

Uses an in-memory ``cachetools.TTLCache`` so repeated lookups within a
short window are free. The directory itself is hard-coded for the
teaching demo; in a real system this would back onto an HRIS API.
"""

from __future__ import annotations

from typing import Any, ClassVar

from cachetools import TTLCache

from app.models.tool import ToolResult
from app.tools.base import BaseTool

EMPLOYEE_DIRECTORY = [
    {
        "id": "EMP001", "name": "Alice Chen", "role": "CEO", "department": "Executive",
        "email": "alice.chen@acmecorp.com", "phone": "+1-555-0101",
    },
    {
        "id": "EMP002", "name": "Bob Kumar", "role": "CTO", "department": "Engineering",
        "email": "bob.kumar@acmecorp.com", "phone": "+1-555-0102",
    },
    {
        "id": "EMP003", "name": "Carol Martinez", "role": "VP of Product", "department": "Product",
        "email": "carol.martinez@acmecorp.com", "phone": "+1-555-0103",
    },
    {
        "id": "EMP004", "name": "David Park", "role": "Senior Engineer", "department": "Engineering",
        "email": "david.park@acmecorp.com", "phone": "+1-555-0104",
    },
    {
        "id": "EMP005", "name": "Emma Wilson", "role": "HR Manager", "department": "Human Resources",
        "email": "emma.wilson@acmecorp.com", "phone": "+1-555-0105",
    },
    {
        "id": "EMP006", "name": "Frank Liu", "role": "Data Scientist", "department": "Engineering",
        "email": "frank.liu@acmecorp.com", "phone": "+1-555-0106",
    },
    {
        "id": "EMP007", "name": "Grace Okafor", "role": "Designer", "department": "Product",
        "email": "grace.okafor@acmecorp.com", "phone": "+1-555-0107",
    },
    {
        "id": "EMP008", "name": "Hasan Al-Rashid", "role": "DevOps Engineer", "department": "Engineering",
        "email": "hasan.alrashid@acmecorp.com", "phone": "+1-555-0108",
    },
]


class EmployeeLookupTool(BaseTool):
    """Searches the in-memory employee directory by name, role, or department."""

    name: ClassVar[str] = "employee_lookup"
    permission: ClassVar[str] = "read"
    definition: ClassVar[dict[str, Any]] = {
        "name": "employee_lookup",
        "description": (
            "Look up Acme Corp employee information by name, department, or role. "
            "Use when asked about team members, org structure, or contact details."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Employee name, department, or role."},
                "include_contact": {
                    "type": "boolean",
                    "description": "Include email and phone in results.",
                },
            },
            "required": ["query"],
        },
    }

    def __init__(self, cache_ttl: int = 300, cache_max: int = 256) -> None:
        """Initialize the tool with a TTL cache.

        Args:
            cache_ttl: Cache entry lifetime in seconds (default 5 minutes).
            cache_max: Maximum number of distinct lookups to retain.
        """
        # TTL cache: same lookup within `cache_ttl` returns the same answer.
        # Worth caching because directory churn is rare.
        self._cache: TTLCache = TTLCache(maxsize=cache_max, ttl=cache_ttl)

    async def run(self, query: str, include_contact: bool = False) -> ToolResult:
        """Search the directory and return formatted matches.

        Args:
            query: Substring to match against name, role, or department.
            include_contact: If True, append email/phone columns to results.

        Returns:
            ``ToolResult.ok`` with a multi-line list of matches (possibly
            empty), or ``ToolResult.fail`` if the query is too short.
        """
        # Reject too-short queries up-front to avoid swamping the LLM with
        # the entire directory from a one-character input.
        if len(query.strip()) < 2:
            return ToolResult.fail("Query must be at least 2 characters.")
        # Cache key includes the contact flag because output formatting differs.
        cache_key = (query.lower(), include_contact)
        if cache_key in self._cache:
            return ToolResult.ok(self._cache[cache_key], cached=True)

        # Linear substring scan over the in-memory directory — small enough
        # that any indexing would be over-engineering for the demo.
        q = query.lower()
        matches = [
            emp for emp in EMPLOYEE_DIRECTORY
            if q in f"{emp['name']} {emp['role']} {emp['department']}".lower()
        ]
        if not matches:
            # Empty result is still a successful lookup — just zero matches.
            return ToolResult.ok(f"No employees found matching '{query}'.")

        lines = []
        for emp in matches:
            line = f"- {emp['name']} | {emp['role']} | {emp['department']}"
            if include_contact:
                line += f" | {emp['email']} | {emp['phone']}"
            lines.append(line)
        out = f"Found {len(matches)} employee(s) matching '{query}':\n" + "\n".join(lines)
        self._cache[cache_key] = out
        return ToolResult.ok(out, cached=False, matches=len(matches))
