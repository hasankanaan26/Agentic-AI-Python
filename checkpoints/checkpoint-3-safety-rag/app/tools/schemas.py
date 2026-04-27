"""Pydantic args_schema for LangChain tools.

LangChain validates inputs against args_schema BEFORE calling the
underlying function — catches bad LLM arguments at the framework
boundary instead of inside our tool code.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class CalculatorInput(BaseModel):
    """args_schema for the ``calculator`` LangChain tool."""

    operation: str = Field(description="add, subtract, multiply, or divide.")
    a: float = Field(description="The first operand.")
    b: float = Field(description="The second operand.")

    @field_validator("operation")
    @classmethod
    def _check_op(cls, v: str) -> str:
        """Validate ``operation`` against the supported set."""
        allowed = {"add", "subtract", "multiply", "divide"}
        if v not in allowed:
            raise ValueError(f"Operation must be one of {allowed}, got '{v}'")
        return v


class ClockInput(BaseModel):
    """args_schema for the ``clock`` LangChain tool."""

    format: str = Field(default="both", description="'date', 'time', or 'both'.")

    @field_validator("format")
    @classmethod
    def _check_fmt(cls, v: str) -> str:
        """Validate ``format`` against the supported set."""
        allowed = {"date", "time", "both"}
        if v not in allowed:
            raise ValueError(f"Format must be one of {allowed}, got '{v}'")
        return v


class EmployeeLookupInput(BaseModel):
    """args_schema for the ``employee_lookup`` LangChain tool."""

    query: str = Field(description="Employee name, department, or role.")
    include_contact: bool = Field(
        default=False,
        description="When true, include email and phone in the rendered output.",
    )

    @field_validator("query")
    @classmethod
    def _min_len(cls, v: str) -> str:
        """Reject queries shorter than 2 characters (after trimming)."""
        if len(v.strip()) < 2:
            raise ValueError("Query must be at least 2 characters.")
        return v.strip()
