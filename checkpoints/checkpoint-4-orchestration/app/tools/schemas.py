"""Pydantic args_schema for LangChain tools.

LangChain validates inputs against args_schema BEFORE calling the
underlying function — catches bad LLM arguments at the framework
boundary instead of inside our tool code.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class CalculatorInput(BaseModel):
    """Validated arguments for :class:`app.tools.calculator.CalculatorTool`."""

    operation: str = Field(description="add, subtract, multiply, or divide.")
    a: float = Field(description="Left operand.")
    b: float = Field(description="Right operand.")

    @field_validator("operation")
    @classmethod
    def _check_op(cls, v: str) -> str:
        """Reject unknown operations before they reach the tool."""
        allowed = {"add", "subtract", "multiply", "divide"}
        if v not in allowed:
            raise ValueError(f"Operation must be one of {allowed}, got '{v}'")
        return v


class ClockInput(BaseModel):
    """Validated arguments for :class:`app.tools.clock.ClockTool`."""

    format: str = Field(default="both", description="'date', 'time', or 'both'.")

    @field_validator("format")
    @classmethod
    def _check_fmt(cls, v: str) -> str:
        """Reject unknown format selectors."""
        allowed = {"date", "time", "both"}
        if v not in allowed:
            raise ValueError(f"Format must be one of {allowed}, got '{v}'")
        return v


class EmployeeLookupInput(BaseModel):
    """Validated arguments for :class:`app.tools.employee_lookup.EmployeeLookupTool`."""

    query: str = Field(description="Employee name, department, or role.")
    include_contact: bool = Field(
        default=False,
        description="When true, include email + phone in each result row.",
    )

    @field_validator("query")
    @classmethod
    def _min_len(cls, v: str) -> str:
        """Trim whitespace and require at least 2 chars (avoids overly broad searches)."""
        if len(v.strip()) < 2:
            raise ValueError("Query must be at least 2 characters.")
        return v.strip()
