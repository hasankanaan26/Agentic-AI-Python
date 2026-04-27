"""Executor prompt template fed into the LangGraph runner per plan step.

Filled with ``step_number``, ``description``, and ``tool_needed`` from the
:class:`app.models.PlanStep` the orchestrator is currently running.
"""

EXECUTOR_PROMPT_TEMPLATE = """Execute this specific step of a larger plan:

Step {step_number}: {description}

Use the {tool_needed} tool to accomplish this. Be precise and return the result clearly."""
