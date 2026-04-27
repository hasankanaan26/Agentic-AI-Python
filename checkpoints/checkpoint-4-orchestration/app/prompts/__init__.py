"""Prompt templates as plain strings — easy to read in code review."""

from app.prompts.agent import AGENT_LOOP_PROMPT, TOOL_AGENT_PROMPT
from app.prompts.executor import EXECUTOR_PROMPT_TEMPLATE
from app.prompts.planner import PLANNER_PROMPT
from app.prompts.safety import SAFETY_SYSTEM_PROMPT

__all__ = [
    "AGENT_LOOP_PROMPT",
    "EXECUTOR_PROMPT_TEMPLATE",
    "PLANNER_PROMPT",
    "SAFETY_SYSTEM_PROMPT",
    "TOOL_AGENT_PROMPT",
]
