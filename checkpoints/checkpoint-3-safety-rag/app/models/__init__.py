"""Pydantic models — the boundary between this app and everything else.

Every HTTP request body and response shape is declared here. Routes import
the names they need so the ``models/`` package functions as a single
catalogue of public types.
"""

from app.models.agent import (
    AgentRequest,
    AgentResponse,
    AgentStep,
    ApprovalAction,
    LangGraphAgentResponse,
    ProposedToolCall,
    SafeAgentRequest,
)
from app.models.rag import IngestResponse, RagStatus
from app.models.safety import PromptCheckRequest, SafetyCheckResult
from app.models.tool import (
    ToolCallInfo,
    ToolCallRequest,
    ToolCallResponse,
    ToolDefinition,
    ToolResult,
)

__all__ = [
    "AgentRequest",
    "AgentResponse",
    "AgentStep",
    "ApprovalAction",
    "IngestResponse",
    "LangGraphAgentResponse",
    "PromptCheckRequest",
    "ProposedToolCall",
    "RagStatus",
    "SafeAgentRequest",
    "SafetyCheckResult",
    "ToolCallInfo",
    "ToolCallRequest",
    "ToolCallResponse",
    "ToolDefinition",
    "ToolResult",
]
