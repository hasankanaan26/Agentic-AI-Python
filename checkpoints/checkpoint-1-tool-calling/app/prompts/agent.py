"""System prompt for the single-step tool-calling agent.

Kept as a plain module-level constant so it shows up cleanly in code
review and is easy to import from routes/tests without dragging in
templating machinery.
"""

# System prompt sent on every `/tools/call` request. The phrasing
# nudges the model toward calling a tool when one fits, while still
# allowing direct answers for general questions.
TOOL_AGENT_PROMPT = """You are a helpful AI assistant with access to tools.

When a user asks a question that can be answered using your tools, call the appropriate tool.
When the question doesn't require a tool, answer directly from your knowledge.

Always prefer using a tool when one is relevant to the question."""
