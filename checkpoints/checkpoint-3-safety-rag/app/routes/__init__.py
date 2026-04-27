"""FastAPI routers grouped by feature area.

Each module exposes an ``APIRouter`` that ``app.main`` mounts onto the
top-level FastAPI application:

- ``health`` — liveness probe.
- ``tools`` — tool listing + single-step tool calls.
- ``agent`` — multi-step LangGraph and raw-loop agent endpoints.
- ``safety`` — prompt-injection detector and tool permission inspector.
- ``rag`` — knowledge-base ingestion and status.
"""
