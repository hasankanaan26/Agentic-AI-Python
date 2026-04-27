"""FastAPI routers, one module per concern.

Routes are intentionally thin: parse the request, call the relevant
service via :mod:`app.deps`, return a Pydantic response. Business logic
lives in :mod:`app.services` / :mod:`app.agents`, never in handlers.
"""
