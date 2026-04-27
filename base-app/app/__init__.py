"""Base-app FastAPI application package.

This package is the minimal scaffold used as the starting point for the
live coding sessions. It contains only the engineering plumbing
(settings, logging, lifespan, dependency injection) so each checkpoint
can layer the agent-specific concepts on top without re-implementing
the foundation. See ``app.main`` for the FastAPI entrypoint.
"""
