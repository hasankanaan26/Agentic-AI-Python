"""Async service clients used by routes, tools, and agents.

Every singleton lives here: the LLM client, the embedding client, the
ChromaDB wrapper, the heuristic safety check, the in-memory trace store,
and the planner+executor orchestrator. Each module documents its own
lifecycle expectations (build once in lifespan, share via Depends).
"""
