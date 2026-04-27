"""Top-level application package for checkpoint-4 (orchestration).

Exposes the FastAPI app via ``app.main`` and shared singletons via
``app.lifespan`` / ``app.deps``. Sub-packages map to a single concern each:

* :mod:`app.models` -- Pydantic edge contracts.
* :mod:`app.services` -- async clients (LLM, embeddings, vector store, tracer).
* :mod:`app.tools` -- the tool catalogue and the registry that executes them.
* :mod:`app.agents` -- raw and LangGraph-based agent loops.
* :mod:`app.rag` -- knowledge-base ingest + retrieval.
* :mod:`app.routes` -- FastAPI routers.
"""
