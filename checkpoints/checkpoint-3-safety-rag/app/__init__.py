"""Top-level FastAPI application package for checkpoint-3 (safety + RAG).

This package wires together the agentic AI service:

- ``settings`` / ``deps`` / ``lifespan`` — configuration, DI, startup/shutdown.
- ``services`` — async LLM, embedding, vector store, and safety primitives.
- ``tools`` — registry of callable tools the agent can invoke.
- ``rag`` — embed/index/retrieve helpers backing the ``knowledge_search`` tool.
- ``agents`` — both the from-scratch loop and the LangGraph ReAct runner.
- ``routes`` — HTTP surface area exposed by ``main.app``.
"""
