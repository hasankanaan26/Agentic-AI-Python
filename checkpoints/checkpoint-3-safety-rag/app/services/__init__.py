"""Async service layer: LLM, embeddings, vector store, and safety helpers.

Services are constructed once at startup (see ``app.lifespan``) and shared
across requests via FastAPI ``Depends``. They wrap the third-party SDKs so
the rest of the codebase only sees a small, async, provider-agnostic API.
"""
