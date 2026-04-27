"""Service layer — long-lived collaborators built once and shared via DI.

Services in this package wrap external systems (LLM providers today; vector
stores and orchestrators in later checkpoints). They are constructed during
the FastAPI lifespan and injected into routes through `app.deps`.
"""
