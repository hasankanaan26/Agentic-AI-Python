"""Retrieval-Augmented Generation building blocks.

Two small modules:

- ``ingest`` — read the JSON knowledge base, embed it, and upsert into Chroma.
- ``retriever`` — embed a user query and search the Chroma collection.

Both are async wrappers around the singletons in ``app.services``.
"""
