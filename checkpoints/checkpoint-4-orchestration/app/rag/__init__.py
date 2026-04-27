"""Retrieval-Augmented Generation helpers.

* :mod:`app.rag.ingest` -- read JSON, embed, upsert into Chroma.
* :mod:`app.rag.retriever` -- embed-then-search in one call.

The vector store and embedding service are passed in as dependencies so
the same logic can be unit tested with fakes.
"""
