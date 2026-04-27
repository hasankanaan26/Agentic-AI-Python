"""Embed-then-search in one call. The "R" in RAG.

Tiny wrapper that turns a natural-language ``query`` into a vector and
forwards it to the store. Kept as a free function rather than a class
because there's nothing to hold onto -- both clients are passed in.
"""

from __future__ import annotations

from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore


async def retrieve(
    query: str, *, embeddings: EmbeddingService, store: VectorStore, top_k: int = 3
) -> list[dict]:
    """Embed ``query`` and return the ``top_k`` nearest chunks.

    Returns an empty list (not an error) when the store hasn't been
    populated yet so callers can show a friendly "no knowledge" message.
    """
    # Cheap pre-check: skip the embedding call entirely on an empty store.
    if await store.count() == 0:
        return []
    vector = await embeddings.embed_text(query)
    return await store.search(vector, top_k=top_k)
