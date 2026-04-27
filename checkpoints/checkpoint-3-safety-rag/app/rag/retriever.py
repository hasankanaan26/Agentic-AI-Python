"""Embed-then-search in one call. The "R" in RAG."""

from __future__ import annotations

from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore


async def retrieve(
    query: str, *, embeddings: EmbeddingService, store: VectorStore, top_k: int = 3
) -> list[dict]:
    """Embed ``query`` and return the ``top_k`` closest chunks.

    Args:
        query: Natural-language query string.
        embeddings: Async embedding client.
        store: Vector store to search.
        top_k: Maximum number of chunks to return.

    Returns:
        List of chunk dicts (``text``, ``source``, ``chunk_index``,
        ``distance``). Empty when the store is unindexed — caller can
        fall back to a "no matches" message rather than crashing.
    """
    # Skip the embedding round-trip entirely if there's nothing to match against.
    if await store.count() == 0:
        return []
    vector = await embeddings.embed_text(query)
    return await store.search(vector, top_k=top_k)
