"""ChromaDB wrapper exposed via async methods.

Chroma's Python client is sync. We wrap each call with `run_in_threadpool`
so the event loop stays free — the route handler awaits, the actual work
happens on anyio's thread pool.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import chromadb
from fastapi.concurrency import run_in_threadpool

from app.logging_config import get_logger

log = get_logger(__name__)

# Single Chroma collection name used across the app — change cautiously,
# any persisted data under the previous name becomes orphaned.
COLLECTION_NAME = "acme_knowledge"


class VectorStore:
    """Async-friendly wrapper around a persistent Chroma collection.

    One instance per process; created in lifespan. Every public method
    hops to the threadpool because Chroma's Python client is synchronous —
    running it inline would block the event loop and stall every other
    in-flight request.
    """

    def __init__(self, chroma_path: Path) -> None:
        self._client = chromadb.PersistentClient(path=str(chroma_path))
        # ``get_or_create_collection`` makes startup idempotent — first boot
        # creates the collection, every subsequent boot reuses the on-disk one.
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        log.info("vector_store_ready", path=str(chroma_path), count=self._collection.count())

    async def add_chunks(self, chunks: list[dict], embeddings: list[list[float]]) -> int:
        """Upsert ``chunks`` and their precomputed ``embeddings`` into the store.

        Args:
            chunks: Each entry must have ``source``, ``chunk_index``, and ``text``.
            embeddings: Aligned with ``chunks`` — same length, same order.

        Returns:
            The number of chunks written.
        """

        def _do() -> int:
            # Stable IDs so re-running ingestion *updates* rather than duplicates.
            ids = [f"{c['source']}_{c['chunk_index']}" for c in chunks]
            self._collection.upsert(
                ids=ids,
                documents=[c["text"] for c in chunks],
                embeddings=embeddings,
                metadatas=[
                    {"source": c["source"], "chunk_index": c["chunk_index"]} for c in chunks
                ],
            )
            return len(chunks)

        # Hop to threadpool — chromadb's API is sync only.
        return await run_in_threadpool(_do)

    async def search(self, query_embedding: list[float], top_k: int = 3) -> list[dict]:
        """Find the ``top_k`` nearest chunks to ``query_embedding``.

        Returns an empty list if the store is empty (rather than raising) so
        callers can render a friendly "no matches" message.
        """

        def _do() -> list[dict]:
            n_stored = self._collection.count()
            if n_stored == 0:
                return []
            # Cap ``n_results`` at the collection size — Chroma errors otherwise.
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, n_stored),
            )
            # Chroma returns parallel arrays; reshape to a list of dicts.
            return [
                {
                    "text": results["documents"][0][i],
                    "source": results["metadatas"][0][i]["source"],
                    "chunk_index": results["metadatas"][0][i]["chunk_index"],
                    "distance": results["distances"][0][i],
                }
                for i in range(len(results["ids"][0]))
            ]

        return await run_in_threadpool(_do)

    async def count(self) -> int:
        """Return the total number of indexed chunks."""
        return await run_in_threadpool(self._collection.count)

    async def clear(self) -> None:
        """Drop and recreate the collection (used by tests / re-ingest flows)."""

        def _do() -> None:
            # Collection may not exist yet on a fresh disk; ignore that case.
            with contextlib.suppress(ValueError):
                self._client.delete_collection(name=COLLECTION_NAME)
            # Recreate so the field on this instance stays valid.
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

        await run_in_threadpool(_do)
