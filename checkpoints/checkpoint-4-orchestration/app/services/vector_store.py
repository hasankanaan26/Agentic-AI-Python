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

COLLECTION_NAME = "acme_knowledge"


class VectorStore:
    """Thin async façade over a persistent ChromaDB collection.

    One instance per process is created in lifespan and shared via DI.
    Cosine distance is used because the embedding models we support
    return non-normalised vectors -- cosine collapses magnitude to 1.
    """

    def __init__(self, chroma_path: Path) -> None:
        self._client = chromadb.PersistentClient(path=str(chroma_path))
        # ``get_or_create`` keeps the call idempotent across restarts.
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        log.info("vector_store_ready", path=str(chroma_path), count=self._collection.count())

    async def add_chunks(self, chunks: list[dict], embeddings: list[list[float]]) -> int:
        """Upsert ``chunks`` with their pre-computed ``embeddings``.

        Args:
            chunks: Each dict must carry ``text``, ``source``, ``chunk_index``.
            embeddings: One vector per chunk, same order.

        Returns:
            Number of chunks written.
        """

        # Inner sync function -- Chroma's client is not async-aware.
        def _do() -> int:
            # IDs are deterministic so re-ingest is idempotent (upsert overwrites).
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

        # Hop to anyio's thread pool so we don't block the event loop on disk I/O.
        return await run_in_threadpool(_do)

    async def search(self, query_embedding: list[float], top_k: int = 3) -> list[dict]:
        """Return up to ``top_k`` nearest chunks to ``query_embedding``.

        Returns an empty list when the collection is empty so callers don't
        need to special-case "before ingest".
        """

        def _do() -> list[dict]:
            n_stored = self._collection.count()
            if n_stored == 0:
                return []
            # Cap ``n_results`` by what's actually present -- Chroma errors
            # if you ask for more than the collection holds.
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, n_stored),
            )
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
        """Return how many vectors are currently indexed (off-thread)."""
        return await run_in_threadpool(self._collection.count)

    async def clear(self) -> None:
        """Drop and recreate the collection. Used by tests and force-ingest."""

        def _do() -> None:
            # ``delete_collection`` raises ValueError if the collection doesn't exist;
            # swallow that case so the call is idempotent.
            with contextlib.suppress(ValueError):
                self._client.delete_collection(name=COLLECTION_NAME)
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

        await run_in_threadpool(_do)
