"""Async ingestion: read JSON -> embed -> upsert into the vector store.

The Acme knowledge file is small and already structured (one entry per
topic), so each entry becomes one chunk. For larger or unstructured
docs, a chunker (week-3-style) plugs in here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.logging_config import get_logger
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore

log = get_logger(__name__)


def _load_entries(knowledge_path: Path) -> list[dict[str, Any]]:
    """Read the knowledge JSON file and turn each entry into one chunk."""
    data = json.loads(knowledge_path.read_text(encoding="utf-8"))
    chunks = []
    for i, entry in enumerate(data):
        # Topic + content + (optional) tags, joined into a single embedding-friendly blob.
        body = f"{entry['topic']}\n\n{entry['content']}"
        if entry.get("tags"):
            body += f"\n\nTags: {', '.join(entry['tags'])}"
        chunks.append({"text": body, "source": "acme-knowledge.json", "chunk_index": i})
    return chunks


async def ingest_knowledge(
    *,
    embeddings: EmbeddingService,
    store: VectorStore,
    knowledge_path: Path,
    force: bool = False,
) -> dict[str, Any]:
    """Embed and index every entry in ``knowledge_path``.

    Args:
        embeddings: Async embedding client.
        store: Vector store to upsert into.
        knowledge_path: Path to the JSON source.
        force: When ``False`` (default), skip work if the store already
            contains data. When ``True``, re-embed anyway (callers should
            usually pair this with :meth:`VectorStore.clear`).

    Returns:
        ``{"chunks_indexed", "skipped", "embedding_dimensions"}`` for the
        ``/rag/ingest`` response.
    """
    existing = await store.count()
    if existing > 0 and not force:
        log.info("ingest_skipped", existing=existing)
        return {"chunks_indexed": existing, "skipped": True, "embedding_dimensions": 0}

    chunks = _load_entries(knowledge_path)
    # One batched embedding call -- embed_texts handles paging internally.
    vectors = await embeddings.embed_texts([c["text"] for c in chunks])
    await store.add_chunks(chunks, vectors)
    dims = len(vectors[0]) if vectors else 0
    log.info("ingest_complete", chunks=len(chunks), dims=dims)
    return {"chunks_indexed": len(chunks), "skipped": False, "embedding_dimensions": dims}


async def ensure_indexed(
    *,
    embeddings: EmbeddingService,
    store: VectorStore,
    knowledge_path: Path,
) -> None:
    """Ingest only if the store is empty. Safe to call at startup.

    Failures during startup ingest are logged and swallowed -- the manual
    ``/rag/ingest`` endpoint is the recovery path. We don't want a flaky
    embedding provider to crash the entire process.
    """
    if await store.count() == 0:
        try:
            await ingest_knowledge(
                embeddings=embeddings, store=store, knowledge_path=knowledge_path, force=False
            )
        except Exception as e:  # noqa: BLE001
            # Don't crash boot on a transient embedding failure; ingest is exposed at /rag/ingest.
            log.warning("ingest_at_startup_failed", error=str(e))
