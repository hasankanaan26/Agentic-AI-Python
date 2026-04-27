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
    """Read the knowledge JSON and convert each entry to a chunk dict."""
    data = json.loads(knowledge_path.read_text(encoding="utf-8"))
    chunks = []
    for i, entry in enumerate(data):
        # Compose the embedding text from topic + content + tags so semantic
        # search can match on any of them.
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
    """Embed the knowledge file and upsert into the vector store.

    Args:
        embeddings: Embedding service used to vectorize chunks.
        store: Vector store receiving the upsert.
        knowledge_path: Path to the JSON source file.
        force: When ``False`` (default) a populated store is left alone;
            when ``True`` we re-embed every entry (upsert handles ID conflicts).

    Returns:
        ``{"chunks_indexed": int, "skipped": bool, "embedding_dimensions": int}``.
    """
    existing = await store.count()
    if existing > 0 and not force:
        # Idempotent fast-path; running ingest twice in a row is a no-op.
        log.info("ingest_skipped", existing=existing)
        return {"chunks_indexed": existing, "skipped": True, "embedding_dimensions": 0}

    chunks = _load_entries(knowledge_path)
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

    Catches any exception so a flaky embedding endpoint doesn't break boot —
    operators can still re-run ingestion via ``POST /rag/ingest``.
    """
    if await store.count() == 0:
        try:
            await ingest_knowledge(
                embeddings=embeddings, store=store, knowledge_path=knowledge_path, force=False
            )
        except Exception as e:  # noqa: BLE001
            # Don't crash boot on a transient embedding failure; ingest is exposed at /rag/ingest.
            log.warning("ingest_at_startup_failed", error=str(e))
