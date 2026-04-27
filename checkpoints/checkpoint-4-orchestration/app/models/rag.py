"""RAG ingest/status response models.

Returned by ``POST /rag/ingest`` and ``GET /rag/status``. The actual
ingest pipeline lives in :mod:`app.rag.ingest`.
"""

from __future__ import annotations

from pydantic import BaseModel


class IngestResponse(BaseModel):
    """Result of running the knowledge ingest job.

    ``embedding_dimensions`` is ``0`` when the ingest was a no-op (collection
    already populated and ``force=False``).
    """

    chunks_indexed: int
    source: str
    embedding_dimensions: int


class RagStatus(BaseModel):
    """Snapshot of the on-disk vector store -- count and persistent path."""

    chunks_indexed: int
    chroma_path: str
