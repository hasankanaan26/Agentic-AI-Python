"""Pydantic response models for the ``/rag`` endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    """Returned by ``POST /rag/ingest`` after embedding the knowledge file."""

    chunks_indexed: int = Field(description="Number of chunks now present in the vector store.")
    source: str = Field(description="Source filename the chunks were drawn from.")
    embedding_dimensions: int = Field(
        description="Dimensionality of the embedding vectors (0 when ingestion was skipped)."
    )


class RagStatus(BaseModel):
    """Returned by ``GET /rag/status`` for liveness/inspection of the index."""

    chunks_indexed: int = Field(description="Total chunks currently indexed in Chroma.")
    chroma_path: str = Field(description="On-disk path of the persistent Chroma store.")
