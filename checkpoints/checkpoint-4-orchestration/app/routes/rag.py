"""/rag — knowledge-base management endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.deps import get_embeddings, get_settings_dep, get_vector_store
from app.models import IngestResponse, RagStatus
from app.rag.ingest import ingest_knowledge
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore
from app.settings import Settings

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    embeddings: Annotated[EmbeddingService, Depends(get_embeddings)],
    store: Annotated[VectorStore, Depends(get_vector_store)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
    force: bool = False,
) -> IngestResponse:
    """Embed and index the bundled knowledge file.

    No-op when the store is already populated unless ``force=True``.
    """
    result = await ingest_knowledge(
        embeddings=embeddings,
        store=store,
        knowledge_path=settings.knowledge_data_path,
        force=force,
    )
    return IngestResponse(
        chunks_indexed=result["chunks_indexed"],
        source="acme-knowledge.json",
        embedding_dimensions=result["embedding_dimensions"],
    )


@router.get("/status", response_model=RagStatus)
async def status(
    store: Annotated[VectorStore, Depends(get_vector_store)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
) -> RagStatus:
    """Report current vector-store size and persistent path."""
    return RagStatus(
        chunks_indexed=await store.count(),
        chroma_path=str(settings.chroma_path),
    )
