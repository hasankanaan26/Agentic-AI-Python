"""Knowledge search — semantic, not keyword. Backed by the week-3 RAG stack.

The tool's JSON schema is unchanged from earlier checkpoints, but the
implementation now embeds the query and searches a Chroma collection.
A query like "how many days off do I get?" matches the "Annual Leave"
document even though no words overlap.

Caching: queries are TTL-cached so repeated identical questions don't
hit the embedding API twice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from cachetools import TTLCache

from app.models.tool import ToolResult
from app.rag.ingest import ensure_indexed
from app.rag.retriever import retrieve
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore
from app.tools.base import BaseTool


class KnowledgeSearchTool(BaseTool):
    """Semantic search over the Acme knowledge base via embeddings + Chroma."""

    name: ClassVar[str] = "knowledge_search"
    permission: ClassVar[str] = "read"
    definition: ClassVar[dict[str, Any]] = {
        "name": "knowledge_search",
        "description": (
            "Search the Acme Corp internal knowledge base for company policies, "
            "procedures, and information. Use when the user asks about company "
            "rules, benefits, IT, onboarding, or any internal topic. Search is "
            "semantic — phrasing the query in natural language is fine."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A natural-language description of what you're looking for.",
                },
            },
            "required": ["query"],
        },
    }

    def __init__(
        self,
        *,
        embeddings: EmbeddingService,
        store: VectorStore,
        knowledge_path: Path,
        cache_ttl: int = 300,
        cache_max: int = 256,
    ) -> None:
        self._embeddings = embeddings
        self._store = store
        self._knowledge_path = knowledge_path
        # TTL cache so repeated identical questions don't re-hit the embedding API.
        self._cache: TTLCache = TTLCache(maxsize=cache_max, ttl=cache_ttl)

    async def run(self, query: str, top_k: int = 3) -> ToolResult:
        """Embed ``query`` and return the top-k nearest chunks.

        Args:
            query: Natural-language description of what to find.
            top_k: Maximum number of chunks to return.

        Returns:
            ``ToolResult.ok`` with the formatted hits, or ``ToolResult.fail``
            on retrieval errors.
        """
        if not query.strip():
            return ToolResult.fail("Query is empty.")

        # Normalise on lowercase so "VPN" and "vpn" share a cache slot.
        cache_key = (query.strip().lower(), top_k)
        if cache_key in self._cache:
            return ToolResult.ok(self._cache[cache_key], cached=True)

        try:
            # Idempotent: only re-ingests if the collection is empty.
            await ensure_indexed(
                embeddings=self._embeddings,
                store=self._store,
                knowledge_path=self._knowledge_path,
            )
            chunks = await retrieve(
                query, embeddings=self._embeddings, store=self._store, top_k=top_k
            )
        except Exception as e:  # noqa: BLE001
            # Surface as structured error so the agent can recover instead of crashing.
            return ToolResult.fail(f"Knowledge search failed: {e}")

        if not chunks:
            return ToolResult.ok("No matching knowledge found.")

        # Render hits compactly: index, distance, source, then the chunk body.
        formatted = "\n\n".join(
            f"[{i}] (distance={c['distance']:.3f}, source={c['source']})\n{c['text']}"
            for i, c in enumerate(chunks, 1)
        )
        self._cache[cache_key] = formatted
        return ToolResult.ok(formatted, cached=False, hits=len(chunks))
