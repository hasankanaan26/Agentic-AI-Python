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
    """Semantic search over the Acme Corp knowledge base.

    Wraps the RAG primitives in ``app.rag``: embed the query, run a
    similarity search against Chroma, format the top hits. Caching the
    rendered string saves a round-trip to the embedding API for repeat
    questions during a single conversation.
    """

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
        self._cache: TTLCache = TTLCache(maxsize=cache_max, ttl=cache_ttl)

    async def run(self, query: str, top_k: int = 3) -> ToolResult:
        """Run RAG over ``query`` and return the formatted top-``top_k`` chunks.

        Args:
            query: Natural-language question.
            top_k: Number of chunks to retrieve.

        Returns:
            ``ToolResult.ok`` with a numbered, distance-annotated answer, or
            ``ToolResult.fail`` if the embedding/search pipeline raised.
        """
        if not query.strip():
            return ToolResult.fail("Query is empty.")

        # Cache key normalizes whitespace + case to share results between
        # near-duplicate phrasings inside a single TTL window.
        cache_key = (query.strip().lower(), top_k)
        if cache_key in self._cache:
            return ToolResult.ok(self._cache[cache_key], cached=True)

        try:
            # ``ensure_indexed`` is idempotent — first call ingests, the rest
            # short-circuit. Cheap enough to call on every search.
            await ensure_indexed(
                embeddings=self._embeddings,
                store=self._store,
                knowledge_path=self._knowledge_path,
            )
            chunks = await retrieve(
                query, embeddings=self._embeddings, store=self._store, top_k=top_k
            )
        except Exception as e:
            # Convert any underlying failure into a ``ToolResult`` so the
            # agent can react instead of seeing a 500.
            return ToolResult.fail(f"Knowledge search failed: {e}")

        if not chunks:
            return ToolResult.ok("No matching knowledge found.")

        # Pretty-print so the LLM gets distance + source context with each hit.
        formatted = "\n\n".join(
            f"[{i}] (distance={c['distance']:.3f}, source={c['source']})\n{c['text']}"
            for i, c in enumerate(chunks, 1)
        )
        self._cache[cache_key] = formatted
        return ToolResult.ok(formatted, cached=False, hits=len(chunks))
