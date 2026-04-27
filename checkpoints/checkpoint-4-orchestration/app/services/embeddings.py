"""Async embedding service. Same provider switch as the chat models.

Engineering standards applied: async clients, retries with backoff,
batching to respect provider payload limits.
"""

from __future__ import annotations

import httpx

from app.logging_config import get_logger
from app.retries import embedding_retry
from app.settings import Settings

log = get_logger(__name__)

EMBED_BATCH_SIZE = 100


class EmbeddingService:
    """Async embedding client wrapper. One instance per process; created in lifespan.

    Uses the same provider switch as :class:`app.services.llm.LLMService`.
    Inputs are batched at :data:`EMBED_BATCH_SIZE` to keep request payloads
    well under provider limits.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._provider = settings.llm_provider
        self._gemini = None
        self._openai = None
        self._azure = None
        self._build_client()

    def _build_client(self) -> None:
        """Instantiate the SDK client matching ``settings.llm_provider``."""
        timeout = httpx.Timeout(self._settings.llm_request_timeout_seconds)
        # Lazy imports keep optional SDK dependencies out of the import graph.
        if self._provider == "gemini":
            from google import genai

            self._gemini = genai.Client(api_key=self._settings.gemini_api_key)
        elif self._provider == "openai":
            from openai import AsyncOpenAI

            self._openai = AsyncOpenAI(api_key=self._settings.openai_api_key, timeout=timeout)
        else:
            from openai import AsyncAzureOpenAI

            self._azure = AsyncAzureOpenAI(
                api_key=self._settings.azure_openai_api_key,
                azure_endpoint=self._settings.azure_openai_endpoint,
                api_version=self._settings.azure_openai_api_version,
                timeout=timeout,
            )

    async def aclose(self) -> None:
        """Close underlying HTTP clients on shutdown (called from lifespan)."""
        if self._openai is not None:
            await self._openai.close()
        if self._azure is not None:
            await self._azure.close()

    async def embed_text(self, text: str) -> list[float]:
        """Convenience wrapper -- embed a single string and return one vector."""
        return (await self.embed_texts([text]))[0]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed an arbitrary number of strings, batching if necessary.

        Args:
            texts: One or more input strings. May be empty.

        Returns:
            One float vector per input, in the same order.
        """
        if not texts:
            return []
        # Fast path: payload fits in a single request.
        if len(texts) <= EMBED_BATCH_SIZE:
            return await self._embed_batch(texts)

        # Otherwise chunk into provider-friendly slices.
        out: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i : i + EMBED_BATCH_SIZE]
            log.info("embedding_batch", offset=i, size=len(batch), total=len(texts))
            out.extend(await self._embed_batch(batch))
        return out

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Run one provider-specific embed call wrapped in tenacity retries."""
        async for attempt in embedding_retry():
            with attempt:
                if self._provider == "gemini":
                    return await self._embed_gemini(texts)
                if self._provider == "openai":
                    return await self._embed_openai(texts)
                return await self._embed_azure(texts)
        raise RuntimeError("unreachable")  # pragma: no cover

    async def _embed_gemini(self, texts: list[str]) -> list[list[float]]:
        """Gemini embedding endpoint via google-genai's async client."""
        result = await self._gemini.aio.models.embed_content(
            model=self._settings.gemini_embedding_model,
            contents=texts,
        )
        return [e.values for e in result.embeddings]

    async def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        """OpenAI embedding endpoint via AsyncOpenAI."""
        resp = await self._openai.embeddings.create(
            model=self._settings.openai_embedding_model,
            input=texts,
        )
        return [item.embedding for item in resp.data]

    async def _embed_azure(self, texts: list[str]) -> list[list[float]]:
        """Azure OpenAI embedding deployment -- same SDK as OpenAI."""
        resp = await self._azure.embeddings.create(
            model=self._settings.azure_openai_embedding_deployment,
            input=texts,
        )
        return [item.embedding for item in resp.data]
