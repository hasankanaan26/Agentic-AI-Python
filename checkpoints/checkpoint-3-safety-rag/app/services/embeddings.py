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

# Most providers cap a single embeddings request at a few hundred inputs.
# 100 stays well under every provider's hard limit and gives clean batches.
EMBED_BATCH_SIZE = 100


class EmbeddingService:
    """Provider-agnostic async embedding client.

    One instance per process; created in lifespan. The active provider is
    determined by ``settings.llm_provider``; only the matching SDK client
    is constructed in ``_build_client``.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._provider = settings.llm_provider
        # Three lazy slots; only the active provider's slot is populated.
        self._gemini = None
        self._openai = None
        self._azure = None
        self._build_client()

    def _build_client(self) -> None:
        """Instantiate the SDK client matching ``self._provider``."""
        timeout = httpx.Timeout(self._settings.llm_request_timeout_seconds)
        if self._provider == "gemini":
            # Imported lazily so installing one provider's deps is enough.
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
        """Close the underlying HTTP connection pool on shutdown."""
        if self._openai is not None:
            await self._openai.close()
        if self._azure is not None:
            await self._azure.close()
        # google-genai manages its own pool; nothing to await here.

    async def embed_text(self, text: str) -> list[float]:
        """Embed a single string. Convenience wrapper around ``embed_texts``."""
        return (await self.embed_texts([text]))[0]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings, splitting into batches as needed.

        Args:
            texts: Inputs to embed. May be empty.

        Returns:
            A list of vectors, one per input, in the same order as ``texts``.
        """
        if not texts:
            return []
        # Fast path — single-batch case avoids the loop machinery entirely.
        if len(texts) <= EMBED_BATCH_SIZE:
            return await self._embed_batch(texts)

        # Slow path — split into ``EMBED_BATCH_SIZE`` chunks and concatenate.
        out: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i : i + EMBED_BATCH_SIZE]
            log.info("embedding_batch", offset=i, size=len(batch), total=len(texts))
            out.extend(await self._embed_batch(batch))
        return out

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed one batch with the embedding-tuned retry policy."""
        async for attempt in embedding_retry():
            with attempt:
                if self._provider == "gemini":
                    return await self._embed_gemini(texts)
                if self._provider == "openai":
                    return await self._embed_openai(texts)
                return await self._embed_azure(texts)
        raise RuntimeError("unreachable")  # pragma: no cover

    async def _embed_gemini(self, texts: list[str]) -> list[list[float]]:
        """Embed via Gemini's async API."""
        result = await self._gemini.aio.models.embed_content(
            model=self._settings.gemini_embedding_model,
            contents=texts,
        )
        return [e.values for e in result.embeddings]

    async def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        """Embed via OpenAI's async API."""
        resp = await self._openai.embeddings.create(
            model=self._settings.openai_embedding_model,
            input=texts,
        )
        return [item.embedding for item in resp.data]

    async def _embed_azure(self, texts: list[str]) -> list[list[float]]:
        """Embed via Azure OpenAI's async API."""
        resp = await self._azure.embeddings.create(
            model=self._settings.azure_openai_embedding_deployment,
            input=texts,
        )
        return [item.embedding for item in resp.data]
