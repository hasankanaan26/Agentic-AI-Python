"""Async LLM service — three behaviours, three providers, one interface.

Engineering standards applied here:
  - **Async all the way down.** AsyncOpenAI / AsyncAzureOpenAI / Gemini's
    `client.aio.*` async APIs. No `requests`, no blocking SDK calls.
  - **Singletons via DI.** The clients are created once at startup
    (lifespan) and handed to routes via Depends.
  - **Bounded retries with backoff.** Every external call is wrapped
    with tenacity (`retries.llm_retry()`).
  - **Pydantic at the boundary.** `call_llm_structured` returns a
    validated model — provider-native JSON mode does the heavy lifting.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from pydantic import BaseModel

from app.logging_config import get_logger
from app.retries import llm_retry
from app.settings import Settings

log = get_logger(__name__)


class LLMService:
    """Async LLM client wrapper. One instance per process; created in lifespan.

    The class hides the per-provider SDK differences behind three uniform
    methods (:meth:`call`, :meth:`call_structured`, :meth:`call_with_tools`).
    Each call is wrapped in :func:`llm_retry` so transient network errors
    cost a retry rather than failing the whole request.
    """

    def __init__(self, settings: Settings) -> None:
        """Build the underlying SDK client matching ``settings.llm_provider``."""
        self._settings = settings
        self._provider = settings.llm_provider
        # Only ONE of these will be non-None depending on the configured provider.
        self._gemini = None
        self._openai = None
        self._azure = None
        self._build_client()

    # -- construction -------------------------------------------------
    def _build_client(self) -> None:
        """Lazy-import and instantiate the SDK for the configured provider."""
        timeout = httpx.Timeout(self._settings.llm_request_timeout_seconds)
        # Lazy imports keep startup fast and avoid pulling SDKs we won't use.
        if self._provider == "gemini":
            from google import genai

            # google-genai manages its own httpx pool internally.
            self._gemini = genai.Client(api_key=self._settings.gemini_api_key)
        elif self._provider == "openai":
            from openai import AsyncOpenAI

            self._openai = AsyncOpenAI(
                api_key=self._settings.openai_api_key,
                timeout=timeout,
            )
        else:
            from openai import AsyncAzureOpenAI

            self._azure = AsyncAzureOpenAI(
                api_key=self._settings.azure_openai_api_key,
                azure_endpoint=self._settings.azure_openai_endpoint,
                api_version=self._settings.azure_openai_api_version,
                timeout=timeout,
            )
        log.info("llm_client_ready", provider=self._provider)

    async def aclose(self) -> None:
        """Close underlying HTTP clients on shutdown.

        Called from the lifespan teardown. We only need to close the
        OpenAI/Azure clients; the google-genai SDK owns its own pool.
        """
        if self._openai is not None:
            await self._openai.close()
        if self._azure is not None:
            await self._azure.close()
        # google-genai's client manages its own pool; nothing to await.

    @property
    def model_name(self) -> str:
        """Identifier of the model/deployment in use, e.g. for response payloads."""
        return self._settings.model_name()

    # -- public API ---------------------------------------------------
    async def call(self, user_prompt: str, system_prompt: str = "") -> str:
        """Plain text completion.

        Args:
            user_prompt: The user message content.
            system_prompt: Optional system instruction.

        Returns:
            The model's response text.
        """
        # Retry transient network errors with exponential backoff.
        async for attempt in llm_retry():
            with attempt:
                if self._provider == "gemini":
                    return await self._call_gemini(user_prompt, system_prompt)
                if self._provider == "openai":
                    return await self._call_openai(user_prompt, system_prompt)
                return await self._call_azure(user_prompt, system_prompt)
        raise RuntimeError("unreachable")  # pragma: no cover

    async def call_structured(
        self,
        user_prompt: str,
        response_model: type[BaseModel],
        system_prompt: str = "",
    ) -> BaseModel:
        """Provider-native JSON mode -> validated Pydantic instance.

        Args:
            user_prompt: The user message content.
            response_model: Pydantic class describing the desired schema.
            system_prompt: Optional system instruction.

        Returns:
            An instance of ``response_model`` parsed from the LLM's JSON output.
        """
        async for attempt in llm_retry():
            with attempt:
                if self._provider == "gemini":
                    return await self._call_gemini_structured(
                        user_prompt, response_model, system_prompt
                    )
                if self._provider == "openai":
                    return await self._call_openai_structured(
                        user_prompt, response_model, system_prompt
                    )
                return await self._call_azure_structured(
                    user_prompt, response_model, system_prompt
                )
        raise RuntimeError("unreachable")  # pragma: no cover

    async def call_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        system_prompt: str = "",
    ) -> dict:
        """Run a tool-aware LLM call.

        Args:
            messages: Conversation history as ``[{"role": ..., "content": ...}, ...]``.
            tools: Tool definitions in the unified ``ToolDefinition`` shape.
            system_prompt: Optional system instruction prepended to the conversation.

        Returns:
            Dict with three keys::

                {
                    "response_text": str | None,    # set when the model answered directly
                    "tool_calls":    list[dict],    # populated when the model asked for tools
                    "finish_reason": "stop" | "tool_calls",
                }
        """
        async for attempt in llm_retry():
            with attempt:
                if self._provider == "gemini":
                    return await self._call_gemini_tools(messages, tools, system_prompt)
                if self._provider == "openai":
                    return await self._call_openai_tools(messages, tools, system_prompt)
                return await self._call_azure_tools(messages, tools, system_prompt)
        raise RuntimeError("unreachable")  # pragma: no cover

    # -- Gemini -------------------------------------------------------
    async def _call_gemini(self, user_prompt: str, system_prompt: str) -> str:
        """Plain text completion via google-genai's async API."""
        from google.genai import types

        config = types.GenerateContentConfig(temperature=0.3)
        if system_prompt:
            # Gemini takes the system prompt as a separate field, not a message.
            config.system_instruction = system_prompt
        response = await self._gemini.aio.models.generate_content(
            model=self._settings.gemini_model,
            contents=user_prompt,
            config=config,
        )
        return response.text

    async def _call_gemini_structured(
        self, user_prompt: str, response_model: type[BaseModel], system_prompt: str
    ) -> BaseModel:
        """Structured-output completion via Gemini's JSON-mode + schema feature."""
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=0.3,
            # Gemini accepts a Pydantic class directly as the response schema.
            response_mime_type="application/json",
            response_schema=response_model,
        )
        if system_prompt:
            config.system_instruction = system_prompt
        response = await self._gemini.aio.models.generate_content(
            model=self._settings.gemini_model,
            contents=user_prompt,
            config=config,
        )
        # Validate the JSON string ourselves so we get clean Pydantic errors.
        return response_model.model_validate_json(response.text)

    async def _call_gemini_tools(
        self, messages: list[dict], tools: list[dict], system_prompt: str
    ) -> dict:
        """Tool-aware completion via Gemini's function-calling API."""
        from google.genai import types

        # Gemini wants its own FunctionDeclaration / Tool wrapper objects.
        function_declarations = [
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=t["parameters"],
            )
            for t in tools
        ]
        config = types.GenerateContentConfig(
            temperature=0.3,
            tools=[types.Tool(function_declarations=function_declarations)],
        )
        if system_prompt:
            config.system_instruction = system_prompt

        # Translate the OpenAI-style role names: Gemini uses "user"/"model",
        # not "user"/"assistant". Anything that isn't "user" becomes "model".
        contents = [
            types.Content(
                role="user" if msg["role"] == "user" else "model",
                parts=[types.Part.from_text(text=msg["content"])],
            )
            for msg in messages
        ]
        response = await self._gemini.aio.models.generate_content(
            model=self._settings.gemini_model,
            contents=contents,
            config=config,
        )
        return _parse_gemini_response(response)

    # -- OpenAI -------------------------------------------------------
    async def _call_openai(self, user_prompt: str, system_prompt: str) -> str:
        """Plain text completion via the OpenAI Async chat-completions API."""
        msgs = _chat_messages(user_prompt, system_prompt)
        resp = await self._openai.chat.completions.create(
            model=self._settings.openai_model,
            messages=msgs,
            temperature=0.3,
        )
        return resp.choices[0].message.content

    async def _call_openai_structured(
        self, user_prompt: str, response_model: type[BaseModel], system_prompt: str
    ) -> BaseModel:
        """Structured-output completion via OpenAI's ``beta.chat.completions.parse``."""
        msgs = _chat_messages(user_prompt, system_prompt)
        # The .parse() helper lets the SDK do the JSON-schema -> Pydantic dance.
        resp = await self._openai.beta.chat.completions.parse(
            model=self._settings.openai_model,
            messages=msgs,
            temperature=0.3,
            response_format=response_model,
        )
        return resp.choices[0].message.parsed

    async def _call_openai_tools(
        self, messages: list[dict], tools: list[dict], system_prompt: str
    ) -> dict:
        """Tool-aware completion via OpenAI's function-calling API."""
        chat = []
        if system_prompt:
            chat.append({"role": "system", "content": system_prompt})
        chat.extend(messages)
        resp = await self._openai.chat.completions.create(
            model=self._settings.openai_model,
            messages=chat,
            tools=_to_openai_tools(tools),
            temperature=0.3,
        )
        return _parse_openai_response(resp)

    # -- Azure --------------------------------------------------------
    async def _call_azure(self, user_prompt: str, system_prompt: str) -> str:
        """Plain text completion via Azure OpenAI (deployment-based addressing)."""
        msgs = _chat_messages(user_prompt, system_prompt)
        resp = await self._azure.chat.completions.create(
            # On Azure, "model" is the *deployment* name, not the base model.
            model=self._settings.azure_openai_deployment,
            messages=msgs,
            temperature=0.3,
        )
        return resp.choices[0].message.content

    async def _call_azure_structured(
        self, user_prompt: str, response_model: type[BaseModel], system_prompt: str
    ) -> BaseModel:
        """Structured-output completion via Azure OpenAI's ``parse`` helper."""
        msgs = _chat_messages(user_prompt, system_prompt)
        resp = await self._azure.beta.chat.completions.parse(
            model=self._settings.azure_openai_deployment,
            messages=msgs,
            temperature=0.3,
            response_format=response_model,
        )
        return resp.choices[0].message.parsed

    async def _call_azure_tools(
        self, messages: list[dict], tools: list[dict], system_prompt: str
    ) -> dict:
        """Tool-aware completion via Azure OpenAI's function-calling API."""
        chat = []
        if system_prompt:
            chat.append({"role": "system", "content": system_prompt})
        chat.extend(messages)
        resp = await self._azure.chat.completions.create(
            model=self._settings.azure_openai_deployment,
            messages=chat,
            tools=_to_openai_tools(tools),
            temperature=0.3,
        )
        return _parse_openai_response(resp)


# ----- helpers ---------------------------------------------------------


def _chat_messages(user_prompt: str, system_prompt: str) -> list[dict]:
    """Build an OpenAI-style ``[system?, user]`` message list."""
    msgs: list[dict] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_prompt})
    return msgs


def _to_openai_tools(tools: list[dict]) -> list[dict]:
    """Wrap our unified tool definitions in OpenAI's function-tool envelope."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in tools
    ]


def _parse_openai_response(resp: Any) -> dict:
    """Normalise an OpenAI/Azure chat response to our unified shape."""
    msg = resp.choices[0].message
    if msg.tool_calls:
        # OpenAI returns tool arguments as a JSON string; deserialize for the caller.
        return {
            "response_text": None,
            "tool_calls": [
                {"name": tc.function.name, "arguments": json.loads(tc.function.arguments)}
                for tc in msg.tool_calls
            ],
            "finish_reason": "tool_calls",
        }
    return {"response_text": msg.content, "tool_calls": [], "finish_reason": "stop"}


def _parse_gemini_response(response: Any) -> dict:
    """Normalise a Gemini response to our unified shape."""
    tool_calls: list[dict] = []
    text: str | None = None
    # A Gemini candidate contains heterogeneous "parts" — text or function calls.
    # We collect them in one pass and decide which finish_reason to surface below.
    for part in response.candidates[0].content.parts:
        if part.function_call:
            tool_calls.append(
                {
                    "name": part.function_call.name,
                    # ``args`` arrives as a proto Struct; coerce to a plain dict.
                    "arguments": dict(part.function_call.args) if part.function_call.args else {},
                }
            )
        elif part.text:
            text = part.text
    if tool_calls:
        return {"response_text": None, "tool_calls": tool_calls, "finish_reason": "tool_calls"}
    return {"response_text": text, "tool_calls": [], "finish_reason": "stop"}
