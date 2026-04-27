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

    Exposes three behaviours — plain text completion, JSON-validated
    completion, and tool-calling — over a single interface that hides the
    differences between Gemini, OpenAI, and Azure OpenAI. Each public
    method dispatches to a provider-specific private helper and is wrapped
    in a tenacity retry policy so transient network errors don't fail
    requests.
    """

    def __init__(self, settings: Settings) -> None:
        """Construct the appropriate provider client based on settings."""
        self._settings = settings
        self._provider = settings.llm_provider
        # Only one of these is non-None for a given process, but we keep
        # all three slots so type checks and lifecycle handling are
        # uniform across providers.
        self._gemini = None
        self._openai = None
        self._azure = None
        self._build_client()

    # -- construction -------------------------------------------------
    def _build_client(self) -> None:
        """Instantiate the provider-specific async client."""
        timeout = httpx.Timeout(self._settings.llm_request_timeout_seconds)
        # Imports are deferred so users only pay the import cost (and need
        # the dependency installed) for the provider they actually use.
        if self._provider == "gemini":
            from google import genai

            self._gemini = genai.Client(api_key=self._settings.gemini_api_key)
        elif self._provider == "openai":
            from openai import AsyncOpenAI

            self._openai = AsyncOpenAI(
                api_key=self._settings.openai_api_key,
                timeout=timeout,
            )
        else:
            from openai import AsyncAzureOpenAI

            # Azure routes by deployment, not model id — the deployment
            # name is supplied later as the `model=` kwarg per request.
            self._azure = AsyncAzureOpenAI(
                api_key=self._settings.azure_openai_api_key,
                azure_endpoint=self._settings.azure_openai_endpoint,
                api_version=self._settings.azure_openai_api_version,
                timeout=timeout,
            )
        log.info("llm_client_ready", provider=self._provider)

    async def aclose(self) -> None:
        """Close underlying HTTP clients on shutdown."""
        if self._openai is not None:
            await self._openai.close()
        if self._azure is not None:
            await self._azure.close()
        # google-genai's client manages its own pool; nothing to await.

    @property
    def model_name(self) -> str:
        """Provider-specific identifier of the model currently in use."""
        return self._settings.model_name()

    # -- public API ---------------------------------------------------
    async def call(self, user_prompt: str, system_prompt: str = "") -> str:
        """Plain text completion.

        Args:
            user_prompt: The user message to send.
            system_prompt: Optional system instruction; empty disables it.

        Returns:
            The model's reply as a plain string.
        """
        # The `async for ... with attempt` pattern is tenacity's idiomatic
        # async retry: each iteration is one attempt, and `with attempt`
        # captures any raised exception for the policy to decide on.
        async for attempt in llm_retry():
            with attempt:
                if self._provider == "gemini":
                    return await self._call_gemini(user_prompt, system_prompt)
                if self._provider == "openai":
                    return await self._call_openai(user_prompt, system_prompt)
                return await self._call_azure(user_prompt, system_prompt)
        # Tenacity always either returns from the with-block or re-raises;
        # this line exists to satisfy type-checkers.
        raise RuntimeError("unreachable")  # pragma: no cover

    async def call_structured(
        self,
        user_prompt: str,
        response_model: type[BaseModel],
        system_prompt: str = "",
    ) -> BaseModel:
        """Provider-native JSON mode -> validated Pydantic instance.

        Args:
            user_prompt: User message describing what JSON to produce.
            response_model: Pydantic model class the response must match.
            system_prompt: Optional system instruction.

        Returns:
            A validated instance of `response_model`.
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
        """Issue a tool-aware chat completion.

        Args:
            messages: Conversation so far in OpenAI-style ``{role, content}`` dicts.
            tools: Tool definitions in the shape produced by
                `ToolRegistry.definitions()`.
            system_prompt: Optional system instruction.

        Returns:
            A normalised dict with three keys:
              - ``response_text``: free-form reply, or None if a tool was called
              - ``tool_calls``: list of ``{name, arguments}`` dicts (possibly empty)
              - ``finish_reason``: ``"tool_calls"`` or ``"stop"``
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
        """Plain-text completion via Gemini's async API."""
        from google.genai import types

        config = types.GenerateContentConfig(temperature=0.3)
        # Gemini puts the system prompt on the config object rather than
        # as a leading message — the surface differs from OpenAI here.
        if system_prompt:
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
        """Gemini structured output: schema enforced by `response_schema`."""
        from google.genai import types

        # `response_mime_type=application/json` plus `response_schema`
        # turns on Gemini's controlled-decoding mode — the response is
        # guaranteed to be JSON conforming to the supplied pydantic model.
        config = types.GenerateContentConfig(
            temperature=0.3,
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
        # We still validate locally — provider promises are not contracts.
        return response_model.model_validate_json(response.text)

    async def _call_gemini_tools(
        self, messages: list[dict], tools: list[dict], system_prompt: str
    ) -> dict:
        """Gemini function-calling: returns the normalised tool-call dict."""
        from google.genai import types

        # Translate our generic tool definitions into Gemini's
        # FunctionDeclaration shape.
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

        # Gemini expects "user" / "model" roles (not "assistant"); we
        # treat anything non-"user" as model output.
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
        """Plain-text completion via OpenAI's async chat API."""
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
        """OpenAI structured output via the `beta.chat.completions.parse` helper."""
        msgs = _chat_messages(user_prompt, system_prompt)
        # `parse` enforces the schema server-side AND returns a typed
        # `parsed` attribute already validated against the pydantic model.
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
        """OpenAI tool calling: returns the normalised tool-call dict."""
        # OpenAI's tool format prepends the system prompt as a regular
        # message rather than via a separate config field.
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
        """Plain-text completion via Azure OpenAI."""
        msgs = _chat_messages(user_prompt, system_prompt)
        # Azure uses the deployment name in place of OpenAI's model id.
        resp = await self._azure.chat.completions.create(
            model=self._settings.azure_openai_deployment,
            messages=msgs,
            temperature=0.3,
        )
        return resp.choices[0].message.content

    async def _call_azure_structured(
        self, user_prompt: str, response_model: type[BaseModel], system_prompt: str
    ) -> BaseModel:
        """Azure OpenAI structured output via the `parse` helper."""
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
        """Azure OpenAI tool calling: returns the normalised tool-call dict."""
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
    """Build an OpenAI-style messages list, optionally with a system message."""
    msgs: list[dict] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_prompt})
    return msgs


def _to_openai_tools(tools: list[dict]) -> list[dict]:
    """Wrap our flat tool definitions in OpenAI's ``{type, function}`` envelope."""
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
    """Normalise an OpenAI/Azure chat response into our common dict shape."""
    msg = resp.choices[0].message
    # When the model decided to invoke tools, `tool_calls` is populated
    # and `content` is None — return the calls in our normalised form.
    if msg.tool_calls:
        return {
            "response_text": None,
            "tool_calls": [
                # OpenAI ships arguments as a JSON string; decode once here.
                {"name": tc.function.name, "arguments": json.loads(tc.function.arguments)}
                for tc in msg.tool_calls
            ],
            "finish_reason": "tool_calls",
        }
    # Plain reply branch — no tools invoked.
    return {"response_text": msg.content, "tool_calls": [], "finish_reason": "stop"}


def _parse_gemini_response(response: Any) -> dict:
    """Normalise a Gemini response into our common dict shape."""
    tool_calls: list[dict] = []
    text: str | None = None
    # Gemini returns a list of parts; each part is either text or a
    # function_call. We collect them separately and then decide which
    # branch of the contract to return.
    for part in response.candidates[0].content.parts:
        if part.function_call:
            tool_calls.append(
                {
                    "name": part.function_call.name,
                    # `args` is a proto Map; coerce to a plain dict so
                    # downstream code can treat it uniformly.
                    "arguments": dict(part.function_call.args) if part.function_call.args else {},
                }
            )
        elif part.text:
            text = part.text
    if tool_calls:
        return {"response_text": None, "tool_calls": tool_calls, "finish_reason": "tool_calls"}
    return {"response_text": text, "tool_calls": [], "finish_reason": "stop"}
