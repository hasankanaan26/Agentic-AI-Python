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

    Hides the provider switch behind a single ``call`` / ``call_structured``
    / ``call_with_tools`` interface so the rest of the codebase doesn't
    care which provider is configured.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._provider = settings.llm_provider
        # Only one of these will be populated; the others stay ``None``.
        self._gemini = None
        self._openai = None
        self._azure = None
        self._build_client()

    # -- construction -------------------------------------------------
    def _build_client(self) -> None:
        """Instantiate the SDK client matching ``settings.llm_provider``."""
        timeout = httpx.Timeout(self._settings.llm_request_timeout_seconds)
        # Lazy imports keep optional SDKs out of the import graph when unused.
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

            self._azure = AsyncAzureOpenAI(
                api_key=self._settings.azure_openai_api_key,
                azure_endpoint=self._settings.azure_openai_endpoint,
                api_version=self._settings.azure_openai_api_version,
                timeout=timeout,
            )
        log.info("llm_client_ready", provider=self._provider)

    async def aclose(self) -> None:
        """Close underlying HTTP clients on shutdown.

        Called from the FastAPI lifespan tear-down so we don't leak sockets.
        """
        if self._openai is not None:
            await self._openai.close()
        if self._azure is not None:
            await self._azure.close()
        # google-genai's client manages its own pool; nothing to await.

    @property
    def model_name(self) -> str:
        """Identifier of the currently active model (for response metadata)."""
        return self._settings.model_name()

    # -- public API ---------------------------------------------------
    async def call(self, user_prompt: str, system_prompt: str = "") -> str:
        """Plain-text completion across providers.

        Args:
            user_prompt: The user message.
            system_prompt: Optional system instruction.

        Returns:
            The assistant's plain-text response.
        """
        # tenacity retries transient transport errors (see retries.py); 4xx
        # validation errors are not retried -- they propagate immediately.
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
        """Provider-native JSON mode that returns a validated Pydantic instance.

        Each provider's structured output ("response_format" / response_schema)
        is configured here so the caller never needs to call
        ``model_validate_json`` itself.

        Args:
            user_prompt: User message; usually the planner request.
            response_model: Target Pydantic class for validation.
            system_prompt: Optional system instruction.

        Returns:
            An instance of ``response_model`` validated by Pydantic.
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
                return await self._call_azure_structured(user_prompt, response_model, system_prompt)
        raise RuntimeError("unreachable")  # pragma: no cover

    async def call_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        system_prompt: str = "",
    ) -> dict:
        """Run one chat turn with tool calling enabled.

        Args:
            messages: Chat history in the OpenAI ``{role, content}`` shape.
            tools: List of tool JSON-schema definitions (registry output).
            system_prompt: Optional system instruction.

        Returns:
            ``{"response_text", "tool_calls", "finish_reason"}``. When the
            model elects to call tools, ``response_text`` is ``None`` and
            ``finish_reason`` is ``"tool_calls"``; otherwise vice versa.
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
    # Each provider has three private helpers: text, structured, tools. The
    # public methods above just dispatch by ``self._provider``.

    async def _call_gemini(self, user_prompt: str, system_prompt: str) -> str:
        """Plain-text completion via the google-genai async client."""
        from google.genai import types

        config = types.GenerateContentConfig(temperature=0.3)
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
        """Gemini JSON mode -- ``response_schema`` constrains the output to ``response_model``."""
        from google.genai import types

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
        return response_model.model_validate_json(response.text)

    async def _call_gemini_tools(
        self, messages: list[dict], tools: list[dict], system_prompt: str
    ) -> dict:
        """Gemini tool-calling: convert tool dicts to ``FunctionDeclaration`` and dispatch."""
        from google.genai import types

        # Gemini wants ``FunctionDeclaration`` objects, not raw dicts.
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

        # Translate OpenAI-style ``{role, content}`` history into Gemini's
        # ``Content`` objects. Anything that isn't user becomes "model".
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
        """Plain-text completion via the OpenAI async client."""
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
        """OpenAI structured output via ``beta.chat.completions.parse``."""
        msgs = _chat_messages(user_prompt, system_prompt)
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
        """OpenAI tool-calling: send raw history + ``tools`` payload."""
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
    # Azure shares the OpenAI Python SDK; only the deployment name and
    # auth shape differ. Behaviour is otherwise identical to ``_call_openai*``.

    async def _call_azure(self, user_prompt: str, system_prompt: str) -> str:
        """Plain-text completion via Azure OpenAI."""
        msgs = _chat_messages(user_prompt, system_prompt)
        resp = await self._azure.chat.completions.create(
            model=self._settings.azure_openai_deployment,
            messages=msgs,
            temperature=0.3,
        )
        return resp.choices[0].message.content

    async def _call_azure_structured(
        self, user_prompt: str, response_model: type[BaseModel], system_prompt: str
    ) -> BaseModel:
        """Azure structured output -- mirrors the OpenAI variant."""
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
        """Azure tool-calling -- mirrors the OpenAI variant."""
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
    """Build the standard OpenAI ``[system, user]`` message list."""
    msgs: list[dict] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_prompt})
    return msgs


def _to_openai_tools(tools: list[dict]) -> list[dict]:
    """Convert our internal tool dicts to OpenAI's ``{type, function}`` shape."""
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
    """Normalise an OpenAI/Azure response into our ``call_with_tools`` shape."""
    msg = resp.choices[0].message
    if msg.tool_calls:
        # OpenAI returns tool args as a JSON string; decode for the agent loop.
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
    """Normalise a Gemini response into our ``call_with_tools`` shape."""
    tool_calls: list[dict] = []
    text: str | None = None
    # Gemini multiplexes function calls and text in ``parts``; iterate and split.
    for part in response.candidates[0].content.parts:
        if part.function_call:
            tool_calls.append(
                {
                    "name": part.function_call.name,
                    "arguments": dict(part.function_call.args) if part.function_call.args else {},
                }
            )
        elif part.text:
            text = part.text
    if tool_calls:
        return {"response_text": None, "tool_calls": tool_calls, "finish_reason": "tool_calls"}
    return {"response_text": text, "tool_calls": [], "finish_reason": "stop"}
