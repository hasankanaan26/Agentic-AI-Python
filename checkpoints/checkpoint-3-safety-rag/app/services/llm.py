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
    """Async LLM client wrapper. One instance per process; created in lifespan."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._provider = settings.llm_provider
        self._gemini = None
        self._openai = None
        self._azure = None
        self._build_client()

    # -- construction -------------------------------------------------
    def _build_client(self) -> None:
        """Instantiate the SDK client matching ``self._provider``."""
        timeout = httpx.Timeout(self._settings.llm_request_timeout_seconds)
        if self._provider == "gemini":
            # Lazy import — only the active provider's SDK has to be installed.
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
        """Close underlying HTTP clients on shutdown."""
        if self._openai is not None:
            await self._openai.close()
        if self._azure is not None:
            await self._azure.close()
        # google-genai's client manages its own pool; nothing to await.

    @property
    def model_name(self) -> str:
        """Return the active model identifier (delegates to ``Settings``)."""
        return self._settings.model_name()

    # -- public API ---------------------------------------------------
    async def call(self, user_prompt: str, system_prompt: str = "") -> str:
        """Run a plain text completion against the active provider.

        Args:
            user_prompt: The user-turn content.
            system_prompt: Optional system instruction (skipped if empty).

        Returns:
            The assistant's text completion.
        """
        # Each retry attempt re-runs the full provider call below; tenacity
        # handles backoff and reraises after ``attempts`` transient failures.
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
        """Force JSON output and validate it against ``response_model``.

        Each provider has a native "JSON mode" (Gemini's response schema,
        OpenAI's beta parse, Azure's beta parse) that constrains the model
        output to the Pydantic schema. The returned instance is fully
        type-checked.
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
        """Run a tool-aware completion (function calling).

        Args:
            messages: OpenAI-style chat history (``role``/``content``).
            tools: JSON-schema tool definitions exposed to the model.
            system_prompt: Optional system instruction.

        Returns:
            ``{"response_text": str | None, "tool_calls": list[dict],
            "finish_reason": "stop" | "tool_calls"}`` — a normalized shape
            agnostic of which provider produced it.
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
        """Plain text completion against Gemini."""
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
        """Structured-output completion against Gemini."""
        from google.genai import types

        # Gemini accepts a Pydantic class directly as ``response_schema`` and
        # constrains the JSON output to validate against it.
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
        """Tool-aware completion against Gemini, returning the normalized dict."""
        from google.genai import types

        # Translate our generic tool definitions into Gemini's typed objects.
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

        # Gemini uses ``model``/``user`` roles instead of OpenAI's
        # ``assistant``/``user``; map our chat history accordingly.
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
        """Plain text completion against OpenAI."""
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
        """Structured-output completion against OpenAI's beta ``parse`` endpoint."""
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
        """Tool-aware completion against OpenAI."""
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
        """Plain text completion against Azure OpenAI."""
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
        """Structured-output completion against Azure OpenAI."""
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
        """Tool-aware completion against Azure OpenAI."""
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
    """Build an OpenAI-style ``messages`` list with an optional system turn."""
    msgs: list[dict] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_prompt})
    return msgs


def _to_openai_tools(tools: list[dict]) -> list[dict]:
    """Wrap our generic tool dicts in the ``{type: function, function: {...}}`` shape."""
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
    """Normalize an OpenAI/Azure chat response to the common return shape."""
    msg = resp.choices[0].message
    if msg.tool_calls:
        # OpenAI emits arguments as a JSON-encoded string — decode here so
        # downstream code never has to think about it.
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
    """Normalize a Gemini response to the common return shape."""
    tool_calls: list[dict] = []
    text: str | None = None
    # A single Gemini response can contain multiple parts mixing text and
    # function_calls; collect them in order.
    for part in response.candidates[0].content.parts:
        if part.function_call:
            tool_calls.append(
                {
                    "name": part.function_call.name,
                    # ``args`` is a proto Map; convert to a plain dict.
                    "arguments": dict(part.function_call.args) if part.function_call.args else {},
                }
            )
        elif part.text:
            text = part.text
    if tool_calls:
        return {"response_text": None, "tool_calls": tool_calls, "finish_reason": "tool_calls"}
    return {"response_text": text, "tool_calls": [], "finish_reason": "stop"}
