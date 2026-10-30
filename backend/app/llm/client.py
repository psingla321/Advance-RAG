from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from app.config import Settings
from app.observability import LLM_CALLS, LLM_TOKENS, tracer
from app.textutil import estimate_tokens, hash_embed

log = logging.getLogger(__name__)

MOCK_GENERAL_ANSWER = (
    "I'm running in offline mock mode, so I can't generate open-domain answers. "
    "Set AZURE_OPENAI_ENDPOINT to enable full LLM responses. Meanwhile I can answer from your documents, "
    "query the sales database, run calculations, tell the time, and create support tickets."
)

Messages = list[dict[str, Any]]


def _record(operation: str, model: str, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
    LLM_CALLS.labels(operation, model).inc()
    if prompt_tokens:
        LLM_TOKENS.labels("prompt", model).inc(prompt_tokens)
    if completion_tokens:
        LLM_TOKENS.labels("completion", model).inc(completion_tokens)


def _usage(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    return (getattr(usage, "prompt_tokens", 0) or 0, getattr(usage, "completion_tokens", 0) or 0) if usage else (0, 0)


class LLMClient:
    is_mock: bool = True

    async def complete(
        self, messages: Messages, *, json_mode: bool = False, temperature: float = 0.1, max_tokens: int = 800
    ) -> str:
        raise NotImplementedError

    def stream(self, messages: Messages, *, temperature: float = 0.1, max_tokens: int = 800) -> AsyncIterator[str]:
        raise NotImplementedError

    async def complete_with_tools(self, messages: Messages, tools: list[dict[str, Any]]) -> dict[str, Any] | None:
        return None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    async def close(self) -> None:
        return None


class MockLLM(LLMClient):
    is_mock = True

    def __init__(self, dim: int) -> None:
        self._dim = dim

    async def complete(self, messages: Messages, **_: Any) -> str:
        _record("chat", "mock", completion_tokens=estimate_tokens(MOCK_GENERAL_ANSWER))
        return MOCK_GENERAL_ANSWER

    async def stream(self, messages: Messages, **_: Any) -> AsyncIterator[str]:
        _record("chat_stream", "mock", completion_tokens=estimate_tokens(MOCK_GENERAL_ANSWER))
        for word in MOCK_GENERAL_ANSWER.split(" "):
            yield word + " "

    async def embed(self, texts: list[str]) -> list[list[float]]:
        _record("embed", "mock", prompt_tokens=sum(estimate_tokens(t) for t in texts))
        return [hash_embed(t, self._dim) for t in texts]


class AzureOpenAIClient(LLMClient):
    is_mock = False

    def __init__(self, settings: Settings) -> None:
        from openai import AsyncAzureOpenAI

        self._credential = None
        kwargs: dict[str, Any] = {
            "azure_endpoint": settings.azure_openai_endpoint,
            "api_version": settings.azure_openai_api_version,
        }
        if settings.azure_openai_api_key:
            kwargs["api_key"] = settings.azure_openai_api_key
        else:
            # Keyless auth via Entra ID (managed identity / az login).
            from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider

            self._credential = DefaultAzureCredential()
            kwargs["azure_ad_token_provider"] = get_bearer_token_provider(
                self._credential, "https://cognitiveservices.azure.com/.default"
            )
        self._client = AsyncAzureOpenAI(**kwargs)
        self._chat_model = settings.azure_openai_chat_deployment
        self._embedding_model = settings.azure_openai_embedding_deployment
        self._dim = settings.embedding_dim

    async def complete(
        self, messages: Messages, *, json_mode: bool = False, temperature: float = 0.1, max_tokens: int = 800
    ) -> str:
        extra: dict[str, Any] = {"response_format": {"type": "json_object"}} if json_mode else {}
        with tracer.start_as_current_span("llm.chat") as span:
            span.set_attribute("gen_ai.request.model", self._chat_model)
            response = await self._client.chat.completions.create(
                model=self._chat_model, messages=messages, temperature=temperature, max_tokens=max_tokens, **extra
            )
            prompt, completion = _usage(response)
            span.set_attribute("gen_ai.usage.input_tokens", prompt)
            span.set_attribute("gen_ai.usage.output_tokens", completion)
        _record("chat", self._chat_model, prompt, completion)
        return response.choices[0].message.content or ""

    async def stream(self, messages: Messages, *, temperature: float = 0.1, max_tokens: int = 800) -> AsyncIterator[str]:
        response = await self._client.chat.completions.create(
            model=self._chat_model, messages=messages, temperature=temperature, max_tokens=max_tokens, stream=True,
            stream_options={"include_usage": True},
        )
        prompt = completion = 0
        async for chunk in response:
            if chunk.usage:
                prompt, completion = _usage(chunk)
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
        _record("chat_stream", self._chat_model, prompt, completion)

    async def complete_with_tools(self, messages: Messages, tools: list[dict[str, Any]]) -> dict[str, Any] | None:
        response = await self._client.chat.completions.create(
            model=self._chat_model, messages=messages, tools=tools, tool_choice="auto", temperature=0
        )
        _record("tool_select", self._chat_model, *_usage(response))
        message = response.choices[0].message
        if not message.tool_calls:
            return None
        call = message.tool_calls[0]
        try:
            arguments = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            log.warning("Model returned invalid tool arguments")
            return None
        return {"name": call.function.name, "arguments": arguments}

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(
            model=self._embedding_model, input=texts, dimensions=self._dim
        )
        _record("embed", self._embedding_model, _usage(response)[0])
        return [item.embedding for item in response.data]

    async def close(self) -> None:
        await self._client.close()
        if self._credential is not None:
            await self._credential.close()


def create_llm(settings: Settings) -> LLMClient:
    if settings.use_mock_llm:
        log.warning("AZURE_OPENAI_ENDPOINT not set - using offline mock LLM")
        return MockLLM(settings.embedding_dim)
    return AzureOpenAIClient(settings)
