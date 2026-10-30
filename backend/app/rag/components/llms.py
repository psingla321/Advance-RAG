"""LLM backends. Every provider import is lazy so the base install stays light."""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import AsyncIterator
from typing import Any

from app.rag.base import LLM
from app.rag.registry import register
from app.textutil import content_term_list, estimate_tokens, split_sentences

log = logging.getLogger(__name__)


@register("llm", "mock")
class MockLLM(LLM):
    """Offline deterministic LLM: extractive answers, no network, no API key.

    Lets the whole pipeline — including evaluation — run in CI and on a laptop.
    """

    is_mock = True

    def __init__(self, max_sentences: int = 3) -> None:
        self._max_sentences = max_sentences

    # Passage blocks rendered by the generators look like: "[3] (file.md — Heading)\ntext"
    _BLOCK = re.compile(r"^\[(\d+)\][^\n]*\n", re.M)

    def _extract(self, messages: list[dict[str, Any]]) -> str:
        """Answer extractively from the prompt's context block, with citations.

        This keeps offline mode genuinely useful: the answer is real text from
        the retrieved passages rather than a placeholder.
        """
        user = str(next((m["content"] for m in reversed(messages) if m.get("role") == "user"), ""))
        # Prompts put the question on its own line; everything else is context.
        question_line = next((l for l in user.splitlines() if l.strip().startswith("Question:")), "")
        question = question_line.partition("Question:")[2] or user
        context = user.replace(question_line, "").split("Context:", 1)[-1]

        # Split the rendered context back into (citation number, passage) pairs.
        passages: list[tuple[int, str]] = []
        matches = list(self._BLOCK.finditer(context))
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(context)
            passages.append((int(match.group(1)), context[match.end() : end]))
        if not passages:
            passages = [(0, context)]

        terms = set(content_term_list(question))
        scored: list[tuple[float, int, int, str]] = []
        for number, passage in passages:
            for position, sentence in enumerate(split_sentences(passage)):
                overlap = len(terms & set(content_term_list(sentence))) / (len(terms) or 1)
                if overlap > 0:
                    scored.append((overlap, -position, number, sentence))
        if not scored:
            return "I could not find an answer to that in the provided context."
        best = sorted(scored, reverse=True)[: self._max_sentences]
        # Restore document order so the answer reads naturally.
        best.sort(key=lambda item: (item[2], -item[1]))
        return " ".join(f"{sentence} [{number}]" if number else sentence for _, _, number, sentence in best)

    async def complete(self, messages: list[dict[str, Any]], *, json_mode: bool = False, **_: Any) -> str:
        if json_mode:
            return "{}"
        return self._extract(messages)

    async def stream(self, messages: list[dict[str, Any]], **_: Any) -> AsyncIterator[str]:
        for word in (await self.complete(messages)).split(" "):
            yield word + " "


@register("llm", "echo")
class EchoLLM(LLM):
    """Returns a canned string. Useful for unit tests and latency baselines."""

    is_mock = True

    def __init__(self, response: str = "ok") -> None:
        self._response = response

    async def complete(self, messages: list[dict[str, Any]], **_: Any) -> str:
        return self._response


@register("llm", "openai", requires=["openai"])
class OpenAILLM(LLM):
    """OpenAI / any OpenAI-compatible endpoint (vLLM, Together, Groq, LM Studio)."""

    def __init__(
        self, model: str = "gpt-4o-mini", api_key: str | None = None, base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        from openai import AsyncOpenAI

        self.model = model
        self._client = AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY", "not-set"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL") or None,
            timeout=timeout,
        )

    async def complete(
        self, messages: list[dict[str, Any]], *, json_mode: bool = False,
        temperature: float = 0.1, max_tokens: int = 800,
    ) -> str:
        extra = {"response_format": {"type": "json_object"}} if json_mode else {}
        response = await self._client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens, **extra
        )
        return response.choices[0].message.content or ""

    async def stream(
        self, messages: list[dict[str, Any]], *, temperature: float = 0.1, max_tokens: int = 800
    ) -> AsyncIterator[str]:
        response = await self._client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens, stream=True
        )
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def close(self) -> None:
        await self._client.close()


@register("llm", "azure_openai", requires=["openai", "azure-identity"])
class AzureOpenAILLM(LLM):
    """Azure OpenAI with API-key or keyless (Entra ID / managed identity) auth."""

    def __init__(
        self, deployment: str | None = None, endpoint: str | None = None, api_key: str | None = None,
        api_version: str = "2024-10-21",
    ) -> None:
        from openai import AsyncAzureOpenAI

        endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise ValueError("azure_openai LLM needs 'endpoint' or AZURE_OPENAI_ENDPOINT.")
        self.model = deployment or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini")
        key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        kwargs: dict[str, Any] = {"azure_endpoint": endpoint, "api_version": api_version}
        self._credential = None
        if key:
            kwargs["api_key"] = key
        else:
            from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider

            self._credential = DefaultAzureCredential()
            kwargs["azure_ad_token_provider"] = get_bearer_token_provider(
                self._credential, "https://cognitiveservices.azure.com/.default"
            )
        self._client = AsyncAzureOpenAI(**kwargs)

    async def complete(
        self, messages: list[dict[str, Any]], *, json_mode: bool = False,
        temperature: float = 0.1, max_tokens: int = 800,
    ) -> str:
        extra = {"response_format": {"type": "json_object"}} if json_mode else {}
        response = await self._client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens, **extra
        )
        return response.choices[0].message.content or ""

    async def stream(
        self, messages: list[dict[str, Any]], *, temperature: float = 0.1, max_tokens: int = 800
    ) -> AsyncIterator[str]:
        response = await self._client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens, stream=True
        )
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def close(self) -> None:
        await self._client.close()
        if self._credential is not None:
            await self._credential.close()


@register("llm", "anthropic", requires=["anthropic"])
class AnthropicLLM(LLM):
    """Claude via the Anthropic SDK."""

    def __init__(self, model: str = "claude-sonnet-5-5", api_key: str | None = None, timeout: float = 60.0) -> None:
        from anthropic import AsyncAnthropic

        self.model = model
        self._client = AsyncAnthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"), timeout=timeout)

    @staticmethod
    def _split(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        system = "\n".join(str(m["content"]) for m in messages if m.get("role") == "system")
        turns = [{"role": m["role"], "content": str(m["content"])} for m in messages if m.get("role") != "system"]
        return system, turns or [{"role": "user", "content": ""}]

    async def complete(
        self, messages: list[dict[str, Any]], *, json_mode: bool = False,
        temperature: float = 0.1, max_tokens: int = 800,
    ) -> str:
        system, turns = self._split(messages)
        if json_mode:
            system = f"{system}\nRespond with a single valid JSON object and nothing else.".strip()
        response = await self._client.messages.create(
            model=self.model, system=system or None, messages=turns,
            temperature=temperature, max_tokens=max_tokens,
        )
        return "".join(block.text for block in response.content if block.type == "text")

    async def stream(
        self, messages: list[dict[str, Any]], *, temperature: float = 0.1, max_tokens: int = 800
    ) -> AsyncIterator[str]:
        system, turns = self._split(messages)
        async with self._client.messages.stream(
            model=self.model, system=system or None, messages=turns,
            temperature=temperature, max_tokens=max_tokens,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def close(self) -> None:
        await self._client.close()


@register("llm", "ollama", requires=["httpx"])
class OllamaLLM(LLM):
    """Local models via Ollama — zero cost, fully offline, good for experimentation."""

    def __init__(self, model: str = "llama3.1", base_url: str = "http://localhost:11434", timeout: float = 180.0) -> None:
        import httpx

        self.model = model
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout)

    async def complete(
        self, messages: list[dict[str, Any]], *, json_mode: bool = False,
        temperature: float = 0.1, max_tokens: int = 800,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model, "messages": messages, "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if json_mode:
            payload["format"] = "json"
        response = await self._client.post("/api/chat", json=payload)
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "")

    async def stream(
        self, messages: list[dict[str, Any]], *, temperature: float = 0.1, max_tokens: int = 800
    ) -> AsyncIterator[str]:
        payload = {
            "model": self.model, "messages": messages, "stream": True,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        async with self._client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                token = json.loads(line).get("message", {}).get("content", "")
                if token:
                    yield token

    async def close(self) -> None:
        await self._client.aclose()


@register("llm", "huggingface", requires=["transformers", "torch"])
class HuggingFaceLLM(LLM):
    """A local transformers pipeline. Heavy, but keeps everything on-premise."""

    def __init__(self, model: str = "Qwen/Qwen2.5-1.5B-Instruct", device: str | None = None, dtype: str = "auto") -> None:
        from transformers import pipeline

        self.model = model
        self._pipe = pipeline("text-generation", model=model, device_map=device or "auto", torch_dtype=dtype)

    async def complete(
        self, messages: list[dict[str, Any]], *, json_mode: bool = False,
        temperature: float = 0.1, max_tokens: int = 800,
    ) -> str:
        import asyncio

        def run() -> str:
            out = self._pipe(
                messages, max_new_tokens=max_tokens, do_sample=temperature > 0, temperature=max(temperature, 0.01)
            )
            generated = out[0]["generated_text"]
            return generated[-1]["content"] if isinstance(generated, list) else str(generated)

        return await asyncio.to_thread(run)


@register("llm", "cached", requires=[])
class CachedLLM(LLM):
    """Wraps any LLM with an on-disk response cache — makes sweeps and reruns free.

    Configure as ``{"type": "cached", "inner": {"type": "openai"}, "path": ".cache/llm"}``.
    """

    def __init__(self, inner: dict[str, Any] | str, path: str = ".nexus_cache/llm") -> None:
        from pathlib import Path

        from app.rag.registry import build
        from app.textutil import sha256_hex

        self._inner: LLM = build("llm", inner)
        self._dir = Path(path)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._hash = sha256_hex
        self.is_mock = self._inner.is_mock
        self.hits = 0
        self.misses = 0

    def _path(self, messages: list[dict[str, Any]], suffix: str):
        key = self._hash(json.dumps([messages, suffix], sort_keys=True, default=str))
        return self._dir / f"{key}.json"

    async def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        path = self._path(messages, f"{kwargs.get('json_mode')}:{kwargs.get('temperature')}")
        if path.exists():
            self.hits += 1
            return json.loads(path.read_text("utf-8"))["response"]
        self.misses += 1
        response = await self._inner.complete(messages, **kwargs)
        path.write_text(json.dumps({"response": response}), "utf-8")
        return response

    async def stream(self, messages: list[dict[str, Any]], **kwargs: Any) -> AsyncIterator[str]:
        text = await self.complete(messages, **kwargs)
        for word in text.split(" "):
            yield word + " "

    def describe(self) -> dict[str, Any]:
        return {"type": "cached", "kind": "llm", "inner": self._inner.describe(),
                "hits": self.hits, "misses": self.misses}

    async def close(self) -> None:
        await self._inner.close()


def estimate_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    """Rough USD cost estimate used by sweep reports. Prices are per 1M tokens."""
    table = {
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4o": (2.50, 10.00),
        "gpt-4.1-mini": (0.40, 1.60),
        "claude-haiku-5-5": (1.00, 5.00),
        "claude-sonnet-5-5": (3.00, 15.00),
        "text-embedding-3-small": (0.02, 0.0),
        "text-embedding-3-large": (0.13, 0.0),
    }
    key = next((k for k in table if k in model), None)
    if key is None:
        return 0.0
    prompt_price, completion_price = table[key]
    return (prompt_tokens * prompt_price + completion_tokens * completion_price) / 1_000_000


__all__ = ["estimate_cost", "estimate_tokens"]
