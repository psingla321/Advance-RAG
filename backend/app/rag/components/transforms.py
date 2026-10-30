"""Query transforms — the "advanced" in advanced RAG.

Each returns one or more query strings; the retriever searches all of them and
fuses the results. They compose: ``{"type": "chain", "steps": [...]}``.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.rag.base import LLM, QueryTransform
from app.rag.registry import build, register

log = logging.getLogger(__name__)


def _parse_lines(text: str, limit: int) -> list[str]:
    """Pull a list out of an LLM response whether it came back as JSON or bullets."""
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            parsed = next((v for v in parsed.values() if isinstance(v, list)), [])
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()][:limit]
    except (json.JSONDecodeError, TypeError):
        pass
    lines = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip() for line in text.splitlines()]
    return [line for line in lines if line and len(line) > 3][:limit]


@register("transform", "identity")
class IdentityTransform(QueryTransform):
    """Pass the query through unchanged (the baseline)."""

    async def transform(self, query: str, context: dict[str, Any] | None = None) -> list[str]:
        return [query]


@register("transform", "multi_query")
class MultiQueryTransform(QueryTransform):
    """Generate N paraphrases so lexical mismatch stops hiding relevant chunks.

    The classic fix for "the document says 'remuneration', the user typed 'salary'".
    """

    PROMPT = (
        "Rewrite the user's question as {n} differently-worded search queries that would "
        "retrieve the same information. Vary vocabulary and specificity. Use synonyms a "
        "document author might have used.\n\nQuestion: {query}\n\n"
        'Return JSON: {{"queries": ["...", "..."]}}'
    )

    def __init__(self, llm: LLM | None = None, n: int = 3, include_original: bool = True) -> None:
        self._llm = llm
        self._n = n
        self._include_original = include_original

    async def transform(self, query: str, context: dict[str, Any] | None = None) -> list[str]:
        if self._llm is None:
            return [query]
        try:
            raw = await self._llm.ask(self.PROMPT.format(n=self._n, query=query), json_mode=True, max_tokens=300)
        except Exception as exc:
            log.warning("multi_query transform failed: %s", exc)
            return [query]
        variants = _parse_lines(raw, self._n)
        return ([query] if self._include_original else []) + [v for v in variants if v.lower() != query.lower()]


@register("transform", "hyde")
class HyDETransform(QueryTransform):
    """Hypothetical Document Embeddings: search with a *fake answer*, not the question.

    Answers live in answer-space, questions in question-space; embedding a
    plausible answer closes that gap. Strong on open-ended queries.
    """

    PROMPT = (
        "Write a short, confident passage (3-4 sentences) that would appear in an internal "
        "company document and that directly answers this question. Invent plausible specifics; "
        "this text is only used to find the real document.\n\nQuestion: {query}\n\nPassage:"
    )

    def __init__(self, llm: LLM | None = None, include_original: bool = True, n: int = 1) -> None:
        self._llm = llm
        self._include_original = include_original
        self._n = n

    async def transform(self, query: str, context: dict[str, Any] | None = None) -> list[str]:
        if self._llm is None:
            return [query]
        variants: list[str] = [query] if self._include_original else []
        for _ in range(self._n):
            try:
                passage = await self._llm.ask(self.PROMPT.format(query=query), temperature=0.7, max_tokens=250)
            except Exception as exc:
                log.warning("hyde transform failed: %s", exc)
                break
            if passage.strip():
                variants.append(passage.strip())
        return variants or [query]


@register("transform", "step_back")
class StepBackTransform(QueryTransform):
    """Ask the broader question first — retrieves the principle, not just the detail.

    "Can I expense a $400 hotel in Zurich?" -> "What are the hotel expense limits?"
    """

    PROMPT = (
        "Given a specific question, write one broader, more general question whose answer "
        "would provide the background principle needed. Return only the question.\n\n"
        "Specific: {query}\nGeneral:"
    )

    def __init__(self, llm: LLM | None = None) -> None:
        self._llm = llm

    async def transform(self, query: str, context: dict[str, Any] | None = None) -> list[str]:
        if self._llm is None:
            return [query]
        try:
            broader = (await self._llm.ask(self.PROMPT.format(query=query), max_tokens=100)).strip()
        except Exception as exc:
            log.warning("step_back transform failed: %s", exc)
            return [query]
        return [query, broader] if broader else [query]


@register("transform", "decompose")
class DecomposeTransform(QueryTransform):
    """Break a multi-hop question into independent sub-questions.

    "How does our parental leave compare to the industry and who approves it?"
    becomes two searches that each have a chance of hitting.
    """

    PROMPT = (
        "Decompose this question into at most {n} simpler, independent sub-questions. "
        "If it is already simple, return it unchanged.\n\nQuestion: {query}\n\n"
        'Return JSON: {{"queries": ["...", "..."]}}'
    )

    def __init__(self, llm: LLM | None = None, n: int = 3, include_original: bool = True) -> None:
        self._llm = llm
        self._n = n
        self._include_original = include_original

    async def transform(self, query: str, context: dict[str, Any] | None = None) -> list[str]:
        if self._llm is None:
            return [query]
        try:
            raw = await self._llm.ask(self.PROMPT.format(n=self._n, query=query), json_mode=True, max_tokens=300)
        except Exception as exc:
            log.warning("decompose transform failed: %s", exc)
            return [query]
        return ([query] if self._include_original else []) + _parse_lines(raw, self._n)


@register("transform", "rewrite")
class RewriteTransform(QueryTransform):
    """Resolve pronouns and ellipsis against conversation history.

    Turns a follow-up ("what about APAC?") into a standalone query. Pass the
    history in ``context={"history": [...]}``.
    """

    PROMPT = (
        "Rewrite the follow-up question as a standalone question using the conversation. "
        "Keep it short, resolve every pronoun, and change nothing if it is already standalone. "
        "Return only the question.\n\nConversation:\n{history}\n\nFollow-up: {query}\nStandalone:"
    )

    def __init__(self, llm: LLM | None = None, max_turns: int = 4) -> None:
        self._llm = llm
        self._max_turns = max_turns

    async def transform(self, query: str, context: dict[str, Any] | None = None) -> list[str]:
        history = (context or {}).get("history") or []
        if self._llm is None or not history:
            return [query]
        rendered = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in history[-self._max_turns :])
        try:
            standalone = (await self._llm.ask(self.PROMPT.format(history=rendered, query=query), max_tokens=120)).strip()
        except Exception as exc:
            log.warning("rewrite transform failed: %s", exc)
            return [query]
        return [standalone or query]


@register("transform", "expand_acronyms")
class AcronymTransform(QueryTransform):
    """Deterministic domain glossary expansion — no LLM call, no latency.

    Configure with ``{"glossary": {"PTO": "paid time off", "WFH": "work from home"}}``.
    """

    def __init__(self, glossary: dict[str, str] | None = None, mode: str = "append") -> None:
        self._glossary = {k.lower(): v for k, v in (glossary or {}).items()}
        self._mode = mode

    async def transform(self, query: str, context: dict[str, Any] | None = None) -> list[str]:
        if not self._glossary:
            return [query]
        expanded = query
        hit = False
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]*", query):
            replacement = self._glossary.get(token.lower())
            if replacement:
                hit = True
                expanded = re.sub(rf"\b{re.escape(token)}\b", replacement, expanded)
        if not hit:
            return [query]
        return [query, expanded] if self._mode == "append" else [expanded]


@register("transform", "chain")
class ChainTransform(QueryTransform):
    """Run several transforms and union their outputs, de-duplicated.

    ``{"type": "chain", "steps": [{"type": "rewrite"}, {"type": "hyde"}]}``
    """

    def __init__(self, steps: list[dict[str, Any] | str] | None = None, llm: LLM | None = None, max_queries: int = 6) -> None:
        self._steps: list[QueryTransform] = [build("transform", step, llm=llm) for step in (steps or [])]
        self._max = max_queries

    async def transform(self, query: str, context: dict[str, Any] | None = None) -> list[str]:
        seen: dict[str, None] = {}
        current = [query]
        for step in self._steps:
            produced: list[str] = []
            for candidate in current:
                produced.extend(await step.transform(candidate, context))
            current = produced or current
        for item in current:
            seen.setdefault(item.strip(), None)
        return list(seen)[: self._max]

    def describe(self) -> dict[str, Any]:
        return {"type": "chain", "kind": "transform", "steps": [s.describe() for s in self._steps]}
