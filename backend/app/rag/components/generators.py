"""Answer generation strategies, from plain stuffing to self-correcting RAG."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator
from typing import Any

from app.rag.base import LLM, Generator
from app.rag.registry import build, register
from app.rag.types import Citation, RAGResult, ScoredChunk
from app.textutil import content_terms, estimate_tokens, split_sentences

log = logging.getLogger(__name__)

DEFAULT_SYSTEM = (
    "You answer strictly from the provided context. Cite the passage number in square "
    "brackets, like [2], after every factual sentence. If the context does not contain "
    "the answer, say so plainly instead of guessing. Be concise and specific."
)


def render_context(contexts: list[ScoredChunk]) -> str:
    blocks = []
    for n, item in enumerate(contexts, 1):
        source = item.chunk.filename or item.chunk.doc_id
        heading = f" — {item.chunk.heading}" if item.chunk.heading else ""
        blocks.append(f"[{n}] ({source}{heading})\n{item.chunk.context_text}")
    return "\n\n".join(blocks)


def build_citations(contexts: list[ScoredChunk], answer: str, only_cited: bool = True) -> list[Citation]:
    used = {int(n) for n in re.findall(r"\[(\d+)\]", answer)} if only_cited else set(range(1, len(contexts) + 1))
    citations = []
    for n, item in enumerate(contexts, 1):
        if only_cited and n not in used:
            continue
        citations.append(
            Citation(
                n=n, chunk_id=item.chunk.id, doc_id=item.chunk.doc_id,
                filename=item.chunk.filename or item.chunk.doc_id, heading=item.chunk.heading,
                snippet=item.chunk.text[:300], score=round(item.score, 4),
            )
        )
    return citations


def _messages(query: str, contexts: list[ScoredChunk], history: list[dict[str, str]] | None, system: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": f"Context:\n{render_context(contexts)}\n\nQuestion: {query}"})
    return messages


@register("generator", "stuff")
class StuffGenerator(Generator):
    """Put all the context in one prompt and answer. The default, and usually right."""

    def __init__(
        self, llm: LLM | None = None, system: str = DEFAULT_SYSTEM, temperature: float = 0.1,
        max_tokens: int = 800, cite: bool = True,
    ) -> None:
        self._llm = llm
        self._system = system
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._cite = cite

    async def generate(
        self, query: str, contexts: list[ScoredChunk], *, history: list[dict[str, str]] | None = None,
        system: str | None = None,
    ) -> RAGResult:
        if not contexts:
            return RAGResult(query=query, answer="I could not find anything relevant in the knowledge base.", abstained=True)
        messages = _messages(query, contexts, history, system or self._system)
        answer = await self._llm.complete(  # type: ignore[union-attr]
            messages, temperature=self._temperature, max_tokens=self._max_tokens
        )
        return RAGResult(
            query=query, answer=answer.strip(), contexts=contexts,
            citations=build_citations(contexts, answer, only_cited=self._cite),
            usage={"prompt_tokens_est": estimate_tokens(render_context(contexts)) + estimate_tokens(query)},
        )

    async def stream(
        self, query: str, contexts: list[ScoredChunk], *, history: list[dict[str, str]] | None = None,
        system: str | None = None,
    ) -> AsyncIterator[str]:
        if not contexts:
            yield "I could not find anything relevant in the knowledge base."
            return
        async for token in self._llm.stream(  # type: ignore[union-attr]
            _messages(query, contexts, history, system or self._system),
            temperature=self._temperature, max_tokens=self._max_tokens,
        ):
            yield token


@register("generator", "refine")
class RefineGenerator(Generator):
    """Walk the passages one at a time, refining a running answer.

    Handles more context than fits in one prompt, and is resilient when evidence
    is spread thin across many documents. Slower: one call per passage.
    """

    FIRST = "Answer the question using only this context.\n\nContext:\n{context}\n\nQuestion: {query}"
    REFINE = (
        "You have a draft answer. Improve it using the new context if — and only if — the new "
        "context adds or corrects something. Otherwise return the draft unchanged.\n\n"
        "Question: {query}\n\nDraft:\n{draft}\n\nNew context:\n{context}\n\nImproved answer:"
    )

    def __init__(self, llm: LLM | None = None, system: str = DEFAULT_SYSTEM, max_tokens: int = 800, batch: int = 2) -> None:
        self._llm = llm
        self._system = system
        self._max_tokens = max_tokens
        self._batch = batch

    async def generate(
        self, query: str, contexts: list[ScoredChunk], *, history: list[dict[str, str]] | None = None,
        system: str | None = None,
    ) -> RAGResult:
        if not contexts:
            return RAGResult(query=query, answer="No relevant context was found.", abstained=True)
        system = system or self._system
        answer = ""
        calls = 0
        for start in range(0, len(contexts), self._batch):
            window = contexts[start : start + self._batch]
            rendered = render_context(window)
            prompt = (
                self.FIRST.format(context=rendered, query=query) if not answer
                else self.REFINE.format(query=query, draft=answer, context=rendered)
            )
            answer = (await self._llm.ask(prompt, system=system, max_tokens=self._max_tokens)).strip()  # type: ignore[union-attr]
            calls += 1
        return RAGResult(
            query=query, answer=answer, contexts=contexts,
            citations=build_citations(contexts, answer, only_cited=False),
            usage={"llm_calls": calls},
        )


@register("generator", "map_reduce")
class MapReduceGenerator(Generator):
    """Answer from each passage in parallel, then synthesise. Fast over wide context."""

    MAP = "Using only this passage, write what it says about the question, or NONE.\n\nPassage:\n{context}\n\nQuestion: {query}"
    REDUCE = (
        "Combine these partial findings into one coherent, non-repetitive answer. "
        "Cite passage numbers in square brackets.\n\nQuestion: {query}\n\nFindings:\n{findings}\n\nAnswer:"
    )

    def __init__(self, llm: LLM | None = None, system: str = DEFAULT_SYSTEM, max_tokens: int = 800, concurrency: int = 8) -> None:
        self._llm = llm
        self._system = system
        self._max_tokens = max_tokens
        self._concurrency = concurrency

    async def generate(
        self, query: str, contexts: list[ScoredChunk], *, history: list[dict[str, str]] | None = None,
        system: str | None = None,
    ) -> RAGResult:
        import asyncio

        if not contexts:
            return RAGResult(query=query, answer="No relevant context was found.", abstained=True)
        semaphore = asyncio.Semaphore(self._concurrency)

        async def map_one(n: int, item: ScoredChunk) -> str:
            async with semaphore:
                out = await self._llm.ask(  # type: ignore[union-attr]
                    self.MAP.format(context=item.chunk.context_text, query=query), max_tokens=300
                )
            return "" if out.strip().upper().startswith("NONE") else f"[{n}] {out.strip()}"

        partials = [p for p in await asyncio.gather(*(map_one(n, c) for n, c in enumerate(contexts, 1))) if p]
        if not partials:
            return RAGResult(query=query, answer="The retrieved passages do not answer this question.",
                             contexts=contexts, abstained=True)
        answer = (
            await self._llm.ask(  # type: ignore[union-attr]
                self.REDUCE.format(query=query, findings="\n\n".join(partials)),
                system=system or self._system, max_tokens=self._max_tokens,
            )
        ).strip()
        return RAGResult(
            query=query, answer=answer, contexts=contexts,
            citations=build_citations(contexts, answer),
            usage={"llm_calls": len(contexts) + 1},
        )


@register("generator", "citation_strict")
class CitationStrictGenerator(Generator):
    """Forces per-sentence citations and drops any sentence the model cannot cite.

    The safest option for compliance-sensitive answers: an uncited claim never
    reaches the user.
    """

    SYSTEM = (
        "Answer only from the context. Every sentence MUST end with one or more citations "
        "like [1] or [2][3]. Never write a sentence you cannot cite. If the context is "
        "insufficient, reply exactly: INSUFFICIENT_CONTEXT"
    )

    def __init__(self, llm: LLM | None = None, max_tokens: int = 800, drop_uncited: bool = True) -> None:
        self._llm = llm
        self._max_tokens = max_tokens
        self._drop = drop_uncited

    async def generate(
        self, query: str, contexts: list[ScoredChunk], *, history: list[dict[str, str]] | None = None,
        system: str | None = None,
    ) -> RAGResult:
        if not contexts:
            return RAGResult(query=query, answer="No relevant context was found.", abstained=True)
        raw = await self._llm.complete(  # type: ignore[union-attr]
            _messages(query, contexts, history, system or self.SYSTEM), temperature=0, max_tokens=self._max_tokens
        )
        raw = raw.strip()
        if "INSUFFICIENT_CONTEXT" in raw.upper():
            return RAGResult(query=query, answer="The knowledge base does not contain an answer to this question.",
                             contexts=contexts, abstained=True)
        sentences = split_sentences(raw)
        kept = [s for s in sentences if re.search(r"\[\d+\]", s)] if self._drop else sentences
        dropped = len(sentences) - len(kept)
        answer = " ".join(kept) or raw
        return RAGResult(
            query=query, answer=answer, contexts=contexts, citations=build_citations(contexts, answer),
            abstained=not kept, metadata={"dropped_uncited_sentences": dropped},
        )


@register("generator", "self_rag")
class SelfRAGGenerator(Generator):
    """Generate, self-critique for groundedness, then repair or abstain.

    A cheap approximation of Self-RAG / CRAG: the second pass catches the
    hallucinated sentence the first pass produced, without a trained critic model.
    """

    CRITIQUE = (
        "For each sentence of the answer, decide whether the context fully supports it.\n\n"
        "Context:\n{context}\n\nAnswer:\n{answer}\n\n"
        'Return JSON: {{"supported": [sentence numbers], "unsupported": [sentence numbers]}}'
    )
    REPAIR = (
        "Rewrite the answer keeping only what the context supports. Remove or correct the "
        "flagged sentences. Keep the citations.\n\nContext:\n{context}\n\n"
        "Answer:\n{answer}\n\nUnsupported sentences: {flagged}\n\nCorrected answer:"
    )

    def __init__(
        self, llm: LLM | None = None, inner: dict[str, Any] | str | None = None,
        groundedness_threshold: float = 0.8, max_repairs: int = 1,
    ) -> None:
        self._llm = llm
        self._inner: Generator = build("generator", inner or {"type": "stuff"}, llm=llm)
        self._threshold = groundedness_threshold
        self._max_repairs = max_repairs

    async def generate(
        self, query: str, contexts: list[ScoredChunk], *, history: list[dict[str, str]] | None = None,
        system: str | None = None,
    ) -> RAGResult:
        result = await self._inner.generate(query, contexts, history=history, system=system)
        if not contexts or result.abstained or self._llm is None:
            return result
        rendered = render_context(contexts)
        for attempt in range(self._max_repairs + 1):
            sentences = split_sentences(result.answer)
            if not sentences:
                break
            numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, 1))
            try:
                raw = await self._llm.ask(
                    self.CRITIQUE.format(context=rendered, answer=numbered), json_mode=True, max_tokens=300
                )
                verdict = json.loads(raw)
                unsupported = [int(n) for n in verdict.get("unsupported", []) if 1 <= int(n) <= len(sentences)]
            except Exception as exc:
                log.warning("self-RAG critique failed: %s", exc)
                break
            grounded = 1.0 - len(unsupported) / len(sentences)
            result.groundedness = round(grounded, 3)
            result.trace.append({"stage": "self_critique", "attempt": attempt, "groundedness": grounded,
                                 "unsupported": unsupported})
            if grounded >= self._threshold or attempt == self._max_repairs:
                break
            flagged = "; ".join(sentences[n - 1] for n in unsupported)
            repaired = await self._llm.ask(
                self.REPAIR.format(context=rendered, answer=result.answer, flagged=flagged), max_tokens=800
            )
            result.answer = repaired.strip() or result.answer
            result.citations = build_citations(contexts, result.answer)
        if result.groundedness is not None and result.groundedness < 0.4:
            result.abstained = True
            result.answer = (
                "I could not verify an answer against the knowledge base, so I would rather not guess. "
                + result.answer
            )
        return result

    async def stream(self, query: str, contexts: list[ScoredChunk], **kwargs: Any) -> AsyncIterator[str]:
        result = await self.generate(query, contexts, **kwargs)
        for word in result.answer.split(" "):
            yield word + " "


@register("generator", "extractive")
class ExtractiveGenerator(Generator):
    """No LLM at all: return the best-matching sentences verbatim.

    Zero hallucination risk, zero cost, and a useful floor to compare against
    when you are deciding whether a generative answer is worth paying for.
    """

    def __init__(self, max_sentences: int = 4) -> None:
        self._max = max_sentences

    async def generate(
        self, query: str, contexts: list[ScoredChunk], *, history: list[dict[str, str]] | None = None,
        system: str | None = None,
    ) -> RAGResult:
        terms = content_terms(query)
        scored: list[tuple[float, int, str]] = []
        for n, item in enumerate(contexts, 1):
            for sentence in split_sentences(item.chunk.context_text):
                overlap = len(terms & content_terms(sentence)) / max(1, len(terms))
                if overlap > 0:
                    scored.append((overlap, n, sentence))
        scored.sort(reverse=True)
        picked = scored[: self._max]
        if not picked:
            return RAGResult(query=query, answer="No matching sentence was found in the knowledge base.",
                             contexts=contexts, abstained=True)
        answer = " ".join(f"{sentence} [{n}]" for _, n, sentence in picked)
        return RAGResult(query=query, answer=answer, contexts=contexts,
                         citations=build_citations(contexts, answer), groundedness=1.0)
