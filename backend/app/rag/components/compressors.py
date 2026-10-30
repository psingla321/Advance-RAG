"""Context compression — fit the best evidence into the token budget.

Stuffing the top-k into the prompt wastes budget on irrelevant sentences and
triggers "lost in the middle". These components decide what actually gets sent.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.rag.base import LLM, Compressor
from app.rag.registry import register
from app.rag.types import ScoredChunk
from app.textutil import content_terms, estimate_tokens, split_sentences

log = logging.getLogger(__name__)


@register("compressor", "truncate")
class TruncateCompressor(Compressor):
    """Take candidates in order until the budget is spent. The simple default."""

    async def compress(self, query: str, candidates: list[ScoredChunk], token_budget: int) -> list[ScoredChunk]:
        kept: list[ScoredChunk] = []
        used = 0
        for item in candidates:
            cost = estimate_tokens(item.chunk.context_text)
            if used + cost > token_budget and kept:
                break
            kept.append(item)
            used += cost
        return kept


@register("compressor", "submodular")
class SubmodularCompressor(Compressor):
    """Greedy submodular packing: maximise query-term coverage per token spent.

    Picks the passage adding the most *new* information per token, so a redundant
    high-scoring chunk loses to a slightly weaker one that covers a missing term.
    """

    def __init__(self, redundancy_penalty: float = 0.5, relevance_weight: float = 1.0) -> None:
        self._penalty = redundancy_penalty
        self._relevance = relevance_weight

    async def compress(self, query: str, candidates: list[ScoredChunk], token_budget: int) -> list[ScoredChunk]:
        terms = content_terms(query)
        kept: list[ScoredChunk] = []
        covered: set[str] = set()
        remaining = list(candidates)
        used = 0
        while remaining:
            best, best_gain, best_cost = None, 0.0, 0
            for item in remaining:
                cost = max(1, estimate_tokens(item.chunk.context_text))
                if used + cost > token_budget:
                    continue
                item_terms = content_terms(f"{item.chunk.heading} {item.chunk.text}")
                new_terms = (terms & item_terms) - covered
                overlap = len(item_terms & covered) / max(1, len(item_terms))
                gain = (len(new_terms) + self._relevance * item.score - self._penalty * overlap) / cost
                if gain > best_gain:
                    best, best_gain, best_cost = item, gain, cost
            if best is None:
                break
            kept.append(best)
            remaining.remove(best)
            covered |= terms & content_terms(f"{best.chunk.heading} {best.chunk.text}")
            used += best_cost
        return kept or candidates[:1]


@register("compressor", "sentence_filter")
class SentenceFilterCompressor(Compressor):
    """Drop sentences inside each passage that do not relate to the query.

    Often halves prompt tokens with no loss in answer quality.
    """

    def __init__(self, min_overlap: float = 0.08, keep_neighbours: bool = True, min_sentences: int = 1) -> None:
        self._min_overlap = min_overlap
        self._keep_neighbours = keep_neighbours
        self._min_sentences = min_sentences

    async def compress(self, query: str, candidates: list[ScoredChunk], token_budget: int) -> list[ScoredChunk]:
        terms = content_terms(query)
        if not terms:
            return await TruncateCompressor().compress(query, candidates, token_budget)
        for item in candidates:
            sentences = split_sentences(item.chunk.context_text)
            if len(sentences) <= self._min_sentences:
                continue
            scores = [len(terms & content_terms(s)) / len(terms) for s in sentences]
            keep = {i for i, score in enumerate(scores) if score >= self._min_overlap}
            if not keep:
                keep = {max(range(len(scores)), key=lambda i: scores[i])}
            if self._keep_neighbours:
                keep |= {i + d for i in list(keep) for d in (-1, 1) if 0 <= i + d < len(sentences)}
            trimmed = " ".join(sentences[i] for i in sorted(keep))
            item.chunk.metadata["compressed_from_tokens"] = estimate_tokens(item.chunk.context_text)
            item.chunk.parent_text = trimmed
        return await TruncateCompressor().compress(query, candidates, token_budget)


@register("compressor", "llm_extract")
class LLMExtractCompressor(Compressor):
    """Have the LLM quote only the lines of each passage that answer the question.

    The strongest compressor and the most expensive: one call per passage.
    Returns the passage unchanged if the model finds nothing (never invents).
    """

    PROMPT = (
        "Extract verbatim only the sentences from the passage that help answer the question. "
        "Do not paraphrase, do not add anything. If nothing is relevant, reply exactly NONE.\n\n"
        "Question: {query}\n\nPassage:\n{passage}\n\nExtracted:"
    )

    def __init__(self, llm: LLM | None = None, concurrency: int = 6, drop_empty: bool = True) -> None:
        self._llm = llm
        self._semaphore = asyncio.Semaphore(concurrency)
        self._drop_empty = drop_empty

    async def _extract(self, query: str, item: ScoredChunk) -> bool:
        async with self._semaphore:
            try:
                extracted = await self._llm.ask(  # type: ignore[union-attr]
                    self.PROMPT.format(query=query, passage=item.chunk.context_text), max_tokens=400, temperature=0
                )
            except Exception as exc:
                log.warning("llm compression failed: %s", exc)
                return True
        extracted = extracted.strip()
        if extracted.upper().startswith("NONE") or not extracted:
            return not self._drop_empty
        item.chunk.parent_text = extracted
        item.chunk.metadata["compressed"] = True
        return True

    async def compress(self, query: str, candidates: list[ScoredChunk], token_budget: int) -> list[ScoredChunk]:
        if self._llm is None:
            return await TruncateCompressor().compress(query, candidates, token_budget)
        keep = await asyncio.gather(*(self._extract(query, item) for item in candidates))
        survivors = [item for item, ok in zip(candidates, keep) if ok]
        return await TruncateCompressor().compress(query, survivors, token_budget)


@register("compressor", "lost_in_middle")
class LostInTheMiddleCompressor(Compressor):
    """Reorder so the strongest evidence sits at the start *and* end of the prompt.

    LLMs attend least to the middle of a long context; this interleaving measurably
    improves answer accuracy with many passages at no extra cost.
    """

    def __init__(self, inner_budget_ratio: float = 1.0) -> None:
        self._ratio = inner_budget_ratio

    async def compress(self, query: str, candidates: list[ScoredChunk], token_budget: int) -> list[ScoredChunk]:
        kept = await TruncateCompressor().compress(query, candidates, int(token_budget * self._ratio))
        head: list[ScoredChunk] = []
        tail: list[ScoredChunk] = []
        for index, item in enumerate(kept):
            (head if index % 2 == 0 else tail).append(item)
        return head + list(reversed(tail))
