"""Evaluation metrics: retrieval quality, answer quality and safety.

Every metric returns a float in [0, 1] where higher is better. Metrics that can
work without an LLM do, so the whole suite runs offline; the LLM-judged ones
degrade to a lexical proxy rather than failing.
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any

from app.rag.base import LLM, Metric
from app.rag.registry import register
from app.rag.types import EvalSample, RAGResult
from app.textutil import content_terms, split_sentences

log = logging.getLogger(__name__)


# --------------------------------------------------------------------- helpers
def has_retrieval_labels(sample: EvalSample) -> bool:
    """True when retrieval metrics can be computed for this sample at all.

    Ground-truth answers count: without ids or reference passages we fall back to
    answer containment, which is a weaker but honest relevance signal.
    """
    return bool(sample.relevant_ids or sample.reference_contexts or sample.ground_truth)


def _relevant_hits(sample: EvalSample, result: RAGResult) -> list[bool]:
    """Per-rank relevance, matching by chunk id, doc id, reference text or answer containment."""
    wanted = set(sample.relevant_ids)
    references = [content_terms(t) for t in sample.reference_contexts if t.strip()]
    # Fallback for datasets that only carry a question and an answer: a passage is
    # "relevant" when it contains most of the answer's content words.
    answer_terms = content_terms(sample.ground_truth or "") if not wanted and not references else set()
    hits = []
    for item in result.contexts:
        if wanted and (item.chunk.id in wanted or item.chunk.doc_id in wanted or item.chunk.filename in wanted):
            hits.append(True)
            continue
        terms = content_terms(item.chunk.context_text)
        if references:
            hits.append(any(len(terms & ref) / max(1, len(ref)) >= 0.5 for ref in references))
        elif answer_terms:
            hits.append(len(terms & answer_terms) / len(answer_terms) >= 0.75)
        else:
            hits.append(False)
    return hits


def _total_relevant(sample: EvalSample) -> int:
    return len(sample.relevant_ids) or len(sample.reference_contexts) or 1


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style token F1 — the standard lexical answer-overlap score."""
    predicted, expected = content_terms(prediction), content_terms(reference)
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = len(predicted & expected)
    if not overlap:
        return 0.0
    precision, recall = overlap / len(predicted), overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def lexical_groundedness(answer: str, contexts: list[str]) -> float:
    """Share of answer sentences whose content words appear in the context.

    An LLM-free proxy for faithfulness; correlates well enough to catch
    regressions and costs nothing.
    """
    sentences = [s for s in split_sentences(answer) if len(content_terms(s)) >= 2]
    if not sentences:
        return 1.0
    pool: set[str] = set()
    for context in contexts:
        pool |= content_terms(context)
    if not pool:
        return 0.0
    supported = sum(1 for s in sentences if len(content_terms(s) & pool) / len(content_terms(s)) >= 0.6)
    return round(supported / len(sentences), 4)


async def _judge(llm: LLM | None, prompt: str, default: float) -> float:
    if llm is None or llm.is_mock:
        return default
    try:
        raw = await llm.ask(prompt, json_mode=True, max_tokens=150, temperature=0)
        parsed = json.loads(raw)
        value = parsed.get("score", parsed.get("value", default))
        return max(0.0, min(1.0, float(value)))
    except Exception as exc:
        log.warning("LLM judge failed (%s); using heuristic fallback", exc)
        return default


# ------------------------------------------------------------------ retrieval
@register("metric", "hit_rate")
class HitRate(Metric):
    """Did at least one relevant chunk make it into the context? The recall floor."""

    needs_ground_truth = True

    def __init__(self, k: int | None = None) -> None:
        self._k = k

    async def score(self, sample: EvalSample, result: RAGResult) -> float:
        hits = _relevant_hits(sample, result)[: self._k] if self._k else _relevant_hits(sample, result)
        return 1.0 if any(hits) else 0.0


@register("metric", "mrr")
class MRR(Metric):
    """Mean Reciprocal Rank: how high the first relevant chunk landed."""

    needs_ground_truth = True

    async def score(self, sample: EvalSample, result: RAGResult) -> float:
        hits = _relevant_hits(sample, result)
        for rank, hit in enumerate(hits, 1):
            if hit:
                return 1.0 / rank
        return 0.0


@register("metric", "ndcg")
class NDCG(Metric):
    """Normalized Discounted Cumulative Gain — rank-sensitive retrieval quality."""

    needs_ground_truth = True

    def __init__(self, k: int = 10) -> None:
        self._k = k

    async def score(self, sample: EvalSample, result: RAGResult) -> float:
        hits = _relevant_hits(sample, result)[: self._k]
        dcg = sum(1 / math.log2(rank + 1) for rank, hit in enumerate(hits, 1) if hit)
        # The ideal ranking puts every relevant item found first. One label (a doc id)
        # can match several retrieved chunks, so the ideal must count the hits we
        # actually saw — otherwise nDCG can exceed 1.
        ideal_count = min(self._k, max(sum(hits), min(_total_relevant(sample), self._k)))
        idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
        return round(min(1.0, dcg / idcg), 4) if idcg else 0.0


@register("metric", "context_precision")
class ContextPrecision(Metric):
    """Fraction of retrieved chunks that are actually relevant — measures noise.

    Low precision wastes tokens and distracts the model even when recall is fine.
    """

    needs_ground_truth = True

    async def score(self, sample: EvalSample, result: RAGResult) -> float:
        hits = _relevant_hits(sample, result)
        return round(sum(hits) / len(hits), 4) if hits else 0.0


@register("metric", "context_recall")
class ContextRecall(Metric):
    """Fraction of the known relevant material that was retrieved.

    The ceiling on your whole system: what retrieval misses, generation cannot fix.
    """

    needs_ground_truth = True

    async def score(self, sample: EvalSample, result: RAGResult) -> float:
        hits = _relevant_hits(sample, result)
        return round(min(1.0, sum(hits) / _total_relevant(sample)), 4)


@register("metric", "context_relevance")
class ContextRelevance(Metric):
    """LLM-judged: what share of the retrieved context is on-topic. No labels needed."""

    needs_llm = True

    def __init__(self, llm: LLM | None = None) -> None:
        self._llm = llm

    async def score(self, sample: EvalSample, result: RAGResult) -> float:
        if not result.contexts:
            return 0.0
        fallback = sum(
            min(1.0, len(content_terms(sample.question) & content_terms(c.chunk.context_text))
                / max(1, len(content_terms(sample.question))))
            for c in result.contexts
        ) / len(result.contexts)
        passages = "\n\n".join(f"[{i}] {c.chunk.context_text[:600]}" for i, c in enumerate(result.contexts, 1))
        prompt = (
            f"Question: {sample.question}\n\nPassages:\n{passages}\n\n"
            'How many passages are relevant to answering the question? '
            'Return JSON {"relevant": <int>, "total": <int>, "score": <relevant/total as 0-1 float>}'
        )
        return await _judge(self._llm, prompt, round(fallback, 4))


# ----------------------------------------------------------------- generation
@register("metric", "faithfulness")
class Faithfulness(Metric):
    """Is every claim in the answer supported by the retrieved context?

    The hallucination metric. Falls back to lexical groundedness without an LLM.
    """

    needs_llm = True

    def __init__(self, llm: LLM | None = None) -> None:
        self._llm = llm

    async def score(self, sample: EvalSample, result: RAGResult) -> float:
        if not result.answer.strip():
            return 0.0
        fallback = lexical_groundedness(result.answer, result.context_texts)
        context = "\n\n".join(result.context_texts)[:8000]
        prompt = (
            "Break the answer into atomic factual claims and count how many are fully "
            f"supported by the context.\n\nContext:\n{context}\n\nAnswer:\n{result.answer}\n\n"
            'Return JSON {"supported": <int>, "total": <int>, "score": <supported/total as 0-1 float>}'
        )
        return await _judge(self._llm, prompt, fallback)


@register("metric", "answer_relevancy")
class AnswerRelevancy(Metric):
    """Does the answer address the question that was asked (not a neighbouring one)?"""

    needs_llm = True

    def __init__(self, llm: LLM | None = None) -> None:
        self._llm = llm

    async def score(self, sample: EvalSample, result: RAGResult) -> float:
        if not result.answer.strip():
            return 0.0
        fallback = token_f1(result.answer, sample.question)
        prompt = (
            f"Question: {sample.question}\n\nAnswer: {result.answer}\n\n"
            "Rate from 0 to 1 how directly this answers the question. Penalise evasive, "
            'partial or off-topic answers. Return JSON {"score": <0-1 float>}'
        )
        return await _judge(self._llm, prompt, round(min(1.0, fallback * 2), 4))


@register("metric", "answer_correctness")
class AnswerCorrectness(Metric):
    """Agreement with the ground-truth answer — the metric that matters most."""

    needs_llm = True
    needs_ground_truth = True

    def __init__(self, llm: LLM | None = None) -> None:
        self._llm = llm

    async def score(self, sample: EvalSample, result: RAGResult) -> float:
        if not sample.ground_truth:
            return 0.0
        fallback = token_f1(result.answer, sample.ground_truth)
        prompt = (
            f"Question: {sample.question}\n\nReference answer: {sample.ground_truth}\n\n"
            f"Candidate answer: {result.answer}\n\n"
            "Rate 0-1 how factually consistent the candidate is with the reference. "
            "Wording may differ; contradictions and missing key facts must be penalised. "
            'Return JSON {"score": <0-1 float>}'
        )
        return await _judge(self._llm, prompt, fallback)


@register("metric", "answer_f1")
class AnswerF1(Metric):
    """Token-level F1 against the ground truth. Deterministic, free, no LLM."""

    needs_ground_truth = True

    async def score(self, sample: EvalSample, result: RAGResult) -> float:
        return round(token_f1(result.answer, sample.ground_truth or ""), 4) if sample.ground_truth else 0.0


@register("metric", "exact_match")
class ExactMatch(Metric):
    """Normalized exact match — only meaningful for short factoid answers."""

    needs_ground_truth = True

    async def score(self, sample: EvalSample, result: RAGResult) -> float:
        if not sample.ground_truth:
            return 0.0
        normalize = lambda s: " ".join(content_terms(s))  # noqa: E731
        return 1.0 if normalize(result.answer) == normalize(sample.ground_truth) else 0.0


@register("metric", "groundedness")
class Groundedness(Metric):
    """Lexical support of the answer by the context. Free and always available."""

    async def score(self, sample: EvalSample, result: RAGResult) -> float:
        if result.groundedness is not None:
            return result.groundedness
        return lexical_groundedness(result.answer, result.context_texts)


@register("metric", "citation_accuracy")
class CitationAccuracy(Metric):
    """Do the bracketed citations point at passages that actually support the sentence?

    Catches the subtle failure where the answer is right but cited to the wrong source.
    """

    async def score(self, sample: EvalSample, result: RAGResult) -> float:
        sentences = [s for s in split_sentences(result.answer) if re.search(r"\[\d+\]", s)]
        if not sentences:
            return 0.0
        correct = 0
        for sentence in sentences:
            cited = {int(n) for n in re.findall(r"\[(\d+)\]", sentence)}
            terms = content_terms(re.sub(r"\[\d+\]", "", sentence))
            if not terms:
                correct += 1
                continue
            supported = any(
                1 <= n <= len(result.contexts)
                and len(terms & content_terms(result.contexts[n - 1].chunk.context_text)) / len(terms) >= 0.5
                for n in cited
            )
            correct += int(supported)
        return round(correct / len(sentences), 4)


@register("metric", "no_answer_accuracy")
class NoAnswerAccuracy(Metric):
    """Does the system abstain exactly when it should?

    Set ``ground_truth`` to an empty string (or ``metadata.unanswerable = true``)
    on samples whose answer is genuinely not in the corpus.
    """

    async def score(self, sample: EvalSample, result: RAGResult) -> float:
        unanswerable = bool(sample.metadata.get("unanswerable")) or sample.ground_truth == ""
        refused = result.abstained or bool(
            re.search(r"(could not find|does not contain|no relevant|cannot answer|not in the)", result.answer, re.I)
        )
        return 1.0 if unanswerable == refused else 0.0


# --------------------------------------------------------------------- safety
@register("metric", "no_pii_leak")
class NoPIILeak(Metric):
    """Penalises answers that emit PII that was not in the question."""

    async def score(self, sample: EvalSample, result: RAGResult) -> float:
        from app.rag.pipeline import RAGPipeline

        leaked = RAGPipeline.detect_pii(result.answer)
        asked = RAGPipeline.detect_pii(sample.question)
        novel = {k: [v for v in values if v not in asked.get(k, [])] for k, values in leaked.items()}
        return 0.0 if any(novel.values()) else 1.0


@register("metric", "latency_budget")
class LatencyBudget(Metric):
    """1.0 inside the budget, decaying to 0 at twice the budget. Quality has a clock."""

    def __init__(self, budget_ms: float = 3000.0) -> None:
        self._budget = budget_ms

    async def score(self, sample: EvalSample, result: RAGResult) -> float:
        if result.latency_ms <= self._budget:
            return 1.0
        return round(max(0.0, 2 - result.latency_ms / self._budget), 4)


DEFAULT_METRICS = ["hit_rate", "mrr", "ndcg", "context_precision", "context_recall",
                   "faithfulness", "answer_relevancy", "answer_correctness", "answer_f1"]
OFFLINE_METRICS = ["hit_rate", "mrr", "ndcg", "context_precision", "context_recall",
                   "groundedness", "answer_f1", "citation_accuracy", "no_answer_accuracy"]
RETRIEVAL_METRICS = ["hit_rate", "mrr", "ndcg", "context_precision", "context_recall"]
