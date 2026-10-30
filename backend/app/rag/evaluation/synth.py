"""Synthesise a golden test set from your own corpus.

Most teams never evaluate their RAG system because nobody wants to hand-write
200 question/answer pairs. This generates a first draft from the indexed chunks —
with the source chunk id attached, so retrieval metrics work immediately.

Review the output before trusting it: treat it as a labelled draft, not truth.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Any

from app.rag.base import LLM
from app.rag.evaluation.dataset import Dataset
from app.rag.types import Chunk, EvalSample
from app.textutil import content_terms, estimate_tokens, split_sentences

log = logging.getLogger(__name__)

QUESTION_TYPES = {
    "factual": "a direct factual question answered by a single sentence of the passage",
    "numeric": "a question whose answer is a number, limit, amount or date in the passage",
    "conditional": "a question about when a rule or exception in the passage applies",
    "comparative": "a question comparing two things mentioned in the passage",
    "multi_hop": "a question that needs this passage plus general reasoning to answer",
}

PROMPT = (
    "You are writing an evaluation set for a retrieval system over internal documents.\n\n"
    "Passage (from {filename}{heading}):\n<<<\n{passage}\n>>>\n\n"
    "Write {n} question/answer pairs of type: {kind} ({kind_description}).\n"
    "Rules:\n"
    "- The question must be answerable from this passage alone.\n"
    "- The question must make sense to someone who has NOT seen the passage "
    "(no 'this document', no 'the above').\n"
    "- The answer must be short and quoted or closely paraphrased from the passage.\n"
    "- Skip the passage entirely if it is boilerplate, a table of contents or navigation.\n\n"
    'Return JSON: {{"pairs": [{{"question": "...", "answer": "..."}}]}}'
)

UNANSWERABLE_PROMPT = (
    "Here are topics covered by a company knowledge base:\n{topics}\n\n"
    "Write {n} questions that sound like they belong to this domain but are NOT answerable "
    "from these topics — a good system must refuse them rather than hallucinate.\n"
    'Return JSON: {{"questions": ["...", "..."]}}'
)


def _parse_pairs(raw: str) -> list[dict[str, str]]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    pairs = parsed.get("pairs") if isinstance(parsed, dict) else parsed
    if not isinstance(pairs, list):
        return []
    return [
        {"question": str(p["question"]).strip(), "answer": str(p.get("answer", "")).strip()}
        for p in pairs
        if isinstance(p, dict) and str(p.get("question", "")).strip()
    ]


def _is_useful(chunk: Chunk, min_tokens: int) -> bool:
    """Skip boilerplate so the generator does not waste calls on tables of contents."""
    text = chunk.text.strip()
    if estimate_tokens(text) < min_tokens:
        return False
    sentences = split_sentences(text)
    if len(sentences) < 2:
        return False
    # Mostly-unique content words, not a repeated navigation block.
    terms = content_terms(text)
    return len(terms) >= 15


async def generate_dataset(
    chunks: list[Chunk],
    llm: LLM,
    *,
    n_questions: int = 50,
    per_chunk: int = 1,
    kinds: list[str] | None = None,
    min_chunk_tokens: int = 60,
    unanswerable_ratio: float = 0.1,
    concurrency: int = 6,
    seed: int = 0,
    show_progress: bool = True,
) -> Dataset:
    """Generate ``n_questions`` QA pairs spread across the corpus.

    ``unanswerable_ratio`` adds questions that *should* be refused — without them
    an evaluation rewards a system that confidently answers everything.
    """
    kinds = kinds or ["factual", "numeric", "conditional"]
    usable = [c for c in chunks if _is_useful(c, min_chunk_tokens)]
    if not usable:
        raise ValueError("No chunks are substantial enough to generate questions from.")

    rng = random.Random(seed)
    answerable_target = max(1, int(n_questions * (1 - unanswerable_ratio)))
    needed = max(1, answerable_target // max(1, per_chunk))
    selected = rng.sample(usable, min(needed, len(usable)))
    semaphore = asyncio.Semaphore(concurrency)
    done = 0

    async def generate_for(chunk: Chunk) -> list[EvalSample]:
        nonlocal done
        kind = rng.choice(kinds)
        async with semaphore:
            try:
                raw = await llm.ask(
                    PROMPT.format(
                        filename=chunk.filename or chunk.doc_id,
                        heading=f", section '{chunk.heading}'" if chunk.heading else "",
                        passage=chunk.context_text[:2500], n=per_chunk,
                        kind=kind, kind_description=QUESTION_TYPES.get(kind, kind),
                    ),
                    json_mode=True, max_tokens=500, temperature=0.4,
                )
            except Exception as exc:
                log.warning("Question generation failed for %s: %s", chunk.id, exc)
                return []
        done += 1
        if show_progress and done % 10 == 0:
            print(f"  generated from {done}/{len(selected)} chunks", flush=True)
        return [
            EvalSample(
                question=pair["question"],
                ground_truth=pair["answer"],
                relevant_ids=[chunk.id, chunk.doc_id],
                reference_contexts=[chunk.context_text],
                metadata={"kind": kind, "source_chunk": chunk.id, "filename": chunk.filename, "synthetic": True},
            )
            for pair in _parse_pairs(raw)
        ]

    batches = await asyncio.gather(*(generate_for(c) for c in selected))
    samples = [s for batch in batches for s in batch][:answerable_target]

    unanswerable_target = n_questions - len(samples)
    if unanswerable_target > 0 and unanswerable_ratio > 0:
        topics = sorted({c.heading or c.filename for c in usable if (c.heading or c.filename)})[:30]
        try:
            raw = await llm.ask(
                UNANSWERABLE_PROMPT.format(topics="\n".join(f"- {t}" for t in topics), n=unanswerable_target),
                json_mode=True, max_tokens=400, temperature=0.7,
            )
            questions = json.loads(raw).get("questions", [])
            samples.extend(
                EvalSample(question=str(q).strip(), ground_truth="",
                           metadata={"kind": "unanswerable", "unanswerable": True, "synthetic": True})
                for q in questions[:unanswerable_target] if str(q).strip()
            )
        except Exception as exc:
            log.warning("Unanswerable question generation failed: %s", exc)

    rng.shuffle(samples)
    return Dataset(samples, "synthetic")


async def generate_from_pipeline(pipeline: Any, **kwargs: Any) -> Dataset:
    """Convenience: pull the indexed chunks straight out of a built pipeline."""
    chunks = await pipeline.vectorstore.all_chunks(pipeline.config.namespace)
    if not chunks:
        raise ValueError("The pipeline has no indexed chunks — ingest documents first.")
    if pipeline.llm.is_mock:
        raise ValueError(
            "Synthetic generation needs a real LLM. Configure one, e.g. "
            "RAG(llm={'type': 'openai', 'model': 'gpt-4o-mini'})."
        )
    return await generate_dataset(chunks, pipeline.llm, **kwargs)
