"""Chunking strategies — the single highest-leverage knob in a RAG pipeline.

All of them honour section boundaries where the loader provided them, so a chunk
never silently merges two unrelated policies.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.rag.base import Chunker
from app.rag.registry import build, register
from app.rag.types import Chunk, Document, Section
from app.textutil import estimate_tokens, split_sentences

log = logging.getLogger(__name__)


def _chunk_id(doc_id: str, ordinal: int) -> str:
    return f"{doc_id}::{ordinal}"


def _make(document: Document, section: Section, ordinal: int, text: str, **metadata: Any) -> Chunk:
    return Chunk(
        id=_chunk_id(document.id, ordinal),
        text=text.strip(),
        doc_id=document.id,
        filename=document.filename,
        heading=section.heading,
        ord=ordinal,
        metadata={**document.metadata, **section.metadata, **metadata},
    )


def _split_oversized(sentence: str, max_tokens: int) -> list[str]:
    words = sentence.split()
    step = max(1, int(max_tokens / 1.3))
    return [" ".join(words[i : i + step]) for i in range(0, len(words), step)]


@register("chunker", "fixed")
class FixedChunker(Chunker):
    """Fixed-size character windows with overlap. The simplest baseline."""

    def __init__(self, size: int = 1000, overlap: int = 150) -> None:
        self._size = size
        self._overlap = min(overlap, size - 1)

    async def chunk(self, document: Document, sections: list[Section]) -> list[Chunk]:
        chunks: list[Chunk] = []
        step = self._size - self._overlap
        for section in sections:
            text = section.text
            for start in range(0, max(len(text), 1), step):
                window = text[start : start + self._size]
                if window.strip():
                    chunks.append(_make(document, section, len(chunks), window))
                if start + self._size >= len(text):
                    break
        return chunks


@register("chunker", "recursive")
class RecursiveChunker(Chunker):
    """Recursive character splitting on paragraph -> line -> sentence -> word.

    The familiar LangChain-style default; keeps natural boundaries where it can.
    """

    def __init__(
        self, max_tokens: int = 350, overlap_tokens: int = 50,
        separators: list[str] | None = None,
    ) -> None:
        self._max = max_tokens
        self._overlap = overlap_tokens
        self._separators = separators or ["\n\n", "\n", ". ", " "]

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if estimate_tokens(text) <= self._max or not separators:
            return [text]
        separator, rest = separators[0], separators[1:]
        parts = [p for p in text.split(separator) if p.strip()]
        out: list[str] = []
        for part in parts:
            out.extend(self._split(part + separator, rest) if estimate_tokens(part) > self._max else [part + separator])
        return out

    async def chunk(self, document: Document, sections: list[Section]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section in sections:
            buffer: list[str] = []
            tokens = 0
            for part in self._split(section.text, self._separators):
                part_tokens = estimate_tokens(part)
                if buffer and tokens + part_tokens > self._max:
                    chunks.append(_make(document, section, len(chunks), "".join(buffer)))
                    carry, carried = [], 0
                    for piece in reversed(buffer):
                        if carried >= self._overlap:
                            break
                        carry.insert(0, piece)
                        carried += estimate_tokens(piece)
                    buffer, tokens = carry, carried
                buffer.append(part)
                tokens += part_tokens
            if any(p.strip() for p in buffer):
                chunks.append(_make(document, section, len(chunks), "".join(buffer)))
        return chunks


@register("chunker", "sentence")
class SentenceChunker(Chunker):
    """Sentence-aligned chunks with sentence overlap — never cuts mid-sentence.

    This is the strategy the production AdvanceRAG ingestion pipeline uses.
    """

    def __init__(self, max_tokens: int = 350, overlap_sentences: int = 1) -> None:
        self._max = max_tokens
        self._overlap = overlap_sentences

    async def chunk(self, document: Document, sections: list[Section]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section in sections:
            sentences: list[str] = []
            for sentence in split_sentences(section.text):
                sentences.extend(
                    _split_oversized(sentence, self._max) if estimate_tokens(sentence) > self._max else [sentence]
                )
            buffer: list[str] = []
            tokens = 0
            for sentence in sentences:
                size = estimate_tokens(sentence)
                if buffer and tokens + size > self._max:
                    chunks.append(_make(document, section, len(chunks), " ".join(buffer)))
                    buffer = buffer[-self._overlap :] if self._overlap else []
                    tokens = sum(estimate_tokens(s) for s in buffer)
                buffer.append(sentence)
                tokens += size
            if buffer:
                chunks.append(_make(document, section, len(chunks), " ".join(buffer)))
        return chunks


@register("chunker", "semantic")
class SemanticChunker(Chunker):
    """Embedding-based splitting: a new chunk starts where meaning shifts.

    Embeds every sentence, then cuts where cosine similarity between consecutive
    sentences drops below the ``percentile`` breakpoint of all gaps. Costs one
    embedding call per sentence at index time but produces very coherent chunks.
    """

    def __init__(self, embedder: Any = None, percentile: float = 20.0, max_tokens: int = 500, min_sentences: int = 2) -> None:
        self._embedder = embedder
        self._percentile = percentile
        self._max = max_tokens
        self._min = min_sentences

    async def chunk(self, document: Document, sections: list[Section]) -> list[Chunk]:
        if self._embedder is None:
            log.warning("semantic chunker has no embedder; falling back to sentence chunking")
            return await SentenceChunker(self._max).chunk(document, sections)

        from app.rag.components.embedders import cosine

        chunks: list[Chunk] = []
        for section in sections:
            sentences = split_sentences(section.text)
            if len(sentences) <= self._min:
                if section.text.strip():
                    chunks.append(_make(document, section, len(chunks), section.text))
                continue
            vectors = await self._embedder.embed(sentences)
            gaps = [1.0 - cosine(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)]
            threshold = _percentile(gaps, self._percentile_complement())
            buffer: list[str] = [sentences[0]]
            tokens = estimate_tokens(sentences[0])
            for i, sentence in enumerate(sentences[1:]):
                size = estimate_tokens(sentence)
                breakpoint_hit = gaps[i] >= threshold and len(buffer) >= self._min
                if buffer and (breakpoint_hit or tokens + size > self._max):
                    chunks.append(_make(document, section, len(chunks), " ".join(buffer), split="semantic"))
                    buffer, tokens = [], 0
                buffer.append(sentence)
                tokens += size
            if buffer:
                chunks.append(_make(document, section, len(chunks), " ".join(buffer), split="semantic"))
        return chunks

    def _percentile_complement(self) -> float:
        # A "20th percentile breakpoint" means we cut at the top 20% largest gaps.
        return 100.0 - self._percentile


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 1.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((percentile / 100.0) * (len(ordered) - 1)))))
    return ordered[index]


@register("chunker", "markdown_header")
class MarkdownHeaderChunker(Chunker):
    """One chunk per markdown section, split further only when over budget.

    Preserves the heading breadcrumb, which makes citations readable.
    """

    def __init__(self, max_tokens: int = 500, overlap_sentences: int = 1) -> None:
        self._inner = SentenceChunker(max_tokens, overlap_sentences)
        self._max = max_tokens

    async def chunk(self, document: Document, sections: list[Section]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section in sections:
            if estimate_tokens(section.text) <= self._max:
                if section.text.strip():
                    chunks.append(_make(document, section, len(chunks), section.text))
                continue
            for sub in await self._inner.chunk(document, [section]):
                sub.id = _chunk_id(document.id, len(chunks))
                sub.ord = len(chunks)
                chunks.append(sub)
        return chunks


@register("chunker", "sentence_window")
class SentenceWindowChunker(Chunker):
    """Index single sentences, but feed the LLM a window of neighbours.

    Precise retrieval (a one-sentence embedding is unambiguous) with enough
    surrounding context to answer. ``parent_text`` carries the window.
    """

    def __init__(self, window: int = 3) -> None:
        self._window = window

    async def chunk(self, document: Document, sections: list[Section]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section in sections:
            sentences = split_sentences(section.text)
            for i, sentence in enumerate(sentences):
                lo = max(0, i - self._window)
                hi = min(len(sentences), i + self._window + 1)
                chunk = _make(document, section, len(chunks), sentence, window=self._window)
                chunk.parent_text = " ".join(sentences[lo:hi])
                chunks.append(chunk)
        return chunks


@register("chunker", "parent_child")
class ParentChildChunker(Chunker):
    """Small children for matching, large parents for context (auto-merging retrieval).

    Children are embedded and searched; ``parent_text`` holds the parent block
    that is actually shown to the model.
    """

    def __init__(self, parent_tokens: int = 1200, child_tokens: int = 200, overlap_sentences: int = 0) -> None:
        self._parent = SentenceChunker(parent_tokens, overlap_sentences)
        self._child = SentenceChunker(child_tokens, overlap_sentences)

    async def chunk(self, document: Document, sections: list[Section]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for parent in await self._parent.chunk(document, sections):
            section = Section(parent.heading, parent.text, parent.metadata)
            for child in await self._child.chunk(document, [section]):
                child.id = _chunk_id(document.id, len(chunks))
                child.ord = len(chunks)
                child.parent_text = parent.text
                child.metadata = {**child.metadata, "parent_id": parent.id}
                chunks.append(child)
        return chunks


@register("chunker", "token", requires=["tiktoken"])
class TokenChunker(Chunker):
    """Exact token-count windows using the model's own tokenizer.

    Use when you must guarantee a context budget rather than estimate it.
    """

    def __init__(self, max_tokens: int = 400, overlap_tokens: int = 50, encoding: str = "cl100k_base") -> None:
        import tiktoken

        self._encoding = tiktoken.get_encoding(encoding)
        self._max = max_tokens
        self._overlap = min(overlap_tokens, max_tokens - 1)

    async def chunk(self, document: Document, sections: list[Section]) -> list[Chunk]:
        chunks: list[Chunk] = []
        step = self._max - self._overlap
        for section in sections:
            tokens = self._encoding.encode(section.text)
            for start in range(0, max(len(tokens), 1), step):
                window = self._encoding.decode(tokens[start : start + self._max])
                if window.strip():
                    chunks.append(_make(document, section, len(chunks), window))
                if start + self._max >= len(tokens):
                    break
        return chunks


@register("chunker", "contextual")
class ContextualChunker(Chunker):
    """Anthropic-style contextual retrieval: prepend an LLM-written situating blurb.

    Each chunk is embedded as ``<context sentence> + <chunk>``, which markedly
    improves recall on documents full of pronouns and implicit references.
    Costs one cheap LLM call per chunk at index time — use a cached LLM.
    """

    PROMPT = (
        "Here is a document:\n<document>\n{document}\n</document>\n\n"
        "Here is a chunk from it:\n<chunk>\n{chunk}\n</chunk>\n\n"
        "Give a short standalone sentence situating this chunk within the document "
        "(what it is about, which section, which entity). Answer with the sentence only."
    )

    def __init__(
        self, llm: Any = None, inner: dict[str, Any] | str | None = None,
        max_document_chars: int = 8000, concurrency: int = 8,
    ) -> None:
        self._llm = llm
        self._inner: Chunker = build("chunker", inner or {"type": "sentence"})
        self._max_doc = max_document_chars
        self._concurrency = concurrency

    async def chunk(self, document: Document, sections: list[Section]) -> list[Chunk]:
        import asyncio

        chunks = await self._inner.chunk(document, sections)
        if self._llm is None or not chunks:
            return chunks
        excerpt = document.text[: self._max_doc]
        semaphore = asyncio.Semaphore(self._concurrency)

        async def annotate(chunk: Chunk) -> None:
            async with semaphore:
                try:
                    blurb = await self._llm.ask(self.PROMPT.format(document=excerpt, chunk=chunk.text), max_tokens=80)
                except Exception as exc:
                    log.warning("Contextual chunking failed for %s: %s", chunk.id, exc)
                    return
                blurb = blurb.strip().replace("\n", " ")
                if blurb:
                    chunk.metadata["context"] = blurb
                    chunk.parent_text = chunk.parent_text or chunk.text
                    chunk.text = f"{blurb} {chunk.text}"

        await asyncio.gather(*(annotate(c) for c in chunks))
        return chunks


@register("chunker", "regex")
class RegexChunker(Chunker):
    """Split on a custom pattern — contract clauses, log records, Q&A pairs."""

    def __init__(self, pattern: str = r"\n(?=\d+\.\s)", keep_separator: bool = True) -> None:
        self._pattern = re.compile(pattern)
        self._keep = keep_separator

    async def chunk(self, document: Document, sections: list[Section]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section in sections:
            for part in self._pattern.split(section.text):
                if part and part.strip():
                    chunks.append(_make(document, section, len(chunks), part))
        return chunks
