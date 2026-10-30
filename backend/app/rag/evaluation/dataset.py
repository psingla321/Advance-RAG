"""Golden datasets: load, save, split and synthesise.

Supported on-disk formats — JSONL (preferred), JSON, CSV — with flexible column
names, so an existing QA sheet usually loads without editing.
"""

from __future__ import annotations

import csv
import json
import logging
import random
from pathlib import Path
from typing import Any, Iterator

from app.rag.types import EvalSample

log = logging.getLogger(__name__)

QUESTION_KEYS = ("question", "query", "input", "prompt", "q")
ANSWER_KEYS = ("ground_truth", "answer", "expected", "output", "reference", "a")
ID_KEYS = ("relevant_ids", "doc_ids", "chunk_ids", "sources", "relevant")
CONTEXT_KEYS = ("reference_contexts", "contexts", "context", "evidence")


def _pick(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    text = str(value).strip()
    if text.startswith("["):
        try:
            return [str(v) for v in json.loads(text)]
        except json.JSONDecodeError:
            pass
    return [part.strip() for part in text.split("|") if part.strip()] if "|" in text else ([text] if text else [])


def _row_to_sample(row: dict[str, Any]) -> EvalSample | None:
    question = _pick(row, QUESTION_KEYS)
    if not question:
        return None
    known = set(QUESTION_KEYS) | set(ANSWER_KEYS) | set(ID_KEYS) | set(CONTEXT_KEYS) | {"metadata"}
    metadata = dict(row.get("metadata") or {})
    metadata.update({k: v for k, v in row.items() if k not in known and v not in (None, "")})
    answer = _pick(row, ANSWER_KEYS)
    return EvalSample(
        question=str(question).strip(),
        ground_truth=None if answer is None else str(answer),
        relevant_ids=_as_list(_pick(row, ID_KEYS)),
        reference_contexts=_as_list(_pick(row, CONTEXT_KEYS)),
        metadata=metadata,
    )


class Dataset:
    """A list of :class:`EvalSample` with IO and slicing helpers."""

    def __init__(self, samples: list[EvalSample], name: str = "dataset") -> None:
        self.samples = samples
        self.name = name

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[EvalSample]:
        return iter(self.samples)

    def __getitem__(self, index: int | slice) -> Any:
        return Dataset(self.samples[index], self.name) if isinstance(index, slice) else self.samples[index]

    def __repr__(self) -> str:
        labelled = sum(1 for s in self.samples if s.ground_truth)
        return f"Dataset({self.name!r}, n={len(self.samples)}, with_answers={labelled})"

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, path: str | Path, name: str | None = None) -> "Dataset":
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
        elif suffix == ".json":
            data = json.loads(path.read_text("utf-8"))
            rows = data if isinstance(data, list) else data.get("samples", data.get("data", []))
        elif suffix in {".csv", ".tsv"}:
            delimiter = "\t" if suffix == ".tsv" else ","
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter=delimiter))
        else:
            raise ValueError(f"Unsupported dataset format {suffix!r}. Use .jsonl, .json, .csv or .tsv")
        samples = [s for s in (_row_to_sample(row) for row in rows) if s is not None]
        if not samples:
            raise ValueError(f"No usable rows in {path}. Expected a 'question' column.")
        return cls(samples, name or path.stem)

    @classmethod
    def from_pairs(cls, pairs: list[tuple[str, str]] | dict[str, str], name: str = "inline") -> "Dataset":
        items = pairs.items() if isinstance(pairs, dict) else pairs
        return cls([EvalSample(question=q, ground_truth=a) for q, a in items], name)

    @classmethod
    def from_questions(cls, questions: list[str], name: str = "unlabelled") -> "Dataset":
        """Reference-free evaluation: faithfulness and relevancy need no labels."""
        return cls([EvalSample(question=q) for q in questions], name)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {"question": s.question, "ground_truth": s.ground_truth, "relevant_ids": s.relevant_ids,
             "reference_contexts": s.reference_contexts, "metadata": s.metadata}
            for s in self.samples
        ]
        if path.suffix.lower() == ".jsonl":
            path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), "utf-8")
        elif path.suffix.lower() == ".csv":
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()} for r in rows)
        else:
            path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), "utf-8")
        return path

    # -------------------------------------------------------------- slicing
    def sample(self, n: int, seed: int = 0) -> "Dataset":
        rng = random.Random(seed)
        return Dataset(rng.sample(self.samples, min(n, len(self.samples))), f"{self.name}[{n}]")

    def split(self, ratio: float = 0.5, seed: int = 0) -> tuple["Dataset", "Dataset"]:
        """Split into dev/test so you do not tune and report on the same rows."""
        shuffled = list(self.samples)
        random.Random(seed).shuffle(shuffled)
        cut = int(len(shuffled) * ratio)
        return Dataset(shuffled[:cut], f"{self.name}-dev"), Dataset(shuffled[cut:], f"{self.name}-test")

    def filter(self, **criteria: Any) -> "Dataset":
        """Slice by metadata, e.g. ``dataset.filter(difficulty="hard")``."""
        return Dataset(
            [s for s in self.samples if all(s.metadata.get(k) == v for k, v in criteria.items())],
            f"{self.name}-filtered",
        )

    def to_pandas(self):  # type: ignore[no-untyped-def]
        import pandas as pd

        return pd.DataFrame(
            [{"question": s.question, "ground_truth": s.ground_truth, "relevant_ids": s.relevant_ids,
              **s.metadata} for s in self.samples]
        )
