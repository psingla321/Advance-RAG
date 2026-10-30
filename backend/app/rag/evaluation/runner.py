"""Run a dataset through a pipeline and score it.

Produces an :class:`EvalReport` with aggregate scores, per-sample rows, latency
percentiles and a failure list — the thing you actually look at when a change
makes the system worse.
"""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.rag.base import Metric
from app.rag.evaluation.dataset import Dataset
from app.rag.evaluation.metrics import DEFAULT_METRICS, OFFLINE_METRICS
from app.rag.registry import build
from app.rag.types import EvalSample, RAGResult

log = logging.getLogger(__name__)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(percentile / 100 * (len(ordered) - 1)))))
    return round(ordered[index], 2)


@dataclass
class SampleResult:
    question: str
    answer: str
    ground_truth: str | None
    scores: dict[str, float]
    latency_ms: float
    abstained: bool
    contexts: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def mean_score(self) -> float:
        return round(statistics.fmean(self.scores.values()), 4) if self.scores else 0.0


@dataclass
class EvalReport:
    """Aggregate and per-sample evaluation results."""

    config_name: str
    fingerprint: str
    dataset: str
    n: int
    scores: dict[str, float]
    latency: dict[str, float]
    samples: list[SampleResult]
    pipeline: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    errors: int = 0

    @property
    def overall(self) -> float:
        """Unweighted mean across metrics — a single number for ranking runs."""
        return round(statistics.fmean(self.scores.values()), 4) if self.scores else 0.0

    def worst(self, n: int = 10, metric: str | None = None) -> list[SampleResult]:
        """The samples to actually read: where the system failed hardest."""
        key = (lambda s: s.scores.get(metric, 0.0)) if metric else (lambda s: s.mean_score)
        return sorted(self.samples, key=key)[:n]

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_name": self.config_name, "fingerprint": self.fingerprint, "dataset": self.dataset,
            "n": self.n, "overall": self.overall, "scores": self.scores, "latency": self.latency,
            "errors": self.errors, "duration_seconds": self.duration_seconds, "pipeline": self.pipeline,
            "samples": [s.__dict__ for s in self.samples],
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=str), "utf-8")
        return path

    def to_pandas(self):  # type: ignore[no-untyped-def]
        import pandas as pd

        return pd.DataFrame(
            [{"question": s.question, "answer": s.answer, "latency_ms": s.latency_ms,
              "abstained": s.abstained, **s.scores} for s in self.samples]
        )

    def table(self) -> str:
        """A terminal-friendly summary."""
        width = max((len(k) for k in self.scores), default=10) + 2
        lines = [
            f"{self.config_name}  ({self.fingerprint})  dataset={self.dataset}  n={self.n}",
            "-" * 52,
        ]
        for name, value in sorted(self.scores.items(), key=lambda kv: -kv[1]):
            # ASCII bar: these reports are printed on Windows terminals too.
            bar = "#" * int(round(value * 20))
            lines.append(f"  {name:<{width}} {value:>6.3f}  {bar}")
        lines += [
            "-" * 52,
            f"  {'overall':<{width}} {self.overall:>6.3f}",
            f"  latency p50/p95  {self.latency.get('p50', 0):.0f} / {self.latency.get('p95', 0):.0f} ms",
        ]
        if self.errors:
            lines.append(f"  errors: {self.errors}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"EvalReport({self.config_name!r}, n={self.n}, overall={self.overall})"


async def evaluate(
    pipeline: Any,
    dataset: Dataset | list[str] | str | Path,
    metrics: list[str | dict[str, Any]] | None = None,
    *,
    concurrency: int = 4,
    namespace: str | None = None,
    filters: dict[str, Any] | None = None,
    show_progress: bool = True,
    keep_contexts: bool = False,
) -> EvalReport:
    """Score ``pipeline`` on ``dataset``.

    With no ``metrics``, picks the LLM-judged suite when a real LLM is configured
    and the free offline suite otherwise, so this always runs.
    """
    if isinstance(dataset, (str, Path)):
        dataset = Dataset.load(dataset)
    elif isinstance(dataset, list):
        dataset = Dataset.from_questions(dataset)

    chosen = metrics or (OFFLINE_METRICS if pipeline.llm.is_mock else DEFAULT_METRICS)
    built: list[tuple[str, Metric]] = []
    for spec in chosen:
        name = spec if isinstance(spec, str) else spec.get("type", "")
        try:
            built.append((name, build("metric", spec, llm=pipeline.llm)))
        except Exception as exc:
            log.warning("Skipping metric %s: %s", name, exc)

    from app.rag.evaluation.metrics import has_retrieval_labels

    has_labels = any(s.ground_truth or s.relevant_ids or s.reference_contexts for s in dataset)
    if not has_labels:
        skipped = [n for n, m in built if m.needs_ground_truth]
        if skipped:
            log.warning(
                "Dataset %r has no ground truth; skipping label-based metrics: %s",
                dataset.name, ", ".join(skipped),
            )
        built = [(n, m) for n, m in built if not m.needs_ground_truth]

    started = time.perf_counter()
    semaphore = asyncio.Semaphore(concurrency)
    completed = 0
    total = len(dataset)

    async def run_one(sample: EvalSample) -> SampleResult:
        nonlocal completed
        async with semaphore:
            error = None
            try:
                result = await pipeline.query(sample.question, namespace=namespace, filters=filters)
            except Exception as exc:
                log.exception("Query failed: %s", sample.question)
                error = str(exc)
                result = RAGResult(query=sample.question, answer="")
            scores: dict[str, float] = {}
            for name, metric in built:
                try:
                    scores[name] = round(float(await metric.score(sample, result)), 4)
                except Exception as exc:
                    log.warning("Metric %s failed on %r: %s", name, sample.question[:60], exc)
        completed += 1
        if show_progress and (completed % max(1, total // 10) == 0 or completed == total):
            print(f"  evaluated {completed}/{total}", flush=True)
        return SampleResult(
            question=sample.question, answer=result.answer, ground_truth=sample.ground_truth,
            scores=scores, latency_ms=result.latency_ms, abstained=result.abstained,
            contexts=result.context_texts[:3] if keep_contexts else [],
            error=error or result.metadata.get("error"),
        )

    samples = list(await asyncio.gather(*(run_one(s) for s in dataset)))
    aggregate = {
        name: round(statistics.fmean([s.scores[name] for s in samples if name in s.scores]), 4)
        for name, _ in built
        if any(name in s.scores for s in samples)
    }
    latencies = [s.latency_ms for s in samples if s.latency_ms]
    return EvalReport(
        config_name=pipeline.config.name,
        fingerprint=pipeline.config.fingerprint(),
        dataset=dataset.name,
        n=len(samples),
        scores=aggregate,
        latency={
            "mean": round(statistics.fmean(latencies), 2) if latencies else 0.0,
            "p50": _percentile(latencies, 50), "p95": _percentile(latencies, 95), "p99": _percentile(latencies, 99),
        },
        samples=samples,
        pipeline=pipeline.describe(),
        duration_seconds=round(time.perf_counter() - started, 2),
        errors=sum(1 for s in samples if s.error),
    )


async def evaluate_retrieval(
    pipeline: Any, dataset: Dataset | str | Path, k_values: list[int] | None = None, concurrency: int = 8
) -> dict[str, Any]:
    """Retrieval-only evaluation at several cut-offs — fast, free, no generation.

    This is where to start tuning: if recall@20 is low, no prompt will save you.
    """
    from app.rag.evaluation.metrics import _relevant_hits, _total_relevant, has_retrieval_labels

    if isinstance(dataset, (str, Path)):
        dataset = Dataset.load(dataset)
    if not any(has_retrieval_labels(s) for s in dataset):
        raise ValueError(
            f"Dataset {dataset.name!r} has no retrieval labels, so every score would be 0. "
            "Add 'relevant_ids', 'reference_contexts' or 'ground_truth' to the samples "
            "(synth.generate_dataset fills these in automatically)."
        )
    k_values = sorted(k_values or [1, 3, 5, 10, 20])
    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(sample: EvalSample) -> list[bool]:
        async with semaphore:
            contexts, _, _ = await pipeline.retrieve(sample.question, top_n=max(k_values))
        return _relevant_hits(sample, RAGResult(query=sample.question, contexts=contexts))

    all_hits = await asyncio.gather(*(run_one(s) for s in dataset))
    report: dict[str, Any] = {"n": len(dataset), "config": pipeline.config.name}
    for k in k_values:
        hits_at_k = [h[:k] for h in all_hits]
        report[f"hit_rate@{k}"] = round(statistics.fmean([1.0 if any(h) else 0.0 for h in hits_at_k]), 4)
        report[f"precision@{k}"] = round(statistics.fmean([sum(h) / k for h in hits_at_k]), 4)
        report[f"recall@{k}"] = round(
            statistics.fmean([min(1.0, sum(h) / _total_relevant(s)) for h, s in zip(hits_at_k, dataset)]), 4
        )
    report["mrr"] = round(
        statistics.fmean([next((1 / r for r, hit in enumerate(h, 1) if hit), 0.0) for h in all_hits]), 4
    )
    return report


def compare(reports: list[EvalReport], baseline: str | None = None) -> str:
    """Side-by-side table of several runs, with deltas against a baseline."""
    if not reports:
        return "(no reports)"
    metrics = sorted({m for r in reports for m in r.scores})
    base = next((r for r in reports if r.config_name == baseline), reports[0])
    name_width = max(len(r.config_name) for r in reports) + 2
    header = f"{'config':<{name_width}}" + "".join(f"{m[:12]:>14}" for m in metrics) + f"{'overall':>10}{'p95 ms':>10}"
    lines = [header, "-" * len(header)]
    for report in sorted(reports, key=lambda r: -r.overall):
        row = f"{report.config_name:<{name_width}}"
        for metric in metrics:
            value = report.scores.get(metric)
            if value is None:
                row += f"{'-':>14}"
                continue
            delta = value - base.scores.get(metric, value)
            marker = "" if report is base or abs(delta) < 0.001 else f" {'+' if delta > 0 else ''}{delta:.2f}"
            row += f"{value:>8.3f}{marker:>6}"
        row += f"{report.overall:>10.3f}{report.latency.get('p95', 0):>10.0f}"
        lines.append(row)
    return "\n".join(lines)
