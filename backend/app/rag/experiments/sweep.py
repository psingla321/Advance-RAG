"""Experiment sweeps: run many configurations over one dataset and rank them.

The point of the whole plug-and-play layer. Instead of arguing about whether
HyDE helps on your corpus, run it::

    sweep = Sweep(base_config, dataset, corpus="./docs")
    sweep.grid(**{"retriever.type": ["dense", "hybrid_rrf", "mmr"],
                  "chunker.max_tokens": [200, 350, 500]})
    results = sweep.run()
    print(results.leaderboard())
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from app.rag.config import RAGConfig
from app.rag.evaluation.dataset import Dataset
from app.rag.evaluation.runner import EvalReport, compare, evaluate
from app.rag.pipeline import RAGPipeline
from app.rag.types import Document

log = logging.getLogger(__name__)


@dataclass
class SweepResults:
    """Every run in a sweep, ranked."""

    reports: list[EvalReport] = field(default_factory=list)
    failures: list[dict[str, str]] = field(default_factory=list)
    duration_seconds: float = 0.0

    def best(self, metric: str | None = None) -> EvalReport | None:
        if not self.reports:
            return None
        key = (lambda r: r.scores.get(metric, 0.0)) if metric else (lambda r: r.overall)
        return max(self.reports, key=key)

    def leaderboard(self, metric: str | None = None, top: int = 20) -> str:
        key = (lambda r: r.scores.get(metric, 0.0)) if metric else (lambda r: r.overall)
        ranked = sorted(self.reports, key=key, reverse=True)[:top]
        if not ranked:
            return "(no successful runs)" + (f" — {len(self.failures)} failed" if self.failures else "")
        lines = [compare(ranked, baseline=ranked[0].config_name)]
        if self.failures:
            lines.append(f"\n{len(self.failures)} configuration(s) failed:")
            lines.extend(f"  {f['name']}: {f['error'][:120]}" for f in self.failures[:5])
        return "\n".join(lines)

    def to_pandas(self):  # type: ignore[no-untyped-def]
        import pandas as pd

        return pd.DataFrame(
            [{"config": r.config_name, "overall": r.overall, "p95_ms": r.latency.get("p95", 0),
              "n": r.n, **r.scores} for r in self.reports]
        ).sort_values("overall", ascending=False)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"duration_seconds": self.duration_seconds, "failures": self.failures,
                 "reports": [r.to_dict() for r in self.reports]},
                indent=2, ensure_ascii=False, default=str,
            ),
            "utf-8",
        )
        return path

    def __repr__(self) -> str:
        best = self.best()
        return f"SweepResults(runs={len(self.reports)}, failed={len(self.failures)}, best={best.config_name if best else None!r})"


class Sweep:
    """Builds and runs a set of configurations against one corpus and dataset."""

    def __init__(
        self,
        base: RAGConfig | dict[str, Any] | str | None = None,
        dataset: Dataset | str | Path | list[str] | None = None,
        corpus: str | Path | list[Document] | list[str] | None = None,
        metrics: list[str] | None = None,
        output_dir: str | Path = "experiments",
    ) -> None:
        self.base = self._resolve_config(base)
        self.dataset = self._resolve_dataset(dataset)
        self.corpus = corpus
        self.metrics = metrics
        self.output_dir = Path(output_dir)
        self.configs: list[RAGConfig] = []

    @staticmethod
    def _resolve_config(base: Any) -> RAGConfig:
        if base is None:
            return RAGConfig()
        if isinstance(base, RAGConfig):
            return base
        if isinstance(base, dict):
            return RAGConfig.from_dict(base)
        return RAGConfig.preset(str(base))

    @staticmethod
    def _resolve_dataset(dataset: Any) -> Dataset | None:
        if dataset is None:
            return None
        if isinstance(dataset, Dataset):
            return dataset
        if isinstance(dataset, list):
            return Dataset.from_questions(dataset)
        return Dataset.load(dataset)

    # ---------------------------------------------------------------- config
    def add(self, config: RAGConfig | dict[str, Any] | str, name: str | None = None) -> "Sweep":
        """Add one explicit configuration to the sweep."""
        resolved = self._resolve_config(config)
        if name:
            resolved = resolved.model_copy(update={"name": name})
        self.configs.append(resolved)
        return self

    def grid(self, **axes: Iterable[Any]) -> "Sweep":
        """Cartesian product over dotted config paths.

        ``grid(**{"retriever.type": ["dense", "hybrid_rrf"], "top_n": [5, 10]})``
        produces four configurations named after the values that vary.
        """
        keys = list(axes)
        for combination in itertools.product(*(list(axes[k]) for k in keys)):
            overrides = dict(zip(keys, combination))
            label = "+".join(
                f"{k.split('.')[-2] if k.endswith('.type') else k.split('.')[-1]}={v}"
                for k, v in overrides.items()
            )
            self.configs.append(self.base.with_overrides(**overrides).model_copy(update={"name": label}))
        return self

    def presets(self, names: list[str]) -> "Sweep":
        """Compare the built-in recipes head to head."""
        for name in names:
            self.configs.append(RAGConfig.preset(name))
        return self

    def ablate(self, **axes: Iterable[Any]) -> "Sweep":
        """One-factor-at-a-time: baseline plus one change per run.

        Linear in the number of options instead of exponential — the right
        default when you want to know which knob matters, not the global optimum.
        """
        self.configs.append(self.base.model_copy(update={"name": "baseline"}))
        for key, values in axes.items():
            for value in values:
                label = f"{key}={value}"
                self.configs.append(self.base.with_overrides(**{key: value}).model_copy(update={"name": label}))
        return self

    # ------------------------------------------------------------------ run
    async def arun(
        self, concurrency: int = 1, query_concurrency: int = 4, save: bool = True, show_progress: bool = True
    ) -> SweepResults:
        """Run every configuration. Pipelines run sequentially by default so local
        models and rate-limited APIs are not thrashed; raise ``concurrency`` for
        cheap offline components."""
        if self.dataset is None:
            raise ValueError("Sweep needs a dataset. Pass one, or generate it with the synth module.")
        if not self.configs:
            raise ValueError("Sweep has no configurations. Call .grid(), .ablate(), .presets() or .add().")

        started = time.perf_counter()
        results = SweepResults()
        semaphore = asyncio.Semaphore(concurrency)
        names: dict[str, int] = {}

        async def run_one(index: int, config: RAGConfig) -> None:
            # Disambiguate configurations that produced the same label.
            count = names.get(config.name, 0)
            names[config.name] = count + 1
            if count:
                config = config.model_copy(update={"name": f"{config.name}#{count + 1}"})
            async with semaphore:
                if show_progress:
                    print(f"\n[{index}/{len(self.configs)}] {config.name}", flush=True)
                pipeline = None
                try:
                    pipeline = await RAGPipeline(config).init()
                    await self._ingest(pipeline, show_progress)
                    report = await evaluate(
                        pipeline, self.dataset, self.metrics,
                        concurrency=query_concurrency, show_progress=False,
                    )
                    results.reports.append(report)
                    if show_progress:
                        print(f"    overall={report.overall:.3f}  p95={report.latency.get('p95', 0):.0f}ms", flush=True)
                except Exception as exc:
                    log.exception("Configuration %s failed", config.name)
                    results.failures.append({"name": config.name, "error": str(exc)})
                finally:
                    if pipeline is not None:
                        await pipeline.close()

        await asyncio.gather(*(run_one(i, c) for i, c in enumerate(self.configs, 1)))
        results.duration_seconds = round(time.perf_counter() - started, 2)
        if save:
            path = self.output_dir / f"sweep-{time.strftime('%Y%m%d-%H%M%S')}.json"
            results.save(path)
            if show_progress:
                print(f"\nSaved {len(results.reports)} runs to {path}", flush=True)
        return results

    def run(self, **kwargs: Any) -> SweepResults:
        """Synchronous entry point — safe to call from a notebook."""
        from app.rag.client import run_sync

        return run_sync(self.arun(**kwargs))

    async def _ingest(self, pipeline: RAGPipeline, show_progress: bool) -> None:
        if self.corpus is None:
            return
        if isinstance(self.corpus, (str, Path)):
            await pipeline.add_directory(self.corpus, show_progress=False)
        elif self.corpus and isinstance(self.corpus[0], Document):
            await pipeline.add_documents(list(self.corpus))  # type: ignore[arg-type]
        else:
            await pipeline.add_texts([str(t) for t in self.corpus])  # type: ignore[union-attr]
        if show_progress:
            print(f"    indexed {await pipeline.count()} chunks", flush=True)
