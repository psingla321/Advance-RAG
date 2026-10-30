"""``nexus-rag`` — command line interface to the RAG lab.

    nexus-rag components                      list everything pluggable
    nexus-rag presets                         list built-in recipes
    nexus-rag ingest ./docs -c hybrid         build an index
    nexus-rag ask "what is the limit?"        query it
    nexus-rag testset -n 50 -o golden.jsonl   synthesise an evaluation set
    nexus-rag eval golden.jsonl -c hybrid     score a configuration
    nexus-rag sweep golden.jsonl --grid retriever.type=dense,hybrid_rrf,mmr
    nexus-rag compare golden.jsonl baseline hybrid advanced

Built on argparse so it needs no extra dependency.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from app.rag.client import run_sync
from app.rag.config import RAGConfig, available_presets
from app.rag.evaluation.dataset import Dataset
from app.rag.evaluation.runner import compare as compare_reports
from app.rag.evaluation.runner import evaluate, evaluate_retrieval
from app.rag.pipeline import build_pipeline
from app.rag.registry import list_components

DEFAULT_STORE = ".nexus_cache/index.json"


def _config(args: argparse.Namespace) -> RAGConfig:
    """Resolve --config plus --set overrides, and persist the index by default.

    Without a persistent store the CLI would forget everything between commands,
    so ``memory`` stores are given a path unless one is already configured.
    """
    source = args.config or "offline"
    path = Path(source)
    config = RAGConfig.from_yaml(path) if path.exists() else RAGConfig.preset(source)

    overrides: dict[str, Any] = {}
    for item in getattr(args, "set", None) or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, _, raw = item.partition("=")
        overrides[key.strip()] = _coerce(raw.strip())
    if overrides:
        config = config.with_overrides(**overrides)

    store = config.vectorstore if isinstance(config.vectorstore, dict) else {"type": config.vectorstore}
    if store.get("type") == "memory" and not store.get("persist_path"):
        config = config.with_overrides(**{"vectorstore.persist_path": getattr(args, "store", None) or DEFAULT_STORE})
    return config


def _coerce(raw: str) -> Any:
    for caster in (int, float):
        try:
            return caster(raw)
        except ValueError:
            pass
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    if raw.startswith(("{", "[")):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return raw


# ------------------------------------------------------------------ commands
def cmd_components(args: argparse.Namespace) -> int:
    catalogue = list_components(args.kind)
    if args.json:
        print(json.dumps(catalogue, indent=2))
        return 0
    for kind, items in catalogue.items():
        if not items:
            continue
        print(f"\n{kind.upper()}")
        for item in items:
            requires = f"  [pip install {' '.join(item['requires'])}]" if item["requires"] else ""
            print(f"  {item['name']:<24} {item['summary'][:84]}{requires}")
            if args.verbose and item["params"]:
                for param in item["params"]:
                    default = "required" if param["required"] else f"default={param['default']!r}"
                    print(f"      - {param['name']}: {param['type']} ({default})")
    return 0


def cmd_presets(args: argparse.Namespace) -> int:
    for name in available_presets():
        try:
            print(f"  {name:<14} {RAGConfig.preset(name).description}")
        except Exception as exc:
            print(f"  {name:<14} (could not load: {exc})")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    async def run() -> int:
        pipeline = await build_pipeline(_config(args))
        try:
            source = Path(args.path)
            report = (
                await pipeline.add_directory(source, show_progress=True) if source.is_dir()
                else await pipeline.add_files([source], show_progress=True)
            )
            print(json.dumps({k: v for k, v in report.items() if k != "errors"}, indent=2))
            for error in report.get("errors", []):
                print(f"  ! {error['file']}: {error['error']}", file=sys.stderr)
            print(f"Index now holds {await pipeline.count()} chunks.")
            return 0
        finally:
            await pipeline.close()

    return run_sync(run())


def cmd_ask(args: argparse.Namespace) -> int:
    async def run() -> int:
        pipeline = await build_pipeline(_config(args))
        try:
            if await pipeline.count() == 0:
                print("The index is empty — run `nexus-rag ingest <path>` first.", file=sys.stderr)
                return 1
            result = await pipeline.query(args.question)
            if args.json:
                print(json.dumps(result.to_dict(), indent=2, default=str))
                return 0
            print(f"\n{result.answer}\n")
            for citation in result.citations:
                heading = f" — {citation.heading}" if citation.heading else ""
                print(f"  [{citation.n}] {citation.filename}{heading}  ({citation.score})")
            if args.explain:
                print("\nStages:")
                for stage in result.trace:
                    extras = " ".join(f"{k}={v}" for k, v in stage.items() if k not in {"stage", "ms"})
                    print(f"  {stage['stage']:<12} {stage['ms']:>7.1f} ms  {extras}")
            grounded = f", groundedness {result.groundedness}" if result.groundedness is not None else ""
            print(f"\n{result.latency_ms:.0f} ms{grounded}")
            return 0
        finally:
            await pipeline.close()

    return run_sync(run())


def cmd_testset(args: argparse.Namespace) -> int:
    from app.rag.evaluation.synth import generate_from_pipeline

    async def run() -> int:
        pipeline = await build_pipeline(_config(args))
        try:
            dataset = await generate_from_pipeline(
                pipeline, n_questions=args.n, unanswerable_ratio=args.unanswerable
            )
            path = dataset.save(args.output)
            print(f"Wrote {len(dataset)} samples to {path}")
            print("Review them before using the set as ground truth.")
            return 0
        finally:
            await pipeline.close()

    return run_sync(run())


def cmd_eval(args: argparse.Namespace) -> int:
    async def run() -> int:
        pipeline = await build_pipeline(_config(args))
        try:
            if args.corpus:
                await pipeline.add_directory(args.corpus)
            if args.retrieval_only:
                print(json.dumps(await evaluate_retrieval(pipeline, args.dataset), indent=2))
                return 0
            report = await evaluate(
                pipeline, args.dataset, args.metrics.split(",") if args.metrics else None,
                concurrency=args.concurrency, keep_contexts=True,
            )
            print("\n" + report.table())
            if args.worst:
                print(f"\nWorst {args.worst} samples:")
                for sample in report.worst(args.worst):
                    print(f"\n  Q: {sample.question}")
                    print(f"  A: {sample.answer[:200]}")
                    if sample.ground_truth:
                        print(f"  expected: {sample.ground_truth[:200]}")
                    print(f"  scores: {sample.scores}")
            if args.output:
                print(f"\nSaved to {report.save(args.output)}")
            return 0
        finally:
            await pipeline.close()

    return run_sync(run())


def cmd_sweep(args: argparse.Namespace) -> int:
    from app.rag.experiments.sweep import Sweep

    axes: dict[str, list[Any]] = {}
    for item in args.grid or []:
        key, _, values = item.partition("=")
        axes[key.strip()] = [_coerce(v.strip()) for v in values.split(",") if v.strip()]
    if not axes and not args.presets:
        raise SystemExit("Give --grid key=a,b,c or --presets name1,name2")

    sweep = Sweep(args.config or "offline", Dataset.load(args.dataset), corpus=args.corpus,
                  metrics=args.metrics.split(",") if args.metrics else None, output_dir=args.output_dir)
    if args.presets:
        sweep.presets(args.presets.split(","))
    if axes:
        sweep.ablate(**axes) if args.ablate else sweep.grid(**axes)
    results = sweep.run(concurrency=args.concurrency, query_concurrency=args.query_concurrency)
    print("\n" + results.leaderboard())
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from app.rag.experiments.sweep import Sweep

    sweep = Sweep(dataset=Dataset.load(args.dataset), corpus=args.corpus,
                  metrics=args.metrics.split(",") if args.metrics else None)
    for name in args.configs:
        sweep.add(name)
    results = sweep.run(save=False)
    print("\n" + compare_reports(results.reports, baseline=args.configs[0]))
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    config = _config(args)
    text = config.to_yaml(args.output)
    print(text if not args.output else f"Wrote {args.output}")
    return 0


# -------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nexus-rag", description="Plug-and-play RAG lab.")
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    sub = parser.add_subparsers(dest="command")

    def shared(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument("-c", "--config", help="preset name or path to a YAML config (default: offline)")
        p.add_argument("--set", action="append", metavar="KEY=VALUE",
                       help="override a config field, e.g. --set retriever.type=mmr (repeatable)")
        p.add_argument("--store", help=f"index file for in-memory stores (default: {DEFAULT_STORE})")
        return p

    p = shared(sub.add_parser("components", help="list every pluggable component"))
    p.add_argument("kind", nargs="?", help="restrict to one kind, e.g. retriever")
    p.add_argument("-v", "--verbose", action="store_true", help="also show constructor parameters")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_components)

    p = sub.add_parser("presets", help="list built-in configuration recipes")
    p.set_defaults(func=cmd_presets)

    p = shared(sub.add_parser("ingest", help="index a file or directory"))
    p.add_argument("path")
    p.set_defaults(func=cmd_ingest)

    p = shared(sub.add_parser("ask", help="query the index"))
    p.add_argument("question")
    p.add_argument("--explain", action="store_true", help="show per-stage timings")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_ask)

    p = shared(sub.add_parser("testset", help="synthesise a golden dataset from the index"))
    p.add_argument("-n", type=int, default=50, help="number of questions")
    p.add_argument("-o", "--output", default="golden.jsonl")
    p.add_argument("--unanswerable", type=float, default=0.1, help="share of should-be-refused questions")
    p.set_defaults(func=cmd_testset)

    p = shared(sub.add_parser("eval", help="score a configuration on a dataset"))
    p.add_argument("dataset")
    p.add_argument("--corpus", help="ingest this directory before evaluating")
    p.add_argument("--metrics", help="comma-separated metric names")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--retrieval-only", action="store_true", help="retrieval metrics only; no generation")
    p.add_argument("--worst", type=int, default=0, help="print the N worst samples")
    p.add_argument("-o", "--output", help="write the full report as JSON")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("sweep", help="run many configurations and rank them")
    p.add_argument("dataset")
    p.add_argument("-c", "--config", help="base preset or config path")
    p.add_argument("--corpus", help="directory to ingest for every run")
    p.add_argument("--grid", action="append", metavar="KEY=A,B,C", help="axis to vary (repeatable)")
    p.add_argument("--ablate", action="store_true", help="one change at a time instead of the full product")
    p.add_argument("--presets", help="comma-separated preset names to include")
    p.add_argument("--metrics")
    p.add_argument("--concurrency", type=int, default=1, help="pipelines in parallel")
    p.add_argument("--query-concurrency", type=int, default=4)
    p.add_argument("--output-dir", default="experiments")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("compare", help="A/B several presets or config files")
    p.add_argument("dataset")
    p.add_argument("configs", nargs="+")
    p.add_argument("--corpus")
    p.add_argument("--metrics")
    p.set_defaults(func=cmd_compare)

    p = shared(sub.add_parser("config", help="print or write the resolved configuration"))
    p.add_argument("-o", "--output")
    p.set_defaults(func=cmd_config)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "version", False):
        from app.rag import __version__

        print(__version__)
        return 0
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
