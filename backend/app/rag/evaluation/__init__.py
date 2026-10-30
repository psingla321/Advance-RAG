"""Evaluation: metrics, golden datasets, synthetic test sets and the runner."""

from app.rag.evaluation import metrics  # noqa: F401  (registers the metric components)
from app.rag.evaluation.dataset import Dataset
from app.rag.evaluation.runner import EvalReport, SampleResult, compare, evaluate, evaluate_retrieval
from app.rag.evaluation.synth import generate_dataset, generate_from_pipeline

__all__ = [
    "Dataset", "EvalReport", "SampleResult", "compare", "evaluate", "evaluate_retrieval",
    "generate_dataset", "generate_from_pipeline",
]
