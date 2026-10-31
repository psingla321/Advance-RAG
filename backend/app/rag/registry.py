"""A tiny component registry — the heart of the plug-and-play layer.

Every swappable part of the pipeline (embedder, vector store, chunker, retriever,
reranker, query transform, compressor, generator, LLM) registers itself under a
``kind`` and a ``name``. A pipeline is then built from plain dicts / YAML:

    {"retriever": {"type": "hybrid_rrf", "top_k": 30}}

Registering your own component is three lines::

    from app.rag import register, Reranker

    @register("reranker", "my_reranker")
    class MyReranker(Reranker):
        async def rerank(self, query, candidates, top_n): ...
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

KINDS = (
    "llm",
    "embedder",
    "vectorstore",
    "loader",
    "chunker",
    "transform",
    "retriever",
    "reranker",
    "compressor",
    "generator",
    "metric",
)

T = TypeVar("T")

_REGISTRY: dict[str, dict[str, type]] = {kind: {} for kind in KINDS}
_META: dict[tuple[str, str], "ComponentInfo"] = {}


@dataclass
class ComponentInfo:
    kind: str
    name: str
    cls: type
    summary: str
    params: list[dict[str, Any]]
    requires: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "summary": self.summary,
            "params": self.params,
            "requires": self.requires,
            "class": f"{self.cls.__module__}.{self.cls.__qualname__}",
        }


class ComponentError(ValueError):
    """Raised when a component name is unknown or its parameters are invalid."""


def _describe_params(cls: type) -> list[dict[str, Any]]:
    try:
        signature = inspect.signature(cls.__init__)
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return []
    params = []
    for name, param in signature.parameters.items():
        if name in {"self", "args", "kwargs"} or param.kind in {param.VAR_POSITIONAL, param.VAR_KEYWORD}:
            continue
        annotation = param.annotation
        params.append(
            {
                "name": name,
                "type": getattr(annotation, "__name__", str(annotation)) if annotation is not param.empty else "Any",
                "default": None if param.default is param.empty else _safe(param.default),
                "required": param.default is param.empty,
            }
        )
    return params


def _safe(value: Any) -> Any:
    return value if isinstance(value, (str, int, float, bool, type(None), list, dict)) else repr(value)


def register(kind: str, name: str, *, requires: list[str] | None = None) -> Callable[[type[T]], type[T]]:
    """Class decorator that registers a component under ``kind``/``name``.

    ``requires`` lists optional pip packages the component imports lazily; it is
    surfaced in ``list_components()`` so users see what to install.
    """
    if kind not in _REGISTRY:
        raise ComponentError(f"Unknown component kind {kind!r}. Valid kinds: {', '.join(KINDS)}")

    def decorator(cls: type[T]) -> type[T]:
        _REGISTRY[kind][name] = cls
        summary = (inspect.getdoc(cls) or "").strip().split("\n\n")[0].replace("\n", " ")
        _META[(kind, name)] = ComponentInfo(kind, name, cls, summary, _describe_params(cls), requires or [])
        setattr(cls, "component_name", name)
        setattr(cls, "component_kind", kind)
        return cls

    return decorator


def get_class(kind: str, name: str) -> type:
    if kind not in _REGISTRY:
        raise ComponentError(f"Unknown component kind {kind!r}.")
    try:
        return _REGISTRY[kind][name]
    except KeyError:
        options = ", ".join(sorted(_REGISTRY[kind])) or "(none registered)"
        raise ComponentError(f"Unknown {kind} {name!r}. Available: {options}") from None


def build(kind: str, spec: str | dict[str, Any] | None, **injected: Any) -> Any:
    """Instantiate a component from ``{"type": name, ...params}`` or a bare name.

    ``injected`` holds shared dependencies (llm, embedder, vectorstore, …). Only
    the ones a component's ``__init__`` actually declares are passed through, so
    every component stays independently constructible.
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        spec = {"type": spec}
    if not isinstance(spec, dict):
        raise ComponentError(f"{kind} spec must be a string or a dict, got {type(spec).__name__}")

    params = {k: v for k, v in spec.items() if k != "type"}
    name = spec.get("type")
    if not name:
        raise ComponentError(f"{kind} spec is missing a 'type' key: {spec!r}")

    cls = get_class(kind, str(name))
    try:
        signature = inspect.signature(cls.__init__)
        accepted = set(signature.parameters)
        takes_kwargs = any(p.kind is p.VAR_KEYWORD for p in signature.parameters.values())
    except (TypeError, ValueError):  # pragma: no cover
        accepted, takes_kwargs = set(), True

    kwargs = dict(params)
    for dep_name, dep in injected.items():
        if dep is not None and dep_name in accepted:
            kwargs[dep_name] = dep
    if not takes_kwargs:
        unknown = set(kwargs) - accepted
        if unknown:
            raise ComponentError(
                f"{kind} {name!r} does not accept {sorted(unknown)}. Accepted: {sorted(accepted - {'self'})}"
            )
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ComponentError(f"Could not build {kind} {name!r}: {exc}") from exc


def list_components(kind: str | None = None) -> dict[str, list[dict[str, Any]]]:
    """Catalogue of everything registered — powers ``advance_rag components`` and /api/lab/components."""
    kinds = [kind] if kind else list(KINDS)
    return {
        k: [_META[(k, n)].to_dict() for n in sorted(_REGISTRY[k])]
        for k in kinds
    }


def names(kind: str) -> list[str]:
    return sorted(_REGISTRY.get(kind, {}))
