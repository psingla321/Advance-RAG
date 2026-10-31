"""Declarative pipeline configuration.

A whole RAG system is one YAML file. Change a line, rerun the evaluation, see
whether it helped — that is the loop this layer exists to make fast.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

ComponentSpec = dict[str, Any] | str

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    """Expand ``${VAR}`` and ``${VAR:-default}`` so configs can be committed safely."""
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.getenv(m.group(1), m.group(2) or ""), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


class RAGConfig(BaseModel):
    """Every stage of the pipeline, as data.

    Each component field takes either a bare name (``"hybrid_rrf"``) or a dict
    with parameters (``{"type": "hybrid_rrf", "dense_weight": 0.7}``).
    """

    name: str = "default"
    description: str = ""

    # --- components -------------------------------------------------------
    llm: ComponentSpec = Field(default_factory=lambda: {"type": "mock"})
    embedder: ComponentSpec = Field(default_factory=lambda: {"type": "hash", "dim": 512})
    vectorstore: ComponentSpec = Field(default_factory=lambda: {"type": "memory"})
    loader: ComponentSpec = Field(default_factory=lambda: {"type": "auto"})
    chunker: ComponentSpec = Field(default_factory=lambda: {"type": "sentence", "max_tokens": 350})
    transform: ComponentSpec = Field(default_factory=lambda: {"type": "identity"})
    retriever: ComponentSpec = Field(default_factory=lambda: {"type": "hybrid_rrf"})
    reranker: ComponentSpec = Field(default_factory=lambda: {"type": "keyword_overlap"})
    compressor: ComponentSpec = Field(default_factory=lambda: {"type": "submodular"})
    generator: ComponentSpec = Field(default_factory=lambda: {"type": "stuff"})

    # --- pipeline knobs ---------------------------------------------------
    top_k: int = Field(default=30, description="Candidates fetched by the retriever.")
    top_n: int = Field(default=8, description="Candidates kept after reranking.")
    context_token_budget: int = Field(default=3000, description="Token budget for packed context.")
    namespace: str = "default"
    min_score: float = Field(default=0.0, description="Drop candidates scoring below this after reranking.")
    abstain_when_empty: bool = True

    # --- optional behaviours ---------------------------------------------
    guardrails: bool = Field(default=True, description="Prompt-injection and PII screening on input.")
    cache: bool = Field(default=False, description="In-process exact-match answer cache.")
    cache_ttl_seconds: int = 3600
    verify: bool = Field(default=False, description="Post-hoc groundedness scoring of the answer.")
    verify_threshold: float = 0.8
    system_prompt: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "RAGConfig":
        import yaml

        raw = yaml.safe_load(Path(path).read_text("utf-8")) or {}
        return cls.model_validate(_expand_env(raw))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RAGConfig":
        return cls.model_validate(_expand_env(data))

    @classmethod
    def preset(cls, name: str) -> "RAGConfig":
        """Load one of the built-in recipes by name (see ``configs/``)."""
        if name in PRESETS:
            return cls.model_validate(PRESETS[name])
        path = CONFIG_DIR / f"{name}.yaml"
        if path.exists():
            return cls.from_yaml(path)
        raise ValueError(f"Unknown preset {name!r}. Available: {', '.join(available_presets())}")

    def to_yaml(self, path: str | Path | None = None) -> str:
        import yaml

        text = yaml.safe_dump(self.model_dump(), sort_keys=False, allow_unicode=True)
        if path:
            Path(path).write_text(text, "utf-8")
        return text

    def with_overrides(self, **overrides: Any) -> "RAGConfig":
        """Return a copy with dotted overrides applied.

            cfg.with_overrides(**{"retriever.type": "mmr", "retriever.lambda_mult": 0.7})

        Changing a component's ``type`` to a *different* component discards the
        previous one's parameters, since they rarely transfer (a ``fixed`` chunker
        has no ``max_tokens``). Parameters set in the same call are then reapplied,
        so ``{"chunker.type": "fixed", "chunker.size": 800}`` works as written.
        """
        data = self.model_dump()
        # Swap components first so sibling parameters in this call survive the reset.
        for key, value in overrides.items():
            parts = key.split(".")
            if len(parts) == 2 and parts[1] == "type":
                current = data.get(parts[0])
                current_type = current.get("type") if isinstance(current, dict) else current
                if current_type != value:
                    data[parts[0]] = {"type": value}
        for key, value in overrides.items():
            target = data
            parts = key.split(".")
            for part in parts[:-1]:
                node = target.get(part)
                if not isinstance(node, dict):
                    node = {"type": node} if isinstance(node, str) else {}
                    target[part] = node
                target = node
            target[parts[-1]] = value
        return RAGConfig.model_validate(data)

    def fingerprint(self) -> str:
        """Stable hash of the component graph — used to key experiment results."""
        from app.textutil import sha256_hex

        payload = self.model_dump(exclude={"name", "description", "metadata"})
        return sha256_hex(json.dumps(payload, sort_keys=True, default=str))[:12]


CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


PRESETS: dict[str, dict[str, Any]] = {
    "offline": {
        "name": "offline",
        "description": "Zero dependencies, zero cost, zero network. Runs anywhere, including CI.",
        "llm": {"type": "mock"},
        "embedder": {"type": "hash", "dim": 512},
        "vectorstore": {"type": "memory"},
        "chunker": {"type": "sentence", "max_tokens": 350},
        "retriever": {"type": "hybrid_rrf"},
        "reranker": {"type": "keyword_overlap"},
        "generator": {"type": "extractive"},
    },
    "baseline": {
        "name": "baseline",
        "description": "Naive RAG: fixed chunks, dense-only retrieval, no reranking. The control group.",
        "embedder": {"type": "hash", "dim": 512},
        "vectorstore": {"type": "memory"},
        "chunker": {"type": "fixed", "size": 1000, "overlap": 150},
        "transform": {"type": "identity"},
        "retriever": {"type": "dense"},
        "reranker": {"type": "identity"},
        "compressor": {"type": "truncate"},
        "generator": {"type": "stuff"},
        "top_k": 10,
        "top_n": 5,
    },
    "hybrid": {
        "name": "hybrid",
        "description": "Dense + BM25 with RRF and heuristic reranking. The sensible default.",
        "chunker": {"type": "sentence", "max_tokens": 350, "overlap_sentences": 1},
        "retriever": {"type": "hybrid_rrf"},
        "reranker": {"type": "keyword_overlap"},
        "compressor": {"type": "submodular"},
        "generator": {"type": "stuff"},
    },
    "advanced": {
        "name": "advanced",
        "description": "Everything on: query expansion, hybrid retrieval, cross-encoder rerank, self-critique.",
        "chunker": {"type": "parent_child", "parent_tokens": 1200, "child_tokens": 220},
        "transform": {"type": "chain", "steps": [{"type": "rewrite"}, {"type": "multi_query", "n": 3}]},
        "retriever": {"type": "parent_document", "inner": {"type": "hybrid_rrf"}},
        "reranker": {"type": "chain", "stages": [
            {"reranker": {"type": "keyword_overlap"}, "top_n": 20},
            {"reranker": {"type": "cross_encoder"}},
        ]},
        "compressor": {"type": "lost_in_middle"},
        "generator": {"type": "self_rag", "inner": {"type": "citation_strict"}},
        "top_k": 40,
        "top_n": 8,
        "verify": True,
    },
    "local": {
        "name": "local",
        "description": "Fully local and private: Ollama + sentence-transformers, no data leaves the machine.",
        "llm": {"type": "ollama", "model": "llama3.1"},
        "embedder": {"type": "cached", "inner": {"type": "sentence_transformers", "model": "BAAI/bge-small-en-v1.5"}},
        "vectorstore": {"type": "memory", "persist_path": ".advance_rag_cache/store.json"},
        "chunker": {"type": "semantic"},
        "retriever": {"type": "hybrid_rrf"},
        "reranker": {"type": "cross_encoder"},
        "generator": {"type": "stuff"},
    },
    "azure": {
        "name": "azure",
        "description": "Matches the production AdvanceRAG engine: Azure OpenAI + Qdrant.",
        "llm": {"type": "azure_openai"},
        "embedder": {"type": "cached", "backend": "redis",
                     "inner": {"type": "azure_openai", "dim": 1024}},
        "vectorstore": {"type": "qdrant", "url": "${QDRANT_URL:-http://localhost:6333}"},
        "chunker": {"type": "sentence", "max_tokens": 350},
        "transform": {"type": "rewrite"},
        "retriever": {"type": "hybrid_rrf"},
        "reranker": {"type": "keyword_overlap"},
        "compressor": {"type": "submodular"},
        "generator": {"type": "self_rag"},
        "verify": True,
    },
    "cheap": {
        "name": "cheap",
        "description": "Minimises token spend: small chunks, tight budget, extractive compression.",
        "chunker": {"type": "sentence", "max_tokens": 200},
        "retriever": {"type": "hybrid_rrf"},
        "reranker": {"type": "keyword_overlap"},
        "compressor": {"type": "sentence_filter"},
        "generator": {"type": "stuff", "max_tokens": 400},
        "top_k": 20,
        "top_n": 4,
        "context_token_budget": 1200,
        "cache": True,
    },
    "high_recall": {
        "name": "high_recall",
        "description": "Optimised for not missing anything: wide fetch, query expansion, graph hops.",
        "chunker": {"type": "sentence", "max_tokens": 300},
        "transform": {"type": "chain", "steps": [{"type": "multi_query", "n": 4}, {"type": "hyde"}]},
        "retriever": {"type": "graph", "inner": {"type": "ensemble"}, "hops": 1},
        "reranker": {"type": "cross_encoder"},
        "compressor": {"type": "submodular"},
        "top_k": 60,
        "top_n": 12,
        "context_token_budget": 5000,
    },
}


def available_presets() -> list[str]:
    from_files = [p.stem for p in CONFIG_DIR.glob("*.yaml")] if CONFIG_DIR.exists() else []
    return sorted(set(PRESETS) | set(from_files))
