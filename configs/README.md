# Configuration recipes

Each file is a complete RAG system. Use one directly:

```bash
nexus-rag ask "..." --config configs/hybrid.yaml
```

```python
from app.rag import RAG
rag = RAG("configs/hybrid.yaml")
```

Or start from one and change a single line — that is the point of the format.
Values support `${ENV_VAR}` and `${ENV_VAR:-default}` so these files are safe to commit.

| File | Use it when |
|---|---|
| `offline.yaml` | You want to try the toolkit right now with nothing installed and no keys. |
| `baseline.yaml` | You need a control group to prove an "improvement" actually improves things. |
| `hybrid.yaml` | Production default: dense + BM25, reranked. Start here for real work. |
| `advanced.yaml` | Recall matters more than latency or cost: query expansion, cross-encoder, self-critique. |
| `local.yaml` | Data cannot leave the machine: Ollama + sentence-transformers. |
| `azure.yaml` | You are deploying alongside the Nexus production engine. |
| `cheap.yaml` | Token spend is the binding constraint. |
