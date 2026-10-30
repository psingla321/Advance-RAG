# nexus-rag (backend)

The FastAPI + LangGraph engine and the plug-and-play RAG lab.

- `app/` — the production engine (chat, routing, agents, ingestion, observability)
- `app/rag/` — the RAG lab: swappable components, evaluation, sweeps, CLI
- `tests/` — `python -m pytest`

Install the toolkit and its `nexus-rag` command:

```bash
pip install -e .            # core only; every backend is optional
pip install -e ".[all]"     # every provider, store and loader
nexus-rag components
```

See the repository README for the full guide.
