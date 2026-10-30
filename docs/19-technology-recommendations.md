# 19 — Technology Recommendations

| Layer | Recommended | Alternatives | Rationale |
|---|---|---|---|
| Edge | Azure Front Door Premium + WAF | Cloudflare | Global anycast, Private Link origins, WAF managed rules |
| API / AI gateway | Azure API Management Premium v2 | Kong, LiteLLM proxy alone | Token limits, LLM metrics, backend pools, MCP gateway, Entra integration |
| Identity | Entra ID, External ID, managed/workload identities | Okta (federated) | Native OBO, Conditional Access, PIM |
| Compute | AKS (Standard tier, private, CNI Overlay, Cilium) | Azure Container Apps for small stamps | GPU pools, KEDA, fine-grained control |
| API framework | Python 3.12 + FastAPI + Pydantic v2 | — | Async, typed, ecosystem |
| Agent framework | LangGraph (checkpointing, interrupts, sub-graphs) | Microsoft Agent Framework, Semantic Kernel | Durable stateful graphs, HITL interrupts, mature |
| Workflow (ingestion) | Temporal on AKS | Durable Functions | Durable, observable, high throughput |
| Messaging | Service Bus Premium (commands), Event Grid (blob events), Event Hubs (telemetry) | Kafka (Confluent) | Managed, geo-DR, sessions, DLQ |
| Vector / hybrid search | **Azure AI Search** (+ semantic ranker) | Milvus/Zilliz (scale-out tier), Qdrant | See [09](09-vector-db-selection.md) |
| Knowledge graph | Neo4j Enterprise / Aura on Azure | Cosmos DB Gremlin, PostgreSQL + Apache AGE | GraphRAG tooling, Cypher, vector index |
| Relational | Azure Database for PostgreSQL Flexible | Azure SQL Database | RLS, extensions (pgvector for small use), cost |
| Document / state | Azure Cosmos DB (NoSQL) | — | Global, multi-region writes, TTL |
| Cache / memory | Azure Managed Redis (RediSearch, geo-replication) | — | Sub-ms, vector search for semantic cache |
| Object storage | ADLS Gen2 | — | Hierarchical namespace, tiering, immutability |
| Analytics | Azure Data Explorer | Fabric Eventhouse | Telemetry/cost analytics at scale |
| OCR / layout | Azure AI Document Intelligence | Unstructured.io, Docling (self-hosted fast path) | Accuracy on tables/forms; commitment pricing |
| Audio / video | Azure AI Speech (batch), Azure AI Video Indexer | Whisper on GPU | Diarization, timestamps |
| Malware scan | Microsoft Defender for Storage | ClamAV sidecar | Native on-upload scanning |
| Content safety | Azure AI Content Safety (Prompt Shields, groundedness) | Llama Guard (local) | Integrated, low-latency |
| PII | Presidio + Azure AI Language PII | — | Custom recognizers |
| LLMs | Azure OpenAI GPT (PTU + Global), Claude via Foundry, Gemini via Vertex, Llama via vLLM/Foundry | — | Multi-provider resiliency & quality routing |
| Model gateway | LiteLLM + custom policy engine behind APIM | Portkey | OpenAI-compatible, fallbacks, budgets |
| Embeddings | text-embedding-3-large (1024-d MRL) | BGE-M3 / E5 (self-hosted) | Quality, MRL support, sovereign fallback |
| Rerank | AI Search semantic ranker; bge-reranker-v2 (self-hosted) | Cohere Rerank via Foundry | Built-in + custom tier |
| Router classifier | Fine-tuned DeBERTa-v3 / multilingual-E5 on ONNX Runtime | SetFit | < 40 ms CPU inference |
| SQL safety | sqlglot (parse/validate), semantic layer | Vanna-style frameworks | AST validation, dialect support |
| Policy engine | OPA (Rego) | Cedar | Mature, sidecar/library |
| Observability | OpenTelemetry, Azure Monitor/App Insights, Managed Prometheus, Managed Grafana, Langfuse (self-hosted) | Arize Phoenix | Open standards + LLM-specific tracing |
| Evals | Langfuse evals, RAGAS, Azure AI Foundry evaluations, promptfoo (CI) | DeepEval | Offline + online |
| Security | Key Vault / Managed HSM, Defender for Cloud, Sentinel, Purview, Azure Policy | — | Native compliance |
| IaC / delivery | Bicep or Terraform, GitHub Actions, Flux, Argo Rollouts, Notation | Azure DevOps | GitOps, progressive delivery |
| Load / chaos | k6, Locust, Azure Chaos Studio | — | |

## Key Python Libraries

FastAPI, Pydantic v2, LangGraph, LangChain-core (selectively), LiteLLM, azure-search-documents, azure-cosmos, asyncpg, redis-py (RedisVL), neo4j driver, azure-ai-documentintelligence, azure-ai-contentsafety, presidio-analyzer, sqlglot, onnxruntime, opentelemetry-sdk + instrumentations, langfuse, temporalio, tenacity, pybreaker.
