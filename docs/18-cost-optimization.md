# 18 — Cost Optimization

> All prices below are **illustrative placeholders** for modeling. Replace with your negotiated Azure/provider rates; the model structure and levers are what matter.

## 1. Cost Drivers (steady state, ranked)

| Rank | Driver | Share of run-rate (typical) | Scales with |
|---|---|---|---|
| 1 | LLM inference tokens | 45–60% | Executed queries × tokens × tier mix |
| 2 | Search (AI Search services, replicas/partitions) | 15–25% | Chunks stored, QPS |
| 3 | Compute (AKS CPU + GPU) | 10–15% | QPS, ingestion throughput, local models |
| 4 | Ingestion AI services (OCR/layout, speech, video, embeddings) | 5–10% steady; **dominant one-time during backfill** | Pages, minutes, tokens ingested |
| 5 | Data stores (Cosmos, PostgreSQL, Redis, Neo4j) | 5–10% | Conversations, cache size |
| 6 | Storage, networking, observability | 3–7% | TB stored, egress, telemetry volume |

## 2. Parameterized Cost Model

### 2.1 LLM

$$
C_{LLM}/\text{day} = Q \cdot (1-h) \cdot \sum_{t\in\{T0,T1,T2\}} s_t \cdot \big[ n \cdot (\text{in} \cdot (1-\rho\,\delta) \cdot p^{in}_t + \text{out}\cdot p^{out}_t) \big]
$$

| Symbol | Meaning | Baseline |
|---|---|---|
| $Q$ | queries/day | 10M |
| $h$ | answer-cache hit rate | 0.40 |
| $s_t$ | call share by tier | 0.35 / 0.55 / 0.10 |
| $n \cdot \text{in}$, $n \cdot \text{out}$ | tokens per executed query (all calls) | 4,000 in / 400 out |
| $\rho$ | fraction of input that is cacheable prefix | 0.40 |
| $\delta$ | prompt-cache discount | 0.75 |
| $p^{in}_t$, $p^{out}_t$ ($/1M) | illustrative | T0 0.10/0.40 · T1 0.40/1.60 · T2 2.50/10.00 |

Illustrative result: blended ≈ $0.50 / 1M input, $2.00 / 1M output → 6M executed queries/day ≈ 24B input + 2.4B output tokens → **≈ $12–15k/day (~$380–450k/month)**. Without caching and routing (all T2, no cache) the same load would cost **≈ $4.2M/month** — routing + caching is a ~10× lever.

### 2.2 Ingestion (backfill of 100M docs, one-time)

| Item | Assumption | Illustrative |
|---|---|---|
| OCR / layout pages | 30% of docs need it × 10 pages = 300M pages | Layout at ~$10 / 1k pages → $3.0M; with digital-PDF fast path + Read model (~$1.5 / 1k) for text-only scans + commitment tier → **$0.6–1.0M** |
| Embeddings | 2B chunks × 600 tokens = 1.2T tokens | @ $0.13 / 1M → $156k; batch API (–50%) → **~$78k** |
| Contextual headers (T0) | 2B chunks × 150 out + cached doc input | **$50–150k** (apply selectively) |
| Audio / video | Per-minute transcription/indexing | Tenant-dependent; batch transcription tiers |

Steady-state ingestion (2M docs/day, ~30% changed chunks re-embedded) ≈ **$1–3k/day**.

### 2.3 Platform (per month, 4 stamps, illustrative)

| Component | Range |
|---|---|
| AI Search | $150–300k |
| AKS CPU | $80–150k (reservations/savings plan) |
| GPU (vLLM, rerankers, NLI) | $40–100k |
| Redis (3 clusters × stamps, geo-replicated) | $30–60k |
| Cosmos DB + PostgreSQL + Neo4j | $50–120k |
| Storage (≈ 200–400 TB with tiering) | $5–15k |
| Observability (Log Analytics, ADX, Langfuse) | $20–50k |
| Networking (Front Door, Firewall, private endpoints, egress) | $15–40k |

**Indicative total run-rate: ~$0.8–1.3M/month** at 10M queries/day ≈ **$0.003–0.004 per query** all-in.

## 3. Mitigation Levers

### 3.1 Cache-first architecture

- 35–50% answer-cache hits remove the entire LLM + search path for those queries (§ [06](06-cache-architecture.md)).
- L5 embedding cache (> 60% query hit rate) and L6 search cache reduce embedding and search load.
- Provider prompt caching on stable prefixes cuts input cost of cached segments by 50–90%.
- **Avoided cost** tracked per tenant to prove savings.

### 3.2 Dynamic model routing

- Lowest sufficient tier per task (T0 35% / T1 55% / T2 ≤ 10%).
- Cascade instead of always-frontier; T2 only after verification failure or high risk.
- Local Llama for high-volume T0 tasks (rewrite, NLI, compression, classification) once GPU utilization > ~40%.
- Batch APIs (–50%) for offline jobs; off-peak scheduling.
- Output caps per route; concise answer style by default.
- PTU for baseline (cheaper per token at high utilization, predictable latency); PAYG for peaks.

### 3.3 Embedding cost reduction

- Content-hash dedup (identical files across users/tenants processed once per tenant; L5 global reuse of embeddings for identical text is allowed since embeddings carry no ACL).
- Chunk-level incremental re-embedding (only changed chunks).
- MRL truncation (1024 vs 3072 dims) → 3× smaller index, lower search cost, negligible recall loss (validate per corpus).
- Batch embedding API for backfill; open-source embedding model on GPU for very large or sovereign corpora.
- Skip embedding for non-text-bearing content (large tables → SQL route).

### 3.4 Incremental indexing

- Metadata/ACL-only updates use partial merges (no re-embed).
- Blue/green re-index re-uses processed DOM and stored embeddings — avoids OCR and embedding re-spend.
- Bulk lane on spot nodes and off-peak.

### 3.5 Storage tiering

| Data | Policy |
|---|---|
| Raw originals | Hot 30 d → Cool 90 d → Cold 1 yr → Archive |
| Processed DOM / chunks | Hot 90 d → Cool (kept for re-index) |
| Embedding Parquet | Cool |
| Old index versions | Deleted 7 d after alias swap |
| Cold tenants (no queries 90 d) | Index moved to binary-quantized / lower-tier service; Weaviate-style offload on Milvus tier |
| Telemetry | Log Analytics 30 d (basic logs for verbose tables) → ADX 1 yr → ADLS archive |

### 3.6 Search cost

- int8 / binary quantization, `stored=false` vectors, MRL dims → 4–30× vector storage reduction.
- Right-size replicas per stamp by QPS; scale replicas down off-peak (scheduled).
- Index-per-large-tenant only when justified; long tail in shared indexes.
- Evaluate Milvus tier when per-vector cost crosses threshold ([09](09-vector-db-selection.md)).

### 3.7 Compute

- Reservations / savings plans for baseline AKS; spot for ingestion bulk and batch GPU.
- VPA rightsizing, bin-packing, Node Auto Provisioning; scale-to-zero for idle tenant-specific workers (KEDA).

## 4. FinOps Governance

- Per-tenant budgets, showback/chargeback from `usage_daily`.
- Cost per query and cost per resolved question as top-line KPIs.
- Weekly review of top cost tenants/routes; anomaly alerts.
- Every new feature requires a cost model entry and a cost guardrail (budget, cap or cache policy).
