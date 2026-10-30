# 16 — Scalability & Performance Strategy

## 1. Targets

| Metric | Target |
|---|---|
| P50 / P95 / P99 end-to-end | < 2 s / < 5 s / < 10 s |
| Concurrent users | 10,000+ |
| Documents | 100M+ (~2B chunks) |
| Daily queries | 10M+ |

## 2. Capacity Math

| Quantity | Derivation | Value |
|---|---|---|
| Average QPS | 10M / 86,400 | ~116 QPS |
| Peak QPS | business-hours concentration (×6) + bursts | **~700 sustained peak, plan 1,500, burst 3,000** |
| Concurrent users → QPS | 10k users × 1 query / 60 s active cadence | ~170 QPS (users are not the bottleneck; integrations/APIs add load) |
| Cache hit (35–50%) | 1,500 × 0.6 | ~900 QPS reach router/execution at plan peak |
| Search QPS | ~70% of executed queries hit retrieval, ×1.5 for multi-query | ~950 QPS across stamps |
| LLM calls | avg 2.2 calls per executed query | ~2,000 calls/s at plan peak |
| LLM tokens | avg 4k in / 400 out per executed query | ~3.6M input + 0.36M output tokens/s at plan peak (sustained average ~10× lower) |
| Daily tokens | 10M × 0.6 × 4.4k | ~26B tokens/day (before prompt-cache discounts) |

These drive PTU sizing (cover ~P70 of daily load with PTU, spill rest to PAYG) and the stamp count (4 stamps at launch, 2 per active region).

## 3. Latency Budget (route B, cache miss, P50)

| Stage | Budget (ms) | Technique |
|---|---|---|
| Edge + gateway + auth | 30 | Front Door anycast, JWT cached keys, APIM in-region |
| Input guardrails | 30 | Prompt Shields async-parallel with cache lookup |
| Query embedding (L5) | 0–40 | Embedding cache; parallel with L1 |
| Cache probe L1–L4 | 10–20 | Pipelined Redis, 50 ms timeout |
| Router (L0 + L1) | 40–80 | ONNX classifier, batched |
| Hybrid search + semantic rerank | 200–250 | Pre-filters, dedicated replicas, int8 |
| Parent expansion + compression | 40–60 | Local T0 / cross-encoder on GPU |
| LLM TTFT | 500–900 | PTU, prompt caching, region affinity |
| Generation (T1 at 150–200 tok/s, streamed) | ~700 for a typical 120-token answer; 2,000–2,700 at the 400-token mean of long answers | Streaming, output caps per route |
| Verification | 100–200 (overlapped with stream tail) | Incremental verification |
| **Time to first token** | **~0.9–1.3 s** | |
| **Full answer** | **~1.7–1.9 s P50 (typical 120-token answer)** | |

P50 across all traffic (with 35–50% cache hits at 10–50 ms) ≈ **0.9–1.2 s**. P95 is dominated by T2 escalations and multi-query retrieval (~3.5–4.5 s). P99 by agent routes F (converted to async beyond 30 s, excluded from synchronous SLO) and fallbacks (~7–9 s).

## 4. Scaling Strategy by Layer

| Layer | Scaling mechanism |
|---|---|
| Edge / gateway | Front Door global; APIM Premium v2 autoscale units per region |
| Stateless services (API, router, orchestrator, agents) | HPA on CPU + custom metric (in-flight requests); async FastAPI (uvicorn, uvloop), connection pooling; pods spread across AZs |
| LangGraph state | Externalized (Redis/Cosmos) → any pod can resume any thread |
| Ingestion | KEDA on Service Bus queue length per stage/lane; spot node pools for bulk; cluster autoscaler / Node Auto Provisioning |
| GPU inference | KEDA on queue depth + GPU utilization; vLLM continuous batching; multi-LoRA |
| AI Search | Replicas for QPS (linear), partitions for storage; tenant sharding across services; read-heavy → replicas, ingest bursts → schedule bulk indexing off-peak |
| Redis | Cluster sharding; separate clusters per workload |
| Cosmos DB | Autoscale RU, hierarchical partition keys to avoid hot partitions |
| PostgreSQL | Vertical + read replicas; pgBouncer; partitioned large tables; per-stamp instances |
| Neo4j | Read replicas for query load; dedicated DBs for large tenants |
| LLM capacity | PTU per region + PAYG/Global spillover + multi-provider; per-tenant TPM limits at APIM |
| Whole platform | **Add stamps** (horizontal scale unit) and rebalance tenants via catalog |

## 5. Performance Techniques

- **Cache-first** (35–50% short-circuit).
- **Parallelism:** guardrails ∥ L1 ∥ embedding; multi-query searches in parallel; independent plan steps in parallel; verification overlapped with streaming.
- **Speculative retrieval:** start hybrid retrieval in parallel with the router when L0 indicates likely route B (> 60% of traffic) — saves ~80 ms; result discarded if route differs.
- **Streaming everywhere** (SSE) and early "thinking/progress" events for agent routes.
- **Prompt caching** and short, stable system prompts.
- **Adaptive retrieval depth** (k=20 when scores are decisive).
- **Connection reuse** (HTTP/2 to model endpoints), regional affinity, no cross-region calls on hot path.
- **Backpressure:** bounded queues, load shedding of low-priority traffic (bulk ingestion, batch evals) under pressure; per-tenant fairness (token buckets) to prevent noisy neighbors.

## 6. Load & Chaos Testing

- k6/Locust replay of production query mixes at 1.5× target peak in `perf`; corpus of 25M docs per stamp.
- Chaos Studio experiments: AZ loss, Redis failover, AI Search replica loss, model 429 storms, Service Bus throttling.
- Pass criteria: SLOs met at 1.5× peak; graceful degradation (cache-only mode, T0/T1-only mode) verified.

## 7. Degradation Modes

| Mode | Trigger | Behavior |
|---|---|---|
| Economy | LLM capacity < 70% healthy | T2 disabled except high-risk; shorter outputs |
| Cache-preferred | Search degraded | Lower L3 threshold to 0.93 with entity guard; disclose cached answer age |
| Retrieval-only | All LLMs unavailable | Return top passages with highlights (no generation) |
| Read-only | Ingestion backlog > SLO × 3 | Pause bulk lane, prioritize interactive |
