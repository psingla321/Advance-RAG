# 09 — Vector Database Selection

## 1. Scale Assumptions

| Item | Value |
|---|---|
| Documents | 100M+ |
| Chunks (avg ~20/doc) | ~2B |
| Vector dims | 1024 (MRL-truncated) |
| Raw float32 size | 2B × 1024 × 4 B ≈ **8.2 TB** |
| int8 scalar quantized | ≈ **2.0 TB** (+ HNSW graph overhead ~10–20%) |
| Binary quantized (with rescoring) | ≈ **0.26 TB** |
| Query load | ~1,500 peak QPS to search after cache (≈ 60% of 2,500 peak) |
| Tenants | Thousands; long tail small, ~50 large |

Quantization is therefore mandatory at this scale regardless of engine.

## 2. Comparison Matrix

Ratings: ●●●●● best … ●○○○○ weakest. Assessment reflects 2026 managed offerings; validate with a PoC on your corpus.

| Criterion | Azure AI Search | Pinecone | Weaviate | Qdrant | Milvus / Zilliz | Elasticsearch |
|---|---|---|---|---|---|---|
| **Hybrid (BM25 + vector) native** | ●●●●● RRF + semantic ranker built-in | ●●●○○ sparse-dense vectors, no full BM25 analyzers | ●●●●○ BM25 + vector hybrid | ●●●●○ sparse + dense, BM25 via sparse | ●●●●○ sparse + dense + BM25 function | ●●●●● mature BM25 + kNN + RRF |
| **Built-in re-ranker** | ●●●●● semantic ranker | ●●●○○ hosted rerank models | ●●●○○ module-based | ●●○○○ external | ●●○○○ external | ●●●●○ ES rerank / ELSER |
| **Raw ANN performance (QPS/latency)** | ●●●○○ | ●●●●○ | ●●●○○ | ●●●●● | ●●●●● (GPU index option) | ●●●○○ |
| **Max scale (billions)** | ●●●○○ multi-service sharding needed beyond partition limits | ●●●●● serverless | ●●●○○ | ●●●●○ | ●●●●● designed for 10B+ | ●●●●○ |
| **Multi-tenancy** | ●●●●○ index-per-tenant or filter; service-level isolation | ●●●●● namespaces | ●●●●● native tenant shards, hot/cold tenants | ●●●●○ payload-partitioned / shard keys | ●●●●○ partition keys, DB per tenant | ●●●●○ index/alias per tenant, DLS |
| **Security trimming / doc-level security** | ●●●●● filters + Entra-based ACL preview features, private endpoints | ●●●○○ metadata filters | ●●●○○ filters, RBAC | ●●●○○ filters, JWT RBAC | ●●●○○ RBAC, filters | ●●●●● document-level security |
| **Azure integration** (Entra, Private Link, CMK, Monitor, Purview) | ●●●●● native | ●●○○○ (Azure Marketplace, private link limited) | ●●●○○ | ●●●○○ (managed cloud on Azure / self-host AKS) | ●●●○○ (Zilliz on Azure / self-host) | ●●●●○ Elastic Cloud on Azure |
| **Compliance** (SOC2, ISO, HIPAA BAA, EU residency) | ●●●●● Azure compliance scope | ●●●●○ | ●●●○○ | ●●●○○ | ●●●○○ | ●●●●○ |
| **Operational burden** | ●●●●● PaaS | ●●●●● SaaS | ●●●○○ | ●●●○○ (self-host) / ●●●●○ managed | ●●○○○ self-host / ●●●●○ Zilliz | ●●●○○ |
| **Cost at 2B vectors** | ●●○○○ partition-based pricing gets expensive at extreme scale | ●●●○○ usage-based; read-heavy can escalate | ●●●○○ | ●●●●○ efficient (quantization, on-disk) | ●●●●● disk-based + GPU indexes | ●●○○○ memory-heavy |
| **Enterprise readiness overall** | ●●●●● | ●●●●○ | ●●●○○ | ●●●○○ | ●●●●○ | ●●●●○ |

### Notes per option

- **Azure AI Search** — strongest fit for an Azure enterprise: hybrid + semantic ranker out of the box, integrated vectorization, private endpoints, CMK, Entra auth, compliance scope, indexers for Azure sources. Constraint: per-service partition/storage limits mean 2B vectors require **sharding across multiple services/indexes** (which aligns with the stamp + tenant model) and aggressive quantization (int8/binary + rescoring, `stored=false` for vector fields).
- **Pinecone** — excellent serverless DX and scale; weaker on lexical analyzers, Azure-native governance and data residency control.
- **Weaviate** — strong native multi-tenancy (per-tenant shards, offloading inactive tenants); operating it at billions requires significant ops.
- **Qdrant** — top raw performance, efficient quantization and on-disk storage; good choice for self-hosted high-QPS tiers; fewer enterprise integrations.
- **Milvus / Zilliz** — designed for 10B+ vectors with disk-based (DiskANN) and GPU indexes; best cost at extreme scale; higher operational complexity (self-host) or vendor dependency (Zilliz).
- **Elasticsearch** — best lexical engine and document-level security; vector performance and memory cost weaker at billions; good if the organization already standardizes on Elastic.

## 3. Recommendation

**Primary: Azure AI Search** (Standard S3 / Storage Optimized L1–L2 depending on stamp), with:

1. **Sharding model aligned to tenancy**
   - Shared tier: shared indexes per stamp, `tenant_id` filter, ≤ ~150M chunks per index.
   - Dedicated tier: index per tenant.
   - Isolated tier: dedicated search service per tenant.
   - A **Search Router** in the Data Access SDK maps `tenant_id → (service, index, alias)` from the tenant catalog.
2. **Compression:** MRL 1024-d, int8 scalar quantization with oversampling + rescoring; vector fields `stored=false`; binary quantization for cold/archival corpora.
3. **Replicas:** ≥ 3 replicas per service for 99.9% read+write SLA and QPS; partitions sized by storage.
4. **Semantic ranker** enabled for route B.

**Scale-out tier (escape hatch): Milvus (self-hosted on AKS) or Zilliz Cloud on Azure**, adopted for a stamp when **any** holds:

- Projected vectors in a stamp > ~2B, or AI Search cost per million vectors exceeds Milvus TCO by > 40% at sustained load;
- Tenants require learned sparse retrieval (SPLADE/BGE-M3 sparse) or GPU-accelerated ANN;
- Very-high-QPS / low-latency vector-only workloads (e.g., recommendation, memory search).

Both tiers sit behind the **same Retrieval Agent interface** (`search(query, filters, k) → PassageSet`), so tenants can be moved without application changes. Re-ranking for the Milvus tier uses self-hosted cross-encoders.

**Redis (RediSearch)** is used only for caches/memory vectors (millions, not billions).

## 4. Decision Record

| Decision | Azure AI Search primary; Milvus/Zilliz scale-out tier |
|---|---|
| Drivers | Enterprise security & compliance, hybrid + semantic ranking built-in, Azure-native ops, multi-tenant isolation options |
| Trade-offs accepted | Higher unit cost at extreme scale; sharding logic in SDK |
| Re-evaluate | Annually, or when vectors per stamp > 1.5B, or when AI Search limits/pricing change |
| PoC exit criteria | Recall@50 ≥ 0.95 vs. exact search on 50M-chunk sample; P95 hybrid+rerank ≤ 250 ms at 500 QPS per service; cost per 1M queries within budget |
