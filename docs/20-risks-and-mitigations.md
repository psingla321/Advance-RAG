# 20 — Risks and Mitigations

Likelihood (L) and Impact (I): H / M / L.

| # | Risk | L | I | Mitigation | Owner |
|---|---|---|---|---|---|
| R1 | Cross-tenant or cross-permission data leakage (search, cache, memory, graph) | M | H | Mandatory Data Access SDK; ACL pre-filters; `acl_hash` in cache keys; entitlement re-check on cited docs; honeytoken canary tenants + automated leakage tests in CI and prod | Security |
| R2 | Hallucinated answers in high-stakes domains | M | H | Grounded-only routes, verification (NLI + groundedness), cross-family critic, abstention, HITL, hallucination KPI < 1% | AI Quality |
| R3 | Prompt injection (direct/indirect) leading to data exfiltration or unwanted actions | M | H | Prompt Shields on input + retrieved content, spotlighting, tool allow-lists, side-effect confirmation, egress control | Security |
| R4 | LLM provider capacity shortages / 429 storms | H | M | PTU in 2 regions, PAYG/Global spillover, multi-provider fallback map, local degraded mode, per-tenant TPM fairness | Platform |
| R5 | Latency SLO misses at peak | M | M | Cache-first, speculative retrieval, PTU, load shedding, stamp scaling, continuous load tests | SRE |
| R6 | Cost overrun (token growth, agent loops, backfills) | H | M | Budgets per request/tenant, routing tiers, cache, batch APIs, anomaly alerts, cost gates for features | FinOps |
| R7 | AI Search scale/cost limits at billions of vectors | M | M | Quantization, MRL, tenant sharding, Milvus scale-out tier behind same interface | Architecture |
| R8 | Stale or wrong cached answers | M | M | Event-driven invalidation, versioned namespaces, entity/number guard, calibrated thresholds, TTLs | Platform |
| R9 | Router misclassification | M | M | Calibrated confidence, parallel top-2 band, fallback ladder, continuous retraining from traces | AI Quality |
| R10 | ACL sync lag (revoked access still visible) | M | H | ACL fast lane (< 60 s), entitlement re-check at answer time, Entra change notifications | Security |
| R11 | Ingestion backlogs / poison files | M | M | Priority lanes, KEDA scaling, DLQ + quarantine, zip-bomb guards, per-stage SLOs | Data Eng |
| R12 | Model / prompt regressions after upgrades | M | M | Pinned versions, eval gates, canary by tenant ring, instant rollback via prompt labels | AI Quality |
| R13 | Vendor lock-in (Azure, specific LLM provider) | M | M | OpenAI-compatible gateway, provider-agnostic agents, open formats (Parquet, OTel, MCP), self-hostable alternatives documented | Architecture |
| R14 | Regulatory change (EU AI Act obligations, new privacy rules) | M | H | AI governance program, ISO/IEC 42001 alignment, transparency notes, logging of AI decisions, residency stamps | Compliance |
| R15 | Regional outage | L | H | Active-active, dual-indexing, geo-replication, quarterly game days | SRE |
| R16 | Knowledge graph quality (bad entity extraction, drift) | M | M | Governed schema, confidence thresholds, entity resolution, human curation for high-value graphs, fallback to RAG | Data Eng |
| R17 | SQL Agent generating expensive or unsafe queries | M | H | SELECT-only AST validation, cost estimation, timeouts, read-only roles, RLS/OBO, LIMIT | Data Eng |
| R18 | Operational complexity (many services, multiple stores) | H | M | Stamp-based standardization, IaC modules, platform team, golden paths, managed services preferred | Platform |
| R19 | GDPR erasure incompleteness across derived data (caches, memory, embeddings, graph) | M | H | Lineage by `doc_id`/`user_id` across all stores, cascade deletion workflow, verification report per DSAR | Compliance |
| R20 | Talent / skills gap in LLMOps | M | M | Training, paved paths, documentation, reference implementations | Leadership |
