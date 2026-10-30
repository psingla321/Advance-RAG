# 11 — Data Flow Diagrams

Sequence diagrams for each execution path. Latencies are typical server-side P50 values.

## 1. Cache Hit (Route G)

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant AFD as Front Door
    participant APIM as APIM
    participant API as Chat API
    participant GR as Guardrails
    participant CM as Cache Manager
    participant R as Redis
    U->>AFD: POST /chat (JWT)
    AFD->>APIM: route to nearest region
    APIM->>APIM: validate JWT, tenant quota
    APIM->>API: request + tenant claims
    API->>GR: prompt shields, PII
    GR-->>API: ok (15 ms)
    API->>CM: lookup(query, tenant, acl_hash)
    CM->>R: L1 exact / L5 embed / L2-L3 vector (pipelined)
    R-->>CM: hit cos 0.96 L3
    CM->>R: entitlement check on cited doc ids
    R-->>CM: all permitted
    CM-->>API: cached answer + citations
    API-->>U: SSE stream (total ~40-80 ms incl. network)
```

## 2. RAG (Route B)

```mermaid
sequenceDiagram
    autonumber
    participant API as Chat API
    participant RT as Router
    participant OR as Orchestrator
    participant RA as Retrieval Agent
    participant AIS as AI Search
    participant MG as Model Gateway
    participant VA as Verification
    participant CM as Cache Manager
    API->>RT: classify (cache miss)
    RT-->>API: B conf 0.91, tier T1 (40 ms)
    API->>OR: execute(B)
    OR->>RA: retrieve(sub-query, filters)
    RA->>AIS: hybrid + vector + semantic rerank, ACL pre-filter
    AIS-->>RA: top 10 (180 ms)
    RA->>RA: parent expansion, compression (40 ms)
    RA-->>OR: PassageSet
    OR->>MG: synthesize (T1, stream)
    MG-->>API: tokens stream (TTFT ~600 ms)
    OR->>VA: verify draft vs evidence
    VA-->>OR: groundedness 0.88 pass (150 ms, parallel with tail of stream)
    OR-->>API: final citations + verdict
    OR->>CM: async write L1/L2/L3 + dep index
```

Verification runs incrementally on completed sentences while streaming; if it fails, the client receives a `revision` event and the corrected answer replaces the draft (or the UI withholds display for high-risk tenants until verification completes).

## 3. SQL (Route D)

```mermaid
sequenceDiagram
    autonumber
    participant OR as Orchestrator
    participant SA as SQL Agent
    participant SL as Semantic Layer
    participant MG as Model Gateway
    participant DB as Customer DB (RLS)
    OR->>SA: question + data source + OBO token
    SA->>SL: schema linking (tables, metrics, synonyms)
    SA->>SA: retrieve verified NL-SQL examples
    SA->>MG: generate SQL (T1, JSON schema)
    MG-->>SA: SQL
    SA->>SA: sqlglot validate: SELECT-only, LIMIT, cost estimate
    SA->>DB: execute as user (read-only, timeout 10 s)
    alt error
        DB-->>SA: error
        SA->>MG: repair (max 2)
    end
    DB-->>SA: rows
    SA-->>OR: TableResult + SQL
    OR->>MG: explain results (numbers from rows only)
```

## 4. Multi-Step Agent (Route F)

```mermaid
sequenceDiagram
    autonumber
    participant OR as Orchestrator
    participant PA as Planning Agent
    participant W as Workers (SQL, Retrieval, KG, Tool)
    participant SU as Summarization
    participant VA as Verification
    participant CR as Critic
    OR->>PA: plan(query, capabilities)
    PA-->>OR: DAG (4 steps)
    par independent steps
        OR->>W: step 1 (SQL)
        OR->>W: step 2 (Retrieval)
    end
    W-->>OR: results 1, 2
    OR->>W: step 3 (KG, depends on 1, 2)
    W-->>OR: result 3
    OR->>SU: synthesize all evidence
    SU-->>OR: draft
    OR->>VA: verify
    VA-->>OR: pass
    OR->>CR: critique (risk >= 0.5)
    CR-->>OR: approve
    OR-->>OR: stream answer + plan trace
```

Long-running (> 30 s) plans are converted to **async jobs**: the client receives a job id and progress events over SSE/WebSocket; results are delivered to the conversation and optionally via Teams/email.

## 5. Human Escalation (Route H)

```mermaid
sequenceDiagram
    autonumber
    participant OR as Orchestrator
    participant HR as Human Review Agent
    participant SB as Service Bus
    participant RC as Reviewer Console
    participant U as User
    OR->>HR: escalate(reason, draft, evidence, risk)
    HR->>SB: enqueue to pool (domain, language, tenant)
    HR-->>U: holding message + ticket id
    SB->>RC: task assigned (SLA timer)
    RC->>HR: approve / edit / reject
    HR-->>U: final answer (notification)
    HR->>HR: store labeled example for evals and router training
```

## 6. Tool Call with Side Effect (Route E)

```mermaid
sequenceDiagram
    autonumber
    participant OR as Orchestrator
    participant TA as Tool Agent
    participant U as User
    participant APIM as APIM MCP Gateway
    participant SYS as ServiceNow
    OR->>TA: "create a P2 incident for VPN outage"
    TA->>TA: select tool, extract args (schema)
    TA-->>U: confirm action card
    U->>TA: confirm
    TA->>APIM: tools/call create_incident (OBO token, idempotency key)
    APIM->>SYS: REST call
    SYS-->>APIM: INC0012345
    APIM-->>TA: result
    TA-->>OR: ToolResult(side_effect=true)
```

## 7. Ingestion Flow

```mermaid
sequenceDiagram
    autonumber
    participant C as Client / Connector
    participant IA as Ingestion API
    participant ST as ADLS
    participant DEF as Defender
    participant EG as Event Grid
    participant SB as Service Bus
    participant WF as Workflow Workers
    participant DI as Doc Intelligence
    participant EM as Embeddings
    participant AIS as AI Search
    participant NEO as Neo4j
    participant REG as Cosmos Registry
    participant CI as Cache Invalidator
    C->>IA: upload (SAS / multipart) + ACL
    IA->>REG: register RECEIVED
    IA->>ST: write raw (quarantine)
    ST->>DEF: on-upload scan
    DEF-->>ST: clean tag
    ST->>EG: BlobCreated
    EG->>SB: ingest.requests
    SB->>WF: start workflow(doc_id)
    WF->>DI: layout / OCR
    DI-->>WF: DOM
    WF->>WF: chunk, classify, PII, entities
    WF->>EM: batch embed (L5 cache first)
    WF->>AIS: upsert chunks
    WF->>NEO: upsert entities, edges
    WF->>REG: INDEXED
    WF->>CI: doc.changed
    CI->>CI: purge dependent cache keys
```
