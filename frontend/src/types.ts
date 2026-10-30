export interface Identity {
  tenantId: string
  userId: string
  groups: string
}

export interface RouteScore {
  route: string
  confidence: number
}

export interface Routing {
  mode: string
  routes: RouteScore[]
  execute: string[]
  confidence: number
  risk: number
  ambiguity: number
  explanation: string
}

export interface Citation {
  n: number
  filename: string
  heading: string
  snippet: string
  doc_id: string
  score: number
}

export interface TraceEntry {
  node: string
  ms: number
  [key: string]: unknown
}

export interface SqlResult {
  sql: string | null
  columns: string[]
  rows: (string | number | null)[][]
  row_count: number
  error: string | null
}

export interface ToolResult {
  tool: string | null
  status: string
  output: string
  side_effect: boolean
}

export interface Verification {
  method: string
  groundedness: number | null
  numeric: number | null
  passed: boolean
  repaired: boolean
}

export interface FinalPayload {
  conversation_id: string
  request_id?: string
  answer: string
  routing: Routing | null
  citations: Citation[]
  sql?: SqlResult | null
  tool?: ToolResult | null
  verification: Verification | null
  escalation?: { id: string; reason: string } | null
  trust: number | null
  cache?: { level: string; similarity: number } | null
  blocked?: boolean
  latency_ms?: number
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  streaming?: boolean
  trace: TraceEntry[]
  routing?: Routing
  final?: FinalPayload
  error?: string
  feedback?: 'up' | 'down'
  traceId?: string
}

export interface Conversation {
  id: string
  serverId: string | null
  title: string
  updatedAt: number
  messages: ChatMessage[]
}

export interface DocumentInfo {
  id: string
  filename: string
  status: string
  chunk_count: number
  acl: string[]
  error: string | null
  created_at: string
}

export interface RecentRequest {
  request_id: string
  ts: string
  query: string
  mode: string
  routes: string[]
  outcome: string
  latency_ms: number
  trust: number | null
  cache_hit: boolean
}

export interface Insights {
  total: number
  cache_hit_rate: number
  escalation_rate: number
  abstention_rate: number
  blocked: number
  avg_trust: number | null
  avg_groundedness: number | null
  latency: { p50: number; p95: number; p99: number; samples: number }
  routes: Record<string, number>
  modes: Record<string, number>
  feedback: { up: number; down: number }
  nodes: { node: string; avg_ms: number; count: number }[]
  timeline: { minute: number; count: number }[]
  recent: RecentRequest[]
  system: { queue_depth: number; inflight: number; max_inflight: number; mock_llm: boolean; embedded_worker: boolean }
}

export type View = 'chat' | 'insights' | 'knowledge' | 'lab'

// ---------------------------------------------------------------- RAG Lab

export interface ComponentInfo {
  kind: string
  name: string
  summary: string
  requires: string[]
  params: { name: string; type: string; default: unknown; required: boolean }[]
}

export interface LabPreset {
  name: string
  description: string
  config: Record<string, any> & { top_k?: number; top_n?: number }
}

export interface LabAnswer {
  query: string
  answer: string
  citations: Citation[]
  contexts: { id: string; score: number; scores: Record<string, number>; filename: string; heading: string; text: string }[]
  expanded_queries: string[]
  groundedness: number | null
  abstained: boolean
  latency_ms: number
  trace: Record<string, unknown>[]
  metadata: Record<string, unknown>
}

export interface LabRetrieval {
  queries: string[]
  trace: Record<string, unknown>[]
  contexts: {
    n: number
    id: string
    score: number
    scores: Record<string, number>
    filename: string
    heading: string
    text: string
  }[]
}

export interface LabSweepRun {
  name: string
  overall: number
  scores: Record<string, number>
  latency: { mean: number; p50: number; p95: number; p99: number }
  pipeline: Record<string, unknown>
}

/** One axis of a configuration sweep, as typed in the Lab UI. */
export interface SweepAxis {
  key: string
  values: string
}
