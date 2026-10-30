import type { TraceEntry } from '../types'

const NODE_COLORS: Record<string, string> = {
  guardrails: '#f472b6',
  memory: '#a78bfa',
  cache: '#facc15',
  router: '#60a5fa',
  retrieval_agent: '#22d3ee',
  sql_agent: '#4ade80',
  tool_agent: '#fb923c',
  direct_llm_agent: '#818cf8',
  synthesis_agent: '#c084fc',
  verification_agent: '#34d399',
  human_review_agent: '#f87171',
  clarification: '#fbbf24',
  finalize: '#94a3b8',
}

function details(entry: TraceEntry): string {
  const { node: _node, ms: _ms, ...rest } = entry
  return Object.entries(rest)
    .filter(([, v]) => v !== null && v !== undefined && v !== '' && !(Array.isArray(v) && v.length === 0))
    .map(([k, v]) => `${k}=${typeof v === 'object' ? JSON.stringify(v) : String(v)}`)
    .join('  ')
}

/** Sequential waterfall of engine steps; parallel agents share a start offset. */
export function TraceWaterfall({ trace }: { trace: TraceEntry[] }) {
  const parallel = new Set(['retrieval_agent', 'sql_agent', 'tool_agent', 'direct_llm_agent'])
  let cursor = 0
  let groupStart = 0
  let groupEnd = 0
  const rows = trace.map((entry, i) => {
    let start: number
    if (parallel.has(entry.node)) {
      if (i === 0 || !parallel.has(trace[i - 1].node)) {
        groupStart = cursor
        groupEnd = cursor
      }
      start = groupStart
      groupEnd = Math.max(groupEnd, groupStart + entry.ms)
      cursor = groupEnd
    } else {
      start = cursor
      cursor += entry.ms
    }
    return { entry, start }
  })
  const total = Math.max(cursor, 1)

  return (
    <div className="waterfall">
      {rows.map(({ entry, start }, i) => (
        <div className="wf-row" key={i} title={details(entry)}>
          <span className="wf-node">{entry.node}</span>
          <div className="wf-track">
            <div
              className="wf-bar"
              style={{
                left: `${(start / total) * 100}%`,
                width: `${Math.max((entry.ms / total) * 100, 0.6)}%`,
                background: NODE_COLORS[entry.node] ?? 'var(--accent)',
              }}
            />
          </div>
          <span className="wf-ms">{entry.ms} ms</span>
          <span className="wf-detail">{details(entry)}</span>
        </div>
      ))}
    </div>
  )
}
