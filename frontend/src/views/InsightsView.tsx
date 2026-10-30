import { useEffect, useState } from 'react'
import { getInsights } from '../api'
import { ROUTE_LABELS } from '../components/MessageBubble'
import type { Identity, Insights } from '../types'

function pct(value: number | null | undefined): string {
  return value == null ? '—' : `${Math.round(value * 100)}%`
}

function Kpi({ label, value, hint, tone }: { label: string; value: string; hint?: string; tone?: string }) {
  return (
    <div className={`kpi ${tone ?? ''}`}>
      <div className="kpi-label">{label}</div>
      <div className="kpi-value">{value}</div>
      {hint && <div className="kpi-hint">{hint}</div>}
    </div>
  )
}

function Bars({ data, labels }: { data: Record<string, number>; labels?: Record<string, string> }) {
  const entries = Object.entries(data).sort((a, b) => b[1] - a[1])
  const max = Math.max(1, ...entries.map(([, v]) => v))
  if (!entries.length) return <p className="muted">No data yet.</p>
  return (
    <div className="bars">
      {entries.map(([k, v]) => (
        <div className="bar-row" key={k}>
          <span className="bar-label">{labels?.[k] ?? k}</span>
          <div className="bar-track"><div className="bar-fill" style={{ width: `${(v / max) * 100}%` }} /></div>
          <span className="bar-value">{v}</span>
        </div>
      ))}
    </div>
  )
}

function Sparkline({ points }: { points: number[] }) {
  const max = Math.max(1, ...points)
  const w = 600
  const h = 80
  const step = w / Math.max(1, points.length - 1)
  const path = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${(i * step).toFixed(1)},${(h - (p / max) * (h - 6) - 3).toFixed(1)}`).join(' ')
  return (
    <svg className="sparkline" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
      <path d={`${path} L${w},${h} L0,${h} Z`} className="spark-area" />
      <path d={path} className="spark-line" />
    </svg>
  )
}

export function InsightsView({ identity }: { identity: Identity }) {
  const [data, setData] = useState<Insights | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let active = true
    const load = () =>
      getInsights(identity)
        .then((d) => active && (setData(d), setError(null)))
        .catch((e) => active && setError((e as Error).message))
    load()
    const timer = setInterval(load, 5000)
    return () => {
      active = false
      clearInterval(timer)
    }
  }, [identity])

  if (error && !data) return <div className="page"><div className="error">{error}</div></div>
  if (!data) return <div className="page"><p className="muted">Loading insights…</p></div>

  const feedbackTotal = data.feedback.up + data.feedback.down
  const loadPct = data.system.max_inflight ? data.system.inflight / data.system.max_inflight : 0

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h2>Insights</h2>
          <p className="muted">Live engine analytics for tenant <code>{identity.tenantId}</code> · refreshes every 5 s</p>
        </div>
        <div className="system-pills">
          <span className={`pill ${data.system.mock_llm ? 'warn' : 'good'}`}>{data.system.mock_llm ? 'Mock LLM' : 'Azure OpenAI'}</span>
          <span className="pill">{data.system.embedded_worker ? 'Embedded worker' : 'Worker pool'}</span>
          <span className={`pill ${data.system.queue_depth > 50 ? 'warn' : ''}`}>Queue {data.system.queue_depth}</span>
          <span className={`pill ${loadPct > 0.8 ? 'bad' : ''}`}>In-flight {data.system.inflight}/{data.system.max_inflight}</span>
        </div>
      </div>

      <div className="kpis">
        <Kpi label="Requests" value={data.total.toLocaleString()} />
        <Kpi label="P50 latency" value={`${Math.round(data.latency.p50)} ms`} hint={`P95 ${Math.round(data.latency.p95)} ms · P99 ${Math.round(data.latency.p99)} ms`}
          tone={data.latency.p95 > 5000 ? 'bad' : data.latency.p95 > 2000 ? 'warn' : 'good'} />
        <Kpi label="Cache hit rate" value={pct(data.cache_hit_rate)} tone={data.cache_hit_rate >= 0.3 ? 'good' : undefined} />
        <Kpi label="Avg trust" value={pct(data.avg_trust)} hint={`groundedness ${pct(data.avg_groundedness)}`}
          tone={data.avg_trust == null ? undefined : data.avg_trust >= 0.75 ? 'good' : 'warn'} />
        <Kpi label="Escalations" value={pct(data.escalation_rate)} hint={`abstained ${pct(data.abstention_rate)} · blocked ${data.blocked}`} />
        <Kpi label="Satisfaction" value={feedbackTotal ? pct(data.feedback.up / feedbackTotal) : '—'} hint={`👍 ${data.feedback.up} · 👎 ${data.feedback.down}`} />
      </div>

      <div className="card wide">
        <div className="card-head">Requests per minute (last 60 min)</div>
        <Sparkline points={data.timeline.map((t) => t.count)} />
      </div>

      <div className="grid-2">
        <div className="card">
          <div className="card-head">Routes executed</div>
          <Bars data={data.routes} labels={ROUTE_LABELS} />
        </div>
        <div className="card">
          <div className="card-head">Execution modes</div>
          <Bars data={data.modes} />
        </div>
        <div className="card">
          <div className="card-head">Average time per engine node</div>
          <Bars data={Object.fromEntries(data.nodes.map((n) => [n.node, n.avg_ms]))} />
        </div>
        <div className="card">
          <div className="card-head">Latency distribution</div>
          <Bars data={{ p50: data.latency.p50, p95: data.latency.p95, p99: data.latency.p99 }} />
          <p className="muted small">{data.latency.samples} samples · targets P50 &lt; 2 s, P95 &lt; 5 s</p>
        </div>
      </div>

      <div className="card wide">
        <div className="card-head">Recent requests</div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Time</th><th>Query</th><th>Mode</th><th>Routes</th><th>Outcome</th><th>Latency</th><th>Trust</th></tr>
            </thead>
            <tbody>
              {data.recent.map((r) => (
                <tr key={r.request_id}>
                  <td className="nowrap">{new Date(r.ts).toLocaleTimeString()}</td>
                  <td className="truncate" title={r.query}>{r.query}</td>
                  <td>{r.cache_hit ? <span className="badge cache">cache</span> : r.mode}</td>
                  <td>{r.routes.map((x) => ROUTE_LABELS[x] ?? x).join(', ')}</td>
                  <td><span className={`outcome ${r.outcome}`}>{r.outcome}</span></td>
                  <td className="nowrap">{Math.round(r.latency_ms)} ms</td>
                  <td>{pct(r.trust)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted small">
          Deep dives: Grafana <code>:3000</code> (metrics), Jaeger <code>:16686</code> (traces), Prometheus <code>/metrics</code>.
        </p>
      </div>
    </div>
  )
}
