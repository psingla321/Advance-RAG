import { useState } from 'react'
import type { ChatMessage } from '../types'
import { copyText } from '../platform'
import { RichText } from './RichText'
import { TraceWaterfall } from './TraceWaterfall'

export const ROUTE_LABELS: Record<string, string> = {
  A: 'Direct LLM',
  B: 'Document RAG',
  D: 'SQL analytics',
  E: 'Tool',
  H: 'Human',
}

function pct(value: number | null | undefined): string {
  return value == null ? '—' : `${Math.round(value * 100)}%`
}

function trustClass(value: number | null | undefined): string {
  if (value == null) return 'neutral'
  if (value >= 0.75) return 'good'
  if (value >= 0.5) return 'warn'
  return 'bad'
}

interface Props {
  message: ChatMessage
  onFeedback?: (message: ChatMessage, rating: 'up' | 'down') => void
}

export function MessageBubble({ message, onFeedback }: Props) {
  const [panel, setPanel] = useState<'none' | 'sources' | 'trace' | 'sql'>('none')
  const [copied, setCopied] = useState(false)

  if (message.role === 'user') {
    return (
      <div className="msg user">
        <div className="bubble">{message.content}</div>
      </div>
    )
  }

  const final = message.final
  const routing = final?.routing ?? message.routing
  const totalMs = message.trace.reduce((sum, t) => sum + (t.ms || 0), 0)
  const toggle = (p: typeof panel) => setPanel((current) => (current === p ? 'none' : p))

  return (
    <div className="msg assistant">
      <div className="avatar">N</div>
      <div className="bubble">
        {(routing || final) && (
          <div className="badges">
            {final?.cache && <span className="badge cache">⚡ {final.cache.level}</span>}
            {routing?.execute.map((r) => (
              <span key={r} className={`badge route route-${r}`}>{ROUTE_LABELS[r] ?? r}</span>
            ))}
            {routing?.mode === 'escalate' && <span className="badge route route-H">Human escalation</span>}
            {routing && <span className="badge muted" title={routing.explanation}>{routing.mode} · conf {pct(routing.confidence)}</span>}
            {final?.verification?.groundedness != null && (
              <span className="badge muted">
                grounded {pct(final.verification.groundedness)}
                {final.verification.repaired ? ' · repaired' : ''}
              </span>
            )}
            {final?.trust != null && <span className={`badge trust ${trustClass(final.trust)}`}>trust {pct(final.trust)}</span>}
            {final?.latency_ms != null && <span className="badge muted">{Math.round(final.latency_ms)} ms</span>}
          </div>
        )}

        {message.content ? (
          <RichText text={message.content} />
        ) : !message.error ? (
          <div className="thinking">
            <span /> <span /> <span />
            {message.trace.length > 0 && <em>{message.trace[message.trace.length - 1].node.replace(/_/g, ' ')}…</em>}
          </div>
        ) : null}
        {message.streaming && message.content && <span className="cursor" />}
        {message.error && <div className="error">{message.error}</div>}

        {!message.streaming && (final || message.trace.length > 0) && (
          <div className="actions">
            {final && final.citations.length > 0 && (
              <button className={`chip ${panel === 'sources' ? 'active' : ''}`} onClick={() => toggle('sources')}>
                📚 {final.citations.length} source{final.citations.length > 1 ? 's' : ''}
              </button>
            )}
            {final?.sql?.sql && (
              <button className={`chip ${panel === 'sql' ? 'active' : ''}`} onClick={() => toggle('sql')}>🗄 SQL</button>
            )}
            {message.trace.length > 0 && (
              <button className={`chip ${panel === 'trace' ? 'active' : ''}`} onClick={() => toggle('trace')}>
                ⏱ Trace · {Math.round(totalMs)} ms
              </button>
            )}
            <span className="spacer" />
            <button
              className="icon"
              title="Copy answer"
              onClick={() =>
                copyText(message.content).then((ok) => {
                  if (!ok) return
                  setCopied(true)
                  setTimeout(() => setCopied(false), 1200)
                })
              }
            >
              {copied ? '✓' : '⧉'}
            </button>
            {final?.request_id && onFeedback && (
              <>
                <button
                  className={`icon ${message.feedback === 'up' ? 'selected' : ''}`}
                  title="Helpful"
                  disabled={!!message.feedback}
                  onClick={() => onFeedback(message, 'up')}
                >
                  👍
                </button>
                <button
                  className={`icon ${message.feedback === 'down' ? 'selected' : ''}`}
                  title="Not helpful"
                  disabled={!!message.feedback}
                  onClick={() => onFeedback(message, 'down')}
                >
                  👎
                </button>
              </>
            )}
          </div>
        )}

        {panel === 'sources' && final && (
          <ol className="sources">
            {final.citations.map((c) => (
              <li key={c.n}>
                <div className="source-title">
                  <sup className="cite">{c.n}</sup> {c.filename}
                  {c.heading && <span className="muted"> › {c.heading}</span>}
                  <span className="score">{c.score.toFixed(2)}</span>
                </div>
                <div className="snippet">{c.snippet}</div>
              </li>
            ))}
          </ol>
        )}
        {panel === 'sql' && final?.sql?.sql && <pre className="sql">{final.sql.sql}</pre>}
        {panel === 'trace' && (
          <div className="trace-panel">
            <TraceWaterfall trace={message.trace} />
            {(final?.request_id || message.traceId) && (
              <div className="ids">
                {final?.request_id && <span>request <code>{final.request_id}</code></span>}
                {message.traceId && <span>trace <code>{message.traceId}</code></span>}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
