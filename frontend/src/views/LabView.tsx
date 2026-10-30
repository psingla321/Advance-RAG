import { useEffect, useMemo, useState } from 'react'
import { askLab, createLabSession, evaluateLab, labComponents, labPresets, runLabSweep, retrieveLab } from '../api'
import type { ComponentInfo, LabAnswer, LabPreset, LabRetrieval, LabSweepRun, SweepAxis } from '../types'

/** The stages a user assembles, in pipeline order. */
const STAGES = [
  { kind: 'chunker', label: 'Chunker', hint: 'How documents are split for indexing' },
  { kind: 'transform', label: 'Query transform', hint: 'Rewrite or expand the question before searching' },
  { kind: 'retriever', label: 'Retriever', hint: 'How candidates are found' },
  { kind: 'reranker', label: 'Reranker', hint: 'Second-pass ordering of candidates' },
  { kind: 'compressor', label: 'Compressor', hint: 'What fits in the context budget' },
  { kind: 'generator', label: 'Generator', hint: 'How the answer is written' },
] as const

const SAMPLE_CORPUS = `The nightly hotel limit is 200 USD in standard cities and 350 USD in high-cost cities such as London, New York, Tokyo and Zurich. Stays above the limit require written approval from a vice president.
---
Hybrid employees must work from the office at least two days per week. Fully remote arrangements require VP sign-off and a 500 USD home office stipend.
---
Passwords must be rotated every 90 days. Multi-factor authentication is mandatory for all administrative accounts.
---
The Software Engineer II salary band is 120,000 to 160,000 USD annually. Senior Engineer is 160,000 to 210,000 USD.`

const SAMPLE_GOLDEN = `What is the hotel limit in high-cost cities? | 350 USD
How many days per week must hybrid employees be in the office? | at least two days
How often must passwords be rotated? | every 90 days
What is the Senior Engineer salary band? | 160,000 to 210,000 USD`

function splitCorpus(raw: string): string[] {
  return raw
    .split(/\n---+\n/)
    .map((t) => t.trim())
    .filter(Boolean)
}

function parseGolden(raw: string): { question: string; ground_truth: string }[] {
  return raw
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const [question, ...rest] = line.split('|')
      return { question: question.trim(), ground_truth: rest.join('|').trim() }
    })
    .filter((s) => s.question)
}

function ScoreBar({ value }: { value: number }) {
  const tone = value >= 0.75 ? 'good' : value >= 0.5 ? 'warn' : 'bad'
  return (
    <div className="lab-score">
      <div className="bar-track">
        <div className={`bar-fill ${tone}`} style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} />
      </div>
      <span className="bar-value">{value.toFixed(3)}</span>
    </div>
  )
}

export function LabView() {
  const [catalogue, setCatalogue] = useState<Record<string, ComponentInfo[]>>({})
  const [presets, setPresets] = useState<LabPreset[]>([])
  const [preset, setPreset] = useState('hybrid')
  const [choices, setChoices] = useState<Record<string, string>>({})
  const [topK, setTopK] = useState(30)
  const [topN, setTopN] = useState(8)

  const [corpus, setCorpus] = useState(SAMPLE_CORPUS)
  const [golden, setGolden] = useState(SAMPLE_GOLDEN)
  const [question, setQuestion] = useState('What is the hotel limit in high-cost cities?')

  const [tab, setTab] = useState<'ask' | 'retrieve' | 'evaluate' | 'sweep'>('ask')
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [answer, setAnswer] = useState<LabAnswer | null>(null)
  const [retrieval, setRetrieval] = useState<LabRetrieval | null>(null)
  const [report, setReport] = useState<any | null>(null)
  const [sweep, setSweep] = useState<LabSweepRun[] | null>(null)
  const [axes, setAxes] = useState<SweepAxis[]>([{ key: 'retriever.type', values: 'dense,bm25,hybrid_rrf,mmr' }])

  useEffect(() => {
    labComponents()
      .then(setCatalogue)
      .catch((e) => setError((e as Error).message))
    labPresets()
      .then((items) => {
        setPresets(items)
        const chosen = items.find((p) => p.name === 'hybrid') ?? items[0]
        if (chosen) applyPreset(chosen)
      })
      .catch((e) => setError((e as Error).message))
  }, [])

  function applyPreset(item: LabPreset) {
    setPreset(item.name)
    const next: Record<string, string> = {}
    for (const stage of STAGES) {
      const spec = (item.config as any)[stage.kind]
      next[stage.kind] = typeof spec === 'string' ? spec : spec?.type ?? ''
    }
    setChoices(next)
    setTopK(item.config.top_k ?? 30)
    setTopN(item.config.top_n ?? 8)
  }

  const overrides = useMemo(() => {
    const out: Record<string, unknown> = { top_k: topK, top_n: topN }
    for (const [kind, type] of Object.entries(choices)) if (type) out[`${kind}.type`] = type
    return out
  }, [choices, topK, topN])

  /** Every action builds a fresh session and re-indexes, so results always match what is on screen. */
  async function withSession<T>(label: string, fn: (sessionId: string) => Promise<T>): Promise<T | null> {
    setBusy(label)
    setError(null)
    try {
      const texts = splitCorpus(corpus)
      if (!texts.length) throw new Error('Add at least one passage to the corpus.')
      const sessionId = await createLabSession(preset, overrides, texts)
      return await fn(sessionId)
    } catch (e) {
      setError((e as Error).message)
      return null
    } finally {
      setBusy(null)
    }
  }

  async function runAsk() {
    setAnswer(null)
    await withSession('ask', async (sessionId) => {
      const result = await askLab(sessionId, question)
      setAnswer(result)
      setTab('ask')
      return result
    })
  }

  async function runRetrieve() {
    setRetrieval(null)
    await withSession('retrieve', async (sessionId) => {
      const result = await retrieveLab(sessionId, question)
      setRetrieval(result)
      setTab('retrieve')
      return result
    })
  }

  async function runEvaluate() {
    setReport(null)
    const samples = parseGolden(golden)
    if (!samples.length) {
      setError('Add at least one golden sample (one per line: question | expected answer).')
      return
    }
    await withSession('evaluate', async (sessionId) => {
      const result = await evaluateLab(sessionId, samples)
      setReport(result)
      setTab('evaluate')
      return result
    })
  }

  async function runSweep() {
    setSweep(null)
    setBusy('sweep')
    setError(null)
    try {
      const grid: Record<string, unknown[]> = {}
      for (const axis of axes) {
        const values = axis.values.split(',').map((v) => v.trim()).filter(Boolean)
        if (axis.key.trim() && values.length) {
          grid[axis.key.trim()] = values.map((v) => (/^-?\d+(\.\d+)?$/.test(v) ? Number(v) : v))
        }
      }
      if (!Object.keys(grid).length) throw new Error('Add at least one sweep axis.')
      const samples = parseGolden(golden)
      if (!samples.length) throw new Error('A sweep needs golden samples to rank configurations.')
      const result = await runLabSweep(preset, grid, samples, splitCorpus(corpus))
      setSweep(result.runs)
      setTab('sweep')
      if (result.failures.length) {
        setError(`${result.failures.length} configuration(s) failed: ${result.failures[0].error}`)
      }
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(null)
    }
  }

  const metricNames = useMemo(() => {
    if (!sweep?.length) return []
    return Object.keys(sweep[0].scores)
  }, [sweep])

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h2>RAG Lab</h2>
          <p className="muted">
            Assemble a pipeline, ask it something, then measure it. Every change here maps to one line of a
            YAML config — nothing is UI-only.
          </p>
        </div>
        <div className="system-pills">
          <span className="pill">{Object.values(catalogue).reduce((n, items) => n + items.length, 0)} components</span>
          <span className="pill">{presets.length} presets</span>
        </div>
      </div>

      {error && <div className="error">{error}</div>}

      <div className="grid-2">
        {/* ------------------------------------------------ pipeline builder */}
        <div className="card">
          <div className="card-head row">
            <span>Pipeline</span>
            <select value={preset} onChange={(e) => {
              const item = presets.find((p) => p.name === e.target.value)
              if (item) applyPreset(item)
            }}>
              {presets.map((p) => <option key={p.name} value={p.name}>{p.name}</option>)}
            </select>
          </div>
          <p className="muted small">{presets.find((p) => p.name === preset)?.description}</p>

          <div className="lab-stages">
            {STAGES.map((stage) => {
              const options = catalogue[stage.kind] ?? []
              const selected = options.find((o) => o.name === choices[stage.kind])
              return (
                <label className="lab-stage" key={stage.kind}>
                  <span className="lab-stage-label">{stage.label}</span>
                  <select
                    value={choices[stage.kind] ?? ''}
                    onChange={(e) => setChoices({ ...choices, [stage.kind]: e.target.value })}
                  >
                    {options.map((o) => <option key={o.name} value={o.name}>{o.name}</option>)}
                  </select>
                  <span className="lab-stage-hint muted small">{selected?.summary ?? stage.hint}</span>
                  {selected?.requires?.length ? (
                    <span className="badge muted">needs {selected.requires.join(', ')}</span>
                  ) : null}
                </label>
              )
            })}
            <div className="lab-knobs">
              <label>top_k<input type="number" min={1} max={200} value={topK} onChange={(e) => setTopK(+e.target.value)} /></label>
              <label>top_n<input type="number" min={1} max={50} value={topN} onChange={(e) => setTopN(+e.target.value)} /></label>
            </div>
          </div>
        </div>

        {/* ------------------------------------------------------ input data */}
        <div className="card">
          <div className="card-head">Corpus</div>
          <p className="muted small">One passage per block, separated by a line containing <code>---</code>.</p>
          <textarea className="lab-text" rows={8} value={corpus} onChange={(e) => setCorpus(e.target.value)} />

          <div className="card-head" style={{ marginTop: 14 }}>Golden set</div>
          <p className="muted small">One per line: <code>question | expected answer</code>.</p>
          <textarea className="lab-text" rows={5} value={golden} onChange={(e) => setGolden(e.target.value)} />
        </div>
      </div>

      {/* ------------------------------------------------------------ actions */}
      <div className="card wide">
        <div className="lab-actions">
          <input
            className="lab-question"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask the pipeline something…"
            onKeyDown={(e) => e.key === 'Enter' && runAsk()}
          />
          <button className="primary" disabled={!!busy} onClick={runAsk}>{busy === 'ask' ? 'Asking…' : 'Ask'}</button>
          <button disabled={!!busy} onClick={runRetrieve}>{busy === 'retrieve' ? 'Retrieving…' : 'Retrieve only'}</button>
          <button disabled={!!busy} onClick={runEvaluate}>{busy === 'evaluate' ? 'Evaluating…' : 'Evaluate'}</button>
          <button disabled={!!busy} onClick={runSweep}>{busy === 'sweep' ? 'Sweeping…' : 'Run sweep'}</button>
        </div>

        <div className="lab-tabs">
          {(['ask', 'retrieve', 'evaluate', 'sweep'] as const).map((t) => (
            <button key={t} className={`lab-tab ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}>{t}</button>
          ))}
        </div>

        {/* ------------------------------------------------------------ ask */}
        {tab === 'ask' && (
          answer ? (
            <div className="lab-result">
              <p className="lab-answer">{answer.answer || '(no answer)'}</p>
              <div className="badges">
                <span className="badge">{answer.latency_ms.toFixed(0)} ms</span>
                {answer.groundedness != null && (
                  <span className={`badge trust ${answer.groundedness >= 0.8 ? 'good' : 'warn'}`}>
                    groundedness {answer.groundedness.toFixed(2)}
                  </span>
                )}
                {answer.abstained && <span className="badge trust warn">abstained</span>}
                {answer.expanded_queries.length > 1 && (
                  <span className="badge">{answer.expanded_queries.length} query variants</span>
                )}
              </div>
              {answer.citations.length > 0 && (
                <ul className="lab-citations">
                  {answer.citations.map((c) => (
                    <li key={c.n}><b>[{c.n}]</b> {c.filename}{c.heading ? ` — ${c.heading}` : ''} <span className="muted small">({c.score})</span></li>
                  ))}
                </ul>
              )}
              <div className="table-wrap">
                <table>
                  <thead><tr><th>stage</th><th>ms</th><th>detail</th></tr></thead>
                  <tbody>
                    {answer.trace.map((stage: any, i: number) => (
                      <tr key={i}>
                        <td>{stage.stage}</td>
                        <td>{Number(stage.ms).toFixed(1)}</td>
                        <td className="muted small">
                          {Object.entries(stage).filter(([k]) => k !== 'stage' && k !== 'ms')
                            .map(([k, v]) => `${k}=${Array.isArray(v) ? v.length : v}`).join('  ')}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : <p className="muted">Ask a question to see the answer, its citations and per-stage timings.</p>
        )}

        {/* -------------------------------------------------------- retrieve */}
        {tab === 'retrieve' && (
          retrieval ? (
            <div className="lab-result">
              <p className="muted small">Searched: {retrieval.queries.map((q, i) => <code key={i}>{q}</code>)}</p>
              <div className="table-wrap">
                <table>
                  <thead><tr><th>#</th><th>score</th><th>signals</th><th>source</th><th>text</th></tr></thead>
                  <tbody>
                    {retrieval.contexts.map((c) => (
                      <tr key={c.n}>
                        <td>{c.n}</td>
                        <td>{c.score.toFixed(3)}</td>
                        <td className="muted small">
                          {Object.entries(c.scores).map(([k, v]) => `${k}=${typeof v === 'number' ? v.toFixed(2) : v}`).join(' ')}
                        </td>
                        <td className="muted small">{c.filename}{c.heading ? ` — ${c.heading}` : ''}</td>
                        <td className="lab-snippet">{c.text.slice(0, 160)}…</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : <p className="muted">Retrieval only — see exactly what the generator would have been given, and why.</p>
        )}

        {/* -------------------------------------------------------- evaluate */}
        {tab === 'evaluate' && (
          report ? (
            <div className="lab-result">
              <div className="kpis">
                <div className="kpi good">
                  <div className="kpi-label">Overall</div>
                  <div className="kpi-value">{report.overall.toFixed(3)}</div>
                  <div className="kpi-hint">mean of {Object.keys(report.scores).length} metrics · n={report.n}</div>
                </div>
                <div className="kpi">
                  <div className="kpi-label">P95 latency</div>
                  <div className="kpi-value">{Math.round(report.latency.p95)} ms</div>
                  <div className="kpi-hint">p50 {Math.round(report.latency.p50)} ms</div>
                </div>
              </div>
              <div className="table-wrap">
                <table>
                  <thead><tr><th>metric</th><th>score</th></tr></thead>
                  <tbody>
                    {Object.entries(report.scores as Record<string, number>)
                      .sort((a, b) => b[1] - a[1])
                      .map(([name, value]) => (
                        <tr key={name}><td>{name}</td><td><ScoreBar value={value} /></td></tr>
                      ))}
                  </tbody>
                </table>
              </div>
              <div className="card-head" style={{ marginTop: 12 }}>Weakest samples</div>
              <p className="muted small">Read these first — they tell you what to change.</p>
              <div className="table-wrap">
                <table>
                  <thead><tr><th>question</th><th>answer</th><th>expected</th></tr></thead>
                  <tbody>
                    {[...report.samples]
                      .sort((a: any, b: any) => {
                        const mean = (s: any) => Object.values(s.scores as Record<string, number>).reduce((x, y) => x + y, 0) / Math.max(1, Object.keys(s.scores).length)
                        return mean(a) - mean(b)
                      })
                      .slice(0, 3)
                      .map((s: any, i: number) => (
                        <tr key={i}>
                          <td className="lab-snippet">{s.question}</td>
                          <td className="lab-snippet">{s.answer.slice(0, 120)}</td>
                          <td className="lab-snippet muted">{s.ground_truth}</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : <p className="muted">Score this pipeline against the golden set.</p>
        )}

        {/* ------------------------------------------------------------ sweep */}
        {tab === 'sweep' && (
          <div className="lab-result">
            <div className="lab-axes">
              {axes.map((axis, i) => (
                <div className="lab-axis" key={i}>
                  <input
                    placeholder="retriever.type"
                    value={axis.key}
                    onChange={(e) => setAxes(axes.map((a, j) => (j === i ? { ...a, key: e.target.value } : a)))}
                  />
                  <input
                    placeholder="dense,bm25,hybrid_rrf"
                    value={axis.values}
                    onChange={(e) => setAxes(axes.map((a, j) => (j === i ? { ...a, values: e.target.value } : a)))}
                  />
                  <button className="icon" onClick={() => setAxes(axes.filter((_, j) => j !== i))}>✕</button>
                </div>
              ))}
              <button className="icon small" onClick={() => setAxes([...axes, { key: '', values: '' }])}>+ axis</button>
            </div>
            {sweep ? (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr><th>configuration</th>{metricNames.map((m) => <th key={m}>{m}</th>)}<th>overall</th><th>p95</th></tr>
                  </thead>
                  <tbody>
                    {sweep.map((run, i) => (
                      <tr key={run.name} className={i === 0 ? 'lab-best' : ''}>
                        <td>{run.name}</td>
                        {metricNames.map((m) => <td key={m}>{run.scores[m]?.toFixed(3) ?? '—'}</td>)}
                        <td><b>{run.overall.toFixed(3)}</b></td>
                        <td className="muted small">{Math.round(run.latency.p95)} ms</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="muted">
                Each axis multiplies the number of runs, and every run re-indexes the corpus. Start with one axis.
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
