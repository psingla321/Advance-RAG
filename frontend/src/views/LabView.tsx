import { useEffect, useMemo, useState } from 'react'
import {
  createLabSession, evaluateLab, explainLab, labComponents, labPresets, playgroundChunk,
  playgroundCompare, playgroundGuard, playgroundTransform, retrieveLab, runLabSweep,
} from '../api'
import type {
  ComponentInfo, LabChunkRun, LabCompareVariant, LabExplain, LabPreset, LabRetrieval, LabSweepRun,
  LabTransformRun, StageChunk, SweepAxis,
} from '../types'

/** The stages a user assembles, in pipeline order, with why each one exists. */
const STAGES = [
  { kind: 'chunker', label: 'Chunker', hint: 'Splits documents for indexing. Too big and the answer drowns in noise; too small and it loses context.' },
  { kind: 'transform', label: 'Query transform', hint: 'Rewrites the question before searching. Fixes the gap between how users ask and how documents are written.' },
  { kind: 'retriever', label: 'Retriever', hint: 'Finds candidates. Dense catches meaning, BM25 catches exact words, hybrid catches both.' },
  { kind: 'reranker', label: 'Reranker', hint: 'Re-orders candidates with a sharper, slower model. Cheap retrieval casts wide, rerank picks the winners.' },
  { kind: 'compressor', label: 'Compressor', hint: 'Fits the survivors into the token budget. Decides what the model actually gets to read.' },
  { kind: 'generator', label: 'Generator', hint: 'Writes the answer from the context, and cites it.' },
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

const CHUNKER_CHOICES = ['fixed', 'recursive', 'sentence', 'semantic', 'markdown_header', 'sentence_window', 'parent_child']
const TRANSFORM_CHOICES = ['identity', 'multi_query', 'hyde', 'step_back', 'decompose']

type Tab = 'xray' | 'retrieve' | 'evaluate' | 'sweep' | 'chunking' | 'transforms' | 'compare'

const TABS: { id: Tab; label: string; blurb: string }[] = [
  { id: 'xray', label: 'X-ray', blurb: 'One question, every stage: what was found, what survived, what the model read.' },
  { id: 'retrieve', label: 'Retrieve', blurb: 'Retrieval only — inspect the context without paying for generation.' },
  { id: 'evaluate', label: 'Evaluate', blurb: 'Score this pipeline against the golden set.' },
  { id: 'sweep', label: 'Sweep', blurb: 'Run many configurations and rank them.' },
  { id: 'chunking', label: 'Chunking', blurb: 'Split the same document several ways and compare the pieces.' },
  { id: 'transforms', label: 'Query rewriting', blurb: 'See the search queries each technique produces.' },
  { id: 'compare', label: 'A/B compare', blurb: 'Change one knob, run the same question both ways.' },
]

function splitCorpus(raw: string): string[] {
  return raw.split(/\n---+\n/).map((t) => t.trim()).filter(Boolean)
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

/** Ranked candidates at one point in the pipeline. */
function StageTable({ rows, limit = 8 }: { rows: StageChunk[]; limit?: number }) {
  if (!rows.length) return <p className="lab-note">Nothing survived this stage.</p>
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr><th>#</th><th>score</th><th>signals</th><th>tokens</th><th>source</th><th>text</th></tr>
        </thead>
        <tbody>
          {rows.slice(0, limit).map((c) => (
            <tr key={c.id}>
              <td>{c.rank}</td>
              <td>{c.score.toFixed(3)}</td>
              <td className="muted small">
                {Object.entries(c.scores).map(([k, v]) => `${k}=${v.toFixed(2)}`).join(' ') || '—'}
              </td>
              <td className="muted small">{c.tokens}</td>
              <td className="muted small">{c.filename}{c.heading ? ` — ${c.heading}` : ''}</td>
              <td className="lab-snippet">{c.text.slice(0, 150)}{c.text.length > 150 ? '…' : ''}</td>
            </tr>
          ))}
        </tbody>
      </table>
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

  const [tab, setTab] = useState<Tab>('xray')
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const [xray, setXray] = useState<LabExplain | null>(null)
  const [retrieval, setRetrieval] = useState<LabRetrieval | null>(null)
  const [report, setReport] = useState<any | null>(null)
  const [sweep, setSweep] = useState<LabSweepRun[] | null>(null)
  const [axes, setAxes] = useState<SweepAxis[]>([{ key: 'retriever.type', values: 'dense,bm25,hybrid_rrf,mmr' }])

  const [chunkPicks, setChunkPicks] = useState<string[]>(['fixed', 'sentence', 'semantic'])
  const [chunkRuns, setChunkRuns] = useState<LabChunkRun[] | null>(null)
  const [transformRuns, setTransformRuns] = useState<{ offline: boolean; transforms: LabTransformRun[] } | null>(null)
  const [guard, setGuard] = useState<{ blocked: boolean; injection_patterns: string[]; pii: Record<string, string[]> } | null>(null)
  const [compareStage, setCompareStage] = useState('retriever')
  const [compareValues, setCompareValues] = useState('dense,bm25')
  const [variants, setVariants] = useState<LabCompareVariant[] | null>(null)

  useEffect(() => {
    labComponents().then(setCatalogue).catch((e) => setError((e as Error).message))
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

  /** Wrapper for the stateless playground probes, which need no session. */
  async function run<T>(label: string, fn: () => Promise<T>): Promise<void> {
    setBusy(label)
    setError(null)
    try {
      await fn()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(null)
    }
  }

  async function runXray() {
    setXray(null)
    await withSession('xray', async (sessionId) => {
      const result = await explainLab(sessionId, question)
      setXray(result)
      setTab('xray')
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
    await run('sweep', async () => {
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
    })
  }

  async function runChunking() {
    setChunkRuns(null)
    await run('chunking', async () => {
      if (!chunkPicks.length) throw new Error('Pick at least one chunker to compare.')
      const text = splitCorpus(corpus).join('\n\n')
      if (!text) throw new Error('Add some text to the corpus first.')
      setChunkRuns(await playgroundChunk(text, chunkPicks))
      setTab('chunking')
    })
  }

  async function runTransforms() {
    setTransformRuns(null)
    setGuard(null)
    await run('transforms', async () => {
      const [t, g] = await Promise.all([
        playgroundTransform(question, TRANSFORM_CHOICES),
        playgroundGuard(question),
      ])
      setTransformRuns(t)
      setGuard(g)
      setTab('transforms')
    })
  }

  async function runCompare() {
    setVariants(null)
    await run('compare', async () => {
      const values = compareValues.split(',').map((v) => v.trim()).filter(Boolean)
      if (values.length < 2) throw new Error('Give at least two comma-separated values to compare.')
      const texts = splitCorpus(corpus)
      if (!texts.length) throw new Error('Add at least one passage to the corpus.')
      const built = values.slice(0, 4).map((v) => ({ label: `${compareStage}=${v}`, overrides: { [`${compareStage}.type`]: v } }))
      setVariants(await playgroundCompare(preset, texts, question, built))
      setTab('compare')
    })
  }

  const metricNames = useMemo(() => (sweep?.length ? Object.keys(sweep[0].scores) : []), [sweep])
  const activeTab = TABS.find((t) => t.id === tab)

  const funnel = useMemo(() => {
    if (!xray?.stages) return []
    return [
      { key: 'retrieve', title: 'Retrieved', note: `top_k = ${xray.pipeline?.knobs?.top_k ?? topK}`, rows: xray.stages.retrieve ?? [] },
      { key: 'rerank', title: 'Reranked', note: `top_n = ${xray.pipeline?.knobs?.top_n ?? topN}`, rows: xray.stages.rerank ?? [] },
      { key: 'compress', title: 'Sent to the model', note: `${xray.budget ?? ''} token budget`, rows: xray.stages.compress ?? [] },
    ]
  }, [xray, topK, topN])

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
            <select
              value={preset}
              onChange={(e) => {
                const item = presets.find((p) => p.name === e.target.value)
                if (item) applyPreset(item)
              }}
            >
              {presets.map((p) => <option key={p.name} value={p.name}>{p.name}</option>)}
            </select>
          </div>
          <p className="lab-note">{presets.find((p) => p.name === preset)?.description}</p>

          <div className="lab-stages">
            {STAGES.map((stage, index) => {
              const options = catalogue[stage.kind] ?? []
              const selected = options.find((o) => o.name === choices[stage.kind])
              return (
                <div className="lab-stage" key={stage.kind}>
                  <span className="lab-stage-num">{index + 1}</span>
                  <div className="lab-stage-top">
                    <span className="lab-stage-label">{stage.label}</span>
                    <select
                      value={choices[stage.kind] ?? ''}
                      onChange={(e) => setChoices({ ...choices, [stage.kind]: e.target.value })}
                    >
                      {options.map((o) => <option key={o.name} value={o.name}>{o.name}</option>)}
                    </select>
                    {selected?.requires?.length ? (
                      <span className="badge muted">needs {selected.requires.join(', ')}</span>
                    ) : null}
                  </div>
                  <span className="lab-stage-hint">{selected?.summary ?? stage.hint}</span>
                </div>
              )
            })}
          </div>

          <div className="lab-knobs">
            <label>
              top_k — candidates retrieved
              <input type="number" min={1} max={200} value={topK} onChange={(e) => setTopK(+e.target.value)} />
            </label>
            <label>
              top_n — kept after rerank
              <input type="number" min={1} max={50} value={topN} onChange={(e) => setTopN(+e.target.value)} />
            </label>
          </div>
        </div>

        {/* ------------------------------------------------------ input data */}
        <div className="card">
          <div className="lab-field">
            <div className="card-head" style={{ marginBottom: 2 }}>Corpus</div>
            <p className="lab-note">One passage per block, separated by a line containing <code>---</code>.</p>
            <textarea className="lab-text" rows={9} value={corpus} onChange={(e) => setCorpus(e.target.value)} />
          </div>

          <div className="lab-field" style={{ marginTop: 14 }}>
            <div className="card-head" style={{ marginBottom: 2 }}>Golden set</div>
            <p className="lab-note">One per line: <code>question | expected answer</code>. Used by Evaluate and Sweep.</p>
            <textarea className="lab-text" rows={5} value={golden} onChange={(e) => setGolden(e.target.value)} />
          </div>
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
            onKeyDown={(e) => e.key === 'Enter' && runXray()}
          />
          <button className="primary" disabled={!!busy} onClick={runXray}>{busy === 'xray' ? 'Running…' : 'Run X-ray'}</button>
          <button disabled={!!busy} onClick={runRetrieve}>{busy === 'retrieve' ? 'Retrieving…' : 'Retrieve only'}</button>
          <button disabled={!!busy} onClick={runEvaluate}>{busy === 'evaluate' ? 'Evaluating…' : 'Evaluate'}</button>
          <button disabled={!!busy} onClick={runSweep}>{busy === 'sweep' ? 'Sweeping…' : 'Run sweep'}</button>
        </div>

        <div className="lab-tabs">
          {TABS.map((t) => (
            <button key={t.id} className={`lab-tab ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          ))}
        </div>
        {activeTab && <p className="lab-note" style={{ marginBottom: 10 }}>{activeTab.blurb}</p>}

        {/* ------------------------------------------------------------ X-ray */}
        {tab === 'xray' && (
          xray ? (
            <div className="lab-result">
              {xray.blocked ? (
                <div className="lab-explain-box">
                  <b>Blocked by guardrails.</b> The question matched a prompt-injection pattern, so no retrieval
                  or generation ran at all. {xray.answer}
                </div>
              ) : (
                <>
                  <div className="lab-funnel">
                    {funnel.map((step) => (
                      <div className="lab-funnel-step" key={step.key}>
                        <h4>{step.title}</h4>
                        <div className="lab-funnel-count">
                          {step.rows.length}<span>chunks</span>
                        </div>
                        <div className="muted small">{step.note}</div>
                      </div>
                    ))}
                  </div>

                  {xray.queries && xray.queries.length > 1 && (
                    <div className="lab-explain-box">
                      <b>The question was expanded into {xray.queries.length} searches.</b> Each one is run and the
                      results are fused, so a document phrased differently from the question can still be found.
                      <div className="lab-chips">
                        {xray.queries.map((q, i) => <code key={i}>{q}</code>)}
                      </div>
                    </div>
                  )}

                  <p className="lab-answer">{xray.answer || '(no answer)'}</p>
                  <div className="badges">
                    <span className="badge">{xray.latency_ms.toFixed(0)} ms</span>
                    {xray.groundedness != null && (
                      <span className={`badge trust ${xray.groundedness >= 0.8 ? 'good' : 'warn'}`}>
                        groundedness {xray.groundedness.toFixed(2)}
                      </span>
                    )}
                    {xray.abstained && <span className="badge trust warn">abstained</span>}
                  </div>

                  {xray.citations.length > 0 && (
                    <ul className="lab-citations">
                      {xray.citations.map((c) => (
                        <li key={c.n}>
                          <b>[{c.n}]</b> {c.filename}{c.heading ? ` — ${c.heading}` : ''}
                          <span className="muted small"> ({c.score})</span>
                        </li>
                      ))}
                    </ul>
                  )}

                  {funnel.map((step) => (
                    <div key={step.key}>
                      <div className="card-head" style={{ marginTop: 6 }}>{step.title} — {step.rows.length} chunks</div>
                      <StageTable rows={step.rows} />
                    </div>
                  ))}

                  <div className="card-head" style={{ marginTop: 6 }}>Stage timings</div>
                  <div className="table-wrap">
                    <table>
                      <thead><tr><th>stage</th><th>ms</th><th>detail</th></tr></thead>
                      <tbody>
                        {xray.trace.map((stage: any, i: number) => (
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
                </>
              )}
            </div>
          ) : <p className="lab-note">Run the X-ray to see how many candidates each stage kept, and which ones reached the model.</p>
        )}

        {/* -------------------------------------------------------- retrieve */}
        {tab === 'retrieve' && (
          retrieval ? (
            <div className="lab-result">
              <p className="lab-note">Searched: {retrieval.queries.map((q, i) => <code key={i}>{q}</code>)}</p>
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
          ) : <p className="lab-note">Retrieval only — see exactly what the generator would have been given, and why.</p>
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
              <p className="lab-note">Read these first — they tell you what to change.</p>
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
          ) : <p className="lab-note">Score this pipeline against the golden set.</p>
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
              <div><button className="icon small" onClick={() => setAxes([...axes, { key: '', values: '' }])}>+ axis</button></div>
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
              <p className="lab-note">
                Each axis multiplies the number of runs, and every run re-indexes the corpus. Start with one axis.
              </p>
            )}
          </div>
        )}

        {/* --------------------------------------------------------- chunking */}
        {tab === 'chunking' && (
          <div className="lab-result">
            <div className="lab-actions">
              {CHUNKER_CHOICES.map((name) => (
                <button
                  key={name}
                  className={`chip ${chunkPicks.includes(name) ? 'active' : ''}`}
                  onClick={() => setChunkPicks(
                    chunkPicks.includes(name) ? chunkPicks.filter((c) => c !== name) : [...chunkPicks, name],
                  )}
                >
                  {name}
                </button>
              ))}
              <button className="primary" disabled={!!busy} onClick={runChunking}>
                {busy === 'chunking' ? 'Splitting…' : 'Split corpus'}
              </button>
            </div>
            {chunkRuns ? (
              <div className="lab-chunk-grid">
                {chunkRuns.map((runResult) => (
                  <div className="lab-chunk-col" key={runResult.chunker}>
                    <div className="card-head">{runResult.chunker}</div>
                    {runResult.error ? (
                      <p className="error small">{runResult.error}</p>
                    ) : (
                      <>
                        <p className="lab-note">
                          {runResult.stats?.count} chunks · avg {runResult.stats?.avg_tokens} tokens
                          {' '}(min {runResult.stats?.min_tokens}, max {runResult.stats?.max_tokens})
                        </p>
                        <div className="lab-chunk-list">
                          {runResult.chunks.map((c) => (
                            <div className="lab-chunk" key={c.ord}>
                              <div className="lab-chunk-meta">
                                <span>#{c.ord}{c.heading ? ` · ${c.heading}` : ''}</span>
                                <span>{c.tokens} tok</span>
                              </div>
                              {c.text}
                              {c.context_text && (
                                <div className="muted small" style={{ marginTop: 4 }}>
                                  ↳ the model is shown a wider window than this
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      </>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <p className="lab-note">
                Pick chunkers and split the corpus. Compare where the boundaries land — a fact cut in half is a
                fact the retriever can no longer find.
              </p>
            )}
          </div>
        )}

        {/* ------------------------------------------------------- transforms */}
        {tab === 'transforms' && (
          <div className="lab-result">
            <div className="lab-actions">
              <button className="primary" disabled={!!busy} onClick={runTransforms}>
                {busy === 'transforms' ? 'Rewriting…' : 'Rewrite the question'}
              </button>
            </div>
            {guard && (
              <div className="lab-explain-box">
                <b>Guardrails: {guard.blocked ? 'blocked' : 'passed'}.</b>{' '}
                {guard.blocked
                  ? `Matched an injection pattern (${guard.injection_patterns.length}), so the engine would refuse before retrieval.`
                  : 'No prompt-injection pattern matched.'}
                {Object.keys(guard.pii).length > 0 && (
                  <> Detected PII: {Object.entries(guard.pii).map(([k, v]) => `${k} (${v.length})`).join(', ')}.</>
                )}
              </div>
            )}
            {transformRuns ? (
              <>
                {transformRuns.offline && (
                  <p className="lab-note">
                    Running offline, so these use deterministic heuristics rather than a model. With Azure OpenAI
                    configured the same techniques call the LLM.
                  </p>
                )}
                <div className="table-wrap">
                  <table>
                    <thead><tr><th>technique</th><th>queries actually searched</th><th>ms</th></tr></thead>
                    <tbody>
                      {transformRuns.transforms.map((t) => (
                        <tr key={t.transform}>
                          <td><b>{t.transform}</b></td>
                          <td>
                            {t.error ? <span className="error small">{t.error}</span> : (
                              <div className="lab-chips">
                                {t.queries.map((q, i) => <code key={i}>{q}</code>)}
                              </div>
                            )}
                          </td>
                          <td className="muted small">{t.ms?.toFixed(1)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            ) : (
              <p className="lab-note">
                One question, five rewriting strategies. More queries means more chances to match the wording the
                document actually used — at the cost of latency.
              </p>
            )}
          </div>
        )}

        {/* ---------------------------------------------------------- compare */}
        {tab === 'compare' && (
          <div className="lab-result">
            <div className="lab-actions">
              <select value={compareStage} onChange={(e) => setCompareStage(e.target.value)}>
                {STAGES.map((s) => <option key={s.kind} value={s.kind}>{s.kind}</option>)}
              </select>
              <input
                className="lab-question"
                value={compareValues}
                onChange={(e) => setCompareValues(e.target.value)}
                placeholder="dense,bm25"
              />
              <button className="primary" disabled={!!busy} onClick={runCompare}>
                {busy === 'compare' ? 'Comparing…' : 'Compare'}
              </button>
            </div>
            {variants ? (
              <div className="lab-variants">
                {variants.map((v) => (
                  <div className="card" key={v.label}>
                    <div className="card-head">{v.label}</div>
                    {v.error ? <p className="error small">{v.error}</p> : (
                      <>
                        <p className="lab-answer">{v.answer || '(no answer)'}</p>
                        <div className="badges">
                          <span className="badge">{v.latency_ms?.toFixed(0)} ms</span>
                          <span className="badge">{v.stages?.compress?.length ?? 0} chunks read</span>
                          {v.groundedness != null && (
                            <span className={`badge trust ${v.groundedness >= 0.8 ? 'good' : 'warn'}`}>
                              groundedness {v.groundedness.toFixed(2)}
                            </span>
                          )}
                        </div>
                      </>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <p className="lab-note">
                Change one component, hold everything else fixed, and read both answers side by side. This is the
                honest way to tell whether a technique earns its latency.
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
