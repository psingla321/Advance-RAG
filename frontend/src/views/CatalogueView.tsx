import { useEffect, useMemo, useState } from 'react'
import { labComponents } from '../api'
import type { ComponentInfo } from '../types'

/** Every pluggable stage, in the order a document travels through the system. */
const KINDS: { kind: string; label: string; blurb: string }[] = [
  { kind: 'loader', label: 'Document types', blurb: 'Bytes in, structured sections out. `auto` dispatches on file extension.' },
  { kind: 'chunker', label: 'Chunking strategies', blurb: 'How a document is split for indexing — the single biggest lever on retrieval quality.' },
  { kind: 'embedder', label: 'Embedding models', blurb: 'Text to vectors. Dimension and provider are swappable per pipeline.' },
  { kind: 'vectorstore', label: 'Vector databases', blurb: 'Where vectors and payloads live. Embedded, self-hosted or fully managed.' },
  { kind: 'transform', label: 'Query transforms', blurb: 'Rewrite the question before searching, to close the gap between asker and author.' },
  { kind: 'retriever', label: 'Retrieval approaches', blurb: 'How candidates are found and fused across query variants.' },
  { kind: 'reranker', label: 'Rerankers', blurb: 'Second-pass ordering with a sharper, slower model.' },
  { kind: 'compressor', label: 'Context packing', blurb: 'What survives into the token budget the model actually reads.' },
  { kind: 'generator', label: 'Generation strategies', blurb: 'How the answer is written from the context, and how it is cited.' },
  { kind: 'llm', label: 'Language models', blurb: 'Generation backends. `mock` keeps everything runnable with no API key.' },
  { kind: 'metric', label: 'Evaluation metrics', blurb: 'How a pipeline is scored against a golden set.' },
]

export function CatalogueView() {
  const [catalogue, setCatalogue] = useState<Record<string, ComponentInfo[]>>({})
  const [query, setQuery] = useState('')
  const [onlyReady, setOnlyReady] = useState(false)
  const [open, setOpen] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    labComponents().then(setCatalogue).catch((e) => setError((e as Error).message))
  }, [])

  const needle = query.trim().toLowerCase()
  const groups = useMemo(() => {
    return KINDS.map((group) => {
      const all = catalogue[group.kind] ?? []
      const items = all.filter((c) => {
        if (onlyReady && c.requires.length) return false
        if (!needle) return true
        return c.name.toLowerCase().includes(needle) || c.summary.toLowerCase().includes(needle)
      })
      return { ...group, items, total: all.length }
    })
  }, [catalogue, needle, onlyReady])

  const total = Object.values(catalogue).reduce((n, items) => n + items.length, 0)
  const shown = groups.reduce((n, g) => n + g.items.length, 0)

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h2>Catalogue</h2>
          <p className="muted">
            Every approach, database and document type this engine can use. Each one is selectable in the RAG Lab
            and writable as one line of YAML — <code>{'{ "retriever": { "type": "hybrid_rrf" } }'}</code>.
          </p>
        </div>
        <div className="system-pills">
          <span className="pill">{total} components</span>
          <span className="pill good">{total - Object.values(catalogue).flat().filter((c) => c.requires.length).length} ready offline</span>
        </div>
      </div>

      {error && <div className="error">{error}</div>}

      <div className="card wide">
        <div className="lab-actions" style={{ marginBottom: 0 }}>
          <input
            className="lab-question"
            placeholder="Filter by name or description — try 'hybrid', 'pdf', 'azure'…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <button className={`chip ${onlyReady ? 'active' : ''}`} onClick={() => setOnlyReady((v) => !v)}>
            {onlyReady ? '✓ ' : ''}No install needed
          </button>
          <span className="muted small">{shown} of {total} shown</span>
        </div>
      </div>

      {groups.map((group) => (
        group.items.length > 0 && (
          <div className="card wide" key={group.kind}>
            <div className="card-head row">
              <span>{group.label}</span>
              <span className="pill">{group.items.length}{group.items.length !== group.total ? ` of ${group.total}` : ''}</span>
            </div>
            <p className="lab-note">{group.blurb}</p>

            <div className="cat-grid">
              {group.items.map((item) => {
                const id = `${group.kind}.${item.name}`
                const expanded = open === id
                const tunable = item.params.filter((p) => !['llm', 'embedder', 'vectorstore', 'inner'].includes(p.name))
                return (
                  <div className={`cat-item ${expanded ? 'open' : ''}`} key={id} onClick={() => setOpen(expanded ? null : id)}>
                    <div className="cat-item-head">
                      <code>{item.name}</code>
                      {item.requires.length > 0
                        ? <span className="badge muted">pip: {item.requires.join(', ')}</span>
                        : <span className="badge trust good">built in</span>}
                    </div>
                    <div className="cat-summary">{item.summary}</div>
                    {expanded && (
                      <div className="cat-params">
                        {tunable.length === 0
                          ? <span className="muted small">No parameters — it works as-is.</span>
                          : tunable.map((p) => (
                              <div className="cat-param" key={p.name}>
                                <code>{p.name}</code>
                                <span className="muted small">{p.type}</span>
                                <span className="muted small">
                                  {p.required ? 'required' : `default ${JSON.stringify(p.default)}`}
                                </span>
                              </div>
                            ))}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </div>
        )
      ))}

      {shown === 0 && total > 0 && (
        <div className="card wide"><p className="lab-note">Nothing matches “{query}”.</p></div>
      )}
    </div>
  )
}
