import { useCallback, useEffect, useRef, useState } from 'react'
import { deleteDocument, listDocuments, listItems, uploadDocument } from '../api'
import type { DocumentInfo, Identity } from '../types'

const ACCEPT = '.pdf,.docx,.txt,.md,.csv,.json'

interface UploadItem {
  name: string
  status: 'uploading' | 'queued' | 'duplicate' | 'error'
  message?: string
}

export function KnowledgeView({ identity }: { identity: Identity }) {
  const [documents, setDocuments] = useState<DocumentInfo[]>([])
  const [queueDepth, setQueueDepth] = useState(0)
  const [acl, setAcl] = useState('all')
  const [uploads, setUploads] = useState<UploadItem[]>([])
  const [dragging, setDragging] = useState(false)
  const [filter, setFilter] = useState('')
  const [queues, setQueues] = useState<{ escalations: any[]; tickets: any[] }>({ escalations: [], tickets: [] })
  const [error, setError] = useState<string | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  const refresh = useCallback(async () => {
    try {
      const [docs, escalations, tickets] = await Promise.all([
        listDocuments(identity),
        listItems('escalations', identity),
        listItems('tickets', identity),
      ])
      setDocuments(docs.documents)
      setQueueDepth(docs.queue_depth)
      setQueues({ escalations, tickets })
      setError(null)
    } catch (err) {
      setError((err as Error).message)
    }
  }, [identity])

  useEffect(() => {
    refresh()
    const timer = setInterval(refresh, 3000)
    return () => clearInterval(timer)
  }, [refresh])

  async function uploadFiles(files: FileList | File[]) {
    const list = Array.from(files)
    setUploads((prev) => [...list.map((f) => ({ name: f.name, status: 'uploading' as const })), ...prev].slice(0, 20))
    await Promise.all(
      list.map(async (file) => {
        try {
          const result = await uploadDocument(file, acl, identity)
          const status = result.status === 'DUPLICATE' ? 'duplicate' : 'queued'
          setUploads((prev) => prev.map((u) => (u.name === file.name && u.status === 'uploading' ? { ...u, status } : u)))
        } catch (err) {
          setUploads((prev) =>
            prev.map((u) => (u.name === file.name && u.status === 'uploading' ? { ...u, status: 'error', message: (err as Error).message } : u)),
          )
        }
      }),
    )
    refresh()
  }

  async function remove(doc: DocumentInfo) {
    if (!confirm(`Delete ${doc.filename}? It will be removed from search and caches.`)) return
    try {
      await deleteDocument(doc.id, identity)
      refresh()
    } catch (err) {
      setError((err as Error).message)
    }
  }

  const visible = documents.filter((d) => d.filename.toLowerCase().includes(filter.toLowerCase()))
  const counts = documents.reduce<Record<string, number>>((acc, d) => ({ ...acc, [d.status]: (acc[d.status] ?? 0) + 1 }), {})
  const chunks = documents.reduce((sum, d) => sum + d.chunk_count, 0)

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h2>Knowledge</h2>
          <p className="muted">Documents are validated, queued, and indexed asynchronously by the worker pool.</p>
        </div>
        <div className="system-pills">
          <span className="pill">{documents.length} documents</span>
          <span className="pill">{chunks.toLocaleString()} chunks</span>
          <span className={`pill ${queueDepth ? 'warn' : 'good'}`}>Queue {queueDepth}</span>
          {counts.FAILED ? <span className="pill bad">{counts.FAILED} failed</span> : null}
        </div>
      </div>

      <div
        className={`dropzone ${dragging ? 'dragging' : ''}`}
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragging(false)
          if (e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files)
        }}
        onClick={() => fileRef.current?.click()}
      >
        <div className="dz-icon">⇪</div>
        <div><strong>Drop files here</strong> or click to browse</div>
        <div className="muted small">PDF, DOCX, TXT, MD, CSV, JSON · up to 20 MB each</div>
        <input
          ref={fileRef}
          type="file"
          multiple
          accept={ACCEPT}
          hidden
          onChange={(e) => {
            if (e.target.files?.length) uploadFiles(e.target.files)
            e.target.value = ''
          }}
        />
      </div>
      <div className="acl-row" onClick={(e) => e.stopPropagation()}>
        <label>
          Access groups for new uploads
          <input value={acl} onChange={(e) => setAcl(e.target.value)} placeholder="all, hr" />
        </label>
      </div>

      {uploads.length > 0 && (
        <ul className="upload-list">
          {uploads.map((u, i) => (
            <li key={`${u.name}-${i}`}>
              <span>{u.name}</span>
              <span className={`status ${u.status}`}>{u.status}</span>
              {u.message && <span className="error small">{u.message}</span>}
            </li>
          ))}
        </ul>
      )}
      {error && <div className="error">{error}</div>}

      <div className="card wide">
        <div className="card-head row">
          <span>Documents</span>
          <input className="search" placeholder="Filter…" value={filter} onChange={(e) => setFilter(e.target.value)} />
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Name</th><th>Status</th><th>Chunks</th><th>Access</th><th>Added</th><th /></tr>
            </thead>
            <tbody>
              {visible.map((d) => (
                <tr key={d.id}>
                  <td className="truncate" title={d.filename}>{d.filename}{d.error && <div className="error small">{d.error}</div>}</td>
                  <td><span className={`status ${d.status.toLowerCase()}`}>{d.status}</span></td>
                  <td>{d.chunk_count}</td>
                  <td>{d.acl.map((g) => <span key={g} className="tag">{g}</span>)}</td>
                  <td className="nowrap">{new Date(d.created_at).toLocaleString()}</td>
                  <td><button className="link danger" onClick={() => remove(d)}>Delete</button></td>
                </tr>
              ))}
              {!visible.length && (
                <tr><td colSpan={6} className="muted">No documents.</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      <div className="grid-2">
        <div className="card">
          <div className="card-head">Human review queue</div>
          {queues.escalations.length ? (
            <ul className="queue">
              {queues.escalations.slice(0, 8).map((e) => (
                <li key={e.id}><code>{e.id}</code><span className="muted">{e.reason}</span><span className="truncate">{e.query}</span></li>
              ))}
            </ul>
          ) : <p className="muted">No escalations.</p>}
        </div>
        <div className="card">
          <div className="card-head">Support tickets</div>
          {queues.tickets.length ? (
            <ul className="queue">
              {queues.tickets.slice(0, 8).map((t) => (
                <li key={t.id}><code>{t.id}</code><span className="tag">{t.priority}</span><span className="truncate">{t.title}</span></li>
              ))}
            </ul>
          ) : <p className="muted">No tickets.</p>}
        </div>
      </div>
    </div>
  )
}
