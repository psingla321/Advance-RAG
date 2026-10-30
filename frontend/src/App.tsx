import { useEffect, useMemo, useRef, useState } from 'react'
import { getHealth, sendFeedback, streamChat } from './api'
import type { ChatMessage, Conversation, FinalPayload, Identity, Routing, TraceEntry, View } from './types'
import { ChatView } from './views/ChatView'
import { InsightsView } from './views/InsightsView'
import { LabView } from './views/LabView'
import { KnowledgeView } from './views/KnowledgeView'

const STORAGE = { identity: 'nexus.identity', conversations: 'nexus.conversations', theme: 'nexus.theme' }
const MAX_CONVERSATIONS = 30
const MAX_MESSAGES = 60

function load<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key)
    return raw ? (JSON.parse(raw) as T) : fallback
  } catch {
    return fallback
  }
}

const newId = () => crypto.randomUUID()
const emptyConversation = (): Conversation => ({ id: newId(), serverId: null, title: 'New chat', updatedAt: Date.now(), messages: [] })

export default function App() {
  const [identity, setIdentity] = useState<Identity>(() => ({ tenantId: 'demo', userId: 'alice', groups: 'all', ...load(STORAGE.identity, {}) }))
  const [conversations, setConversations] = useState<Conversation[]>(() => {
    const stored = load<Conversation[]>(STORAGE.conversations, [])
    return stored.length ? stored : [emptyConversation()]
  })
  const [activeId, setActiveId] = useState(() => conversations[0].id)
  const [view, setView] = useState<View>('chat')
  const [theme, setTheme] = useState<'dark' | 'light'>(() => load(STORAGE.theme, 'dark'))
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [health, setHealth] = useState<{ mock: boolean; ok: boolean } | null>(null)
  const [showSettings, setShowSettings] = useState(false)
  const abortRef = useRef<AbortController | null>(null)

  const active = useMemo(() => conversations.find((c) => c.id === activeId) ?? conversations[0], [conversations, activeId])

  useEffect(() => localStorage.setItem(STORAGE.identity, JSON.stringify(identity)), [identity])
  useEffect(() => {
    document.documentElement.dataset.theme = theme
    localStorage.setItem(STORAGE.theme, JSON.stringify(theme))
  }, [theme])
  useEffect(() => {
    if (busy) return
    const trimmed = conversations
      .filter((c) => c.messages.length)
      .slice(0, MAX_CONVERSATIONS)
      .map((c) => ({ ...c, messages: c.messages.slice(-MAX_MESSAGES) }))
    localStorage.setItem(STORAGE.conversations, JSON.stringify(trimmed))
  }, [conversations, busy])
  useEffect(() => {
    const check = () =>
      getHealth()
        .then((h) => setHealth({ mock: h.mock_llm, ok: h.status === 'ok' }))
        .catch(() => setHealth({ mock: false, ok: false }))
    check()
    const timer = setInterval(check, 15000)
    return () => clearInterval(timer)
  }, [])

  function patchConversation(id: string, fn: (c: Conversation) => Conversation) {
    setConversations((prev) => prev.map((c) => (c.id === id ? fn(c) : c)))
  }

  function patchMessage(convId: string, msgId: string, fn: (m: ChatMessage) => ChatMessage) {
    patchConversation(convId, (c) => ({ ...c, messages: c.messages.map((m) => (m.id === msgId ? fn(m) : m)) }))
  }

  async function send(text: string) {
    const message = text.trim()
    if (!message || busy) return
    const convId = active.id
    const assistantId = newId()
    setInput('')
    setBusy(true)
    patchConversation(convId, (c) => ({
      ...c,
      title: c.messages.length ? c.title : message.slice(0, 48),
      updatedAt: Date.now(),
      messages: [
        ...c.messages,
        { id: newId(), role: 'user', content: message, trace: [] },
        { id: assistantId, role: 'assistant', content: '', streaming: true, trace: [] },
      ],
    }))
    setConversations((prev) => [...prev].sort((a, b) => b.updatedAt - a.updatedAt))

    const controller = new AbortController()
    abortRef.current = controller
    const update = (fn: (m: ChatMessage) => ChatMessage) => patchMessage(convId, assistantId, fn)
    try {
      await streamChat(
        message,
        active.serverId,
        identity,
        (event, data) => {
          switch (event) {
            case 'meta':
              patchConversation(convId, (c) => ({ ...c, serverId: data.conversation_id }))
              update((m) => ({ ...m, traceId: data.trace_id }))
              break
            case 'trace':
              update((m) => ({ ...m, trace: [...m.trace, data as TraceEntry] }))
              break
            case 'routing':
              update((m) => ({ ...m, routing: data as Routing }))
              break
            case 'token':
              update((m) => ({ ...m, content: m.content + data.text }))
              break
            case 'revision':
              update((m) => ({ ...m, content: data.answer }))
              break
            case 'final': {
              const final = data as FinalPayload
              update((m) => ({ ...m, content: final.answer, final, streaming: false }))
              break
            }
            case 'error':
              update((m) => ({ ...m, error: `${data.message} (request ${data.request_id})`, streaming: false }))
              break
          }
        },
        controller.signal,
      )
    } catch (err) {
      if ((err as Error).name !== 'AbortError') update((m) => ({ ...m, error: (err as Error).message }))
    } finally {
      update((m) => ({ ...m, streaming: false }))
      setBusy(false)
      abortRef.current = null
    }
  }

  async function feedback(message: ChatMessage, rating: 'up' | 'down') {
    if (!message.final?.request_id) return
    patchMessage(active.id, message.id, (m) => ({ ...m, feedback: rating }))
    try {
      await sendFeedback(message.final.request_id, rating, identity)
    } catch {
      patchMessage(active.id, message.id, (m) => ({ ...m, feedback: undefined }))
    }
  }

  function newChat() {
    abortRef.current?.abort()
    const fresh = emptyConversation()
    setConversations((prev) => [fresh, ...prev.filter((c) => c.messages.length)])
    setActiveId(fresh.id)
    setView('chat')
  }

  function removeConversation(id: string) {
    setConversations((prev) => {
      const rest = prev.filter((c) => c.id !== id)
      const next = rest.length ? rest : [emptyConversation()]
      if (id === activeId) setActiveId(next[0].id)
      return next
    })
  }

  const navItems: { id: View; icon: string; label: string }[] = [
    { id: 'chat', icon: '💬', label: 'Chat' },
    { id: 'knowledge', icon: '📚', label: 'Knowledge' },
    { id: 'insights', icon: '📈', label: 'Insights' },
    { id: 'lab', icon: '🧪', label: 'RAG Lab' },
  ]

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">
          <div className="logo">N</div>
          <div>
            <div className="brand-name">Nexus AI</div>
            <div className="brand-sub">Agentic enterprise engine</div>
          </div>
        </div>

        <nav className="nav">
          {navItems.map((item) => (
            <button key={item.id} className={`nav-item ${view === item.id ? 'active' : ''}`} onClick={() => setView(item.id)}>
              <span>{item.icon}</span>
              {item.label}
            </button>
          ))}
        </nav>

        <button className="primary" onClick={newChat}>+ New chat</button>

        <div className="conv-list">
          <div className="section-title">Recent conversations</div>
          {conversations
            .filter((c) => c.messages.length || c.id === active.id)
            .map((c) => (
              <div
                key={c.id}
                className={`conv ${c.id === active.id && view === 'chat' ? 'active' : ''}`}
                onClick={() => {
                  setActiveId(c.id)
                  setView('chat')
                }}
              >
                <span className="conv-title">{c.title}</span>
                {c.messages.length > 0 && (
                  <button
                    className="icon small"
                    title="Delete conversation"
                    onClick={(e) => {
                      e.stopPropagation()
                      removeConversation(c.id)
                    }}
                  >
                    ✕
                  </button>
                )}
              </div>
            ))}
        </div>

        <div className="sidebar-footer">
          {showSettings && (
            <div className="settings">
              <label>Tenant<input value={identity.tenantId} onChange={(e) => setIdentity({ ...identity, tenantId: e.target.value })} /></label>
              <label>User<input value={identity.userId} onChange={(e) => setIdentity({ ...identity, userId: e.target.value })} /></label>
              <label>Groups<input value={identity.groups} onChange={(e) => setIdentity({ ...identity, groups: e.target.value })} /></label>
              <p className="hint">Use groups <code>all,hr</code> to unlock HR-only documents.</p>
            </div>
          )}
          <div className="footer-row">
            <span className={`status-dot ${health ? (health.ok ? 'ok' : 'down') : ''}`} />
            <span className="muted small">{health ? (health.ok ? (health.mock ? 'Online · mock LLM' : 'Online · Azure OpenAI') : 'Degraded') : 'Connecting…'}</span>
            <span className="spacer" />
            <button className="icon" title="Identity settings" onClick={() => setShowSettings((s) => !s)}>⚙</button>
            <button className="icon" title="Toggle theme" onClick={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}>
              {theme === 'dark' ? '☀' : '☾'}
            </button>
          </div>
          <div className="whoami muted small">{identity.userId} @ {identity.tenantId} · {identity.groups}</div>
        </div>
      </aside>

      {view === 'chat' && (
        <ChatView
          messages={active.messages}
          busy={busy}
          input={input}
          onInput={setInput}
          onSend={send}
          onStop={() => abortRef.current?.abort()}
          onFeedback={feedback}
        />
      )}
      {view === 'insights' && <InsightsView identity={identity} />}
      {view === 'knowledge' && <KnowledgeView identity={identity} />}
      {view === 'lab' && <LabView />}
    </div>
  )
}
