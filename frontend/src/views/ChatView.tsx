import { useEffect, useRef } from 'react'
import { MessageBubble } from '../components/MessageBubble'
import type { ChatMessage } from '../types'

const SUGGESTIONS = [
  { icon: '📄', text: 'What is the hotel limit in high-cost cities?' },
  { icon: '📊', text: 'Total revenue by region in 2025' },
  { icon: '🏆', text: 'Which product had the highest units in Q4 2025?' },
  { icon: '🔐', text: 'What are the password requirements and how do I report phishing?' },
  { icon: '🧮', text: 'What is 1250 * 12 * 0.15?' },
  { icon: '🎫', text: 'Create a P2 ticket for VPN outage in the Berlin office' },
]

interface Props {
  messages: ChatMessage[]
  busy: boolean
  input: string
  onInput: (value: string) => void
  onSend: (text: string) => void
  onStop: () => void
  onFeedback: (message: ChatMessage, rating: 'up' | 'down') => void
}

export function ChatView({ messages, busy, input, onInput, onSend, onStop, onFeedback }: Props) {
  const endRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages])

  useEffect(() => {
    const el = textareaRef.current
    if (el) {
      el.style.height = 'auto'
      el.style.height = `${Math.min(el.scrollHeight, 180)}px`
    }
  }, [input])

  return (
    <main className="chat">
      <div className="messages">
        {messages.length === 0 ? (
          <div className="empty">
            <div className="hero-logo">N</div>
            <h1>How can I help today?</h1>
            <p>
              Nexus routes every question to the best path — documents, SQL analytics, tools, the LLM, or a human —
              verifies the answer, and shows you exactly how it got there.
            </p>
            <div className="suggestions">
              {SUGGESTIONS.map((s) => (
                <button key={s.text} onClick={() => onSend(s.text)}>
                  <span>{s.icon}</span>
                  {s.text}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((m) => <MessageBubble key={m.id} message={m} onFeedback={onFeedback} />)
        )}
        <div ref={endRef} />
      </div>

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault()
          onSend(input)
        }}
      >
        <div className="composer-box">
          <textarea
            ref={textareaRef}
            value={input}
            placeholder="Ask about your documents, sales data, or anything else…"
            onChange={(e) => onInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                onSend(input)
              }
            }}
            rows={1}
            maxLength={4000}
          />
          {busy ? (
            <button type="button" className="send stop" onClick={onStop} title="Stop">■</button>
          ) : (
            <button type="submit" className="send" disabled={!input.trim()} title="Send">↑</button>
          )}
        </div>
        <div className="composer-hint">
          Shortcuts: <code>/docs</code> <code>/sql</code> <code>/tool</code> <code>/chat</code> <code>/human</code> · Shift+Enter for a new line
        </div>
      </form>
    </main>
  )
}
