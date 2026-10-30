import { Fragment, useState, type ReactNode } from 'react'

function renderInline(text: string, key: string): ReactNode[] {
  return text
    .split(/(\*\*[^*]+\*\*|\[\d+\]|`[^`]+`|_[^_]+_)/g)
    .filter(Boolean)
    .map((part, i) => {
      const k = `${key}-${i}`
      if (part.startsWith('**') && part.endsWith('**')) return <strong key={k}>{part.slice(2, -2)}</strong>
      if (/^\[\d+\]$/.test(part)) return <sup key={k} className="cite">{part.slice(1, -1)}</sup>
      if (part.startsWith('`') && part.endsWith('`')) return <code key={k}>{part.slice(1, -1)}</code>
      if (/^_[^_]+_$/.test(part)) return <em key={k}>{part.slice(1, -1)}</em>
      return <Fragment key={k}>{part}</Fragment>
    })
}

function parseRow(line: string): string[] {
  return line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map((c) => c.trim())
}

function CodeBlock({ code, lang }: { code: string; lang: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <div className="codeblock">
      <div className="codeblock-bar">
        <span>{lang || 'text'}</span>
        <button
          className="link"
          onClick={() => {
            navigator.clipboard.writeText(code).then(() => {
              setCopied(true)
              setTimeout(() => setCopied(false), 1200)
            })
          }}
        >
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre>{code}</pre>
    </div>
  )
}

/** XSS-safe markdown subset: headings, paragraphs, bullets, numbered lists, tables, code fences, inline marks, citations. */
export function RichText({ text }: { text: string }) {
  const lines = text.split('\n')
  const blocks: ReactNode[] = []
  let i = 0
  while (i < lines.length) {
    const line = lines[i]
    const trimmed = line.trim()

    if (trimmed.startsWith('```')) {
      const lang = trimmed.slice(3).trim()
      const code: string[] = []
      i++
      while (i < lines.length && !lines[i].trim().startsWith('```')) code.push(lines[i++])
      i++
      blocks.push(<CodeBlock key={`c${i}`} code={code.join('\n')} lang={lang} />)
      continue
    }
    if (trimmed.startsWith('|')) {
      const rows: string[][] = []
      while (i < lines.length && lines[i].trim().startsWith('|')) {
        if (!/^\|\s*:?-{3,}/.test(lines[i].trim())) rows.push(parseRow(lines[i]))
        i++
      }
      const [head, ...body] = rows
      if (!head) continue
      blocks.push(
        <div className="table-wrap" key={`t${i}`}>
          <table>
            <thead>
              <tr>{head.map((c, j) => <th key={j}>{c}</th>)}</tr>
            </thead>
            <tbody>
              {body.map((r, ri) => (
                <tr key={ri}>{r.map((c, j) => <td key={j}>{c}</td>)}</tr>
              ))}
            </tbody>
          </table>
        </div>,
      )
      continue
    }
    const heading = /^(#{1,4})\s+(.*)$/.exec(trimmed)
    if (heading) {
      blocks.push(<div key={`h${i}`} className={`md-h md-h${heading[1].length}`}>{renderInline(heading[2], `h${i}`)}</div>)
      i++
      continue
    }
    if (/^\s*([-*•]|\d+\.)\s+/.test(line)) {
      const ordered = /^\s*\d+\./.test(line)
      const items: string[] = []
      while (i < lines.length && /^\s*([-*•]|\d+\.)\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*([-*•]|\d+\.)\s+/, ''))
        i++
      }
      const children = items.map((item, j) => <li key={j}>{renderInline(item, `l${i}-${j}`)}</li>)
      blocks.push(ordered ? <ol key={`o${i}`}>{children}</ol> : <ul key={`u${i}`}>{children}</ul>)
      continue
    }
    if (trimmed) blocks.push(<p key={`p${i}`}>{renderInline(line, `p${i}`)}</p>)
    i++
  }
  return <div className="rich">{blocks}</div>
}
