import type {
  ComponentInfo,
  DocumentInfo,
  Identity,
  Insights,
  LabAnswer,
  LabPreset,
  LabRetrieval,
  LabSweepRun,
} from './types'

function identityHeaders(identity: Identity): Record<string, string> {
  return { 'X-Tenant-Id': identity.tenantId, 'X-User-Id': identity.userId || 'anonymous', 'X-User-Groups': identity.groups }
}

async function errorMessage(res: Response): Promise<string> {
  const retry = res.headers.get('Retry-After')
  let detail = `Request failed (${res.status})`
  try {
    const body = await res.json()
    if (typeof body.detail === 'string') detail = body.detail
  } catch {
    /* non-JSON error body */
  }
  return retry ? `${detail} Retry in ${retry}s.` : detail
}

export type ChatEventHandler = (event: string, data: any) => void

export async function streamChat(
  message: string,
  conversationId: string | null,
  identity: Identity,
  onEvent: ChatEventHandler,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...identityHeaders(identity) },
    body: JSON.stringify({ message, conversation_id: conversationId }),
    signal,
  })
  if (!res.ok || !res.body) throw new Error(await errorMessage(res))

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let boundary: number
    while ((boundary = buffer.indexOf('\n\n')) !== -1) {
      const raw = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)
      let event = 'message'
      const data: string[] = []
      for (const line of raw.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim()
        else if (line.startsWith('data:')) data.push(line.slice(5).trimStart())
      }
      if (data.length) onEvent(event, JSON.parse(data.join('\n')))
    }
  }
}

async function getJson<T>(path: string, identity: Identity): Promise<T> {
  const res = await fetch(path, { headers: identityHeaders(identity) })
  if (!res.ok) throw new Error(await errorMessage(res))
  return res.json()
}

export const listDocuments = (identity: Identity) =>
  getJson<{ documents: DocumentInfo[]; queue_depth: number }>('/api/documents', identity)

export const getInsights = (identity: Identity) => getJson<Insights>('/api/insights', identity)

export const listItems = async (kind: 'escalations' | 'tickets', identity: Identity): Promise<any[]> =>
  (await getJson<{ items: any[] }>(`/api/${kind}`, identity)).items

export async function uploadDocument(file: File, acl: string, identity: Identity): Promise<{ status: string }> {
  const form = new FormData()
  form.append('file', file)
  form.append('acl', acl)
  const res = await fetch('/api/documents', { method: 'POST', headers: identityHeaders(identity), body: form })
  if (!res.ok) throw new Error(await errorMessage(res))
  return res.json()
}

export async function deleteDocument(id: string, identity: Identity): Promise<void> {
  const res = await fetch(`/api/documents/${encodeURIComponent(id)}`, { method: 'DELETE', headers: identityHeaders(identity) })
  if (!res.ok) throw new Error(await errorMessage(res))
}

export async function sendFeedback(requestId: string, rating: 'up' | 'down', identity: Identity): Promise<void> {
  const res = await fetch('/api/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...identityHeaders(identity) },
    body: JSON.stringify({ request_id: requestId, rating }),
  })
  if (!res.ok) throw new Error(await errorMessage(res))
}

export async function getHealth(): Promise<{ status: string; mock_llm: boolean; checks: Record<string, boolean> }> {
  const res = await fetch('/api/health')
  if (!res.ok) throw new Error(await errorMessage(res))
  return res.json()
}

// ---------------------------------------------------------------- RAG Lab
// The lab endpoints are tenant-agnostic: pipelines are built per session from
// the configuration on screen, so no identity headers are required.

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(await errorMessage(res))
  return res.json()
}

export const labComponents = async (): Promise<Record<string, ComponentInfo[]>> =>
  (await getJsonPublic<{ components: Record<string, ComponentInfo[]> }>('/api/lab/components')).components

export const labPresets = async (): Promise<LabPreset[]> =>
  (await getJsonPublic<{ presets: LabPreset[] }>('/api/lab/presets')).presets

async function getJsonPublic<T>(path: string): Promise<T> {
  const res = await fetch(path)
  if (!res.ok) throw new Error(await errorMessage(res))
  return res.json()
}

/** Build a pipeline and index the corpus; returns the session id to query. */
export async function createLabSession(
  preset: string,
  overrides: Record<string, unknown>,
  texts: string[],
): Promise<string> {
  const session = await postJson<{ session_id: string }>('/api/lab/sessions', { config: preset, overrides })
  await postJson(`/api/lab/sessions/${session.session_id}/texts`, { texts })
  return session.session_id
}

export const askLab = (sessionId: string, question: string): Promise<LabAnswer> =>
  postJson(`/api/lab/sessions/${sessionId}/ask`, { question })

export const retrieveLab = (sessionId: string, question: string): Promise<LabRetrieval> =>
  postJson(`/api/lab/sessions/${sessionId}/retrieve`, { question })

export const evaluateLab = (sessionId: string, samples: { question: string; ground_truth: string }[]): Promise<any> =>
  postJson(`/api/lab/sessions/${sessionId}/evaluate`, { samples })

export const runLabSweep = (
  base: string,
  grid: Record<string, unknown[]>,
  samples: { question: string; ground_truth: string }[],
  texts: string[],
): Promise<{ runs: LabSweepRun[]; failures: { name: string; error: string }[] }> =>
  postJson('/api/lab/sweep', { base, grid, samples, texts })
