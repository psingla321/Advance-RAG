/** Browser APIs that only exist in a secure context, with plain-HTTP fallbacks.
 *
 * `crypto.randomUUID` and `navigator.clipboard` are undefined unless the page is
 * served over HTTPS or from localhost. A lab or demo deployment reached by IP
 * (http://10.0.0.5:8080) is *not* a secure context, and an unguarded call there
 * throws — which, from a `useState` initialiser, stops React mounting at all.
 */

/** RFC 4122 v4 id. Falls back to `getRandomValues`, which plain HTTP does provide. */
export function uuid(): string {
  const webCrypto = globalThis.crypto
  if (typeof webCrypto?.randomUUID === 'function') return webCrypto.randomUUID()

  const bytes = new Uint8Array(16)
  if (typeof webCrypto?.getRandomValues === 'function') {
    webCrypto.getRandomValues(bytes)
  } else {
    for (let i = 0; i < bytes.length; i++) bytes[i] = Math.floor(Math.random() * 256)
  }
  bytes[6] = (bytes[6] & 0x0f) | 0x40 // version 4
  bytes[8] = (bytes[8] & 0x3f) | 0x80 // variant 10xx

  const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, '0'))
  return [
    hex.slice(0, 4).join(''), hex.slice(4, 6).join(''), hex.slice(6, 8).join(''),
    hex.slice(8, 10).join(''), hex.slice(10).join(''),
  ].join('-')
}

/** Copy text to the clipboard. Resolves to false when the browser refuses. */
export async function copyText(text: string): Promise<boolean> {
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text)
      return true
    } catch {
      /* permission denied or not focused — fall through to the legacy path */
    }
  }
  // execCommand is deprecated but is the only clipboard route on plain HTTP.
  try {
    const field = document.createElement('textarea')
    field.value = text
    field.setAttribute('readonly', '')
    field.style.cssText = 'position:fixed;top:-9999px;opacity:0'
    document.body.appendChild(field)
    field.select()
    const ok = document.execCommand('copy')
    document.body.removeChild(field)
    return ok
  } catch {
    return false
  }
}
