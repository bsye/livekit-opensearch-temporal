export async function requestJson<T>(
  method: string,
  url: string,
  body?: unknown,
  contentType = 'application/json',
): Promise<T> {
  const res = await fetch(url, {
    method,
    headers: body === undefined ? undefined : { 'content-type': contentType },
    body: body === undefined || typeof body === 'string' ? body : JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`${method} ${url}: ${res.status} ${await res.text()}`)
  return (await res.json()) as T
}

export const postJson = <T>(url: string, body: unknown) => requestJson<T>('POST', url, body)

export interface ChatMessage {
  content: string | null
  tool_calls?: { id: string; function: { name: string; arguments: string } }[]
}

export async function completeChat(baseUrl: string, body: Record<string, unknown>): Promise<ChatMessage> {
  const res = await postJson<{ choices: { message: ChatMessage }[] }>(`${baseUrl}/chat/completions`, body)
  return res.choices[0].message
}
