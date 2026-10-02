import WebSocket from 'ws'

export const FRAME_SAMPLES = 1280
export const FRAME_MS = 80

export interface ToolSpec {
  name: string
  description: string
  parameters: Record<string, unknown>
}

export interface ToolCall {
  name: string
  arguments: Record<string, unknown>
}

export function systemPrompt(instructions: string, tools: ToolSpec[]): string {
  if (!tools.length) return instructions
  return (
    `${instructions}\n\nYou can use the following tools to assist the user if required:` +
    `\n<AVAILABLE_TOOLS>[${tools.map((t) => JSON.stringify(t)).join(', ')}]</AVAILABLE_TOOLS>\n\n` +
    'If you decide to call any tool(s), use the following format:\n' +
    '<TOOLCALL>[{"name": "tool_name1", "arguments": "tool_args1"}, {"name": "tool_name2", "arguments": "tool_args2"}]</TOOLCALL>\n\n' +
    'The user will execute tool-calls and return responses from tool(s) in this format:\n' +
    '<TOOL_RESPONSE>[{"tool_response1"}, {"tool_response2"}]</TOOL_RESPONSE>\n\n' +
    'Based on the tool responses, you can call additional tools if needed, correct tool calls if any errors are found, or just respond to the user.'
  )
}

export function stripToolResponses(functionText: string): string {
  return functionText.replace(/<TOOL_RESPONSE>[\s\S]*?<\/TOOL_RESPONSE>/g, '')
}

export function parseToolCalls(functionText: string): ToolCall[] {
  const calls: ToolCall[] = []
  for (const m of functionText.matchAll(/<TOOLCALL>([\s\S]*?)<\/TOOLCALL>/g)) {
    const parsed = safeJson(m[1]) ?? safeJson(repair(m[1]))
    if (Array.isArray(parsed)) {
      for (const c of parsed as { name: string; arguments: unknown }[]) {
        const args = typeof c.arguments === 'string' ? (safeJson(c.arguments) ?? { raw: c.arguments }) : c.arguments
        calls.push({ name: c.name, arguments: (args ?? {}) as Record<string, unknown> })
      }
      continue
    }
    const name = m[1].match(/"name"\s*:?\s*"([^"]+)"/)?.[1]
    const args = m[1].match(/"arguments"?\s*:?\s*(\{[^{}]*\})/)?.[1]
    calls.push({
      name: name ?? '(unparsable)',
      arguments: ((args && safeJson(args)) || { raw: m[1] }) as Record<string, unknown>,
    })
  }
  return calls
}

function repair(s: string): string {
  let out = s.replace(/"(\w+) \{/g, '"$1": {').replace(/"(\w+) "/g, '"$1": "')
  const open = (out.match(/\[/g) ?? []).length - (out.match(/\]/g) ?? []).length
  if (open > 0) out += ']'.repeat(open)
  return out
}

function formatCall(call: ToolCall): string {
  return JSON.stringify(call).replace(/":/g, '": ').replace(/,"/g, ', "')
}

function safeJson(s: string): unknown {
  try {
    return JSON.parse(s)
  } catch {
    return undefined
  }
}

export function ascii(s: string): string {
  return s
    .replace(/[‘’]/g, "'")
    .replace(/[“”]/g, '"')
    .replace(/[–—]/g, ' - ')
    .normalize('NFKD')
    .replace(/[^\x20-\x7e\n]/g, '')
}

export interface TextEvent {
  index: number
  at: number
  delta: string
}

export class VoiceChatSession {
  userText = ''
  assistantText = ''
  functionText = ''
  textEvents: TextEvent[] = []
  onFunction?: (text: string) => void
  onUserText?: (text: string) => void
  onAudio?: (pcm: Int16Array) => void
  private ws!: WebSocket
  private waiters: { type: string; resolve: (e: Record<string, unknown>) => void }[] = []

  static async open(url: string, prompt: string): Promise<VoiceChatSession> {
    const s = new VoiceChatSession()
    s.ws = new WebSocket(`${url}/v1/realtime`)
    s.ws.on('message', (raw) => s.handle(JSON.parse(raw.toString())))
    await new Promise((resolve, reject) => {
      s.ws.once('open', resolve)
      s.ws.once('error', reject)
    })
    const ready = s.next('session.updated')
    s.ws.send(JSON.stringify({ type: 'session.update', session: { system_prompt: prompt } }))
    await ready
    return s
  }

  private handle(e: Record<string, unknown>) {
    switch (e.type) {
      case 'response.text.delta':
        this.assistantText = e.text as string
        this.textEvents.push({ index: e.frame_index as number, at: Date.now(), delta: e.delta as string })
        break
      case 'response.function.delta':
        this.functionText = e.text as string
        this.onFunction?.(this.functionText)
        break
      case 'conversation.item.input_audio_transcription.delta':
        this.userText = e.text as string
        this.onUserText?.(this.userText)
        break
      case 'response.audio.delta':
        if (this.onAudio) {
          const b = Buffer.from(e.audio as string, 'base64')
          this.onAudio(new Int16Array(b.buffer, b.byteOffset, b.byteLength / 2))
        }
        break
      case 'error':
        console.error('voicechat error:', JSON.stringify(e.error))
        break
    }
    const i = this.waiters.findIndex((w) => w.type === e.type)
    if (i >= 0) this.waiters.splice(i, 1)[0].resolve(e)
  }

  private next(type: string): Promise<Record<string, unknown>> {
    return new Promise((resolve) => this.waiters.push({ type, resolve }))
  }

  push(frame: Int16Array) {
    const audio = Buffer.from(frame.buffer, frame.byteOffset, frame.byteLength).toString('base64')
    this.ws.send(JSON.stringify({ type: 'input_audio_buffer.append', audio }))
  }

  async toolOutput(output: string, call?: ToolCall) {
    const done = this.next('conversation.item.injected')
    const item = { type: 'function_call_output', output: ascii(output), call: call && ascii(formatCall(call)) }
    this.ws.send(JSON.stringify({ type: 'conversation.item.create', item }))
    await done
  }

  async stats(): Promise<{ frames: number; frame_ms_p50: number; frame_ms_p95: number }> {
    const r = this.next('session.stats')
    this.ws.send(JSON.stringify({ type: 'session.stats' }))
    return (await r) as never
  }

  close() {
    if (this.ws.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify({ type: 'session.cancel' }))
    this.ws.close()
  }
}
