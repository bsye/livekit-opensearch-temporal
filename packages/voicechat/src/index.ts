import { between, count, removeBetween } from '@voice/text'
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
  return removeBetween(functionText, '<TOOL_RESPONSE>', '</TOOL_RESPONSE>')
}

export function parseToolCalls(functionText: string): ToolCall[] {
  const calls: ToolCall[] = []
  for (const raw of between(functionText, '<TOOLCALL>', '</TOOLCALL>')) {
    const parsed = safeJson(raw) ?? safeJson(repair(raw))
    if (Array.isArray(parsed)) {
      for (const c of parsed as { name: string; arguments: unknown }[]) {
        const args = typeof c.arguments === 'string' ? (safeJson(c.arguments) ?? { raw: c.arguments }) : c.arguments
        calls.push({ name: c.name, arguments: (args ?? {}) as Record<string, unknown> })
      }
      continue
    }
    const name = quotedValueAfter(raw, '"name"')
    const args = objectAfter(raw, 'arguments')
    calls.push({
      name: name ?? '(unparsable)',
      arguments: ((args && safeJson(args)) || { raw }) as Record<string, unknown>,
    })
  }
  return calls
}

function repair(json: string): string {
  let out = ''
  for (let i = 0; i < json.length; i++) {
    out += json[i]
    if (json[i] !== '"') continue
    let j = i + 1
    while (j < json.length && isWordChar(json[j])) j++
    if (j > i + 1 && json[j] === ' ' && (json[j + 1] === '{' || json[j + 1] === '"')) {
      out += `${json.slice(i + 1, j)}": `
      i = j
    }
  }
  const unclosed = count(out, '[') - count(out, ']')
  return unclosed > 0 ? out + ']'.repeat(unclosed) : out
}

const isWordChar = (c: string) => c === '_' || c.toLowerCase() !== c.toUpperCase() || (c >= '0' && c <= '9')

function quotedValueAfter(text: string, key: string): string | undefined {
  const at = text.indexOf(key)
  if (at === -1) return undefined
  const open = text.indexOf('"', at + key.length)
  const close = open === -1 ? -1 : text.indexOf('"', open + 1)
  return close === -1 ? undefined : text.slice(open + 1, close)
}

function objectAfter(text: string, key: string): string | undefined {
  const at = text.indexOf(key)
  const open = at === -1 ? -1 : text.indexOf('{', at)
  const close = open === -1 ? -1 : text.indexOf('}', open)
  return close === -1 ? undefined : text.slice(open, close + 1)
}

function formatCall(call: ToolCall): string {
  return JSON.stringify(call).replaceAll('":', '": ').replaceAll(',"', ', "')
}

function safeJson(s: string): unknown {
  try {
    return JSON.parse(s)
  } catch {
    return undefined
  }
}

const ASCII_LOOKALIKES: Record<string, string> = { '‘': "'", '’': "'", '“': '"', '”': '"', '–': ' - ', '—': ' - ' }

export function ascii(text: string): string {
  return [...text]
    .map((c) => ASCII_LOOKALIKES[c] ?? c)
    .join('')
    .normalize('NFKD')
    .split('')
    .filter((c) => c === '\n' || (c >= ' ' && c <= '~'))
    .join('')
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
