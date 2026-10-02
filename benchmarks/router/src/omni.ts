import { appendFileSync, mkdirSync } from 'node:fs'
import { median, option, percent } from '@bench/shared'
import { speak } from '@bench/shared/speech'
import { exampleMessages, instructions } from '@voice/agent/omni-prompt'
import { toolSpecs } from '@voice/agent/tool-specs'
import { dataPath, env } from '@voice/config'
import { completeChat } from '@voice/http'
import { fileStamp } from '@voice/text'
import { type Label, routerTurns, type Turn } from './turns.js'

const MODES = option('--modes', 'audio,audio+text,text').split(',')
const LIMIT = Number(option('--limit', '1000'))
const PROMPT = option('--prompt', 'plain')
const OMNI = env('OMNI_BASE_URL')
const ACTION_TOOLS = new Set(['set_reminder', 'cancel_reminder', 'send_email'])

type Decision = 'recall' | 'action' | 'none'
interface Row extends Turn {
  mode: string
  transcript: string
  tools: string[]
  decision: Decision
  reply: string
  ms: number
}

const tools = toolSpecs()
const EXAMPLES = exampleMessages()
const system = instructions(new Date())

function wav(pcm: Int16Array, rate = 16_000): Blob {
  const header = Buffer.alloc(44)
  header.write('RIFF', 0)
  header.writeUInt32LE(36 + pcm.byteLength, 4)
  header.write('WAVEfmt ', 8)
  header.writeUInt32LE(16, 16)
  header.writeUInt16LE(1, 20)
  header.writeUInt16LE(1, 22)
  header.writeUInt32LE(rate, 24)
  header.writeUInt32LE(rate * 2, 28)
  header.writeUInt16LE(2, 32)
  header.writeUInt16LE(16, 34)
  header.write('data', 36)
  header.writeUInt32LE(pcm.byteLength, 40)
  return new Blob([Uint8Array.from(header), new Uint8Array(pcm.buffer, pcm.byteOffset, pcm.byteLength).slice()], {
    type: 'audio/wav',
  })
}

async function transcribe(pcm: Int16Array): Promise<string> {
  const form = new FormData()
  form.append('file', wav(pcm), 'turn.wav')
  form.append('model', env('STT_MODEL'))
  const res = await fetch(`${env('SPEECH_BASE_URL')}/audio/transcriptions`, { method: 'POST', body: form })
  if (!res.ok) throw new Error(`stt: ${res.status} ${await res.text()}`)
  return ((await res.json()) as { text: string }).text.trim()
}

async function uploadAudio(pcm: Int16Array): Promise<string> {
  const res = await fetch(`${OMNI}/audio/turns?rate=16000`, {
    method: 'POST',
    headers: { 'content-type': 'application/octet-stream' },
    body: new Uint8Array(pcm.buffer, pcm.byteOffset, pcm.byteLength).slice(),
  })
  if (!res.ok) throw new Error(`upload: ${res.status}`)
  return ((await res.json()) as { id: string }).id
}

async function decide(mode: string, transcript: string, audioId: string) {
  const content = mode === 'text' ? transcript : `${transcript} <audio:${audioId}>`
  const started = Date.now()
  const message = await completeChat(OMNI, {
    model: env('OMNI_MODEL'),
    temperature: 0,
    max_tokens: 120,
    tools,
    transcript_with_audio: mode === 'audio+text',
    messages: [
      { role: 'system', content: system },
      ...(PROMPT === 'example' ? EXAMPLES.slice(0, 4) : PROMPT === 'examples' ? EXAMPLES : []),
      { role: 'user', content },
    ],
  })
  const called = (message.tool_calls ?? []).map((c) => c.function.name)
  const decision: Decision = called.includes('recall')
    ? 'recall'
    : called.some((t) => ACTION_TOOLS.has(t))
      ? 'action'
      : 'none'
  return { tools: called, decision, reply: message.content ?? '', ms: Date.now() - started }
}

const byLabel = (label: Label) =>
  routerTurns()
    .filter((t) => t.label === label)
    .slice(0, LIMIT)
const turns = [...byLabel('past'), ...byLabel('action'), ...byLabel('chat')]
const outDir = dataPath('benchmarks', 'router')
mkdirSync(outDir, { recursive: true })
const log = `${outDir}/omni-${PROMPT}-${fileStamp()}.jsonl`
console.log(`${turns.length} turns × ${MODES.join(', ')}, prompt ${PROMPT} → ${log}`)

const rows: Row[] = []
for (const turn of turns) {
  const pcm = await speak(turn.text)
  const [transcript, audioId] = await Promise.all([transcribe(pcm), uploadAudio(pcm)])
  for (const mode of MODES) {
    const row: Row = { ...turn, mode, transcript, ...(await decide(mode, transcript, audioId)) }
    rows.push(row)
    appendFileSync(log, `${JSON.stringify(row)}\n`)
  }
  process.stdout.write('.')
}

console.log(
  '\n\nmode        | memory turns → recall | action turns → action tool | chat turns → no tool | non-memory turns → recall | p50',
)
for (const mode of MODES) {
  const rs = rows.filter((r) => r.mode === mode)
  const of = (label: Label) => rs.filter((r) => r.label === label)
  const share = (label: Label, d: Decision) =>
    percent(of(label).filter((r) => r.decision === d).length, of(label).length)
  const others = rs.filter((r) => r.label !== 'past')
  console.log(
    `${mode.padEnd(11)} | ${share('past', 'recall').padStart(21)} | ${share('action', 'action').padStart(26)} | ${share('chat', 'none').padStart(20)} | ${percent(others.filter((r) => r.decision === 'recall').length, others.length).padStart(25)} | ${median(rs.map((r) => r.ms))} ms`,
  )
}
console.log(
  'reference   | Laya 3-way choice: 71% of memory turns, 14% of other turns; Laya OR (1st person AND match >= 4): 84% / 21%',
)
