import { createReadStream, writeFileSync } from 'node:fs'
import { option } from '@bench/shared'
import { bulkIndex, deleteRoom, ensureIndex, type MemoryDoc, refresh, toMemoryDoc } from '@voice/memory'
import { tableCell } from '@voice/text'
import StreamArray from 'stream-json/streamers/StreamArray.js'
import { LIBRARY_DIR, LIBRARY_ROOM, LIBRARY_SHEET } from './library.js'

const USER = 'me'
const AGENT = 'assistant'
const N_QUESTIONS = Number(option('--questions', '100'))

interface Turn {
  role: 'user' | 'assistant'
  content: string
}
interface Question {
  question_id: string
  question_type: string
  question: string
  answer: string | number
  question_date: string
  answer_session_ids: string[]
  haystack_session_ids: string[]
  haystack_dates: string[]
  haystack_sessions: Turn[][]
}
type Summary = Omit<Question, 'haystack_sessions' | 'haystack_dates' | 'haystack_session_ids'> & {
  haystack: Set<string>
}

async function* questions(): AsyncGenerator<Question> {
  const stream = createReadStream(`${LIBRARY_DIR}/longmemeval_m_cleaned.json`).pipe(StreamArray.withParser())
  for await (const { value } of stream as AsyncIterable<{ value: Question }>) yield value
}

function parseDate(s: string): number {
  const [date, , time] = s.split(' ')
  const [year, month, day] = (date ?? '').split('/').map(Number)
  const [hours, minutes] = (time ?? '').split(':').map(Number)
  const at = new Date(year, month - 1, day, hours, minutes).getTime()
  if (Number.isNaN(at)) throw new Error(`bad date ${s}`)
  return at
}

console.log('scanning LongMemEval_M …')
const all: Summary[] = []
for await (const q of questions()) {
  const { haystack_sessions: _s, haystack_dates: _d, haystack_session_ids, ...rest } = q
  all.push({ ...rest, haystack: new Set(haystack_session_ids) })
}
const answerable = (h: Set<string>) =>
  all.filter((q) => q.answer_session_ids.length && q.answer_session_ids.every((id) => h.has(id)))
const base = all.map((q) => ({ q, n: answerable(q.haystack).length })).sort((a, b) => b.n - a.n)[0].q

const byType = new Map<string, Summary[]>()
const seenText = new Set<string>()
for (const q of all) {
  if (q.question_id.endsWith('_abs') || !q.answer_session_ids.length || seenText.has(q.question)) continue
  seenText.add(q.question)
  byType.set(q.question_type, [...(byType.get(q.question_type) ?? []), q])
}
const chosen: Summary[] = []
for (let i = 0; chosen.length < N_QUESTIONS && [...byType.values()].some((l) => l.length > i); i++) {
  for (const list of byType.values()) if (list[i] && chosen.length < N_QUESTIONS) chosen.push(list[i])
}
const needed = new Set<string>([...base.haystack, ...chosen.flatMap((q) => q.answer_session_ids)])
const offset = Date.now() - Math.max(...chosen.map((q) => parseDate(q.question_date)), parseDate(base.question_date))
console.log(
  `library: history of ${base.question_id} (${base.haystack.size} sessions) + evidence for ${chosen.length} questions → ${needed.size} sessions`,
)

await ensureIndex()
await deleteRoom(LIBRARY_ROOM)
let batch: MemoryDoc[] = []
let total = 0
const indexed = new Set<string>()
for await (const q of questions()) {
  for (const [s, session] of q.haystack_sessions.entries()) {
    const sid = q.haystack_session_ids[s]
    if (!needed.has(sid) || indexed.has(sid)) continue
    indexed.add(sid)
    const sessionStart = parseDate(q.haystack_dates[s]) + offset
    let minute = 0
    for (let i = 0; i < session.length - 1; i++) {
      if (session[i].role !== 'user' || session[i + 1].role !== 'assistant') continue
      batch.push(
        await toMemoryDoc({
          roomSid: `lme-${sid}`,
          roomName: LIBRARY_ROOM,
          user: USER,
          agent: AGENT,
          userText: session[i].content,
          agentText: session[i + 1].content,
          startedAt: sessionStart + minute * 60_000,
          endedAt: sessionStart + (minute + 1) * 60_000,
        }),
      )
      minute += 2
      total++
      if (batch.length >= 500) {
        await bulkIndex(batch)
        batch = []
      }
    }
    if (indexed.size % 100 === 0) console.log(`  ${indexed.size}/${needed.size} sessions, ${total} exchanges`)
  }
  if (indexed.size === needed.size) break
}
await bulkIndex(batch)
await refresh()
console.log(`indexed ${total} exchanges as room "${LIBRARY_ROOM}"`)

const fmt = (ms: number) => new Date(ms).toLocaleString('en-GB', { dateStyle: 'medium', timeStyle: 'short' })
const sheet = [
  `# Library questions (${chosen.length}, each answerable from the loaded memory)`,
  '',
  `Library: LongMemEval_M history of ${base.question_id} + evidence sessions for these questions: ${indexed.size} sessions,`,
  `${total} exchanges, dates shifted by ${(offset / 86_400_000).toFixed(0)} days so the timeline ends today. Each question was originally asked`,
  'on its own date (shown shifted); answers are as of that date, which matters for knowledge-update and temporal questions.',
  '',
  '| type | asked (shifted) | question | expected answer |',
  '|---|---|---|---|',
  ...chosen
    .sort((a, b) => a.question_type.localeCompare(b.question_type))
    .map(
      (q) =>
        `| ${q.question_type} | ${fmt(parseDate(q.question_date) + offset)} | ${tableCell(q.question)} | ${tableCell(q.answer)} |`,
    ),
].join('\n')
writeFileSync(LIBRARY_SHEET, `${sheet}\n`)
console.log(`question sheet: ${LIBRARY_SHEET}`)
