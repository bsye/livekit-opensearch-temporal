import { createReadStream, existsSync, readFileSync, writeFileSync } from 'node:fs'
import { env } from '@voice/config'
import { rerankScores, type SearchHit, search, TOPICS } from '@voice/memory'
import StreamArray from 'stream-json/streamers/StreamArray.js'
import { rrf } from './embeddings.js'
import { LIBRARY_DIR, LIBRARY_ROOM, libraryQuestions } from './library.js'

const DEPTH = 20
const TOPIC_BOOST = 2
const CACHE = `${LIBRARY_DIR}/results/library.omni-topic-queries.json`

const TOOL = {
  type: 'function',
  function: {
    name: 'recall',
    description: "Search the user's past conversations.",
    parameters: {
      type: 'object',
      properties: {
        queries: {
          type: 'array',
          items: { type: 'string' },
          minItems: 1,
          maxItems: 3,
          description: '1 to 3 short keyword queries',
        },
        topic: { type: 'string', enum: Object.keys(TOPICS), description: `Only if clear: ${JSON.stringify(TOPICS)}` },
      },
      required: ['queries'],
    },
  },
}
const SYSTEM =
  "You find what the user said in past conversations. For the user's question, call recall with 1 to 3 short keyword " +
  'queries (the words the user most likely used back then) and, only if it is clear, the topic. Never guess the answer.'
const EXAMPLE = [
  { role: 'user', content: 'How much did I spend on the flights for my Lisbon trip?' },
  {
    role: 'assistant',
    content: '',
    tool_calls: [
      {
        id: 'example',
        type: 'function',
        function: {
          name: 'recall',
          arguments: JSON.stringify({
            queries: ['Lisbon flights price', 'plane tickets Portugal cost'],
            topic: 'travel',
          }),
        },
      },
    ],
  },
  {
    role: 'tool',
    tool_call_id: 'example',
    content: '[Mar 2] User: I just booked my flights to Lisbon, 420 euros return.',
  },
  { role: 'assistant', content: 'You paid 420 euros for the return flights.' },
]

type Call = { queries: string[]; topic?: string }
const cache: Record<string, Call> = existsSync(CACHE) ? JSON.parse(readFileSync(CACHE, 'utf8')) : {}

async function modelCall(question: string): Promise<Call> {
  if (cache[question]) return cache[question]
  const res = await fetch(`${env('OMNI_BASE_URL')}/chat/completions`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      model: env('OMNI_MODEL'),
      temperature: 0,
      max_tokens: 120,
      tools: [TOOL],
      messages: [{ role: 'system', content: SYSTEM }, ...EXAMPLE, { role: 'user', content: question }],
    }),
  })
  const message = (
    (await res.json()) as { choices: { message: { tool_calls?: { function: { arguments: string } }[] } }[] }
  ).choices[0].message
  const args = message.tool_calls?.[0]?.function.arguments
  const parsed = (args ? JSON.parse(args) : {}) as Partial<Call>
  const call = {
    queries: (parsed.queries ?? []).filter((q) => typeof q === 'string' && q.trim()).slice(0, 3),
    topic: parsed.topic && parsed.topic in TOPICS ? parsed.topic : undefined,
  }
  cache[question] = call
  writeFileSync(CACHE, JSON.stringify(cache))
  return call
}

async function evidence(questions: Set<string>): Promise<Map<string, Set<string>>> {
  const out = new Map<string, Set<string>>()
  const stream = createReadStream(`${LIBRARY_DIR}/longmemeval_m_cleaned.json`).pipe(StreamArray.withParser())
  for await (const { value } of stream as AsyncIterable<{
    value: { question: string; answer_session_ids: string[] }
  }>) {
    if (questions.has(value.question) && !out.has(value.question))
      out.set(value.question, new Set(value.answer_session_ids.map((s) => `lme-${s}`)))
    if (out.size === questions.size) break
  }
  return out
}

const library = { term: { roomName: LIBRARY_ROOM } }

async function bm25(text: string, topic?: { filter?: string; boost?: string }): Promise<SearchHit[]> {
  const filter: unknown[] = [library]
  if (topic?.filter) filter.push({ term: { topic: topic.filter } })
  const should: unknown[] = topic?.boost ? [{ term: { topic: { value: topic.boost, boost: TOPIC_BOOST } } }] : []
  return search({ size: DEPTH, query: { bool: { filter, must: [{ match: { userText: text } }], should } } })
}

async function merged(question: string, call: Call, topic?: { filter?: string; boost?: string }): Promise<SearchHit[]> {
  const lists = await Promise.all([question, ...call.queries].map((t) => bm25(t, topic)))
  const docs = new Map<string, SearchHit>()
  for (const l of lists) for (const h of l) docs.set(h._id, h)
  const ids = [...docs.keys()]
  const order = rrf(
    lists.map((l) => l.map((h) => ids.indexOf(h._id))),
    ids.length,
  )
  return order.slice(0, DEPTH).map((i) => docs.get(ids[i])!)
}

async function rerank(question: string, hits: SearchHit[]): Promise<SearchHit[]> {
  if (!hits.length) return []
  const scores = await rerankScores(
    question,
    hits.map((h) => h._source.userText),
  )
  return hits
    .map((h, i) => ({ h, s: scores[i] }))
    .sort((a, b) => b.s - a.s)
    .map((x) => x.h)
}

const questions = libraryQuestions()
const gold = await evidence(new Set(questions.map((q) => q.question)))
const methods = ['minilm', 'mq', 'mq+filter', 'mq+boost'] as const
const score: Record<string, { top1: number; top5: number; ms: number[] }> = Object.fromEntries(
  methods.map((m) => [m, { top1: 0, top5: 0, ms: [] }]),
)
const byType: Record<string, Record<string, number>> = {}
let topicGiven = 0
for (const q of questions) {
  const want = gold.get(q.question)
  if (!want) continue
  const call = await modelCall(q.question)
  if (call.topic) topicGiven++
  for (const m of methods) {
    const t = Date.now()
    const hits =
      m === 'minilm'
        ? await rerank(q.question, await bm25(q.question))
        : m === 'mq'
          ? await rerank(q.question, await merged(q.question, call))
          : m === 'mq+filter'
            ? await rerank(q.question, await merged(q.question, call, { filter: call.topic }))
            : await rerank(q.question, await merged(q.question, call, { boost: call.topic }))
    score[m].ms.push(Date.now() - t)
    const ok5 = hits.slice(0, 5).some((h) => want.has(h._source.roomSid))
    score[m].top5 += ok5 ? 1 : 0
    score[m].top1 += hits[0] && want.has(hits[0]._source.roomSid) ? 1 : 0
    ;(byType[q.type] ??= {})[m] = (byType[q.type]?.[m] ?? 0) + (ok5 ? 1 : 0)
  }
}
const n = [...gold.values()].length
const p50 = (xs: number[]) => [...xs].sort((a, b) => a - b)[xs.length >> 1]
console.log(`${n} library questions; the model gave a topic for ${topicGiven}\n`)
console.log('| method | evidence in top 5 | first | p50 ms (search + re-rank) |\n|---|---|---|---|')
for (const m of methods)
  console.log(`| ${m} | ${(score[m].top5 / n).toFixed(2)} | ${(score[m].top1 / n).toFixed(2)} | ${p50(score[m].ms)} |`)
console.log(`\n| question type | ${methods.join(' | ')} |\n|---|${methods.map(() => '---').join('|')}|`)
for (const [type, s] of Object.entries(byType).sort()) {
  const total = questions.filter((q) => q.type === type && gold.has(q.question)).length
  console.log(`| ${type} (${total}) | ${methods.map((m) => `${s[m] ?? 0}/${total}`).join(' | ')} |`)
}
