import { ask } from '@voice/laya'
import { expandQuery } from './expansion.js'
import { rerankScores } from './rerank.js'
import { ensureIndex, type MemoryDoc, search } from './store.js'

export interface RecallQuery {
  question: string
  from?: number
  to?: number
  limit?: number
  excludeRoomSid?: string
}

export interface RecallHit {
  doc: MemoryDoc
  score: number
}

export interface RecallResult {
  hits: RecallHit[]
  route: 'rerank' | 'expand'
  timings: Record<string, number>
  ms: number
}

const RERANK_DEPTH = 20
const MIN_RERANK_SCORE = 0
const EXPANSION_WEIGHT = 0.5
const PREFERENCE_QUESTION =
  'Is the user asking for a recommendation or suggestion that should fit their own preferences or interests?'

export async function recall(q: RecallQuery): Promise<RecallResult> {
  const started = Date.now()
  const timings: Record<string, number> = {}
  const timed = async <T>(stage: string, fn: () => Promise<T>): Promise<T> => {
    const t = Date.now()
    const result = await fn()
    timings[stage] = Date.now() - t
    return result
  }
  await ensureIndex()
  const limit = q.limit ?? 5
  const filter = [{ range: { endedAt: { gte: q.from ?? 0, lte: q.to ?? Date.now() } } }]
  const must_not: unknown[] = [{ term: { fromMemory: true } }]
  if (q.excludeRoomSid) must_not.push({ term: { roomSid: q.excludeRoomSid } })

  const preference = await timed('route', () => isPreferenceQuestion(q.question))
  if (preference) {
    const expansion = await timed('expand', () => expandQuery(q.question))
    const hits = await timed('bm25', () =>
      search({
        size: limit,
        query: {
          bool: {
            filter,
            must_not,
            should: [
              { match: { userText: q.question } },
              { match: { userText: { query: expansion, boost: EXPANSION_WEIGHT } } },
            ],
            minimum_should_match: 1,
          },
        },
      }),
    )
    return {
      hits: hits.map((h) => ({ doc: h._source, score: h._score })),
      route: 'expand',
      timings,
      ms: Date.now() - started,
    }
  }

  const candidates = await timed('bm25', () =>
    search({ size: RERANK_DEPTH, query: { bool: { filter, must_not, must: [{ match: { userText: q.question } }] } } }),
  )
  const scores = await timed('rerank', async () =>
    candidates.length
      ? rerankScores(
          q.question,
          candidates.map((c) => c._source.userText),
        )
      : [],
  )
  const hits = candidates
    .map((c, i) => ({ doc: c._source, score: scores[i] }))
    .filter((h) => h.score >= MIN_RERANK_SCORE)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
  return { hits, route: 'rerank', timings, ms: Date.now() - started }
}

async function isPreferenceQuestion(question: string): Promise<boolean> {
  const { answers } = await ask(`User question: ${question}`, {
    q: { type: 'noul', instructions: PREFERENCE_QUESTION },
  })
  return (answers.q.noul ?? 0) >= 0.5
}

export const NOTHING_FOUND = 'Nothing found about that in earlier conversations (this conversation is above).'

export async function recallBrief(q: RecallQuery, sentences = 3): Promise<{ text: string; top: number; hits: number }> {
  const { hits } = await recall(q)
  const nothing = { text: NOTHING_FOUND, top: -Infinity, hits: 0 }
  if (!hits.length) return nothing
  const top = Math.max(...hits.map((h) => h.score))
  const candidates = hits.flatMap(({ doc }) =>
    doc.userText
      .split(/(?<=[.!?])\s+|\n+/)
      .filter((s) => s.trim().length > 12 && !s.trim().endsWith('?'))
      .map((s) => ({ s: s.trim().slice(0, 200), at: doc.endedAt })),
  )
  if (!candidates.length) return nothing
  const scores = await rerankScores(
    q.question,
    candidates.map((c) => c.s),
  )
  const best = candidates
    .map((c, i) => ({ ...c, score: scores[i] }))
    .sort((a, b) => b.score - a.score)
    .slice(0, sentences)
    .sort((a, b) => a.at - b.at)
    .map((c) => `${new Date(c.at).toDateString().slice(4, 10)}: ${c.s}`)
  const text = `${best.join(' | ')} (oldest first; the last one is the most recent)`
  return { text, top, hits: hits.length }
}
