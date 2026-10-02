import { ask } from '@voice/laya';
import { expandQuery } from './expansion.js';
import { rerankScores } from './rerank.js';
import { ensureIndex, search, type MemoryDoc } from './store.js';

export interface RecallQuery {
  question: string;
  from?: number;
  to?: number;
  limit?: number;
  /** The current conversation isn't "the past" yet. */
  excludeRoomSid?: string;
}

export interface RecallHit {
  doc: MemoryDoc;
  score: number;
}

export interface RecallResult {
  hits: RecallHit[];
  route: 'rerank' | 'expand';
  timings: Record<string, number>;
  ms: number;
}

// Chosen on LongMemEval_M (benchmarks/longmemeval): BM25 top 20 re-ranked by MiniLM.
const RERANK_DEPTH = 20;
// MiniLM logits below 0 mean "not relevant"; passing them on let the LLM blend unrelated messages into answers.
const MIN_RERANK_SCORE = 0;
const EXPANSION_WEIGHT = 0.5;
const PREFERENCE_QUESTION =
  'Is the user asking for a recommendation or suggestion that should fit their own preferences or interests?';

/**
 * BM25 over the user's words, re-ranked by a cross-encoder. Re-rankers trained on web search hurt
 * "what would I like?" questions, so Laya routes those to LLM query expansion instead.
 */
export async function recall(q: RecallQuery): Promise<RecallResult> {
  const started = Date.now();
  const timings: Record<string, number> = {};
  const timed = async <T>(stage: string, fn: () => Promise<T>): Promise<T> => {
    const t = Date.now();
    const result = await fn();
    timings[stage] = Date.now() - t;
    return result;
  };
  await ensureIndex();
  const limit = q.limit ?? 5;
  const filter = [{ range: { endedAt: { gte: q.from ?? 0, lte: q.to ?? Date.now() } } }];
  const must_not: unknown[] = [{ term: { fromMemory: true } }];
  if (q.excludeRoomSid) must_not.push({ term: { roomSid: q.excludeRoomSid } });

  const preference = await timed('route', () => isPreferenceQuestion(q.question));
  if (preference) {
    const expansion = await timed('expand', () => expandQuery(q.question));
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
    );
    return { hits: hits.map((h) => ({ doc: h._source, score: h._score })), route: 'expand', timings, ms: Date.now() - started };
  }

  const candidates = await timed('bm25', () =>
    search({ size: RERANK_DEPTH, query: { bool: { filter, must_not, must: [{ match: { userText: q.question } }] } } }),
  );
  const scores = await timed('rerank', async () =>
    candidates.length ? rerankScores(q.question, candidates.map((c) => c._source.userText)) : [],
  );
  const hits = candidates
    .map((c, i) => ({ doc: c._source, score: scores[i] }))
    .filter((h) => h.score >= MIN_RERANK_SCORE)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);
  return { hits, route: 'rerank', timings, ms: Date.now() - started };
}

async function isPreferenceQuestion(question: string): Promise<boolean> {
  const { answers } = await ask(`User question: ${question}`, { q: { type: 'noul', instructions: PREFERENCE_QUESTION } });
  return (answers.q.noul ?? 0) >= 0.5;
}

// says *earlier* conversations: the current one is in the model's chat history, and a bare "nothing
// found" made it deny what the user had said a minute before
export const NOTHING_FOUND = 'Nothing found about that in earlier conversations (this conversation is above).';

/**
 * For the speech-to-speech model, which reads a tool result one token per 80 ms frame: the few
 * sentences the re-ranker scores highest across the top hits, dated and oldest first.
 */
export async function recallBrief(q: RecallQuery, sentences = 3): Promise<{ text: string; top: number; hits: number }> {
  const { hits } = await recall(q);
  const nothing = { text: NOTHING_FOUND, top: -Infinity, hits: 0 };
  if (!hits.length) return nothing;
  const top = Math.max(...hits.map((h) => h.score));
  const candidates = hits.flatMap(({ doc }) =>
    doc.userText
      .split(/(?<=[.!?])\s+|\n+/)
      // statements only: a question isn't evidence ("What is Rachel's specialty?" asked in an earlier
      // call came back as the answer to the same question)
      .filter((s) => s.trim().length > 12 && !s.trim().endsWith('?'))
      .map((s) => ({ s: s.trim().slice(0, 200), at: doc.endedAt })),
  );
  if (!candidates.length) return nothing;
  const scores = await rerankScores(q.question, candidates.map((c) => c.s));
  const text =
    candidates
      .map((c, i) => ({ ...c, score: scores[i] }))
      .sort((a, b) => b.score - a.score)
      .slice(0, sentences)
      .sort((a, b) => a.at - b.at)
      .map((c) => `${new Date(c.at).toDateString().slice(4, 10)}: ${c.s}`)
      .join(' | ') + ' (oldest first; the last one is the most recent)';
  return { text, top, hits: hits.length };
}
