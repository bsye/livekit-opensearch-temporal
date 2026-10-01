// Conversation memory without embeddings: every exchange (user turn(s) + agent reply) is stored
// in OpenSearch with exact times and Laya categories. recall: BM25 over the user's words within the
// time range, re-ranked by a small cross-encoder (MiniLM); Laya routes preference questions to LLM
// query expansion instead. Pipeline chosen by benchmarks/longmemeval (README there).
// Used by the Temporal worker (indexing activity) and the agent (recall tool). Not workflow-safe.

import { rerankScores } from './rerank.js';

const OPENSEARCH_URL = process.env.OPENSEARCH_URL ?? 'http://localhost:9201';
const INDEX = process.env.MEMORY_INDEX ?? 'conversation-memory';
const LAYA = process.env.LAYA_BASE_URL ?? 'http://localhost:8100';

// Fixed taxonomy: Laya picks from options, it doesn't invent labels
export const TOPICS = {
  work: 'work, projects, meetings, colleagues',
  personal: 'family, friends, home, plans',
  travel: 'trips, flights, hotels, places',
  health: 'health, doctors, fitness',
  money: 'money, payments, bills, shopping',
  tech: 'software, computers, technical topics',
  smalltalk: 'greetings, chit-chat, thanks',
} as const;
export type Topic = keyof typeof TOPICS;

export interface Exchange {
  roomSid: string;
  roomName?: string;
  user: string;
  agent: string;
  userText: string;
  agentText: string;
  startedAt: number; // unix ms, first user turn
  endedAt: number; // unix ms, agent reply
  // the agent answered from memory: asking about the past isn't new evidence, and a wrong answer
  // must not be recalled later as if the user had said it
  fromMemory?: boolean;
}

export interface MemoryDoc extends Exchange {
  text: string;
  topic: Topic;
  topicConfidence: number;
  hasTask: boolean; // contains a task, reminder or commitment
}

export interface RecallHit {
  doc: MemoryDoc;
  score: number; // Laya P(the excerpt is about the question)
}

const MAPPING = {
  mappings: {
    properties: {
      roomSid: { type: 'keyword' },
      roomName: { type: 'keyword' },
      user: { type: 'keyword' },
      agent: { type: 'keyword' },
      userText: { type: 'text', analyzer: 'english' },
      agentText: { type: 'text', analyzer: 'english' },
      text: { type: 'text', analyzer: 'english' },
      startedAt: { type: 'date', format: 'epoch_millis' },
      endedAt: { type: 'date', format: 'epoch_millis' },
      topic: { type: 'keyword' },
      topicConfidence: { type: 'float' },
      hasTask: { type: 'boolean' },
      fromMemory: { type: 'boolean' },
    },
  },
};

let indexReady: Promise<void> | undefined;

export function ensureIndex(): Promise<void> {
  indexReady ??= (async () => {
    const exists = await fetch(`${OPENSEARCH_URL}/${INDEX}`, { method: 'HEAD' });
    if (exists.ok) return;
    const res = await fetch(`${OPENSEARCH_URL}/${INDEX}`, {
      method: 'PUT',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(MAPPING),
    });
    // a concurrent creator may have won the race
    if (!res.ok && !(await res.text()).includes('resource_already_exists_exception')) {
      throw new Error(`create index ${INDEX}: ${res.status}`);
    }
  })().catch((err) => {
    indexReady = undefined;
    throw err;
  });
  return indexReady;
}

export function exchangeText(e: Pick<Exchange, 'userText' | 'agentText'>): string {
  return `User: ${e.userText}\nAssistant: ${e.agentText}`;
}

/** Laya categories for one exchange: topic (choice) and whether it holds a task (noul), one call. */
export async function categorise(text: string): Promise<Pick<MemoryDoc, 'topic' | 'topicConfidence' | 'hasTask'>> {
  const res = await layaPost('/v1/systemone', {
    state: text,
    questions: {
      topic: { type: 'choice', instructions: 'What is this conversation mainly about?', criteria: TOPICS },
      task: { type: 'noul', instructions: 'Does the user ask for a task, reminder or commitment?' },
    },
  });
  const answers = res.answers as {
    topic: { choice: Topic; confidence: number };
    task: { noul: number };
  };
  return {
    topic: answers.topic.choice,
    topicConfidence: answers.topic.confidence,
    hasTask: answers.task.noul >= 0.5,
  };
}

/** Index one exchange (idempotent: the id is derived from room and time). */
export async function indexExchange(e: Exchange): Promise<MemoryDoc> {
  await ensureIndex();
  const text = exchangeText(e);
  const doc: MemoryDoc = { ...e, text, ...(await categorise(text)) };
  const id = encodeURIComponent(`${e.roomSid}-${e.endedAt}`);
  const res = await fetch(`${OPENSEARCH_URL}/${INDEX}/_doc/${id}`, {
    method: 'PUT',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(doc),
  });
  if (!res.ok) throw new Error(`index exchange: ${res.status} ${await res.text()}`);
  return doc;
}

export interface RecallQuery {
  question: string; // the user's question in their own words, e.g. "what did I say about the Lisbon trip?"
  from?: number; // unix ms
  to?: number; // unix ms
  limit?: number; // hits returned (default 5)
  excludeRoomSid?: string; // the current conversation: not "the past" yet
}

export interface RecallResult {
  hits: RecallHit[];
  route: 'rerank' | 'expand'; // re-ranked BM25, or LLM-expanded BM25 for preference questions
  timings: Record<string, number>; // ms per stage
  ms: number;
}

// Benchmarked pipeline (benchmarks/longmemeval, LongMemEval_M): BM25 over the user's words, top 20
// re-ranked by MiniLM (0.652 recall@5, ~70 ms). Re-rankers trained on web search hurt questions
// asking for the user's preferences, so Laya routes those (~5 ms) to LLM query expansion instead.
const RERANK_DEPTH = 20;
// MiniLM scores are logits: below 0 the model judges the passage not relevant. Passing those on
// let the LLM blend unrelated messages into answers ("Computer Science from UCLA" from a message
// about applying to a master's), so they're dropped; no hit left means "I don't remember".
const MIN_RERANK_SCORE = 0;
const PREFERENCE_THRESHOLD = 0.5;
const EXPANSION_WEIGHT = 0.5;
const PREFERENCE_QUESTION =
  'Is the user asking for a recommendation or suggestion that should fit their own preferences or interests?';

export async function recall(q: RecallQuery): Promise<RecallResult> {
  const started = Date.now();
  const timings: Record<string, number> = {};
  const lap = (stage: string, t: number) => (timings[stage] = Date.now() - t);
  await ensureIndex();
  const limit = q.limit ?? 5;
  const filter = [{ range: { endedAt: { gte: q.from ?? 0, lte: q.to ?? Date.now() } } }];
  const must_not: unknown[] = [{ term: { fromMemory: true } }];
  if (q.excludeRoomSid) must_not.push({ term: { roomSid: q.excludeRoomSid } });

  let t = Date.now();
  const preference = await isPreferenceQuestion(q.question);
  lap('route', t);

  if (preference) {
    t = Date.now();
    const expansion = await expandQuery(q.question);
    lap('expand', t);
    t = Date.now();
    const hits = await search({
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
    });
    lap('bm25', t);
    return { hits: hits.map((h) => ({ doc: h._source, score: h._score })), route: 'expand', timings, ms: Date.now() - started };
  }

  t = Date.now();
  const candidates = await search({ size: RERANK_DEPTH, query: { bool: { filter, must_not, must: [{ match: { userText: q.question } }] } } });
  lap('bm25', t);
  t = Date.now();
  const scores = candidates.length ? await rerankScores(q.question, candidates.map((c) => c._source.userText)) : [];
  lap('rerank', t);
  const hits = candidates
    .map((c, i) => ({ doc: c._source, score: scores[i] }))
    .filter((h) => h.score >= MIN_RERANK_SCORE)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);
  return { hits, route: 'rerank', timings, ms: Date.now() - started };
}

async function isPreferenceQuestion(question: string): Promise<boolean> {
  const res = await layaPost('/v1/systemone', {
    state: `User question: ${question}`,
    questions: { q: { type: 'noul', instructions: PREFERENCE_QUESTION } },
  });
  return Number((res.answers as { q: { noul: number } }).q.noul) >= PREFERENCE_THRESHOLD;
}

// LLM query expansion: words the user most likely used back then (benchmarks/longmemeval/src/expansion.ts)
const EXPANSION_SYSTEM =
  'You generate search keywords. Given a question a user asks about their own past chats, output the words and short ' +
  'phrases they most likely used when they first talked about it: specific nouns, synonyms, related terms. Never guess ' +
  'the answer. Output only 10 to 20 comma-separated terms.';

async function expandQuery(question: string): Promise<string> {
  const res = await fetch(`${process.env.LLM_BASE_URL ?? 'http://localhost:1234/v1'}/chat/completions`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      model: process.env.LLM_MODEL,
      reasoning_effort: 'none',
      temperature: 0,
      max_tokens: 120,
      messages: [
        { role: 'system', content: EXPANSION_SYSTEM },
        { role: 'user', content: 'Question: What was the name of the restaurant I liked in Rome?' },
        {
          role: 'assistant',
          content: 'restaurant, Rome, Italy, dinner, trattoria, pizzeria, pasta, food, ate, meal, trip, vacation, recommend, favorite, loved',
        },
        { role: 'user', content: `Question: ${question}` },
      ],
    }),
  });
  if (!res.ok) throw new Error(`query expansion: ${res.status}`);
  return ((await res.json()) as { choices: { message: { content: string } }[] }).choices[0].message.content;
}

async function search(body: unknown): Promise<{ _id: string; _score: number; _source: MemoryDoc }[]> {
  const res = await fetch(`${OPENSEARCH_URL}/${INDEX}/_search`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`search: ${res.status} ${await res.text()}`);
  return ((await res.json()) as { hits: { hits: { _id: string; _score: number; _source: MemoryDoc }[] } }).hits.hits;
}

async function layaPost(path: string, body: unknown): Promise<Record<string, unknown>> {
  const res = await fetch(`${LAYA}${path}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`laya ${path}: ${res.status} ${await res.text()}`);
  return (await res.json()) as Record<string, unknown>;
}
