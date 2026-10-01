// Conversation memory without embeddings: every exchange (user turn(s) + agent reply) is stored
// in OpenSearch with exact times and Laya categories; recall narrows by time and keywords in
// OpenSearch, then scores every candidate in parallel with Laya (batched on the GPU, ~0.5ms each).
// Used by the Temporal worker (indexing activity) and the agent (recall tool). Not workflow-safe.

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
  question: string; // what to look for, e.g. "the trip to Lisbon"
  from?: number; // unix ms
  to?: number; // unix ms
  limit?: number; // hits returned (default 5)
}

// Candidates scanned by Laya: the most recent in range, plus keyword matches that may be older
const RECENT_CANDIDATES = 2000;
const KEYWORD_CANDIDATES = 200;
const MIN_SCORE = 0.5;

/** Time + keyword narrowing in OpenSearch, then a parallel Laya scan of every candidate. */
export async function recall(q: RecallQuery): Promise<{ hits: RecallHit[]; scanned: number; ms: number }> {
  const started = Date.now();
  await ensureIndex();
  const range = { range: { endedAt: { gte: q.from ?? 0, lte: q.to ?? Date.now() } } };
  const [recent, keyword] = await Promise.all([
    search({ query: { bool: { filter: [range] } }, sort: [{ endedAt: 'desc' }], size: RECENT_CANDIDATES }),
    search({
      query: { bool: { filter: [range], must: [{ match: { text: q.question } }] } },
      size: KEYWORD_CANDIDATES,
    }),
  ]);
  const candidates = new Map<string, MemoryDoc>();
  for (const hit of [...keyword, ...recent]) candidates.set(hit._id, hit._source);
  const docs = [...candidates.values()];
  if (docs.length === 0) return { hits: [], scanned: 0, ms: Date.now() - started };

  const { scores } = (await layaPost('/v1/scan', {
    states: docs.map((d) => d.text),
    question: `Does this conversation talk about ${q.question}?`,
  })) as { scores: number[] };
  const hits = docs
    .map((doc, i) => ({ doc, score: scores[i] }))
    .filter((h) => h.score >= MIN_SCORE)
    .sort((a, b) => b.score - a.score)
    .slice(0, q.limit ?? 5);
  return { hits, scanned: docs.length, ms: Date.now() - started };
}

async function search(body: unknown): Promise<{ _id: string; _source: MemoryDoc }[]> {
  const res = await fetch(`${OPENSEARCH_URL}/${INDEX}/_search`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`search: ${res.status} ${await res.text()}`);
  return ((await res.json()) as { hits: { hits: { _id: string; _source: MemoryDoc }[] } }).hits.hits;
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
