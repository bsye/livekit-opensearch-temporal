// Load a LongMemEval_M-based library into conversation memory as "your" past, to try recall by voice at
// realistic scale. A single LongMemEval history only holds evidence for 1-2 questions (its other ~480
// sessions are shared filler), so the library is: one full history (the bulk, ~480 sessions over
// months) + the evidence sessions of N questions spread over all question types, each at its real date.
// One shared shift moves the timeline so the latest question date is now. Every user→assistant exchange
// is indexed with Laya categories, and a question sheet with expected answers is written.
//
//   npm run load-library -w livekit-temporal [-- --questions 100]   (re-running replaces the library)
import { createReadStream, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import StreamArray from 'stream-json/streamers/StreamArray.js';
import { categorise, ensureIndex, exchangeText, type MemoryDoc } from './memory.js';

const OPENSEARCH_URL = process.env.OPENSEARCH_URL ?? 'http://localhost:9201';
const INDEX = process.env.MEMORY_INDEX ?? 'conversation-memory';
const DATA = new URL('../../../data/benchmarks/longmemeval/', import.meta.url);
const ROOM_NAME = 'longmemeval-library';
const USER = 'me';
const AGENT = 'assistant';
const N_QUESTIONS = Number(process.argv.includes('--questions') ? process.argv[process.argv.indexOf('--questions') + 1] : 100);

interface Turn {
  role: 'user' | 'assistant';
  content: string;
}
interface Question {
  question_id: string;
  question_type: string;
  question: string;
  answer: string | number;
  question_date: string;
  answer_session_ids: string[];
  haystack_session_ids: string[];
  haystack_dates: string[];
  haystack_sessions: Turn[][];
}
type Summary = Omit<Question, 'haystack_sessions' | 'haystack_dates' | 'haystack_session_ids'> & { haystack: Set<string> };

async function* questions(): AsyncGenerator<Question> {
  const stream = createReadStream(fileURLToPath(new URL('longmemeval_m_cleaned.json', DATA))).pipe(StreamArray.withParser());
  for await (const { value } of stream as AsyncIterable<{ value: Question }>) yield value;
}

/** "2023/05/20 (Sat) 02:21" → unix ms (local time). */
function parseDate(s: string): number {
  const m = s.match(/(\d{4})\/(\d{2})\/(\d{2}) \(\w+\) (\d{2}):(\d{2})/);
  if (!m) throw new Error(`bad date ${s}`);
  return new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5]).getTime();
}

// Pass 1: which history covers the evidence of the most questions?
console.log('scanning LongMemEval_M …');
const all: Summary[] = [];
for await (const q of questions()) {
  const { haystack_sessions: _s, haystack_dates: _d, haystack_session_ids, ...rest } = q;
  all.push({ ...rest, haystack: new Set(haystack_session_ids) });
}
const answerable = (h: Set<string>) => all.filter((q) => q.answer_session_ids.length && q.answer_session_ids.every((id) => h.has(id)));
const base = all.map((q) => ({ q, n: answerable(q.haystack).length })).sort((a, b) => b.n - a.n)[0].q;

// N questions spread across types (abstention excluded: their evidence is deliberately absent),
// skipping duplicate question texts so every question has one unambiguous answer
const byType = new Map<string, Summary[]>();
const seenText = new Set<string>();
for (const q of all) {
  if (q.question_id.endsWith('_abs') || !q.answer_session_ids.length || seenText.has(q.question)) continue;
  seenText.add(q.question);
  byType.set(q.question_type, [...(byType.get(q.question_type) ?? []), q]);
}
const chosen: Summary[] = [];
for (let i = 0; chosen.length < N_QUESTIONS && [...byType.values()].some((l) => l.length > i); i++) {
  for (const list of byType.values()) if (list[i] && chosen.length < N_QUESTIONS) chosen.push(list[i]);
}
const needed = new Set<string>([...base.haystack, ...chosen.flatMap((q) => q.answer_session_ids)]);
const offset = Date.now() - Math.max(...chosen.map((q) => parseDate(q.question_date)), parseDate(base.question_date));
console.log(`library: history of ${base.question_id} (${base.haystack.size} sessions) + evidence for ${chosen.length} questions → ${needed.size} sessions`);

// Pass 2: index that history's exchanges, dates shifted so its question date is now
await ensureIndex();
await post(`${OPENSEARCH_URL}/${INDEX}/_delete_by_query?refresh=true`, { query: { term: { roomName: ROOM_NAME } } });
let lines: string[] = [];
let total = 0;
const flush = async () => {
  if (!lines.length) return;
  const res = await post(`${OPENSEARCH_URL}/_bulk`, `${lines.join('\n')}\n`, 'application/x-ndjson');
  if ((res as { errors: boolean }).errors) throw new Error('bulk indexing errors');
  lines = [];
};
const indexed = new Set<string>();
for await (const q of questions()) {
  for (const [s, session] of q.haystack_sessions.entries()) {
    const sid = q.haystack_session_ids[s];
    if (!needed.has(sid) || indexed.has(sid)) continue;
    indexed.add(sid);
    const sessionStart = parseDate(q.haystack_dates[s]) + offset;
    let minute = 0;
    for (let i = 0; i < session.length - 1; i++) {
      if (session[i].role !== 'user' || session[i + 1].role !== 'assistant') continue;
      const exchange = {
        roomSid: `lme-${sid}`,
        roomName: ROOM_NAME,
        user: USER,
        agent: AGENT,
        userText: session[i].content,
        agentText: session[i + 1].content,
        startedAt: sessionStart + minute * 60_000,
        endedAt: sessionStart + (minute + 1) * 60_000,
      };
      minute += 2;
      const text = exchangeText(exchange);
      const doc: MemoryDoc = { ...exchange, text, ...(await categorise(text)) };
      lines.push(JSON.stringify({ index: { _index: INDEX, _id: `${exchange.roomSid}-${exchange.endedAt}` } }), JSON.stringify(doc));
      total++;
      if (lines.length >= 1000) await flush();
    }
    if (indexed.size % 100 === 0) console.log(`  ${indexed.size}/${needed.size} sessions, ${total} exchanges`);
  }
  if (indexed.size === needed.size) break;
}
await flush();
await post(`${OPENSEARCH_URL}/${INDEX}/_refresh`, {});
console.log(`indexed ${total} exchanges as room "${ROOM_NAME}"`);

// Question sheet
const fmt = (ms: number) => new Date(ms).toLocaleString('en-GB', { dateStyle: 'medium', timeStyle: 'short' });
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
    .map((q) => `| ${q.question_type}${q.question_id.endsWith('_abs') ? ' (abstention)' : ''} | ${fmt(parseDate(q.question_date) + offset)} | ${q.question.replace(/\|/g, '/')} | ${String(q.answer).replace(/\|/g, '/').replace(/\n/g, ' ')} |`),
].join('\n');
const sheetFile = new URL('library-questions.md', DATA);
writeFileSync(sheetFile, `${sheet}\n`);
console.log(`question sheet: ${fileURLToPath(sheetFile)}`);

async function post(url: string, body: unknown, type = 'application/json'): Promise<unknown> {
  const res = await fetch(url, { method: 'POST', headers: { 'content-type': type }, body: typeof body === 'string' ? body : JSON.stringify(body) });
  if (!res.ok) throw new Error(`${url}: ${res.status} ${await res.text()}`);
  return res.json();
}
