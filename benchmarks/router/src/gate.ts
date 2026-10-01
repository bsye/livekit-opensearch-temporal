// Alternative to a classifier: run memory search on every turn (BM25 top 20 -> MiniLM, ~60 ms) and
// add results only if the best match is relevant (re-ranker score above a threshold). Same 220 turns.
//
//   npm run gate -w router-bench
import { readFileSync } from 'node:fs';
import { rerankScores, warmReranker } from 'livekit-temporal/rerank';
import { ACTION, CHAT, PAST_EXTRA } from './turns.js';

const OPENSEARCH = process.env.OPENSEARCH_URL ?? 'http://localhost:9201';
const INDEX = process.env.MEMORY_INDEX ?? 'conversation-memory';
const sheet = readFileSync(new URL('../../../data/benchmarks/longmemeval/library-questions.md', import.meta.url), 'utf8');
const library = sheet.split('\n').filter((l) => l.startsWith('| ') && !l.startsWith('| type')).map((l) => ({ text: l.split(' | ')[2].trim(), past: true, group: 'library' }));
const turns = [
  ...library,
  ...PAST_EXTRA.map((text) => ({ text, past: true, group: 'past:conversational' })),
  ...ACTION.map((text) => ({ text, past: false, group: 'action' })),
  ...CHAT.map((text) => ({ text, past: false, group: 'chat' })),
];

await warmReranker();
const rows: { text: string; past: boolean; group: string; top: number; ms: number }[] = [];
for (const t of turns) {
  const start = performance.now();
  const res = await fetch(`${OPENSEARCH}/${INDEX}/_search`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ size: 20, _source: ['userText'], query: { match: { userText: t.text } } }),
  });
  const hits = ((await res.json()) as { hits: { hits: { _source: { userText: string } }[] } }).hits.hits;
  const scores = hits.length ? await rerankScores(t.text, hits.map((h) => h._source.userText)) : [];
  rows.push({ ...t, top: scores.length ? Math.max(...scores) : -99, ms: performance.now() - start });
}
const median = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
console.log(`${turns.length} turns, search + re-rank p50 ${median(rows.map((r) => r.ms)).toFixed(0)} ms\n`);
console.log('threshold | past turns with a relevant match | non-past turns that would get memory added (action / chat)');
for (const th of [0, 2, 4]) {
  const past = rows.filter((r) => r.past);
  const action = rows.filter((r) => r.group === 'action');
  const chat = rows.filter((r) => r.group === 'chat');
  const pct = (xs: typeof rows) => `${((100 * xs.filter((r) => r.top >= th).length) / xs.length).toFixed(0)}%`;
  console.log(`   ${String(th).padEnd(6)} | ${pct(past).padEnd(33)} | ${pct(action)} / ${pct(chat)}`);
}
const lib = rows.filter((r) => r.group === 'library');
const conv = rows.filter((r) => r.group === 'past:conversational');
console.log(`\nat threshold 0: library ${lib.filter((r) => r.top >= 0).length}/${lib.length}, conversational past ${conv.filter((r) => r.top >= 0).length}/${conv.length}`);
console.log('non-past turns above 0 (would get memory added):');
for (const r of rows.filter((r) => !r.past && r.top >= 0)) console.log(`  ${r.top.toFixed(1).padStart(5)}  ${r.group.padEnd(7)} ${r.text}`);
