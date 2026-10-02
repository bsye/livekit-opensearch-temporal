/**
 * Instead of classifying the turn: search memory on every turn (BM25 top 20 → MiniLM) and use the
 * results only when the best match is relevant enough. Same 220 turns as run.ts.
 *   npm run gate -w @bench/router
 */
import { rerankScores, search, warmReranker } from '@voice/memory';
import { routerTurns, type Turn } from './turns.js';

const turns = routerTurns();
await warmReranker();
const rows: (Turn & { top: number; ms: number })[] = [];
for (const t of turns) {
  const start = performance.now();
  const hits = await search({ size: 20, _source: ['userText'], query: { match: { userText: t.text } } });
  const scores = hits.length ? await rerankScores(t.text, hits.map((h) => h._source.userText)) : [];
  rows.push({ ...t, top: scores.length ? Math.max(...scores) : -99, ms: performance.now() - start });
}

const median = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
const pct = (xs: typeof rows, th: number) => `${((100 * xs.filter((r) => r.top >= th).length) / xs.length).toFixed(0)}%`;
const past = rows.filter((r) => r.label === 'past');
const action = rows.filter((r) => r.label === 'action');
const chat = rows.filter((r) => r.label === 'chat');
console.log(`${turns.length} turns, search + re-rank p50 ${median(rows.map((r) => r.ms)).toFixed(0)} ms\n`);
console.log('threshold | past turns with a relevant match | non-past turns that would get memory (action / chat)');
for (const th of [0, 2, 4]) console.log(`   ${String(th).padEnd(6)} | ${pct(past, th).padEnd(33)} | ${pct(action, th)} / ${pct(chat, th)}`);
console.log('\nnon-past turns above 0:');
for (const r of rows.filter((r) => r.label !== 'past' && r.top >= 0)) console.log(`  ${r.top.toFixed(1).padStart(5)}  ${r.group.padEnd(7)} ${r.text}`);
