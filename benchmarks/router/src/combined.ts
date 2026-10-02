// Router + relevance gate together: search memory only when Laya routes the turn to "past" AND the
// best re-ranked match is relevant (score >= threshold). For the speech-to-speech agent, which then
// forces the result into the model before it answers. Same 220 turns as run.ts / gate.ts.
//
//   npm run combined -w router-bench
import { readFileSync } from 'node:fs';
import { recall } from 'livekit-temporal/memory';
import { warmReranker } from 'livekit-temporal/rerank';
import { ACTION, CHAT, PAST_EXTRA } from './turns.js';

const LAYA = process.env.LAYA_BASE_URL ?? 'http://localhost:8100';
const sheet = readFileSync(new URL('../../../data/benchmarks/longmemeval/library-questions.md', import.meta.url), 'utf8');
const library = sheet.split('\n').filter((l) => l.startsWith('| ') && !l.startsWith('| type')).map((l) => l.split(' | ')[2].trim());
const turns = [
  ...library.map((text) => ({ text, past: true, group: 'library' })),
  ...PAST_EXTRA.map((text) => ({ text, past: true, group: 'conversational past' })),
  ...ACTION.map((text) => ({ text, past: false, group: 'action' })),
  ...CHAT.map((text) => ({ text, past: false, group: 'chat' })),
];

async function route(text: string): Promise<string> {
  const res = await fetch(`${LAYA}/v1/systemone`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      state: `User: ${text}`,
      questions: {
        q: {
          type: 'choice',
          instructions: 'What does the user want from the assistant in this turn?',
          criteria: {
            A: 'Information from their own past or from earlier conversations with the assistant',
            B: 'An action: a reminder, timer, message, email, calendar entry, call or other task',
            C: 'Conversation, general knowledge, advice or anything else',
          },
        },
      },
    }),
  });
  return ((await res.json()) as { answers: { q: { choice: string } } }).answers.q.choice;
}

await warmReranker();
const rows = [];
for (const t of turns) {
  const start = performance.now();
  const [choice, r] = await Promise.all([route(t.text), recall({ question: t.text, limit: 1 })]);
  rows.push({ ...t, laya: choice === 'A', top: r.hits[0]?.score ?? -99, ms: performance.now() - start });
}
const pct = (n: number, d: number) => `${((100 * n) / d).toFixed(0)}%`;
const past = rows.filter((r) => r.past);
const other = rows.filter((r) => !r.past);
console.log(`${rows.length} turns; Laya + recall in parallel p50 ${rows.map((r) => r.ms).sort((a, b) => a - b)[rows.length >> 1].toFixed(0)} ms\n`);
console.log('rule                         | past turns searched | non-past turns searched');
const rule = (name: string, f: (r: (typeof rows)[number]) => boolean) =>
  console.log(`${name.padEnd(28)} | ${pct(past.filter(f).length, past.length).padStart(19)} | ${pct(other.filter(f).length, other.length).padStart(5)} (${other.filter(f).length})`);
rule('laya only', (r) => r.laya);
for (const th of [0, 2, 4, 6]) rule(`gate >= ${th} only`, (r) => r.top >= th);
for (const th of [0, 2, 4, 6]) rule(`laya AND gate >= ${th}`, (r) => r.laya && r.top >= th);
for (const th of [4, 6, 8]) rule(`laya OR gate >= ${th}`, (r) => r.laya || r.top >= th);
// first person ("my degree", "did I"): a strong match then counts even when Laya says action/chat
const FIRST_PERSON = /\b(i|my|me|mine|i'm|i've|i'd|i'll)\b/i;
for (const th of [2, 4, 6]) rule(`laya OR (1st person AND >= ${th})`, (r) => r.laya || (FIRST_PERSON.test(r.text) && r.top >= th));
console.log('\nnon-past turns searched under "laya OR (1st person AND >= 4)":');
for (const r of other.filter((r) => r.laya || (FIRST_PERSON.test(r.text) && r.top >= 4))) console.log(`  ${r.group}: ${r.text} (${r.top.toFixed(1)})`);
console.log('\npast turns missed under "laya OR (1st person AND >= 4)":');
for (const r of past.filter((r) => !(r.laya || (FIRST_PERSON.test(r.text) && r.top >= 4))).slice(0, 12)) console.log(`  ${r.group}: ${r.text} (laya ${r.laya}, ${r.top.toFixed(1)})`);
