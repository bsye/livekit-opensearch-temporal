// Router test: can Laya tell, from one user turn, whether to run memory recall (past), leave it to
// the LLM's tools (action), or just answer (chat)? Three phrasings, all fixed before running.
//
//   npm run bench -w router-bench
import { readFileSync } from 'node:fs';
import { ACTION, CHAT, PAST_EXTRA } from './turns.js';

const LAYA = process.env.LAYA_BASE_URL ?? 'http://localhost:8100';
type Label = 'past' | 'action' | 'chat';

// the 100 LongMemEval library questions (all need memory), with their types for the breakdown
const sheet = readFileSync(new URL('../../../data/benchmarks/longmemeval/library-questions.md', import.meta.url), 'utf8');
const library = sheet
  .split('\n')
  .filter((l) => l.startsWith('| ') && !l.startsWith('| type'))
  .map((l) => {
    const cells = l.split(' | ');
    return { text: cells[2].trim(), label: 'past' as Label, group: `library:${cells[0].replace('| ', '').replace(' (abstention)', '')}` };
  });
const turns = [
  ...library,
  ...PAST_EXTRA.map((text) => ({ text, label: 'past' as Label, group: 'past:conversational' })),
  ...ACTION.map((text) => ({ text, label: 'action' as Label, group: 'action' })),
  ...CHAT.map((text) => ({ text, label: 'chat' as Label, group: 'chat' })),
];

const PHRASINGS = {
  // binary: does this turn need memory?
  'noul-past': {
    type: 'noul',
    instructions: 'Is the user asking about something from their own past, or something they told the assistant earlier?',
  },
  'noul-needs-memory': {
    type: 'noul',
    instructions:
      'To answer well, does the assistant need something the user said in earlier conversations (their life, facts about them, or their preferences)?',
  },
  // three-way choice with neutral keys (Laya's noul labels can dominate; see its model card)
  choice: {
    type: 'choice',
    instructions: 'What does the user want from the assistant in this turn?',
    criteria: {
      A: 'Information from their own past or from earlier conversations with the assistant',
      B: 'An action: a reminder, timer, message, email, calendar entry, call or other task',
      C: 'Conversation, general knowledge, advice or anything else',
    },
  },
} as const;

async function ask(text: string, question: Record<string, unknown>) {
  const t = performance.now();
  const res = await fetch(`${LAYA}/v1/systemone`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ state: `User: ${text}`, questions: { q: question } }),
  });
  const body = (await res.json()) as { answers: { q: Record<string, unknown> }; latency_ms: number };
  return { answer: body.answers.q, httpMs: performance.now() - t, modelMs: body.latency_ms };
}

const pct = (n: number, d: number) => `${((100 * n) / Math.max(d, 1)).toFixed(0)}%`;
const median = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
await ask('warm up', PHRASINGS['noul-past']);

console.log(`${turns.length} turns: ${turns.filter((t) => t.label === 'past').length} past, ${ACTION.length} action, ${CHAT.length} chat\n`);
for (const [name, question] of Object.entries(PHRASINGS)) {
  const results = [];
  for (const turn of turns) {
    const { answer, httpMs, modelMs } = await ask(turn.text, question);
    const predicted: Label =
      question.type === 'noul'
        ? Number(answer.noul) >= 0.5 ? 'past' : 'chat'
        : ({ A: 'past', B: 'action', C: 'chat' } as const)[answer.choice as 'A' | 'B' | 'C'];
    results.push({ ...turn, predicted, httpMs, modelMs });
  }
  const isPast = (l: Label) => l === 'past';
  const tp = results.filter((r) => isPast(r.label) && isPast(r.predicted)).length;
  const fp = results.filter((r) => !isPast(r.label) && isPast(r.predicted)).length;
  const fn = results.filter((r) => isPast(r.label) && !isPast(r.predicted)).length;
  console.log(`== ${name}`);
  console.log(`   memory routing: recall ${pct(tp, tp + fn)} of past turns, precision ${pct(tp, tp + fp)}, ${fp} non-past turns sent to memory`);
  if (question.type === 'choice') {
    const correct = results.filter((r) => r.label === r.predicted).length;
    console.log(`   3-way accuracy ${pct(correct, results.length)}; actions recognised ${pct(results.filter((r) => r.label === 'action' && r.predicted === 'action').length, ACTION.length)}`);
  }
  console.log(`   latency: model p50 ${median(results.map((r) => r.modelMs)).toFixed(1)} ms, with HTTP p50 ${median(results.map((r) => r.httpMs)).toFixed(1)} ms`);
  const groups = [...new Set(results.map((r) => r.group))];
  for (const g of groups) {
    const rs = results.filter((r) => r.group === g);
    const wantPast = rs[0].label === 'past';
    const ok = rs.filter((r) => isPast(r.predicted) === wantPast).length;
    console.log(`   ${g.padEnd(38)} ${wantPast ? 'sent to memory' : 'kept out of memory'}: ${pct(ok, rs.length)} (n=${rs.length})`);
  }
  const misses = results.filter((r) => isPast(r.label) !== isPast(r.predicted)).slice(0, 6);
  for (const m of misses) console.log(`     ✗ ${m.label}→${m.predicted}: ${m.text.slice(0, 90)}`);
  console.log();
}
