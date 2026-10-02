/**
 * Can Laya tell from one user turn whether to search memory (past), leave it to the tools (action),
 * or just answer (chat)? Three phrasings, all fixed before running.
 *   npm run bench -w @bench/router
 */
import { ask, type Question, ROUTE_QUESTION } from '@voice/laya'
import { ACTION, CHAT, type Label, routerTurns } from './turns.js'

const PHRASINGS: Record<string, Question> = {
  'noul-past': {
    type: 'noul',
    instructions:
      'Is the user asking about something from their own past, or something they told the assistant earlier?',
  },
  'noul-needs-memory': {
    type: 'noul',
    instructions:
      'To answer well, does the assistant need something the user said in earlier conversations (their life, facts about them, or their preferences)?',
  },
  // neutral keys: Laya's yes/no labels can dominate a noul question
  choice: ROUTE_QUESTION,
}
const CHOICES: Record<string, Label> = { A: 'past', B: 'action', C: 'chat' }

const turns = routerTurns()
const pct = (n: number, d: number) => `${((100 * n) / Math.max(d, 1)).toFixed(0)}%`
const median = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)]
const isPast = (l: Label) => l === 'past'

await ask('warm up', { q: PHRASINGS['noul-past'] })
console.log(
  `${turns.length} turns: ${turns.filter((t) => isPast(t.label)).length} past, ${ACTION.length} action, ${CHAT.length} chat\n`,
)

for (const [name, question] of Object.entries(PHRASINGS)) {
  const results = []
  for (const turn of turns) {
    const t = performance.now()
    const { answers, latencyMs } = await ask(`User: ${turn.text}`, { q: question })
    const predicted: Label =
      question.type === 'noul' ? ((answers.q.noul ?? 0) >= 0.5 ? 'past' : 'chat') : CHOICES[answers.q.choice ?? 'C']
    results.push({ ...turn, predicted, httpMs: performance.now() - t, modelMs: latencyMs })
  }
  const tp = results.filter((r) => isPast(r.label) && isPast(r.predicted)).length
  const fp = results.filter((r) => !isPast(r.label) && isPast(r.predicted)).length
  const fn = results.filter((r) => isPast(r.label) && !isPast(r.predicted)).length
  console.log(`== ${name}`)
  console.log(
    `   memory routing: recall ${pct(tp, tp + fn)} of past turns, precision ${pct(tp, tp + fp)}, ${fp} non-past turns sent to memory`,
  )
  if (question.type === 'choice') {
    const correct = results.filter((r) => r.label === r.predicted).length
    const actions = results.filter((r) => r.label === 'action' && r.predicted === 'action').length
    console.log(`   3-way accuracy ${pct(correct, results.length)}; actions recognised ${pct(actions, ACTION.length)}`)
  }
  console.log(
    `   latency: model p50 ${median(results.map((r) => r.modelMs)).toFixed(1)} ms, with HTTP p50 ${median(results.map((r) => r.httpMs)).toFixed(1)} ms`,
  )
  for (const group of new Set(results.map((r) => r.group))) {
    const rs = results.filter((r) => r.group === group)
    const wantPast = isPast(rs[0].label)
    const ok = rs.filter((r) => isPast(r.predicted) === wantPast).length
    console.log(
      `   ${group.padEnd(38)} ${wantPast ? 'sent to memory' : 'kept out of memory'}: ${pct(ok, rs.length)} (n=${rs.length})`,
    )
  }
  for (const m of results.filter((r) => isPast(r.label) !== isPast(r.predicted)).slice(0, 6)) {
    console.log(`     ✗ ${m.label}→${m.predicted}: ${m.text.slice(0, 90)}`)
  }
  console.log()
}
