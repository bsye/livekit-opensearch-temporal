import { route } from '@voice/laya'
import { recall, warmReranker } from '@voice/memory'
import { routerTurns, type Turn } from './turns.js'

const FIRST_PERSON = /\b(i|my|me|mine|i'm|i've|i'd|i'll)\b/i

interface Row extends Turn {
  laya: boolean
  top: number
  ms: number
}

await warmReranker()
const rows: Row[] = []
for (const t of routerTurns()) {
  const start = performance.now()
  const [laya, r] = await Promise.all([route(t.text), recall({ question: t.text, limit: 1 })])
  rows.push({ ...t, laya: laya === 'past', top: r.hits[0]?.score ?? -99, ms: performance.now() - start })
}

const pct = (n: number, d: number) => `${((100 * n) / d).toFixed(0)}%`
const past = rows.filter((r) => r.label === 'past')
const other = rows.filter((r) => r.label !== 'past')
const p50 = rows.map((r) => r.ms).sort((a, b) => a - b)[rows.length >> 1]
console.log(`${rows.length} turns; Laya + recall in parallel p50 ${p50.toFixed(0)} ms\n`)
console.log('rule                          | past turns searched | non-past turns searched')
const rule = (name: string, f: (r: Row) => boolean) =>
  console.log(
    `${name.padEnd(29)} | ${pct(past.filter(f).length, past.length).padStart(19)} | ${pct(other.filter(f).length, other.length).padStart(5)} (${other.filter(f).length})`,
  )
rule('laya only', (r) => r.laya)
for (const th of [0, 2, 4, 6]) rule(`gate >= ${th} only`, (r) => r.top >= th)
for (const th of [0, 2, 4, 6]) rule(`laya AND gate >= ${th}`, (r) => r.laya && r.top >= th)
for (const th of [4, 6, 8]) rule(`laya OR gate >= ${th}`, (r) => r.laya || r.top >= th)
for (const th of [2, 4, 6])
  rule(`laya OR (1st person AND >= ${th})`, (r) => r.laya || (FIRST_PERSON.test(r.text) && r.top >= th))

const chosen = (r: Row) => r.laya || (FIRST_PERSON.test(r.text) && r.top >= 4)
console.log('\nnon-past turns searched under "laya OR (1st person AND >= 4)":')
for (const r of other.filter(chosen)) console.log(`  ${r.group}: ${r.text} (${r.top.toFixed(1)})`)
console.log('\npast turns missed under "laya OR (1st person AND >= 4)":')
for (const r of past.filter((r) => !chosen(r)).slice(0, 12))
  console.log(`  ${r.group}: ${r.text} (laya ${r.laya}, ${r.top.toFixed(1)})`)
