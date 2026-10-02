import { median, percent } from '@bench/shared'
import { isFirstPerson } from '@voice/agent/s2s-routing'
import { route } from '@voice/laya'
import { recall, warmReranker } from '@voice/memory'
import { routerTurns, type Turn } from './turns.js'

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

const past = rows.filter((r) => r.label === 'past')
const other = rows.filter((r) => r.label !== 'past')
console.log(`${rows.length} turns; Laya + recall in parallel p50 ${median(rows.map((r) => r.ms)).toFixed(0)} ms\n`)
console.log('rule                          | past turns searched | non-past turns searched')
const rule = (name: string, f: (r: Row) => boolean) =>
  console.log(
    `${name.padEnd(29)} | ${percent(past.filter(f).length, past.length).padStart(19)} | ${percent(other.filter(f).length, other.length).padStart(5)} (${other.filter(f).length})`,
  )
rule('laya only', (r) => r.laya)
for (const th of [0, 2, 4, 6]) rule(`gate >= ${th} only`, (r) => r.top >= th)
for (const th of [0, 2, 4, 6]) rule(`laya AND gate >= ${th}`, (r) => r.laya && r.top >= th)
for (const th of [4, 6, 8]) rule(`laya OR gate >= ${th}`, (r) => r.laya || r.top >= th)
for (const th of [2, 4, 6])
  rule(`laya OR (1st person AND >= ${th})`, (r) => r.laya || (isFirstPerson(r.text) && r.top >= th))

const chosen = (r: Row) => r.laya || (isFirstPerson(r.text) && r.top >= 4)
console.log('\nnon-past turns searched under "laya OR (1st person AND >= 4)":')
for (const r of other.filter(chosen)) console.log(`  ${r.group}: ${r.text} (${r.top.toFixed(1)})`)
console.log('\npast turns missed under "laya OR (1st person AND >= 4)":')
for (const r of past.filter((r) => !chosen(r)).slice(0, 12))
  console.log(`  ${r.group}: ${r.text} (laya ${r.laya}, ${r.top.toFixed(1)})`)
