import { appendFileSync, createReadStream, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { pathToFileURL } from 'node:url'
import { flag, option, percentile } from '@bench/shared'
import { dataPath, env } from '@voice/config'
import { postJson, requestJson } from '@voice/http'
import { scan } from '@voice/laya'
import { RERANKERS, Reranker, type RerankerName } from '@voice/memory'
import StreamArray from 'stream-json/streamers/StreamArray.js'
import { bm25OkapiRanking } from './bm25okapi.js'
import { contrieverRanking, loadContriever } from './contriever.js'
import { EMBEDDERS, Embedder, rrf } from './embeddings.js'
import { QueryExpander } from './expansion.js'
import { QueryWriter } from './multiquery.js'
import {
  buildCorpus,
  type Doc,
  evaluateRetrieval,
  evaluateRetrievalTurn2Session,
  excluded,
  type Granularity,
  KS,
  type Question,
  type Scores,
} from './official.js'
import { PreferenceTags } from './preference.js'
import { rm3Query } from './rm3.js'
import { encodeDocs, loadSparse, queryWeights } from './sparse.js'

const OPENSEARCH = env('OPENSEARCH_URL')
const DATA = dataPath('benchmarks', 'longmemeval')
const RERANK_MAX_LENGTH = Number(option('--rerank-max-length', '512'))
const RERANK_DEPTH = Number(option('--rerank-depth', '50'))
const ALL_METHODS = [
  'contriever',
  'bm25-paper',
  'bm25',
  'bm25+qe',
  'rm3',
  'dense-qwen3',
  'rrf-bm25+qwen3',
  'rrf-bm25qe+qwen3',
  'dense-nomic',
  'rrf-bm25+nomic',
  'rrf-bm25qe+nomic',
  'bm25+pref',
  'bm25+qe+pref',
  'sparse',
  'bm25-fuzzy',
  'laya',
  'bm25+laya',
  'minilm:bm25',
  'minilm:bm25+qe',
  'bge:bm25+qe',
  'minilm:rrf-bm25qe+qwen3',
  'bge:rrf-bm25qe+qwen3',
  'minilm:rrf-bm25qe+nomic',
  'bge:rrf-bm25qe+nomic',
  'bm25+oqe',
  'minilm:bm25+oqe',
  'mq',
  'minilm:mq',
  'bm25+mqe',
  'minilm:bm25+mqe',
] as const
const EXPANSION_WEIGHT = 0.5
const PREFERENCE_WEIGHT = 0.5
const PREFERENCE_QUESTION_MIN = 0.5
type Method = (typeof ALL_METHODS)[number]

const granularity = option('--granularity', 'turn') as Granularity
const limit = Number(option('--limit', '0'))
const dataset = option('--dataset', 'longmemeval_s_cleaned')
const methods = option('--methods', 'bm25-paper,bm25,bm25-fuzzy,laya,bm25+laya').split(',') as Method[]
for (const m of methods) if (!ALL_METHODS.includes(m)) throw new Error(`unknown method ${m}`)
for (const m of methods) {
  const base = m.split(':')[1]
  if (base && !methods.includes(base as Method)) throw new Error(`${m} needs ${base} in --methods`)
}
const index = `bench-${dataset.replaceAll('_', '-')}-${granularity}`
const dataFile = `${DATA}/${dataset}.json`
const outDir = pathToFileURL(`${DATA}/results/`)
mkdirSync(outDir, { recursive: true })

await indexCorpus()
if (methods.includes('contriever')) await loadContriever()
if (methods.includes('sparse')) {
  await loadSparse()
  await indexSparseCorpus()
}
const expander = new QueryExpander(new URL(`${dataset}.expansions.json`, outDir))
const omni = () => ({ baseUrl: env('OMNI_BASE_URL'), model: env('OMNI_MODEL') })
const omniExpander = methods.some((m) => m.endsWith('bm25+oqe'))
  ? new QueryExpander(new URL(`${dataset}.omni-expansions.json`, outDir), omni())
  : undefined
const queryWriter = methods.some((m) => m.endsWith('mq') || m.endsWith('mqe'))
  ? new QueryWriter(new URL(`${dataset}.omni-queries.json`, outDir), omni())
  : undefined
const prefTags = new PreferenceTags(new URL(`${dataset}.preference-tags.json`, outDir))
const rerankers = Object.fromEntries(
  (Object.keys(RERANKERS) as RerankerName[])
    .filter((r) => methods.some((m) => m.startsWith(`${r}:`)))
    .map((r) => [r, new Reranker(r, RERANK_MAX_LENGTH)]),
)
for (const r of Object.values(rerankers)) await r.load()
const embedders = Object.fromEntries(
  Object.keys(EMBEDDERS)
    .filter((e) => methods.some((m) => m.endsWith(e)))
    .map((e) => [e, new Embedder(EMBEDDERS[e], new URL('embeddings/', outDir))]),
)

type Row = {
  question_id: string
  question_type: string
  latencyMs: Partial<Record<Method, number>>
  scanned: Partial<Record<Method, number>>
  metrics: Partial<Record<Method, Record<string, number>>>
  corpusSize: number
  layaPeakMb?: number
}
const tag = option('--tag', '')
const progressFile = new URL(`${dataset}-${granularity}${tag ? `-${tag}` : ''}.progress.jsonl`, outDir)
if (flag('--fresh')) rmSync(progressFile, { force: true })
const rows: Row[] = existsSync(progressFile)
  ? readFileSync(progressFile, 'utf8')
      .split('\n')
      .filter(Boolean)
      .map((line) => JSON.parse(line) as Row)
  : []
const done = new Set(rows.map((r) => r.question_id))
if (done.size) console.log(`resuming: ${done.size} questions already done`)

let layaPeakMb = 0
let evaluatedCount = done.size
let skipped = 0
const started = Date.now()
let processed = 0
for await (const q of questions()) {
  if (excluded(q)) {
    skipped++
    continue
  }
  if (limit && evaluatedCount >= limit) break
  if (done.has(q.question_id)) continue
  rows.push(await evaluate(q))
  appendFileSync(progressFile, `${JSON.stringify(rows.at(-1))}\n`)
  evaluatedCount++
  processed++
  if (processed % 25 === 0) {
    const rate = ((Date.now() - started) / processed / 1000).toFixed(2)
    console.log(`${evaluatedCount} questions (${rate}s each, laya peak ${layaPeakMb} MB)`)
  }
}
prefTags.save()
console.log(
  `${evaluatedCount} questions evaluated, ${skipped} excluded (abstention/no-target, as in the official script)`,
)
report()

async function evaluate(q: Question): Promise<Row> {
  const corpus = buildCorpus(q, granularity)
  const ids = corpus.map((d) => d.id)
  const correct = [...new Set(ids.filter((id) => id.includes('answer')))]
  const position = new Map(ids.map((id, i) => [id, i]))
  const row: Row = {
    question_id: q.question_id,
    question_type: q.question_type,
    latencyMs: {},
    scanned: {},
    metrics: {},
    corpusSize: corpus.length,
  }
  const rankings: Partial<Record<Method, number[]>> = {}

  let t = Date.now()
  if (methods.includes('contriever')) {
    rankings.contriever = await contrieverRanking(
      corpus.map((d) => d.text),
      q.question,
    )
    row.latencyMs.contriever = Date.now() - t
    row.scanned.contriever = corpus.length
    t = Date.now()
  }
  if (methods.includes('bm25-paper')) {
    rankings['bm25-paper'] = bm25OkapiRanking(
      corpus.map((d) => d.text),
      q.question,
    )
    row.latencyMs['bm25-paper'] = Date.now() - t
    row.scanned['bm25-paper'] = 0
    t = Date.now()
  }
  const bm25 = complete(await bm25Ranking(q, position, false), corpus.length)
  const bm25Ms = Date.now() - t
  if (methods.includes('bm25')) {
    rankings.bm25 = bm25
    row.latencyMs.bm25 = bm25Ms
    row.scanned.bm25 = 0
  }
  if (methods.includes('bm25+qe')) {
    const expansion = await expander.expand(q.question_id, q.question)
    t = Date.now()
    rankings['bm25+qe'] = complete(await bm25Ranking(q, position, false, expansion), corpus.length)
    row.latencyMs['bm25+qe'] = Date.now() - t
    row.scanned['bm25+qe'] = 0
  }
  if (omniExpander && methods.some((m) => m.endsWith('bm25+oqe'))) {
    const expansion = await omniExpander.expand(q.question_id, q.question)
    t = Date.now()
    rankings['bm25+oqe'] = complete(await bm25Ranking(q, position, false, expansion), corpus.length)
    row.latencyMs['bm25+oqe'] = Date.now() - t
    row.scanned['bm25+oqe'] = 0
  }
  if (queryWriter && methods.some((m) => m.endsWith('mq'))) {
    const queries = await queryWriter.queries(q.question_id, q.question)
    t = Date.now()
    const lists = await Promise.all(
      [q.question, ...queries].map((text) => bm25Ranking({ ...q, question: text }, position, false)),
    )
    rankings.mq = rrf(
      lists.map((l) => complete(l, corpus.length)),
      corpus.length,
    )
    row.latencyMs.mq = Date.now() - t
    row.scanned.mq = 0
  }
  if (queryWriter && methods.some((m) => m.endsWith('bm25+mqe'))) {
    const queries = await queryWriter.queries(q.question_id, q.question)
    t = Date.now()
    rankings['bm25+mqe'] = complete(
      await bm25Ranking(q, position, false, queries.join(', ') || undefined),
      corpus.length,
    )
    row.latencyMs['bm25+mqe'] = Date.now() - t
    row.scanned['bm25+mqe'] = 0
  }
  if (methods.includes('rm3')) {
    t = Date.now()
    rankings.rm3 = complete(await rm3Ranking(q, position), corpus.length)
    row.latencyMs.rm3 = Date.now() - t
    row.scanned.rm3 = 0
  }
  for (const [e, embedder] of Object.entries(embedders)) {
    await embedder.embedDocs(corpus.map((d) => d.text))
    t = Date.now()
    const dense = await embedder.ranking(
      corpus.map((d) => d.text),
      q.question,
    )
    const denseMs = Date.now() - t
    const dm = `dense-${e}` as Method
    if (methods.includes(dm)) {
      rankings[dm] = dense
      row.latencyMs[dm] = denseMs
      row.scanned[dm] = corpus.length
    }
    const hm = `rrf-bm25+${e}` as Method
    if (methods.includes(hm)) {
      rankings[hm] = rrf([bm25, dense], corpus.length)
      row.latencyMs[hm] = Math.max(bm25Ms, denseMs)
      row.scanned[hm] = corpus.length
    }
    const qm = `rrf-bm25qe+${e}` as Method
    if (methods.includes(qm)) {
      const expansion = await expander.expand(q.question_id, q.question)
      t = Date.now()
      const bm25qe = complete(await bm25Ranking(q, position, false, expansion), corpus.length)
      rankings[qm] = rrf([bm25qe, dense], corpus.length)
      row.latencyMs[qm] = Math.max(Date.now() - t, denseMs)
      row.scanned[qm] = corpus.length
    }
  }
  for (const m of ['bm25+pref', 'bm25+qe+pref'] as const) {
    if (!methods.includes(m)) continue
    const expansion = m === 'bm25+qe+pref' ? await expander.expand(q.question_id, q.question) : undefined
    t = Date.now()
    const scores = await bm25Scores(q, position, expansion)
    const pq = await prefTags.queryScore(q.question)
    if (pq >= PREFERENCE_QUESTION_MIN) {
      const max = Math.max(...scores.values(), 1e-9)
      const tags = await prefTags.turnScores(corpus.map((d) => d.text))
      const final = corpus.map((_, i) => (scores.get(i) ?? 0) / max + PREFERENCE_WEIGHT * pq * tags[i])
      rankings[m] = byScore(
        corpus.map((_, i) => i),
        final,
      )
    } else {
      rankings[m] = complete([...scores.keys()], corpus.length)
    }
    row.latencyMs[m] = Date.now() - t
    row.scanned[m] = pq >= PREFERENCE_QUESTION_MIN ? corpus.length : 0
  }
  if (methods.includes('sparse')) {
    t = Date.now()
    rankings.sparse = complete(await sparseRanking(q, position), corpus.length)
    row.latencyMs.sparse = Date.now() - t
    row.scanned.sparse = 0
  }
  if (methods.includes('bm25-fuzzy')) {
    t = Date.now()
    rankings['bm25-fuzzy'] = complete(await bm25Ranking(q, position, true), corpus.length)
    row.latencyMs['bm25-fuzzy'] = Date.now() - t
    row.scanned['bm25-fuzzy'] = 0
  }
  if (methods.includes('laya')) {
    t = Date.now()
    const scores = await layaScan(corpus, q.question)
    rankings.laya = byScore(
      corpus.map((_, i) => i),
      scores,
    )
    row.latencyMs.laya = Date.now() - t
    row.scanned.laya = corpus.length
  }
  if (methods.includes('bm25+laya')) {
    t = Date.now()
    const head = bm25.slice(0, RERANK_DEPTH)
    const headScores = await layaScan(
      head.map((i) => corpus[i]),
      q.question,
    )
    rankings['bm25+laya'] = [...byScore(head, headScores), ...bm25.slice(RERANK_DEPTH)]
    row.latencyMs['bm25+laya'] = bm25Ms + (Date.now() - t)
    row.scanned['bm25+laya'] = head.length
  }
  for (const m of methods.filter((x) => x.includes(':'))) {
    const [r, base] = m.split(':') as [RerankerName, Method]
    t = Date.now()
    const head = rankings[base]!.slice(0, RERANK_DEPTH)
    const scores = await rerankers[r].score(
      q.question,
      head.map((i) => corpus[i].text),
    )
    rankings[m] = [...byScore(head, scores), ...rankings[base]!.slice(RERANK_DEPTH)]
    row.latencyMs[m] = (row.latencyMs[base] ?? 0) + (Date.now() - t)
    row.scanned[m] = head.length
  }
  for (const m of methods) row.metrics[m] = metricsFor(rankings[m]!, correct, ids)
  row.layaPeakMb = layaPeakMb
  return row
}

async function bm25Ranking(
  q: Question,
  position: Map<string, number>,
  fuzzy: boolean,
  expansion?: string,
): Promise<number[]> {
  const match = fuzzy ? { match: { text: { query: q.question, fuzziness: 'AUTO' } } } : { match: { text: q.question } }
  const should = expansion ? [match, { match: { text: { query: expansion, boost: EXPANSION_WEIGHT } } }] : [match]
  const res = await post(`${OPENSEARCH}/${index}/_search`, {
    size: 10000,
    _source: ['docId'],
    query: { bool: { filter: [{ term: { qid: q.question_id } }], should, minimum_should_match: 1 } },
  })
  const hits = (res as { hits: { hits: { _source: { docId: string } }[] } }).hits.hits
  return hits.map((h) => position.get(h._source.docId)!).filter((i) => i !== undefined)
}

async function bm25Scores(
  q: Question,
  position: Map<string, number>,
  expansion?: string,
): Promise<Map<number, number>> {
  const match = { match: { text: q.question } }
  const should = expansion ? [match, { match: { text: { query: expansion, boost: EXPANSION_WEIGHT } } }] : [match]
  const res = await post(`${OPENSEARCH}/${index}/_search`, {
    size: 10000,
    _source: ['docId'],
    query: { bool: { filter: [{ term: { qid: q.question_id } }], should, minimum_should_match: 1 } },
  })
  const hits = (res as { hits: { hits: { _score: number; _source: { docId: string } }[] } }).hits.hits
  return new Map(
    hits.filter((h) => position.has(h._source.docId)).map((h) => [position.get(h._source.docId)!, h._score]),
  )
}

async function rm3Ranking(q: Question, position: Map<string, number>): Promise<number[]> {
  const filter = [{ term: { qid: q.question_id } }]
  const first = (await post(`${OPENSEARCH}/${index}/_search`, {
    size: 10,
    _source: false,
    query: { bool: { filter, must: [{ match: { text: q.question } }] } },
  })) as { hits: { hits: { _id: string; _score: number }[] } }
  const should = await rm3Query(OPENSEARCH, index, post, q.question, first.hits.hits)
  const res = (await post(`${OPENSEARCH}/${index}/_search`, {
    size: 10000,
    _source: ['docId'],
    query: { bool: { filter, should, minimum_should_match: 1 } },
  })) as { hits: { hits: { _source: { docId: string } }[] } }
  return res.hits.hits.map((h) => position.get(h._source.docId)!).filter((i) => i !== undefined)
}

async function sparseRanking(q: Question, position: Map<string, number>): Promise<number[]> {
  const weights = queryWeights(q.question)
  const res = await post(`${OPENSEARCH}/${index}-sparse/_search`, {
    size: 10000,
    _source: ['docId'],
    query: {
      bool: {
        filter: [{ term: { qid: q.question_id } }],
        should: Object.entries(weights).map(([k, w]) => ({
          rank_feature: { field: `sparse.${k}`, linear: {}, boost: w },
        })),
        minimum_should_match: 1,
      },
    },
  })
  const hits = (res as { hits: { hits: { _source: { docId: string } }[] } }).hits.hits
  return hits.map((h) => position.get(h._source.docId)!).filter((i) => i !== undefined)
}

async function layaScan(docs: Doc[], question: string): Promise<number[]> {
  if (docs.length === 0) return []
  const { scores, peakMb } = await scan(
    docs.map((d) => d.text),
    `Does this message contain information that helps answer the question: "${question}"?`,
  )
  layaPeakMb = Math.max(layaPeakMb, peakMb ?? 0)
  return scores
}

function byScore(indices: number[], scores: number[]): number[] {
  return indices
    .map((idx, i) => ({ idx, s: scores[i], i }))
    .sort((a, b) => b.s - a.s || a.i - b.i)
    .map((x) => x.idx)
}

function complete(ranking: number[], n: number): number[] {
  const seen = new Set(ranking)
  return [...ranking, ...Array.from({ length: n }, (_, i) => i).filter((i) => !seen.has(i))]
}

function metricsFor(ranking: number[], correct: string[], ids: string[]): Record<string, number> {
  const out: Record<string, number> = {}
  for (const k of KS) {
    const put = (prefix: string, s: Scores) => {
      out[`${prefix}recall_any@${k}`] = s.recall_any
      out[`${prefix}recall_all@${k}`] = s.recall_all
      out[`${prefix}ndcg_any@${k}`] = s.ndcg_any
    }
    put(`${granularity}.`, evaluateRetrieval(ranking, correct, ids, k))
    if (granularity === 'turn') put('session.', evaluateRetrievalTurn2Session(ranking, correct, ids, k))
  }
  return out
}

async function* questions(): AsyncGenerator<Question> {
  const pipeline = createReadStream(dataFile).pipe(StreamArray.withParser())
  for await (const { value } of pipeline as AsyncIterable<{ key: number; value: Question }>) yield value
}

function indexCorpus() {
  return buildIndex(
    index,
    {
      qid: { type: 'keyword' },
      docId: { type: 'keyword' },
      text: { type: 'text', analyzer: 'english' },
      timestamp: { type: 'keyword' },
    },
    async (q, docs) =>
      docs.map((d) => JSON.stringify({ qid: q.question_id, docId: d.id, text: d.text, timestamp: d.timestamp })),
  )
}

function indexSparseCorpus() {
  return buildIndex(
    `${index}-sparse`,
    { qid: { type: 'keyword' }, docId: { type: 'keyword' }, sparse: { type: 'rank_features' } },
    async (q, docs) => {
      const vectors = await encodeDocs(docs.map((d) => d.text))
      return docs.map(
        (d, i) => `{"qid":${JSON.stringify(q.question_id)},"docId":${JSON.stringify(d.id)},"sparse":${vectors[i]}}`,
      )
    },
  )
}

/** Rebuilt only when the marker file and the index's document count disagree. */
async function buildIndex(
  name: string,
  properties: Record<string, unknown>,
  documents: (q: Question, docs: Doc[]) => Promise<string[]>,
): Promise<void> {
  const marker = new URL(`${name}.indexed`, outDir)
  const { count } = await requestJson<{ count: number }>('GET', `${OPENSEARCH}/${name}/_count`).catch(() => ({
    count: -1,
  }))
  if (existsSync(marker) && Number(readFileSync(marker, 'utf8')) === count) {
    return console.log(`index ${name} ready (${count} docs)`)
  }
  console.log(`indexing ${dataset} (${granularity}) into ${name} …`)
  await requestJson('DELETE', `${OPENSEARCH}/${name}`).catch(() => undefined)
  await put(`${OPENSEARCH}/${name}`, {
    settings: { number_of_shards: 1, number_of_replicas: 0, refresh_interval: '-1' },
    mappings: { properties },
  })
  let lines: string[] = []
  let total = 0
  const started = Date.now()
  const flush = async () => {
    if (!lines.length) return
    const res = await requestJson<{ errors: boolean }>(
      'POST',
      `${OPENSEARCH}/_bulk`,
      `${lines.join('\n')}\n`,
      'application/x-ndjson',
    )
    if (res.errors) throw new Error('bulk indexing errors')
    lines = []
  }
  for await (const q of questions()) {
    const docs = buildCorpus(q, granularity)
    for (const doc of await documents(q, docs)) lines.push(JSON.stringify({ index: { _index: name } }), doc)
    total += docs.length
    if (lines.length >= 2000) await flush()
    if (total % 100000 < docs.length)
      console.log(`  ${total} docs (${((Date.now() - started) / 60000).toFixed(1)} min)`)
  }
  await flush()
  await put(`${OPENSEARCH}/${name}/_settings`, { index: { refresh_interval: '1s' } })
  await requestJson('POST', `${OPENSEARCH}/${name}/_refresh`)
  writeFileSync(marker, String(total))
  console.log(`indexed ${total} docs`)
}

function report(): void {
  const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / (xs.length || 1)
  const ran = ALL_METHODS.filter((m) => rows.some((r) => r.metrics[m]))
  const keys = [
    `${granularity}.recall_any@1`,
    `${granularity}.recall_any@5`,
    `${granularity}.recall_any@10`,
    `${granularity}.ndcg_any@10`,
    ...(granularity === 'turn'
      ? ['session.recall_any@5', 'session.recall_all@5', 'session.recall_any@10', 'session.ndcg_any@10']
      : []),
  ]

  const lines: string[] = []
  lines.push(
    `# LongMemEval (${dataset}, ${granularity} granularity, ${rows.length} questions, ~${mean(rows.map((r) => r.corpusSize)).toFixed(0)} docs per question)\n`,
  )
  lines.push(`| retriever | ${keys.join(' | ')} | p50 ms | p95 ms | docs scanned |`)
  lines.push(`|---|${keys.map(() => '---').join('|')}|---|---|---|`)
  for (const m of ran) {
    const rs = rows.filter((r) => r.metrics[m])
    const vals = keys.map((k) => mean(rs.map((r) => r.metrics[m]![k])).toFixed(3))
    const lat = rs.map((r) => r.latencyMs[m]!)
    lines.push(
      `| ${m} | ${vals.join(' | ')} | ${percentile(lat, 0.5)} | ${percentile(lat, 0.95)} | ${mean(rs.map((r) => r.scanned[m]!)).toFixed(0)} |`,
    )
  }
  const key = 'session.recall_any@5'
  lines.push(`\n## ${key} by question type\n`)
  lines.push(`| question type | n | ${ran.join(' | ')} |`)
  lines.push(`|---|---|${ran.map(() => '---').join('|')}|`)
  for (const type of [...new Set(rows.map((r) => r.question_type))].sort()) {
    const subset = rows.filter((r) => r.question_type === type)
    lines.push(
      `| ${type} | ${subset.length} | ${ran.map((m) => mean(subset.filter((r) => r.metrics[m]).map((r) => r.metrics[m]![key])).toFixed(3)).join(' | ')} |`,
    )
  }
  const text = lines.join('\n')
  console.log(`\n${text}`)
  const stamp = new Date().toISOString().replaceAll(':', '-').replaceAll('.', '-')
  writeFileSync(new URL(`${dataset}-${granularity}-${stamp}.md`, outDir), `${text}\n`)
}

function post(url: string, body: unknown) {
  return postJson<unknown>(url, body)
}
function put(url: string, body: unknown) {
  return requestJson<unknown>('PUT', url, body)
}
