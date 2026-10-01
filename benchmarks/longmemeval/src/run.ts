// LongMemEval retrieval benchmark for conversation memory without embeddings.
//
//   npm run bench -w longmemeval-bench -- [--dataset longmemeval_s_cleaned|longmemeval_m_cleaned]
//        [--granularity turn|session] [--methods bm25,bm25-fuzzy,bm25+laya,laya] [--limit N] [--fresh]
//
// Retrievers (same corpus, labels and metrics as the official run_retrieval.py):
//   bm25        OpenSearch BM25 (english analyzer), the paper's lexical baseline
//   bm25-fuzzy  BM25 with fuzzy term matching (tolerates misspellings / speech-to-text errors)
//   laya        parallel Laya scan: every document of the question's history scored on the GPU
//   bm25+laya   BM25 top 50 re-ordered by Laya, then the rest of the BM25 ranking
//
// The dataset is streamed one question at a time (LongMemEval_M is 2.7 GB), once to index the
// corpus into OpenSearch and once to evaluate. Resumable: each question's result is appended to
// results/<dataset>-<granularity>.progress.jsonl; a rerun skips questions already there.
import { createReadStream, appendFileSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import StreamArray from 'stream-json/streamers/StreamArray.js';
import {
  buildCorpus,
  evaluateRetrieval,
  evaluateRetrievalTurn2Session,
  excluded,
  KS,
  type Doc,
  type Granularity,
  type Question,
  type Scores,
} from './official.js';

const OPENSEARCH = process.env.OPENSEARCH_URL ?? 'http://localhost:9201';
const LAYA = process.env.LAYA_BASE_URL ?? 'http://localhost:8100';
const DATA = new URL('../../../data/benchmarks/longmemeval/', import.meta.url);
const RERANK_DEPTH = 50;
const ALL_METHODS = ['bm25', 'bm25-fuzzy', 'laya', 'bm25+laya'] as const;
type Method = (typeof ALL_METHODS)[number];

const arg = (name: string, fallback: string) => {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1] : fallback;
};
const granularity = arg('granularity', 'turn') as Granularity;
const limit = Number(arg('limit', '0'));
const dataset = arg('dataset', 'longmemeval_s_cleaned');
const methods = arg('methods', 'bm25,bm25-fuzzy,laya,bm25+laya').split(',') as Method[];
for (const m of methods) if (!ALL_METHODS.includes(m)) throw new Error(`unknown method ${m}`);
const index = `bench-${dataset.replace(/_/g, '-')}-${granularity}`;
const dataFile = fileURLToPath(new URL(`${dataset}.json`, DATA));
const outDir = new URL('results/', DATA);
mkdirSync(outDir, { recursive: true });

await indexCorpus();

type Row = {
  question_id: string;
  question_type: string;
  latencyMs: Partial<Record<Method, number>>;
  scanned: Partial<Record<Method, number>>;
  metrics: Partial<Record<Method, Record<string, number>>>;
  corpusSize: number;
  layaPeakMb?: number;
};
const progressFile = new URL(`${dataset}-${granularity}.progress.jsonl`, outDir);
if (process.argv.includes('--fresh')) rmSync(progressFile, { force: true });
const rows: Row[] = existsSync(progressFile)
  ? readFileSync(progressFile, 'utf8').split('\n').filter(Boolean).map((line) => JSON.parse(line) as Row)
  : [];
const done = new Set(rows.map((r) => r.question_id));
if (done.size) console.log(`resuming: ${done.size} questions already done`);

let layaPeakMb = 0;
let evaluatedCount = done.size;
let skipped = 0;
const started = Date.now();
let processed = 0;
for await (const q of questions()) {
  if (excluded(q)) {
    skipped++;
    continue;
  }
  if (limit && evaluatedCount >= limit) break;
  if (done.has(q.question_id)) continue;
  rows.push(await evaluate(q));
  appendFileSync(progressFile, `${JSON.stringify(rows.at(-1))}\n`);
  evaluatedCount++;
  processed++;
  if (processed % 25 === 0) {
    const rate = ((Date.now() - started) / processed / 1000).toFixed(2);
    console.log(`${evaluatedCount} questions (${rate}s each, laya peak ${layaPeakMb} MB)`);
  }
}
console.log(`${evaluatedCount} questions evaluated, ${skipped} excluded (abstention/no-target, as in the official script)`);
report();

// --- evaluation --------------------------------------------------------------------------

async function evaluate(q: Question): Promise<Row> {
  const corpus = buildCorpus(q, granularity);
  const ids = corpus.map((d) => d.id);
  const correct = [...new Set(ids.filter((id) => id.includes('answer')))];
  const position = new Map(ids.map((id, i) => [id, i]));
  const row: Row = { question_id: q.question_id, question_type: q.question_type, latencyMs: {}, scanned: {}, metrics: {}, corpusSize: corpus.length };
  const rankings: Partial<Record<Method, number[]>> = {};

  let t = Date.now();
  const bm25 = complete(await bm25Ranking(q, position, false), corpus.length);
  const bm25Ms = Date.now() - t;
  if (methods.includes('bm25')) {
    rankings.bm25 = bm25;
    row.latencyMs.bm25 = bm25Ms;
    row.scanned.bm25 = 0;
  }
  if (methods.includes('bm25-fuzzy')) {
    t = Date.now();
    rankings['bm25-fuzzy'] = complete(await bm25Ranking(q, position, true), corpus.length);
    row.latencyMs['bm25-fuzzy'] = Date.now() - t;
    row.scanned['bm25-fuzzy'] = 0;
  }
  if (methods.includes('laya')) {
    t = Date.now();
    const scores = await layaScan(corpus, q.question);
    rankings.laya = byScore(corpus.map((_, i) => i), scores);
    row.latencyMs.laya = Date.now() - t;
    row.scanned.laya = corpus.length;
  }
  if (methods.includes('bm25+laya')) {
    t = Date.now();
    const head = bm25.slice(0, RERANK_DEPTH);
    const headScores = await layaScan(head.map((i) => corpus[i]), q.question);
    rankings['bm25+laya'] = [...byScore(head, headScores), ...bm25.slice(RERANK_DEPTH)];
    row.latencyMs['bm25+laya'] = bm25Ms + (Date.now() - t);
    row.scanned['bm25+laya'] = head.length;
  }
  for (const m of methods) row.metrics[m] = metricsFor(rankings[m]!, correct, ids);
  row.layaPeakMb = layaPeakMb;
  return row;
}

async function bm25Ranking(q: Question, position: Map<string, number>, fuzzy: boolean): Promise<number[]> {
  const match = fuzzy ? { match: { text: { query: q.question, fuzziness: 'AUTO' } } } : { match: { text: q.question } };
  const res = await post(`${OPENSEARCH}/${index}/_search`, {
    size: 10000,
    _source: ['docId'],
    query: { bool: { filter: [{ term: { qid: q.question_id } }], must: [match] } },
  });
  const hits = (res as { hits: { hits: { _source: { docId: string } }[] } }).hits.hits;
  return hits.map((h) => position.get(h._source.docId)!).filter((i) => i !== undefined);
}

async function layaScan(docs: Doc[], question: string): Promise<number[]> {
  if (docs.length === 0) return [];
  const res = (await post(`${LAYA}/v1/scan`, {
    states: docs.map((d) => d.text),
    question: `Does this message contain information that helps answer the question: "${question}"?`,
  })) as { scores: number[]; peak_mb?: number };
  layaPeakMb = Math.max(layaPeakMb, res.peak_mb ?? 0);
  return res.scores;
}

/** Stable sort of indices by score, highest first. */
function byScore(indices: number[], scores: number[]): number[] {
  return indices.map((idx, i) => ({ idx, s: scores[i], i })).sort((a, b) => b.s - a.s || a.i - b.i).map((x) => x.idx);
}

/** Documents a retriever didn't return go after its ranking, in corpus order (needed for turn→session k). */
function complete(ranking: number[], n: number): number[] {
  const seen = new Set(ranking);
  return [...ranking, ...Array.from({ length: n }, (_, i) => i).filter((i) => !seen.has(i))];
}

function metricsFor(ranking: number[], correct: string[], ids: string[]): Record<string, number> {
  const out: Record<string, number> = {};
  for (const k of KS) {
    const put = (prefix: string, s: Scores) => {
      out[`${prefix}recall_any@${k}`] = s.recall_any;
      out[`${prefix}recall_all@${k}`] = s.recall_all;
      out[`${prefix}ndcg_any@${k}`] = s.ndcg_any;
    };
    put(`${granularity}.`, evaluateRetrieval(ranking, correct, ids, k));
    if (granularity === 'turn') put('session.', evaluateRetrievalTurn2Session(ranking, correct, ids, k));
  }
  return out;
}

// --- data --------------------------------------------------------------------------------

/** Stream the dataset's top-level array one question at a time. */
async function* questions(): AsyncGenerator<Question> {
  const pipeline = createReadStream(dataFile).pipe(StreamArray.withParser());
  for await (const { value } of pipeline as AsyncIterable<{ key: number; value: Question }>) yield value;
}

async function indexCorpus(): Promise<void> {
  const marker = new URL(`${index}.indexed`, outDir);
  const count = (await fetch(`${OPENSEARCH}/${index}/_count`).then((r) => (r.ok ? r.json() : { count: -1 }))) as { count: number };
  if (existsSync(marker) && Number(readFileSync(marker, 'utf8')) === count.count) {
    return console.log(`index ${index} ready (${count.count} docs)`);
  }

  console.log(`indexing ${dataset} (${granularity}) into ${index} …`);
  await fetch(`${OPENSEARCH}/${index}`, { method: 'DELETE' });
  await put(`${OPENSEARCH}/${index}`, {
    settings: { number_of_shards: 1, number_of_replicas: 0, refresh_interval: '-1' },
    mappings: {
      properties: {
        qid: { type: 'keyword' },
        docId: { type: 'keyword' },
        text: { type: 'text', analyzer: 'english' },
        timestamp: { type: 'keyword' },
      },
    },
  });
  let lines: string[] = [];
  let total = 0;
  const flush = async () => {
    if (!lines.length) return;
    const res = await fetch(`${OPENSEARCH}/_bulk`, { method: 'POST', headers: { 'content-type': 'application/x-ndjson' }, body: `${lines.join('\n')}\n` });
    const body = (await res.json()) as { errors: boolean };
    if (body.errors) throw new Error('bulk indexing errors');
    lines = [];
  };
  for await (const q of questions()) {
    for (const d of buildCorpus(q, granularity)) {
      lines.push(JSON.stringify({ index: { _index: index } }), JSON.stringify({ qid: q.question_id, docId: d.id, text: d.text, timestamp: d.timestamp }));
      total++;
      if (lines.length >= 4000) await flush();
    }
    if (total % 100000 < buildCorpus(q, granularity).length) console.log(`  ${total} docs`);
  }
  await flush();
  await put(`${OPENSEARCH}/${index}/_settings`, { index: { refresh_interval: '1s' } });
  await fetch(`${OPENSEARCH}/${index}/_refresh`, { method: 'POST' });
  writeFileSync(marker, String(total));
  console.log(`indexed ${total} docs`);
}

// --- reporting ---------------------------------------------------------------------------

function report(): void {
  const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / (xs.length || 1);
  const pct = (xs: number[], p: number) => [...xs].sort((a, b) => a - b)[Math.min(xs.length - 1, Math.floor(xs.length * p))];
  const ran = ALL_METHODS.filter((m) => rows.some((r) => r.metrics[m]));
  const keys = [`${granularity}.recall_any@1`, `${granularity}.recall_any@5`, `${granularity}.recall_any@10`, `${granularity}.ndcg_any@10`,
    ...(granularity === 'turn' ? ['session.recall_any@5', 'session.recall_all@5', 'session.recall_any@10', 'session.ndcg_any@10'] : [])];

  const lines: string[] = [];
  lines.push(`# LongMemEval (${dataset}, ${granularity} granularity, ${rows.length} questions, ~${mean(rows.map((r) => r.corpusSize)).toFixed(0)} docs per question)\n`);
  lines.push(`| retriever | ${keys.join(' | ')} | p50 ms | p95 ms | docs scanned |`);
  lines.push(`|---|${keys.map(() => '---').join('|')}|---|---|---|`);
  for (const m of ran) {
    const rs = rows.filter((r) => r.metrics[m]);
    const vals = keys.map((k) => mean(rs.map((r) => r.metrics[m]![k])).toFixed(3));
    const lat = rs.map((r) => r.latencyMs[m]!);
    lines.push(`| ${m} | ${vals.join(' | ')} | ${pct(lat, 0.5)} | ${pct(lat, 0.95)} | ${mean(rs.map((r) => r.scanned[m]!)).toFixed(0)} |`);
  }
  const key = 'session.recall_any@5';
  lines.push(`\n## ${key} by question type\n`);
  lines.push(`| question type | n | ${ran.join(' | ')} |`);
  lines.push(`|---|---|${ran.map(() => '---').join('|')}|`);
  for (const type of [...new Set(rows.map((r) => r.question_type))].sort()) {
    const subset = rows.filter((r) => r.question_type === type);
    lines.push(`| ${type} | ${subset.length} | ${ran.map((m) => mean(subset.filter((r) => r.metrics[m]).map((r) => r.metrics[m]![key])).toFixed(3)).join(' | ')} |`);
  }
  const text = lines.join('\n');
  console.log(`\n${text}`);
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  writeFileSync(new URL(`${dataset}-${granularity}-${stamp}.md`, outDir), `${text}\n`);
}

async function post(url: string, body: unknown): Promise<unknown> {
  const res = await fetch(url, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
  if (!res.ok) throw new Error(`${url}: ${res.status} ${await res.text()}`);
  return res.json();
}

async function put(url: string, body: unknown): Promise<void> {
  const res = await fetch(url, { method: 'PUT', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
  if (!res.ok) throw new Error(`${url}: ${res.status} ${await res.text()}`);
}
