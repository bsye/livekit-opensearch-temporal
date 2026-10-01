// LongMemEval retrieval benchmark for conversation memory without embeddings.
//
//   npm run bench -w longmemeval-bench -- [--granularity turn|session] [--limit N] [--fresh]
//
// Resumable: each question's result is appended to results/<dataset>-<granularity>.progress.jsonl
// as soon as it finishes; a rerun skips questions already there (--fresh starts over).
//
// Retrievers (same corpus, labels and metrics as the official run_retrieval.py):
//   bm25        OpenSearch BM25 (english analyzer), the paper's lexical baseline
//   laya        parallel Laya scan: every document of the question's history scored on the GPU
//   bm25+laya   BM25 top 50 re-ordered by Laya, then the rest of the BM25 ranking
import { appendFileSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
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
const METHODS = ['bm25', 'laya', 'bm25+laya'] as const;
type Method = (typeof METHODS)[number];

const arg = (name: string, fallback: string) => {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1] : fallback;
};
const granularity = arg('granularity', 'turn') as Granularity;
const limit = Number(arg('limit', '0'));
const dataset = arg('dataset', 'longmemeval_s_cleaned');
const index = `bench-${dataset.replace(/_/g, '-')}-${granularity}`;

console.log(`loading ${dataset}.json …`);
const questions = JSON.parse(readFileSync(new URL(`${dataset}.json`, DATA), 'utf8')) as Question[];
const evaluated = questions.filter((q) => !excluded(q)).slice(0, limit || undefined);
console.log(`${questions.length} questions, ${evaluated.length} evaluated (abstention/no-target excluded as in the official script)`);

await indexCorpus();

type Row = { question_id: string; question_type: string; latencyMs: Record<Method, number>; scanned: Record<Method, number>; metrics: Record<Method, Record<string, number>>; layaPeakMb?: number };
const outDir = new URL('results/', DATA);
mkdirSync(outDir, { recursive: true });
const progressFile = new URL(`${dataset}-${granularity}.progress.jsonl`, outDir);
if (process.argv.includes('--fresh')) rmSync(progressFile, { force: true });
const rows: Row[] = existsSync(progressFile)
  ? readFileSync(progressFile, 'utf8').split('\n').filter(Boolean).map((line) => JSON.parse(line) as Row)
  : [];
const done = new Set(rows.map((r) => r.question_id));
const todo = evaluated.filter((q) => !done.has(q.question_id));
if (done.size) console.log(`resuming: ${done.size} done, ${todo.length} to go`);
let layaPeakMb = 0;
const started = Date.now();
for (const [n, q] of todo.entries()) {
  const corpus = buildCorpus(q, granularity);
  const ids = corpus.map((d) => d.id);
  const correct = [...new Set(ids.filter((id) => id.includes('answer')))];
  const position = new Map(ids.map((id, i) => [id, i]));
  const row: Row = { question_id: q.question_id, question_type: q.question_type, latencyMs: {} as Row['latencyMs'], scanned: {} as Row['scanned'], metrics: {} as Row['metrics'] };

  // bm25
  let t = Date.now();
  const bm25 = complete(await bm25Ranking(q, position), corpus.length);
  row.latencyMs.bm25 = Date.now() - t;
  row.scanned.bm25 = 0;

  // laya: score every document
  t = Date.now();
  const allScores = await layaScan(corpus, q.question);
  const laya = byScore(corpus.map((_, i) => i), allScores);
  row.latencyMs.laya = Date.now() - t;
  row.scanned.laya = corpus.length;

  // bm25+laya: re-order the BM25 head
  t = Date.now();
  const head = bm25.slice(0, RERANK_DEPTH);
  const headScores = await layaScan(head.map((i) => corpus[i]), q.question);
  const hybrid = [...byScore(head, headScores), ...bm25.slice(RERANK_DEPTH)];
  row.latencyMs['bm25+laya'] = row.latencyMs.bm25 + (Date.now() - t);
  row.scanned['bm25+laya'] = head.length;

  for (const [method, ranking] of [['bm25', bm25], ['laya', laya], ['bm25+laya', hybrid]] as const) {
    row.metrics[method] = metricsFor(ranking, correct, ids);
  }
  row.layaPeakMb = layaPeakMb;
  rows.push(row);
  appendFileSync(progressFile, `${JSON.stringify(row)}\n`);
  if ((n + 1) % 25 === 0 || n + 1 === todo.length) {
    const rate = ((Date.now() - started) / (n + 1) / 1000).toFixed(2);
    console.log(`${rows.length}/${evaluated.length} questions (${rate}s each, laya peak ${layaPeakMb} MB)`);
  }
}

report();

// --- retrieval ---------------------------------------------------------------------------

async function bm25Ranking(q: Question, position: Map<string, number>): Promise<number[]> {
  const res = await post(`${OPENSEARCH}/${index}/_search`, {
    size: 10000,
    _source: ['docId'],
    query: { bool: { filter: [{ term: { qid: q.question_id } }], must: [{ match: { text: q.question } }] } },
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

// --- indexing ----------------------------------------------------------------------------

async function indexCorpus(): Promise<void> {
  const expected = questions.reduce((sum, q) => sum + buildCorpus(q, granularity).length, 0);
  const count = await fetch(`${OPENSEARCH}/${index}/_count`).then((r) => (r.ok ? r.json() : { count: 0 })) as { count: number };
  if (count.count === expected) return console.log(`index ${index} ready (${expected} docs)`);

  console.log(`indexing ${expected} ${granularity} docs into ${index} …`);
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
  const flush = async () => {
    if (!lines.length) return;
    const res = await fetch(`${OPENSEARCH}/_bulk`, { method: 'POST', headers: { 'content-type': 'application/x-ndjson' }, body: `${lines.join('\n')}\n` });
    const body = (await res.json()) as { errors: boolean };
    if (body.errors) throw new Error('bulk indexing errors');
    lines = [];
  };
  for (const q of questions) {
    for (const d of buildCorpus(q, granularity)) {
      lines.push(JSON.stringify({ index: { _index: index } }), JSON.stringify({ qid: q.question_id, docId: d.id, text: d.text, timestamp: d.timestamp }));
      if (lines.length >= 4000) await flush();
    }
  }
  await flush();
  await put(`${OPENSEARCH}/${index}/_settings`, { index: { refresh_interval: '1s' } });
  await fetch(`${OPENSEARCH}/${index}/_refresh`, { method: 'POST' });
  console.log(`indexed ${expected} docs`);
}

// --- reporting ---------------------------------------------------------------------------

function report(): void {
  const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / (xs.length || 1);
  const pct = (xs: number[], p: number) => [...xs].sort((a, b) => a - b)[Math.min(xs.length - 1, Math.floor(xs.length * p))];
  const keys = [`${granularity}.recall_any@1`, `${granularity}.recall_any@5`, `${granularity}.recall_any@10`, `${granularity}.ndcg_any@10`,
    ...(granularity === 'turn' ? ['session.recall_any@5', 'session.recall_all@5', 'session.recall_any@10', 'session.ndcg_any@10'] : [])];

  const lines: string[] = [];
  lines.push(`# LongMemEval (${dataset}, ${granularity} granularity, ${rows.length} questions)\n`);
  lines.push(`| retriever | ${keys.join(' | ')} | p50 ms | p95 ms | docs scanned |`);
  lines.push(`|---|${keys.map(() => '---').join('|')}|---|---|---|`);
  for (const m of METHODS) {
    const vals = keys.map((k) => mean(rows.map((r) => r.metrics[m][k])).toFixed(3));
    const lat = rows.map((r) => r.latencyMs[m]);
    lines.push(`| ${m} | ${vals.join(' | ')} | ${pct(lat, 0.5)} | ${pct(lat, 0.95)} | ${mean(rows.map((r) => r.scanned[m])).toFixed(0)} |`);
  }
  const key = granularity === 'turn' ? 'session.recall_any@5' : 'session.recall_any@5';
  lines.push(`\n## ${key} by question type\n`);
  lines.push(`| question type | n | ${METHODS.join(' | ')} |`);
  lines.push(`|---|---|${METHODS.map(() => '---').join('|')}|`);
  for (const type of [...new Set(rows.map((r) => r.question_type))].sort()) {
    const subset = rows.filter((r) => r.question_type === type);
    lines.push(`| ${type} | ${subset.length} | ${METHODS.map((m) => mean(subset.map((r) => r.metrics[m][key])).toFixed(3)).join(' | ')} |`);
  }
  const text = lines.join('\n');
  console.log(`\n${text}`);
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  writeFileSync(new URL(`${dataset}-${granularity}-${stamp}.md`, outDir), `${text}\n`);
  writeFileSync(new URL(`${dataset}-${granularity}-${stamp}.json`, outDir), JSON.stringify(rows));
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
