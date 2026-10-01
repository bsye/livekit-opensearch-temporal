// Dense retrieval with modern embedding models served by LM Studio on the Apple GPU
// (OpenAI-compatible /v1/embeddings). Benchmark only. Vectors are L2-normalised and cached per
// distinct text, in memory and on disk, so later runs (hybrids, re-ranking) don't re-embed.
import { createHash } from 'node:crypto';
import { appendFileSync, existsSync, mkdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const LLM = process.env.LLM_BASE_URL ?? 'http://localhost:1234/v1';
const BATCH = 64;

export interface EmbedderSpec {
  name: string; // method suffix, e.g. "qwen3"
  model: string; // LM Studio model id
  dim: number;
  queryPrefix: string; // as each model card prescribes
  docPrefix: string;
}

export const EMBEDDERS: Record<string, EmbedderSpec> = {
  qwen3: {
    name: 'qwen3',
    model: 'text-embedding-qwen3-embedding-0.6b',
    dim: 1024,
    // Qwen3-Embedding: instruction on the query side only
    queryPrefix: "Instruct: Given a question about the user's past conversations, retrieve the user message that helps answer it\nQuery:",
    docPrefix: '',
  },
  nomic: {
    name: 'nomic',
    model: 'text-embedding-nomic-embed-text-v1.5',
    dim: 768,
    queryPrefix: 'search_query: ',
    docPrefix: 'search_document: ',
  },
};

const hash = (text: string) => createHash('sha1').update(text).digest('base64').slice(0, 16);

export class Embedder {
  private vectors = new Map<string, Float32Array>();
  private keysFile: string;
  private dataFile: string;

  constructor(private spec: EmbedderSpec, cacheDir: URL) {
    const dir = fileURLToPath(new URL(`${spec.name}/`, cacheDir));
    mkdirSync(dir, { recursive: true });
    this.keysFile = `${dir}keys.txt`;
    this.dataFile = `${dir}vectors.f32`;
    if (existsSync(this.keysFile) && existsSync(this.dataFile)) {
      const keys = readFileSync(this.keysFile, 'utf8').split('\n').filter(Boolean);
      const data = readFileSync(this.dataFile);
      const floats = new Float32Array(data.buffer, data.byteOffset, data.byteLength / 4);
      // an interrupted append can leave one side longer; use what both files agree on
      const n = Math.min(keys.length, Math.floor(floats.length / spec.dim));
      for (let i = 0; i < n; i++) this.vectors.set(keys[i], floats.slice(i * spec.dim, (i + 1) * spec.dim));
      console.log(`${spec.name}: ${n} cached embeddings loaded`);
    }
  }

  async embedQuery(question: string): Promise<Float32Array> {
    return (await this.request([this.spec.queryPrefix + question]))[0];
  }

  /** Document vectors, embedding (and persisting) only texts not seen before. */
  async embedDocs(texts: string[]): Promise<Float32Array[]> {
    const missing = [...new Set(texts.filter((t) => !this.vectors.has(hash(t))))].sort((a, b) => a.length - b.length);
    for (let s = 0; s < missing.length; s += BATCH) {
      const batch = missing.slice(s, s + BATCH);
      const vecs = await this.request(batch.map((t) => this.spec.docPrefix + t));
      const keys = batch.map(hash);
      keys.forEach((k, i) => this.vectors.set(k, vecs[i]));
      const buf = Buffer.alloc(vecs.length * vecs[0].length * 4);
      vecs.forEach((v, i) => Buffer.from(v.buffer, v.byteOffset, v.byteLength).copy(buf, i * v.byteLength));
      appendFileSync(this.dataFile, buf);
      appendFileSync(this.keysFile, `${keys.join('\n')}\n`);
    }
    return texts.map((t) => this.vectors.get(hash(t))!);
  }

  /** Ranking by cosine similarity (vectors are normalised), highest first. */
  async ranking(corpus: string[], question: string): Promise<number[]> {
    const q = await this.embedQuery(question);
    const docs = await this.embedDocs(corpus);
    const scores = docs.map((d) => dot(d, q));
    return Array.from(scores.keys()).sort((a, b) => scores[b] - scores[a] || a - b);
  }

  private async request(input: string[]): Promise<Float32Array[]> {
    const res = await fetch(`${LLM}/embeddings`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model: this.spec.model, input }),
    });
    if (!res.ok) throw new Error(`embeddings ${this.spec.model}: ${res.status} ${await res.text()}`);
    const data = ((await res.json()) as { data: { embedding: number[] }[] }).data;
    return data.map((d) => normalise(Float32Array.from(d.embedding)));
  }
}

function normalise(v: Float32Array): Float32Array {
  let n = 0;
  for (const x of v) n += x * x;
  n = Math.sqrt(n) || 1;
  for (let i = 0; i < v.length; i++) v[i] /= n;
  return v;
}

function dot(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** Reciprocal rank fusion (Cormack et al. 2009) with the standard k = 60. */
export function rrf(rankings: number[][], n: number, k = 60): number[] {
  const scores = new Float64Array(n);
  for (const ranking of rankings) ranking.forEach((doc, rank) => (scores[doc] += 1 / (k + rank + 1)));
  return Array.from(scores.keys()).sort((a, b) => scores[b] - scores[a] || a - b);
}
