/**
 * Learned sparse retrieval with OpenSearch's doc-only model: documents are expanded into weighted
 * vocabulary terms at index time (rank_features); queries only need the tokenizer and IDF weights.
 */
import { readFileSync } from 'node:fs';
import { AutoModel, AutoTokenizer, env, type PreTrainedModel, type PreTrainedTokenizer, type Tensor } from '@huggingface/transformers';
import { dataPath } from '@voice/config';

const DIR = 'sparse-doc-v3-distill';
const MODELS = `${dataPath('models')}/`;
const BATCH = 32;

let tokenizer: PreTrainedTokenizer;
let model: PreTrainedModel;
let vocab: string[];
let idf: Map<number, number>;
let special: Set<number>;
// distinct text → features as a JSON string (LongMemEval_M repeats each turn ~4x)
const cache = new Map<string, string>();

export async function loadSparse(): Promise<void> {
  env.allowRemoteModels = false;
  env.localModelPath = MODELS;
  tokenizer ??= await AutoTokenizer.from_pretrained(DIR);
  model ??= await AutoModel.from_pretrained(DIR, { dtype: 'fp32', device: 'cpu' });
  vocab = readFileSync(`${MODELS}${DIR}/vocab.txt`, 'utf8').split('\n');
  const idfByToken = JSON.parse(readFileSync(`${MODELS}${DIR}/idf.json`, 'utf8')) as Record<string, number>;
  idf = new Map(vocab.map((t, i) => [i, idfByToken[t] ?? 0]));
  special = new Set(['[CLS]', '[SEP]', '[PAD]', '[MASK]', '[UNK]'].map((t) => vocab.indexOf(t)));
}

const key = (id: number) => `t${id}`; // rank_features keys can't contain dots; use token ids

/** Sparse document vectors as JSON {"t<tokenId>": weight, ...}, cached per distinct text. */
export async function encodeDocs(texts: string[]): Promise<string[]> {
  const missing = [...new Set(texts.filter((t) => !cache.has(t)))].sort((a, b) => a.length - b.length);
  for (let s = 0; s < missing.length; s += BATCH) {
    const batch = missing.slice(s, s + BATCH);
    const inputs = tokenizer(batch, { padding: true, truncation: true, max_length: 512 });
    const { sparse } = (await model({ input_ids: inputs.input_ids, attention_mask: inputs.attention_mask })) as { sparse: Tensor };
    const [n, v] = sparse.dims as [number, number];
    const data = sparse.data as Float32Array;
    for (let b = 0; b < n; b++) {
      const features: Record<string, number> = {};
      for (let id = 0; id < v; id++) {
        const w = data[b * v + id];
        if (w > 0 && !special.has(id)) features[key(id)] = w;
      }
      cache.set(batch[b], JSON.stringify(features));
    }
  }
  return texts.map((t) => cache.get(t)!);
}

/** Query terms weighted by the model's IDF (no model inference at query time). */
export function queryWeights(query: string): Record<string, number> {
  const ids = (tokenizer(query, { add_special_tokens: false }).input_ids as Tensor).tolist() as unknown as number[][];
  const weights: Record<string, number> = {};
  for (const id of new Set(ids.flat().map(Number))) {
    const w = idf.get(id) ?? 0;
    if (w > 0 && !special.has(id)) weights[key(id)] = w;
  }
  return weights;
}

export function tokenOf(featureKey: string): string {
  return vocab[Number(featureKey.slice(1))];
}
