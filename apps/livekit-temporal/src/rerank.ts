// Cross-encoder re-ranker for memory recall: ms-marco-MiniLM-L-6-v2 (22M params) reads the question
// and each candidate together and scores relevance. ONNX via @huggingface/transformers (CPU).
// Benchmarked in benchmarks/longmemeval: BM25 top 20 -> MiniLM is +0.10 recall@5 for ~40 ms.
import { fileURLToPath } from 'node:url';
import {
  AutoModelForSequenceClassification,
  AutoTokenizer,
  env,
  type PreTrainedModel,
  type PreTrainedTokenizer,
  type Tensor,
} from '@huggingface/transformers';

const REPO = 'Xenova/ms-marco-MiniLM-L-6-v2';
const BATCH = 16;
let loading: Promise<{ tokenizer: PreTrainedTokenizer; model: PreTrainedModel }> | undefined;

function load() {
  loading ??= (async () => {
    env.cacheDir = fileURLToPath(new URL('../../../data/models/hf-cache/', import.meta.url));
    const tokenizer = await AutoTokenizer.from_pretrained(REPO);
    const model = await AutoModelForSequenceClassification.from_pretrained(REPO, { dtype: 'fp32', device: 'cpu' });
    return { tokenizer, model };
  })();
  return loading;
}

/** Load the model ahead of the first recall (call at agent start). */
export async function warmReranker(): Promise<void> {
  await rerankScores('warm up', ['warm up']);
}

/** Relevance score of each passage for the question (higher is more relevant). */
export async function rerankScores(question: string, passages: string[]): Promise<number[]> {
  const { tokenizer, model } = await load();
  const scores = new Array<number>(passages.length);
  const order = passages.map((_, i) => i).sort((a, b) => passages[a].length - passages[b].length);
  for (let s = 0; s < order.length; s += BATCH) {
    const idx = order.slice(s, s + BATCH);
    const inputs = tokenizer(idx.map(() => question), {
      text_pair: idx.map((i) => passages[i]),
      padding: true,
      truncation: true,
      max_length: 512,
    });
    const { logits } = (await model(inputs)) as { logits: Tensor };
    const data = logits.data as Float32Array;
    idx.forEach((i, b) => (scores[i] = data[b]));
  }
  return scores;
}
