// Cross-encoder re-ranking: the model reads (question, candidate) together and scores relevance.
// ONNX models from the Hugging Face Hub run with @huggingface/transformers (ONNX Runtime, CPU).
// Benchmark only.
import { fileURLToPath } from 'node:url';
import {
  AutoModelForSequenceClassification,
  AutoTokenizer,
  env,
  type PreTrainedModel,
  type PreTrainedTokenizer,
  type Tensor,
} from '@huggingface/transformers';

export const RERANKERS = {
  // BAAI bge-reranker-v2-m3 (568M); fp32 weights ship as external data (model.onnx_data)
  bge: { repo: 'onnx-community/bge-reranker-v2-m3-ONNX', maxLength: 512, externalData: true },
  minilm: { repo: 'Xenova/ms-marco-MiniLM-L-6-v2', maxLength: 512, externalData: false }, // classic MS MARCO cross-encoder (22M)
} as const;
export type RerankerName = keyof typeof RERANKERS;

const BATCH = 16;

export class Reranker {
  private tokenizer!: PreTrainedTokenizer;
  private model!: PreTrainedModel;

  constructor(private name: RerankerName) {}

  async load(): Promise<void> {
    env.allowRemoteModels = true;
    env.cacheDir = fileURLToPath(new URL('../../../data/models/hf-cache/', import.meta.url));
    const { repo, externalData } = RERANKERS[this.name];
    this.tokenizer = await AutoTokenizer.from_pretrained(repo);
    this.model = await AutoModelForSequenceClassification.from_pretrained(repo, {
      dtype: 'fp32',
      device: 'cpu',
      use_external_data_format: externalData,
    });
  }

  /** Relevance score of each passage for the question (higher is more relevant). */
  async score(question: string, passages: string[]): Promise<number[]> {
    const scores = new Array<number>(passages.length);
    // shortest first so batches pad to similar lengths
    const order = passages.map((_, i) => i).sort((a, b) => passages[a].length - passages[b].length);
    for (let s = 0; s < order.length; s += BATCH) {
      const idx = order.slice(s, s + BATCH);
      const inputs = this.tokenizer(idx.map(() => question), {
        text_pair: idx.map((i) => passages[i]),
        padding: true,
        truncation: true,
        max_length: RERANKERS[this.name].maxLength,
      });
      const { logits } = (await this.model(inputs)) as { logits: Tensor };
      const data = logits.data as Float32Array;
      idx.forEach((i, b) => (scores[i] = data[b]));
    }
    return scores;
  }
}
