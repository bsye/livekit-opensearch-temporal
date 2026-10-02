import {
  AutoModelForSequenceClassification,
  AutoTokenizer,
  env,
  type PreTrainedModel,
  type PreTrainedTokenizer,
  type Tensor,
} from '@huggingface/transformers';
import { dataPath } from '@voice/config';

export const RERANKERS = {
  minilm: { repo: 'Xenova/ms-marco-MiniLM-L-6-v2', externalData: false }, // 22M, used by recall
  bge: { repo: 'onnx-community/bge-reranker-v2-m3-ONNX', externalData: true }, // 568M, benchmark only
} as const;
export type RerankerName = keyof typeof RERANKERS;

const BATCH = 16;

/** Cross-encoder: reads the question and each passage together; scores are logits (> 0 = relevant). */
export class Reranker {
  private loaded?: Promise<{ tokenizer: PreTrainedTokenizer; model: PreTrainedModel }>;

  constructor(
    private name: RerankerName,
    private maxLength = 512,
  ) {}

  load() {
    this.loaded ??= (async () => {
      env.allowRemoteModels = true;
      env.cacheDir = dataPath('models', 'hf-cache');
      const { repo, externalData } = RERANKERS[this.name];
      const tokenizer = await AutoTokenizer.from_pretrained(repo);
      const model = await AutoModelForSequenceClassification.from_pretrained(repo, {
        dtype: 'fp32',
        device: 'cpu',
        use_external_data_format: externalData,
      });
      return { tokenizer, model };
    })();
    return this.loaded;
  }

  async score(question: string, passages: string[]): Promise<number[]> {
    const { tokenizer, model } = await this.load();
    const scores = new Array<number>(passages.length);
    // shortest first, so each batch pads to similar lengths
    const order = passages.map((_, i) => i).sort((a, b) => passages[a].length - passages[b].length);
    for (let s = 0; s < order.length; s += BATCH) {
      const idx = order.slice(s, s + BATCH);
      const inputs = tokenizer(idx.map(() => question), {
        text_pair: idx.map((i) => passages[i]),
        padding: true,
        truncation: true,
        max_length: this.maxLength,
      });
      const { logits } = (await model(inputs)) as { logits: Tensor };
      const data = logits.data as Float32Array;
      idx.forEach((i, b) => (scores[i] = data[b]));
    }
    return scores;
  }
}

// 256 tokens: same LongMemEval accuracy as 512 (top-1 0.652, recall_any@5 0.862), cheaper on long messages
const minilm = new Reranker('minilm', 256);

export const rerankScores = (question: string, passages: string[]) => minilm.score(question, passages);

export async function warmReranker(): Promise<void> {
  await rerankScores('warm up', ['warm up']);
}
