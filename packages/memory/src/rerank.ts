import {
  AutoModelForSequenceClassification,
  AutoTokenizer,
  env,
  type PreTrainedModel,
  type PreTrainedTokenizer,
  type Tensor,
} from '@huggingface/transformers'
import { dataPath } from '@voice/config'

export const RERANKERS = {
  minilm: { repo: 'Xenova/ms-marco-MiniLM-L-6-v2', externalData: false },
  bge: { repo: 'onnx-community/bge-reranker-v2-m3-ONNX', externalData: true },
} as const
export type RerankerName = keyof typeof RERANKERS

const BATCH = 16

export class Reranker {
  private loaded?: Promise<{ tokenizer: PreTrainedTokenizer; model: PreTrainedModel }>

  constructor(
    private name: RerankerName,
    private maxLength = 512,
  ) {}

  load() {
    this.loaded ??= (async () => {
      env.allowRemoteModels = true
      env.cacheDir = dataPath('models', 'hf-cache')
      const { repo, externalData } = RERANKERS[this.name]
      const tokenizer = await AutoTokenizer.from_pretrained(repo)
      const model = await AutoModelForSequenceClassification.from_pretrained(repo, {
        dtype: 'fp32',
        device: 'cpu',
        use_external_data_format: externalData,
      })
      return { tokenizer, model }
    })()
    return this.loaded
  }

  async score(question: string, passages: string[]): Promise<number[]> {
    const { tokenizer, model } = await this.load()
    const scores = new Array<number>(passages.length)
    const order = passages.map((_, i) => i).sort((a, b) => passages[a].length - passages[b].length)
    for (let s = 0; s < order.length; s += BATCH) {
      const idx = order.slice(s, s + BATCH)
      const inputs = tokenizer(
        idx.map(() => question),
        {
          text_pair: idx.map((i) => passages[i]),
          padding: true,
          truncation: true,
          max_length: this.maxLength,
        },
      )
      const { logits } = (await model(inputs)) as { logits: Tensor }
      const data = logits.data as Float32Array
      for (const [b, i] of idx.entries()) scores[i] = data[b]
    }
    return scores
  }
}

const minilm = new Reranker('minilm', 256)

export const rerankScores = (question: string, passages: string[]) => minilm.score(question, passages)

export async function warmReranker(): Promise<void> {
  await rerankScores('warm up', ['warm up'])
}
