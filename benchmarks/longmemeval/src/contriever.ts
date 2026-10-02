import {
  AutoModel,
  AutoTokenizer,
  env,
  type PreTrainedModel,
  type PreTrainedTokenizer,
  type Tensor,
} from '@huggingface/transformers'
import { dataPath } from '@voice/config'

env.allowRemoteModels = false
env.localModelPath = `${dataPath('models')}/`

const BATCH = 32
let tokenizer: PreTrainedTokenizer
let model: PreTrainedModel
const cache = new Map<string, Float32Array>()
const CACHE_MAX = 400_000

export async function loadContriever(): Promise<void> {
  tokenizer ??= await AutoTokenizer.from_pretrained('contriever')
  model ??= await AutoModel.from_pretrained('contriever', { dtype: 'fp32', device: 'cpu' })
}

export async function embed(texts: string[]): Promise<Float32Array[]> {
  const out: (Float32Array | undefined)[] = texts.map((t) => cache.get(t))
  const missing = [...new Set(texts.filter((_, i) => !out[i]))].sort((a, b) => a.length - b.length)
  for (let s = 0; s < missing.length; s += BATCH) {
    const batch = missing.slice(s, s + BATCH)
    const inputs = tokenizer(batch, { padding: true, truncation: true, max_length: 512 })
    const { last_hidden_state } = (await model(inputs)) as { last_hidden_state: Tensor }
    const [n, seq, dim] = last_hidden_state.dims as [number, number, number]
    const hidden = last_hidden_state.data as Float32Array
    const mask = (inputs.attention_mask as Tensor).data as BigInt64Array
    for (let b = 0; b < n; b++) {
      const v = new Float32Array(dim)
      let count = 0
      for (let t = 0; t < seq; t++) {
        if (!mask[b * seq + t]) continue
        count++
        const base = (b * seq + t) * dim
        for (let d = 0; d < dim; d++) v[d] += hidden[base + d]
      }
      for (let d = 0; d < dim; d++) v[d] /= count
      if (cache.size >= CACHE_MAX) cache.delete(cache.keys().next().value!)
      cache.set(batch[b], v)
    }
  }
  return texts.map((t, i) => out[i] ?? cache.get(t)!)
}

export async function contrieverRanking(corpus: string[], query: string): Promise<number[]> {
  const [q] = await embed([query])
  const docs = await embed(corpus)
  const scores = docs.map((d) => {
    let s = 0
    for (let i = 0; i < d.length; i++) s += d[i] * q[i]
    return s
  })
  return Array.from(scores.keys())
    .sort((a, b) => scores[a] - scores[b] || a - b)
    .reverse()
}
