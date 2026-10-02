// Port of rank_bm25's BM25Okapi (https://github.com/dorianbrown/rank_bm25, Apache-2.0), the paper's
// BM25: text split on spaces only, no lowercasing or stemming.
export function bm25OkapiRanking(corpus: string[], query: string, k1 = 1.5, b = 0.75, epsilon = 0.25): number[] {
  const docs = corpus.map((d) => d.split(' '))
  const docFreqs = docs.map((doc) => {
    const f = new Map<string, number>()
    for (const w of doc) f.set(w, (f.get(w) ?? 0) + 1)
    return f
  })
  const docLen = docs.map((d) => d.length)
  const avgdl = docLen.reduce((a, b) => a + b, 0) / docs.length

  const nd = new Map<string, number>() // word → number of documents containing it
  for (const f of docFreqs) for (const w of f.keys()) nd.set(w, (nd.get(w) ?? 0) + 1)
  const idf = new Map<string, number>()
  let idfSum = 0
  const negative: string[] = []
  for (const [word, freq] of nd) {
    const v = Math.log(docs.length - freq + 0.5) - Math.log(freq + 0.5)
    idf.set(word, v)
    idfSum += v
    if (v < 0) negative.push(word)
  }
  const eps = epsilon * (idfSum / idf.size)
  for (const w of negative) idf.set(w, eps)

  const scores = new Float64Array(docs.length)
  for (const q of query.split(' ')) {
    const w = idf.get(q) ?? 0
    for (let i = 0; i < docs.length; i++) {
      const tf = docFreqs[i].get(q) ?? 0
      scores[i] += w * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + (b * docLen[i]) / avgdl)))
    }
  }
  // np.argsort(scores)[::-1]: ascending then reversed, so ties end up in descending index order
  return Array.from(scores.keys())
    .sort((a, c) => scores[a] - scores[c] || a - c)
    .reverse()
}
