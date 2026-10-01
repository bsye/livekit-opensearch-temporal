// RM3 pseudo-relevance feedback (Lavrenko & Croft relevance model, interpolated), following Anserini's
// implementation and defaults: fbDocs=10, fbTerms=10, originalQueryWeight=0.5. Feedback documents are
// the top BM25 hits; P(w|D) uses OpenSearch's own analyzed term vectors (english analyzer), so terms
// are the same stems the index uses. Classic query expansion with no model and no LLM.
const FB_DOCS = 10;
const FB_TERMS = 10;
const ORIGINAL_WEIGHT = 0.5;

type Post = (url: string, body: unknown) => Promise<unknown>;

export async function rm3Query(opensearch: string, index: string, post: Post, question: string, hits: { _id: string; _score: number }[]) {
  const top = hits.slice(0, FB_DOCS);
  // original query terms, analyzed like the index, uniform weights
  const analyzed = (await post(`${opensearch}/${index}/_analyze`, { field: 'text', text: question })) as { tokens: { token: string }[] };
  const qTerms = [...new Set(analyzed.tokens.map((t) => t.token))];
  const original = new Map(qTerms.map((t) => [t, 1 / qTerms.length]));
  if (top.length === 0) return termQuery(original);

  // relevance model: sum over feedback docs of P(w|D) * P(D|Q), with P(D|Q) the normalised BM25 score
  const tv = (await post(`${opensearch}/${index}/_mtermvectors`, {
    ids: top.map((h) => h._id),
    parameters: { fields: ['text'], term_statistics: false, field_statistics: false, positions: false, offsets: false },
  })) as { docs: { _id: string; term_vectors?: { text?: { terms: Record<string, { term_freq: number }> } } }[] };
  const scoreSum = top.reduce((a, h) => a + h._score, 0) || 1;
  const weightOf = new Map(top.map((h) => [h._id, h._score / scoreSum]));
  const fb = new Map<string, number>();
  for (const doc of tv.docs) {
    const terms = doc.term_vectors?.text?.terms ?? {};
    const len = Object.values(terms).reduce((a, t) => a + t.term_freq, 0) || 1;
    const pd = weightOf.get(doc._id) ?? 0;
    for (const [term, { term_freq }] of Object.entries(terms)) fb.set(term, (fb.get(term) ?? 0) + (term_freq / len) * pd);
  }
  const best = [...fb.entries()].sort((a, b) => b[1] - a[1]).slice(0, FB_TERMS);
  const fbSum = best.reduce((a, [, w]) => a + w, 0) || 1;

  // interpolate the two L1-normalised term distributions
  const final = new Map<string, number>();
  for (const [t, w] of original) final.set(t, ORIGINAL_WEIGHT * w);
  for (const [t, w] of best) final.set(t, (final.get(t) ?? 0) + (1 - ORIGINAL_WEIGHT) * (w / fbSum));
  return termQuery(final);
}

/** bool of term queries on the analyzed field, each boosted by its weight. */
function termQuery(weights: Map<string, number>) {
  return [...weights.entries()].map(([term, w]) => ({ term: { text: { value: term, boost: w } } }));
}
