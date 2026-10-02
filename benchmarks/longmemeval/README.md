# Conversation-memory retrieval benchmark (LongMemEval)

How well, and how fast, can the voice agent find the right moment in past conversations —
and do we need embeddings to do it? This package answers that on **LongMemEval**, the standard
long-term chat memory benchmark (Wu et al., ICLR 2025), using its official evaluation protocol.
Raw per-run tables are in [RESULTS.md](RESULTS.md).

## TL;DR

On LongMemEval_M (419 questions, ~2,400 past messages each, 1.2M in total):

| pipeline | recall@5 | top-1 | query time | needs |
|---|---|---|---|---|
| BM25 | 0.556 | 0.516 | ~30 ms | OpenSearch |
| **BM25 → MiniLM re-rank of top 20** | **0.652** | **0.652** | **~70 ms** | OpenSearch + 22M model, no LLM |
| BM25 + query expansion → MiniLM top 50, Laya-routed | 0.699 | 0.649 | ~130 ms + ~0.5 s LLM | + LLM |
| Best measured (hybrid BM25 + query expansion + Qwen3 embeddings → bge re-rank, Laya-routed) | 0.728 | 0.644 | ~2.3 s + LLM | + embeddings everywhere, 568M re-ranker |
| Contriever (the paper's dense baseline) | 0.456 | 0.372 | ~3 s CPU | embeddings |

(recall@5 = turn-level `recall_all@5`; top-1 = turn-level `recall_any@1`.)

- Properly configured **BM25 beats the paper's dense baseline** on the same data, and a modern
  embedder (Qwen3-Embedding-0.6B) alone is only BM25-level.
- **A small cross-encoder re-ranker is the biggest lever**: +0.10 recall@5 over BM25 for ~40 ms.
- Embeddings add at most +0.03 on top of the best no-embedding stack.
- **Laya** (fast decision model) is not a good relevance ranker, but it is a good **router**:
  deciding per question whether to re-rank lifts every metric.

## Methodology

### Dataset
- **LongMemEval** ([paper](https://arxiv.org/abs/2410.10813), [code](https://github.com/xiaowu0162/LongMemEval),
  MIT). 500 questions, each with its own timestamped user↔assistant chat history (the "haystack")
  and labels marking which sessions and which turns contain the evidence.
- **LongMemEval_S**: ~48 sessions (~250 user turns) per question. **LongMemEval_M**: ~500 sessions
  (~2,400 user turns, 1.2M turns overall, 247,666 distinct). M is the headline setting.
- We use the 2025 **cleaned** release (the only one distributed now); the paper used the original.
  Absolute numbers therefore differ from the paper's; **comparisons are only made between retrievers
  run here on identical data**.

### Protocol (ported line for line from the official code, `src/official.ts`)
- **Corpus**: each user turn is one document (turn granularity) or a session's user turns joined
  (session granularity). Assistant turns are not indexed, as in the paper.
- **Query**: the question text only.
- **Labels**: a document is relevant if it holds evidence (`has_answer`).
- **Exclusions**: the 30 abstention questions and 51 questions whose evidence is only in assistant
  turns are left out of the averages → **419 evaluated questions**.
- **Metrics**: `recall_any@k`, `recall_all@k`, `ndcg_any@k` for k ∈ {1, 3, 5, 10, 30, 50}, at turn level
  and converted to session level (`evaluate_retrieval_turn2session`).
- **Sanity checks**: our port of the paper's BM25 and our Contriever reproduce the paper's *ordering*
  (Contriever beats that BM25 by +0.100 here vs +0.117 in the paper); model implementations were
  checked for parity with their references (Contriever vs PyTorch to 5 decimals; the sparse encoder's
  similarity matches its model card exactly).

### Fairness rules
- **Nothing is tuned on the test questions.** Every weight and threshold was fixed before running:
  query-expansion weight 0.5, RRF k = 60, RM3 with Anserini defaults (10 docs, 10 terms, 0.5),
  re-rank depth 50 (20 and 10 reported as speed variants), Laya threshold 0.5.
- **Gold question types are never used** by any method (only to break results down by category).
- **Latency is query time only**: embedding or encoding the corpus is index-time work and timed
  separately. LLM query expansion (~0.5 s per question, cached for reruns) is reported separately.

### Environment
Apple M5 Max (64 GB), OpenSearch 3.9 (single node, 1 GB heap), LM Studio (embeddings, LLM on the GPU),
ONNX Runtime CPU for Contriever, sparse encoder and re-rankers (CoreML was slower for these), Laya on
MLX. All benchmark code is TypeScript; Python is used only for one-time ONNX exports of two models.

## Retrievers tested

| name in results | what it is |
|---|---|
| `bm25-paper` | The paper's BM25: `rank_bm25` BM25Okapi on text split on spaces (no lowercasing or stemming) |
| `bm25` | OpenSearch BM25 with the English analyzer (lowercasing, stemming, stopwords) |
| `bm25-fuzzy` | BM25 tolerating misspellings (`fuzziness: AUTO`) |
| `bm25+qe` | BM25 over the question plus LLM-generated query-expansion terms (weight 0.5) |
| `rm3` | BM25 + RM3 pseudo-relevance feedback (expansion terms from the top hits, no LLM) |
| `sparse` | Learned sparse retrieval: OpenSearch's doc-only neural sparse model, `rank_features` |
| `contriever` | facebook/contriever, the paper's dense baseline |
| `dense-qwen3`, `dense-nomic` | Qwen3-Embedding-0.6B / nomic-embed-text v1.5 via LM Studio |
| `rrf-bm25+<e>`, `rrf-bm25qe+<e>` | Hybrid: reciprocal rank fusion of BM25 (or BM25+qe) with embeddings |
| `<r>:<base>` | Cross-encoder re-ranking (`minilm` = ms-marco-MiniLM-L-6-v2, `bge` = bge-reranker-v2-m3) of `<base>`'s top N |
| `laya`, `bm25+laya` | Laya scan of every turn / Laya re-ordering BM25's top 50 |
| `bm25+pref` | BM25 with Laya preference tags boosted for preference questions |
| Laya-routed | Laya decides per question whether to re-rank (skipped for preference questions) |

## Results summary (LongMemEval_M, turn level)

| retriever | recall@5 (all) | recall@5 (any) | top-1 | NDCG@10 | preference Qs | query time |
|---|---|---|---|---|---|---|
| bm25-paper | 0.356 | 0.573 | 0.360 | 0.435 | 0.233 | 14 ms |
| contriever | 0.456 | 0.740 | 0.372 | 0.541 | 0.667 | ~3 s |
| rm3 | 0.492 | 0.759 | 0.411 | 0.568 | 0.367 | 31 ms |
| dense-nomic | 0.520 | 0.800 | 0.489 | 0.599 | 0.433 | 19 ms |
| dense-qwen3 | 0.547 | 0.833 | 0.516 | 0.628 | 0.600 | 36 ms |
| **bm25** | 0.556 | 0.800 | 0.516 | 0.630 | 0.333 | 29 ms |
| sparse | 0.573 | 0.816 | 0.532 | 0.655 | 0.400 | 175 ms |
| rrf-bm25+qwen3 | 0.601 | 0.866 | 0.532 | 0.681 | 0.600 | 37 ms |
| bm25+qe | 0.621 | 0.859 | 0.556 | 0.684 | 0.633 | 21 ms + LLM |
| rrf-bm25qe+qwen3 | 0.621 | 0.883 | 0.556 | 0.694 | 0.700 | 37 ms + LLM |
| minilm:bm25 (top 10) | 0.616 | 0.847 | 0.654 | 0.693 | 0.467 | 46 ms |
| **minilm:bm25 (top 20)** | **0.652** | 0.862 | **0.652** | 0.716 | 0.433 | **68 ms** |
| minilm:bm25 (top 50) | 0.671 | 0.876 | 0.644 | 0.731 | 0.433 | 125 ms |
| minilm:bm25+qe (top 20) | 0.680 | 0.874 | 0.647 | 0.735 | 0.467 | 66 ms + LLM |
| minilm:bm25+qe (top 50) | 0.690 | 0.893 | 0.649 | 0.749 | 0.467 | 121 ms + LLM |
| bge:bm25+qe (top 50) | 0.704 | 0.893 | 0.654 | 0.755 | 0.433 | 2.3 s + LLM |
| bge:rrf-bm25qe+qwen3 (top 50) | 0.726 | 0.895 | 0.644 | 0.758 | 0.433 | 2.3 s + LLM |
| Laya-routed minilm:bm25+qe (top 50) | 0.699 | 0.900 | – | 0.756 | 0.567 | 121 ms + LLM |
| Laya-routed bge:rrf-bm25qe+qwen3 | **0.728** | **0.905** | – | **0.766** | 0.667 | 2.3 s + LLM |
| *Laya-based, did not help:* bm25+laya | 0.320 | 0.642 | 0.258 | 0.436 | 0.333 | 120 ms |
| *Laya-based, did not help:* bm25+pref | 0.556 | 0.795 | 0.513 | 0.628 | 0.300 | 29 ms |

Published reference (paper Table 9, original data, session level, recall@5): BM25 0.634, Contriever
0.723, Stella V5 1.5B 0.720.

## What we learned
1. **Tokenisation matters more than the retriever family.** Lowercasing, stemming and stopwords lift
   BM25 by +0.20 recall@5 on identical data, more than the gap the paper reports between dense
   retrievers and its BM25.
2. **Re-ranking is the best use of a model.** A 22M-parameter cross-encoder over the top 20 gives
   +0.10 recall@5 and the best top-1 for ~40 ms; the 568M bge adds little more for 15× the time.
3. **Paraphrase is the remaining gap** (preference questions: "what would I enjoy?"). Query expansion
   fixes most of it; re-rankers trained on web search make it worse; embeddings help a little.
4. **Laya** is fast (~5 ms), excellent at **classifying one text against a fixed question** (83% of
   preference questions flagged, 0–5% false alarms), and poor at **relevance between two texts**.
   Use it to route and to label, not to rank.

## Running

```sh
npm run setup                                   # repo root: tools, dependencies, containers
services/laya/run.sh                            # for the Laya-based methods
# download longmemeval_s_cleaned.json / longmemeval_m_cleaned.json into data/benchmarks/longmemeval/
npm run bench -w @bench/longmemeval -- --dataset longmemeval_m_cleaned \
  --methods 'bm25,bm25+qe,minilm:bm25' --rerank-depth 20 --tag mytag
```

`--granularity turn|session`, `--limit N`, `--fresh` (restart), `--tag` (separate progress file;
runs are resumable). Dense methods need LM Studio with the embedding model loaded; `contriever` and
`sparse` need the one-time ONNX exports (`services/onnx-export/run.sh contriever|sparse`); `bm25+qe`
needs the LLM in LM Studio (expansions are cached after the first run).

`npm run load-library -w @bench/longmemeval` loads a LongMemEval_M-based library into the agent's
memory to try recall by voice, and writes the question sheet the router and voicechat benchmarks use.

## Glossary

- **BM25**: the standard keyword-ranking formula in search engines (Lucene, Elasticsearch, OpenSearch).
  Scores documents by how many query words they contain, weighting rare words more and long
  documents less.
- **Tokenisation / analyzer**: how text is split into the words BM25 matches. The English analyzer
  lowercases, removes punctuation and stopwords ("the", "a") and stems ("graduated" → "graduat").
- **Stemming**: reducing words to a common root so different forms match.
- **Fuzzy matching**: also matching words within a small edit distance (typos).
- **Embedding / dense retrieval**: a neural model turns each text into a vector; texts with similar
  meaning get nearby vectors, so retrieval matches meaning rather than words.
- **Learned sparse retrieval**: a model expands each document into weighted related words at index
  time; search stays a keyword lookup in an inverted index (no vectors).
- **Inverted index**: the core search-engine structure: for each word, the list of documents containing it.
- **Hybrid search**: combining keyword and embedding retrieval.
- **RRF (reciprocal rank fusion)**: merges several rankings by summing 1 / (60 + rank) per document;
  needs no score calibration.
- **Query expansion (QE)**: adding related words to the query before searching; here generated by an LLM.
- **Pseudo-relevance feedback / RM3**: query expansion without an LLM: assume the top hits are
  relevant and add their most characteristic words to the query.
- **Cross-encoder / re-ranker**: a model that reads the question and one candidate *together* and
  outputs a relevance score; too slow for a whole corpus, so it re-orders the top N of a first stage.
- **First stage / re-ranking depth**: the fast retriever that produces candidates / how many of them
  the re-ranker re-orders (top N).
- **Laya**: an open, fast (~5 ms) "System 1" decision model that answers typed questions (yes/no,
  choice, score) about a text; runs locally on MLX.
- **Routing**: deciding per question which pipeline to use.
- **Haystack**: the past conversations a question is asked against. **Session**: one past
  conversation. **Turn**: one message in it.
- **Turn / session granularity**: whether the indexed documents are single user turns or whole sessions.
- **recall_any@k**: 1 if *at least one* evidence document is in the top k, else 0, averaged over questions.
- **recall_all@k**: 1 if *all* evidence documents are in the top k (stricter; matters for questions
  whose answer spans several sessions).
- **Top-1**: `recall_any@1`, the right message ranked first.
- **NDCG@k**: rewards putting evidence higher in the top k (1.0 = all evidence at the top).
- **p50 / p95**: median / 95th-percentile latency per question.
- **Abstention question**: a question with no answer in the history; the right behaviour is "I don't
  know". Excluded from retrieval metrics.
- **Question types** (LongMemEval): single-session-user / -assistant / -preference (answer in one
  session; preference = recommendations fitting the user's taste), multi-session (combine several
  sessions), temporal-reasoning (time-dependent), knowledge-update (a fact that changed over time).
