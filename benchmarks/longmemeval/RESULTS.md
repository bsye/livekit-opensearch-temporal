# LongMemEval retrieval results

Official protocol (ports of LongMemEval's `eval_utils.py` / `run_retrieval.py`): user turns as documents, question as query, abstention and no-target questions excluded. Machine: M5 Max. Run: `npm run bench -w longmemeval-bench`.

## LongMemEval (longmemeval_s_cleaned, turn granularity, 419 questions)
| retriever | turn.recall_any@1 | turn.recall_any@5 | turn.recall_any@10 | turn.ndcg_any@10 | session.recall_any@5 | session.recall_all@5 | session.recall_any@10 | session.ndcg_any@10 | p50 ms | p95 ms | docs scanned |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.609 | 0.895 | 0.926 | 0.745 | 0.921 | 0.759 | 0.943 | 0.753 | 12 | 32 | 0 |
| laya | 0.167 | 0.542 | 0.721 | 0.368 | 0.587 | 0.284 | 0.773 | 0.383 | 923 | 1653 | 244 |
| bm25+laya | 0.325 | 0.754 | 0.862 | 0.531 | 0.795 | 0.465 | 0.895 | 0.546 | 185 | 497 | 50 |
### session.recall_any@5 by question type
| question type | n | bm25 | laya | bm25+laya |
|---|---|---|---|---|
| knowledge-update | 72 | 0.986 | 0.806 | 0.958 |
| multi-session | 121 | 0.942 | 0.760 | 0.901 |
| single-session-assistant | 5 | 1.000 | 0.200 | 0.800 |
| single-session-preference | 30 | 0.633 | 0.333 | 0.500 |
| single-session-user | 64 | 0.969 | 0.656 | 0.828 |
| temporal-reasoning | 127 | 0.906 | 0.339 | 0.654 |

## Published reference (LongMemEval_M, not S; paper Table 9, value = session, K = V, Recall@5)

| BM25 | Contriever | Stella V5 1.5B |
|---|---|---|
| 0.634 | 0.723 | 0.720 |

The paper does not state whether Recall is recall_any or recall_all; both are reported above.

## LongMemEval_M (cleaned), 419 questions, ~2,400 user turns / ~475 sessions per question

Controlled comparison, same data and protocol. Recall = recall_all (the paper's Recall@k sits
between our recall_all and recall_any; it does not define which it reports).

| retriever | session R@5 | session NDCG@5 | session R@10 | session NDCG@10 | turn R@5 | turn NDCG@5 | turn R@10 | turn NDCG@10 | p50 ms |
|---|---|---|---|---|---|---|---|---|---|
| bm25-paper (exact port of the paper's rank_bm25 BM25Okapi, space-split) | 0.578 | 0.614 | 0.659 | 0.639 | 0.356 | 0.408 | 0.437 | 0.435 | 11–14 |
| **bm25** (OpenSearch, english analyzer) | **0.745** | **0.779** | **0.828** | **0.803** | **0.556** | **0.598** | **0.654** | **0.630** | 17–23 |
| bm25-fuzzy | 0.680 | 0.711 | 0.752 | 0.732 | 0.508 | 0.555 | 0.611 | 0.588 | 13–27 |
| bm25 top-50 → Laya (zero-shot) | – | – | – | – | 0.320 | 0.387 | 0.451 | 0.436 | 120 |

Published (paper Table 9, LongMemEval_M **original** data, K = V):

| retriever | session R@5 / N@5 / R@10 / N@10 | round R@5 / N@5 / R@10 / N@10 |
|---|---|---|
| BM25 | 0.634 / 0.516 / 0.710 / 0.540 | 0.472 / 0.352 / 0.538 / 0.372 |
| Contriever | 0.723 / 0.634 / 0.823 / 0.663 | 0.589 / 0.454 / 0.747 / 0.495 |
| Stella V5 1.5B | 0.720 / 0.594 / 0.794 / 0.615 | 0.660 / 0.498 / 0.784 / 0.528 |

Findings: tokenisation alone (lowercase, punctuation, stemming, stopwords) lifts BM25 by +0.17
(session) / +0.20 (turn) R@5 on identical data, more than the +0.09 the paper reports for dense
retrievers over its BM25. Fuzzy matching hurts on typed text. Zero-shot Laya re-ordering hurts.
Caveat: our port lands on either side of the paper's BM25 numbers (data version: cleaned vs
original), so comparisons with the paper's dense rows cross data versions; running a dense
baseline on the cleaned data is needed to claim parity.

## Dense baseline on the same data: Contriever vs BM25 (LongMemEval_M cleaned, turn granularity)

`facebook/contriever` exactly as the paper's flat-contriever (masked mean pooling, dot product,
512-token truncation), ONNX in TS, parity with PyTorch to 5 decimals. CPU (CoreML was slower).

| retriever | turn R_all@5 | turn R_any@5 | turn NDCG@5 | turn R_all@10 | turn NDCG@10 | session R_any@5 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| bm25-paper | 0.356 | 0.573 | 0.408 | 0.437 | 0.435 | 0.582 | 21 | 28 |
| contriever | 0.456 | 0.740 | 0.491 | 0.604 | 0.541 | 0.809 | 3067 | 16752 |
| **bm25 (OpenSearch)** | **0.556** | **0.800** | **0.598** | **0.654** | **0.630** | **0.823** | **46** | **66** |

session recall_any@5 by question type:

| question type | n | contriever | bm25-paper | bm25 |
|---|---|---|---|---|
| knowledge-update | 72 | 0.944 | 0.792 | **0.958** |
| multi-session | 121 | **0.868** | 0.529 | 0.843 |
| single-session-assistant | 5 | 0.800 | 0.400 | 0.800 |
| single-session-preference | 30 | **0.667** | 0.233 | 0.333 |
| single-session-user | 64 | 0.766 | 0.609 | **0.922** |
| temporal-reasoning | 127 | 0.732 | 0.591 | **0.795** |

- Harness check: contriever beats bm25-paper by +0.100 turn R@5 here vs +0.117 in the paper (round R@5).
- Properly tokenised BM25 beats the paper's dense baseline overall (+0.10 turn R_all@5, +0.09 NDCG@10)
  at ~65x lower latency; dense wins on preference questions (paraphrase, no shared words) and slightly
  on multi-session.
- Not measured: Stella V5 1.5B (paper: ~+0.07 over Contriever at round level).

## Closing the paraphrase gap without embeddings (LongMemEval_M cleaned, turn granularity)

All settings fixed before running (expansion weight 0.5, preference weight 0.5, question threshold 0.5);
nothing tuned on the test questions. Gold question types are never used by any method.

| retriever | turn R_all@5 | turn R_any@5 | turn NDCG@10 | session R_any@5 | preference (n=30) | query-time cost |
|---|---|---|---|---|---|---|
| contriever (dense baseline) | 0.456 | 0.740 | 0.541 | 0.809 | **0.667** | ~3 s (CPU, cached) |
| bm25 | 0.556 | 0.800 | 0.630 | 0.823 | 0.333 | ~30 ms |
| **bm25+qe** (LLM query expansion) | **0.621** | **0.859** | **0.684** | **0.883** | 0.633 | ~30 ms + ~0.5 s LLM |
| sparse (doc-only learned sparse, rank_features) | 0.573 | 0.816 | 0.655 | 0.845 | 0.400 | ~175 ms (no model at query time) |
| bm25+pref (Laya preference tags) | 0.556 | 0.795 | 0.628 | 0.821 | 0.300 | ~30 ms |
| bm25+qe+pref | 0.616 | 0.852 | 0.677 | 0.876 | 0.533 | ~30 ms + LLM |

- Query expansion is the clear win: +0.065 R_all@5, +0.054 NDCG@10 over BM25, preference 0.33 → 0.63
  (≈ Contriever's 0.67), multi-session 0.92 (Contriever 0.87).
- Learned sparse helps modestly (+0.017 R_all@5) at 5.4 GB index and ~30 min encoding for 1.2M turns.
- Laya preference tags hurt: Laya classifies preference *questions* well (83% of them flagged vs 0–5% of
  other types) but boosting every preference-stating turn pushes irrelevant ones up.
