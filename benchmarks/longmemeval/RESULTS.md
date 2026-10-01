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
