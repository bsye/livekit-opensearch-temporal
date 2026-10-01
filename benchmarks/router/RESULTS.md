# Router test: memory, action or chat?

Can a fast model decide, per user turn, whether to run memory recall before the LLM answers?
220 turns: 120 needing memory (the 100 LongMemEval library questions + 20 conversational ones),
50 action requests, 50 chat / general knowledge (`src/turns.ts`). Phrasings fixed before running.

## Laya (`npm run bench -w router-bench`), ~5 ms per turn (4 ms model + 1 ms HTTP)

| phrasing | memory turns caught | false alarms (action / chat → memory) |
|---|---|---|
| noul "asking about their own past?" | 13% | 0% / 0% |
| noul "needs earlier conversations?" | 1% | 0% / 0% |
| choice A/B/C (past / action / chat) | **71%** | 6% / 22% |

The yes/no phrasings fail (Laya's known label-following on noul). The 3-way choice is usable but
misses 29% of memory questions (single-session-assistant 29% caught, preference 44%) and 3-way
accuracy is 65%.

## Relevance gate (`npm run gate -w router-bench`): search memory on every turn, ~44 ms

| re-ranker threshold | memory turns with a relevant match | action / chat turns that would get memory |
|---|---|---|
| 0 | 61% | 10% / 34% |
| 2 | 48% | 6% / 24% |
| 4 | 33% | 4% / 12% |

Worse: relevance answers "did you say something like this before?" (often yes for chit-chat the
user has said before), not "is the user asking about their past?".

Conclusion: zero-shot, neither is reliable enough to route alone. The routing decision is a narrow,
fixed-label task, the kind Laya's authors report fine-tuning lifts from ~0.36 to ~0.77+.
