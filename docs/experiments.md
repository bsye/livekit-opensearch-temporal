# What we tested

The project's log of experiments, in the order they happened: the question, what we tried, what we
measured, and what we decided. The README tells the story; this file keeps the evidence. Raw tables
live next to each benchmark (`benchmarks/*/RESULTS.md`).

All numbers come from one Apple M5 Max (64 GB) running everything locally.

---

## 1. Can the whole stack run on a laptop?

**Question.** Can LiveKit's self-hosted production setup and a voice agent run on macOS with only
local models?

**Tried.** LiveKit's generated Linux setup (Caddy, Redis, egress, ingress) in Docker under OrbStack,
which gives containers the host networking LiveKit needs. Models can't run in Docker on macOS (no
Metal), so speech and the LLM run natively.

**Measured** (the cascade: Silero VAD → STT → LLM → TTS):

| stage | choice | alternative tried | latency |
|---|---|---|---|
| STT | Parakeet TDT 0.6B v2 | Whisper large-v3-turbo: ~720 ms | 60–110 ms |
| LLM first token | Gemma 4 26B-A4B, reasoning off | Gemma 4 E4B: leaks its reasoning when reasoning is off, thinks 1–7 s when on | 530–620 ms |
| TTS first audio | Kokoro 82M, 8-bit | bf16: ~390 ms | ~200 ms |
| voice to voice | | | 2–2.7 s |

**Decided.** Native MLX sidecars for models, containers for everything else.

## 2. Turn-taking

**Question.** When has the user finished speaking?

**Tried and measured.** Silence thresholds alone: short ones split sentences at natural pauses ("There
is a… agent orchestration" became two turns), long ones slowed every reply. Preemptive generation with
preemptive TTS made the agent speak discarded drafts and sound like it repeated itself.

**Decided.** LiveKit's local audio turn detector (v1-mini; v1 only runs on LiveKit Cloud) after
300 ms of silence, endpointing 300–2500 ms. Barge-in needs 600 ms and two words, so "okay" or echo
doesn't interrupt. The LLM starts on the final transcript, but TTS waits for the confirmed turn.

## 3. Temporal as the record

**Question.** How do we get a trustworthy record of every session: who said what, when, and what the
agent did?

**Tried.** LiveKit webhooks → a translator → one Temporal workflow per room session (signal-with-start,
id = room sid, deduplicated by event id, grace period for late events). Then three ways of showing it:

1. Signals named after who did what, so the timeline reads as the conversation.
2. One child workflow per participant ("lanes"). Every event then appeared twice in the room view
   (received, then forwarded), and the user found it hard to follow.
3. One child workflow per exchange (`conversationTurn`), with each step as a labelled row, plus a
   separate telemetry workflow for timings. The timings were still hard to read, cut off from the turn
   they belonged to.

**Decided.** One row per turn, with the timings inside it: the user's side (spoke, recognised, end of
turn), memory, each tool call with its duration, the agent's side (first token, first audio, voice to
voice). Reminders are durable timers; memory indexing is a retried activity.

## 4. Can a small decision model block wrong actions?

**Question.** Laya answers typed questions about a text in ~4 ms. Can it gate tool calls?

**Tried.** 24 labelled request/action pairs (`services/laya/eval_gate.py`).

**Measured.** The best setting got 21/24, but every miss was a wrong argument approved with confidence:
"Transfer 50 euros" → `amount=500` at P = 1.00, "3pm" → `13:00` at 0.96.
([results](../benchmarks/action-gate/RESULTS.md))

**Decided.** Not a gate. Consequential tools ask the user (an approval step inside the tool, so the LLM
can't skip it); reversible ones run and have an undo. Laya audits actions afterwards, for review.

## 5. Memory without embeddings

**Question (the project's assumption).** Can a search engine over the text a conversation already
produces retrieve past conversations as well as embeddings?

**First attempt.** Laya scanning every past message in parallel on the GPU. On LongMemEval_S it found
the evidence in the top 5 for 54% of questions in 923 ms; plain BM25 found 90% in 12 ms. Laya
classifies a text well but judges relevance between two texts poorly.

**Then.** LongMemEval (ICLR 2025) with its official protocol ported line for line, every setting fixed
before running, on LongMemEval_M (419 questions, ~2,400 past messages each):

| round | finding |
|---|---|
| tokenisation | Lowercasing, stemming and stopwords alone lift BM25 by +0.20 recall@5 over the paper's BM25. |
| dense baseline | BM25 beats Contriever by +0.10 recall@5 at 1/65 of the latency. |
| paraphrase gap | LLM query expansion closes most of it; learned sparse retrieval helps a little; Laya preference tags hurt. |
| embeddings, hybrids, RM3 | Qwen3-Embedding alone is BM25-level; hybrids add little over expansion; RM3 hurts. |
| re-rankers | A 22M cross-encoder (MiniLM) over the top 20 is the largest single gain: top-1 0.52 → 0.65. |
| routing | Re-rankers hurt preference questions; Laya flags those questions (83% caught) and skipping the re-ranker for them helps every metric. |

**Decided.** BM25 over the user's words → MiniLM re-rank of the top 20; preference questions (flagged
by Laya) go through LLM query expansion. Embeddings add at most +0.03 on top.
([methodology](../benchmarks/longmemeval/README.md), [tables](../benchmarks/longmemeval/RESULTS.md))

## 6. Memory in a live call

**Tried.** A LongMemEval_M history loaded as the user's past (632 sessions, 3,407 exchanges, dates
shifted to end today) and 100 questions asked by voice.

**Measured.** The evidence was in the top 5 for 86% of questions and first for 79%, at 62 ms p50.

**Found and fixed.** Facts that changed over time (now given to the LLM oldest first, with "the latest
statement wins"); replies made from memory being indexed as if the user had said them (now tagged and
excluded); unrelated hits blended into answers (hits the re-ranker scores below 0 are dropped).

## 7. Who decides when to search memory?

With an LLM that calls tools, the LLM decides. Speech-to-speech models rarely called `recall`, so
we tried deciding outside the model. 220 test turns: 120 that need memory, 50 actions, 50 chat.

| who decides | memory turns searched | other turns searched |
|---|---|---|
| Laya yes/no question | 1–13% | 0% |
| Laya 3-way choice (past / action / chat) | 71% | 14% |
| re-ranker relevance only | 47% | 23% |
| Laya OR (first person AND relevant match) | 84% | 21% |

**Used for a while.** The last rule, plus regexes for "you told me" and named people. In live calls it
broke in ways each new rule patched and the next call broke again: "Rachel is a chef" (made up; the
search found nothing and the model filled the gap); "Business Administration" given correctly, then
denied a turn later when a follow-up searched only the follow-up.

**Then: let the model decide** (Qwen3-Omni, the omni agent's model, with the agent's real tools):

| prompt | memory turns → recall | actions → tool | other turns → recall |
|---|---|---|---|
| instructions only | 19–31% | 0–4% | 0–1% |
| + one worked recall example | 82–84% | 80–86% | 1–4% |
| + a second example (a preference) | **89%** | 82% | 7% |

Told in prose to call recall, the model doesn't; shown one example, it does. That also explains why
live "remind me at 6" made no reminder.

**Decided.** The omni agent decides itself, with two worked examples seeded into its conversation. No
per-turn Laya routing, no regexes. ([results](../benchmarks/router/RESULTS.md))

## 8. Speech-to-speech and the omni hybrid

**NVIDIA NemotronLabs VoiceChat** (full duplex: listens and speaks every 80 ms). Built an MLX sidecar
that injects a tool result in one batched pass (~0.1 s instead of one 80 ms step per token), and
memory forced in from outside (section 7). A model step takes 77–86 ms on smoke runs, at the edge of
real time. A full comparison is still to run.

**Qwen3-Omni hybrid** (built in a parallel session): Qwen3-Omni hears the user's audio and answers in
text; Kokoro speaks it. Reported working on its spoken test (4/5, ~1.3 s voice to voice). Problems
found in live calls:

| symptom | cause | fix |
|---|---|---|
| a reply took 41 s, then 3.6 s | Gemma (18 GB) still loaded next to Qwen3-Omni (~24 GB): 33 GB of swap | starting an agent frees the models it doesn't use |
| replies queued behind each other | cancelled requests kept generating on the single GPU thread | stop generating when the client disconnects; 300-token cap |
| uncached retries after tool calls | prefix cache broke on audio mid-prompt | only the last message is heard as audio |
| greeting continued the seeded examples | the model took the examples as real history | a fixed greeting |

## 9. Making recall fast

**Prefetch.** Recall starts on the user's words as soon as speech-to-text finalises them, during the
~0.7 s end-of-turn wait. It removes the search from the wait when the model then calls `recall`.

**Shorter re-ranker inputs.** 256 tokens instead of 512: identical accuracy (top-1 0.652, recall_any@5
0.862), slightly faster. Re-ranking 10 instead of 20 is faster but loses accuracy, so 20 stays.

**Found.** In a live call, recall took 1.7 s on the wait anyway: Laya's preference check inside recall
waited 2.5 s for the GPU behind Qwen3-Omni (BM25 30 ms, MiniLM 27 ms).

## 10. Can the model write the search?

**Question.** Can the model that decides to search also write the search, and does that replace
MiniLM? LongMemEval_M, model calls cached:

| method | top-1 | recall_any@5 | extra model call |
|---|---|---|---|
| BM25 → MiniLM (live) | 0.652 | 0.862 | no |
| model keywords, no MiniLM | 0.561 | 0.847 | yes |
| model's 1–3 tool-call queries, merged by rank, no MiniLM | 0.432 | 0.797 | no |
| **model's tool-call queries as expansion → MiniLM** | **0.649** | **0.874** | **no** |
| separate keyword call → MiniLM (best) | 0.649 | 0.888 | yes |

**Learned.** Model-written terms don't replace the re-ranker. Merging separate short queries by rank
dilutes the ranking; appending them to the question helps. The queries the model writes in its recall
call get most of the gain of a separate expansion call for free.

**Topic filters** (the model picks a topic; every stored exchange has one from Laya): a filter loses a
quarter of the answers (0.86 → 0.61), a boost doesn't help (0.85). Rejected.

---

## Where it stands

- **Retrieval.** BM25 + a 22M re-ranker is within a few points of the best embedding stack, at a
  fraction of the latency and infrastructure. The assumption held.
- **Laya.** Not a ranker, not a gate, and as a per-turn router beaten by the model itself once the
  model is shown an example. It still pays off for fixed-label work: flagging preference questions
  inside recall, labelling exchanges, auditing actions.
- **Proposed next, not yet applied.**
  - `recall` takes the model's 1–3 queries as expansion terms (benchmark above).
  - Drop Laya's preference check inside recall: 2.5 s under GPU contention, against preference
    questions scoring 0.47 instead of up to 0.70.
  - Drop the prefetch, which can't use the model's queries.
  - Estimated memory turn: ~1.3 s from end of turn to speech, of which the search is 60–90 ms.
