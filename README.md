# LiveKit · OpenSearch · Temporal: a local voice agent that remembers

## Goal

- **Build and test a reliable local voice agent that can retrieve anything from the user's past
  conversations, without the usual RAG stack**: no embeddings, no vector database. Memory is a
  search engine over what was actually said.
- **Experiment with open models beyond the transcribe → process → speak pipeline**: speech-to-speech
  (NVIDIA VoiceChat) and a hybrid where the model hears the user's audio directly (Qwen3-Omni).

Everything runs on one Apple Silicon laptop: self-hosted LiveKit for the call, open models on the GPU,
OpenSearch for memory, Temporal as the record of every session.

## Assumptions

1. **BM25 has been the standard for relevance search for decades.** Applied to a search engine over
   timestamped conversation text, it should return relevant results, and fast, without embeddings.
2. **Query expansion closes the gap where RAG is traditionally strong.** Embeddings win on paraphrase
   ("what would I enjoy?" when the user once said "I love history podcasts"). Having a language model
   write the words the user most likely used should recover that.

## The result

The best setup is **Qwen3-Omni with our recall tooling** (`AGENT=omni`):

```
 user's voice ─▶ Parakeet transcript ─┐
             └─▶ Qwen3-Omni hears the audio + transcript
                   │ decides itself: answer, call an action tool, or call recall
                   ▼
   recall ─▶ BM25 in OpenSearch over the user's words ─▶ top 20 ─▶ MiniLM re-rank ─▶ top 5
             (started early, while the turn detector is still deciding)
                   │
                   ▼
             Qwen3-Omni answers ─▶ Kokoro speaks ─▶ every step recorded in Temporal
```

| what | measured |
|---|---|
| retrieval on LongMemEval_M (419 questions, ~2,400 past messages each) | evidence first for 65%, in the top 5 for 86%, ~70 ms |
| the same on a voice library loaded as the user's past (100 questions) | evidence first for 79%, in the top 5 for 86%, 62 ms |
| the model deciding to search memory (220 spoken turns) | 89% of memory questions, 7% of other turns |
| the model calling action tools (reminders, email) | 82% of action requests |
| search on the critical path in a live call | 0 ms (prefetched); voice to voice 3.2 s on a memory question |

**Assumption 1 held.** Properly tokenised BM25 beats the LongMemEval paper's dense retriever, and
with a 22M-parameter re-ranker on top it beats a modern embedding model by a wide margin. The best
stack we measured, with embeddings, adds only a few points on top, at 30× the latency.

**Assumption 2 held, with a correction.** Query expansion does close the paraphrase gap: preference
questions go from 33% to 63–70%, level with the dense baseline. But it doesn't replace the
re-ranker; the two together are best. The model can write the expansion terms itself, inside its
recall tool call, at no extra cost.

## Methodology and benchmarks

All numbers come from one M5 Max (64 GB). Every setting was fixed before a benchmark ran, never tuned
on its test questions. The full log of what was tried, including what didn't work, is in
[docs/experiments.md](docs/experiments.md); the reasons behind non-obvious code are in
[docs/design-notes.md](docs/design-notes.md).

### 1. The recall tool, on LongMemEval

**LongMemEval** (Wu et al., ICLR 2025) is the standard long-term chat memory benchmark: 500
questions, each with its own timestamped history and labels marking which messages hold the answer.
We ported its official retrieval evaluation line for line, so numbers are comparable with the paper,
and ran on the larger LongMemEval_M (~2,400 past messages per question). Our harness reproduces the
paper's Contriever-vs-BM25 gap (+0.100 vs +0.117).
[Methodology, retrievers and glossary](benchmarks/longmemeval/README.md) ·
[all tables](benchmarks/longmemeval/RESULTS.md)

| retriever | first | in top 5 | NDCG@10 | query time |
|---|---|---|---|---|
| the paper's BM25 (space-split, no stemming) | 0.360 | 0.573 | 0.435 | 14 ms |
| Contriever (the paper's dense baseline) | 0.372 | 0.740 | 0.541 | ~3 s |
| Qwen3-Embedding-0.6B | 0.516 | 0.833 | 0.628 | 36 ms |
| BM25, English analyzer | 0.516 | 0.800 | 0.630 | ~30 ms |
| BM25 + query expansion | 0.556 | 0.859 | 0.684 | ~30 ms + LLM |
| **BM25 → MiniLM re-rank (live recall)** | **0.652** | 0.862 | 0.716 | **~70 ms** |
| **BM25 + the model's own tool-call queries → MiniLM** | 0.649 | 0.874 | 0.732 | ~60 ms |
| BM25 + query expansion → MiniLM | 0.649 | 0.888 | 0.741 | ~70 ms + LLM |
| everything: expansion + embeddings + 568M re-ranker | 0.644 | 0.895 | 0.758 | ~2.3 s + LLM |

What decided the design:

- **Tokenisation matters more than the retriever family.** Stemming and stopwords alone lift BM25 by
  +0.20 over the paper's version, more than the gap the paper reports for dense retrievers.
- **A small re-ranker is the best use of a model.** MiniLM (22M) gets ~90% of the 568M bge's gain at
  1/15 of the time. Re-ranking 256 tokens per message instead of 512 costs nothing.
- **Query expansion fixes paraphrase; it doesn't replace re-ranking.** Model-written terms without
  MiniLM rank the right message first only 43–56% of the time.
- **Rejected:** embeddings (≤ +0.03), RM3, learned sparse retrieval, re-ranking with a small decision
  model (Laya: 0.32 vs BM25's 0.56 for all evidence in the top 5), topic filters (lost a quarter of the answers).

**In a live call.** A LongMemEval_M history was loaded as the user's past (632 sessions, 3,407
exchanges) and queried by voice: the evidence came first for 79% of questions and in the top 5 for
86%, at 62 ms. Voice testing also found three failure modes, all fixed: facts that changed over time
(the latest statement now wins), the agent recalling its own earlier answers as if the user had said
them, and unrelated hits blended into answers (hits the re-ranker scores below 0 are dropped).

**Latency.** Recall starts on the user's words while the turn detector is still deciding whether
they're done (~0.7 s), so in a live call the search adds nothing to the wait.

### 2. Tool calling: actions and when to search

**Can a small model block wrong actions?** Laya, a ~4 ms decision model, checked 24 labelled
request/action pairs and got 21 right, but approved "transfer 50 euros" → `amount=500` with full
confidence. So consequential tools ask the user first (the tool pauses inside its own execution, so
the model can't skip the question), reversible ones run and can be undone, and Laya only audits
afterwards. [Results](benchmarks/action-gate/RESULTS.md)

**Who decides when to search memory?** 220 spoken turns: 120 that need memory, 50 actions, 50 chat.
[Results](benchmarks/router/RESULTS.md)

| who decides | memory turns searched | other turns searched | action requests → tool |
|---|---|---|---|
| Laya, yes/no question | 1–13% | 0% | – |
| Laya, 3-way choice (past / action / chat) | 71% | 14% | – |
| Laya + relevance + hand-written rules | 84% | 21% | – |
| Qwen3-Omni, told in its instructions | 19–31% | 0–1% | 0–4% |
| Qwen3-Omni, one worked example of a recall call | 82–84% | 1–4% | 80–86% |
| **Qwen3-Omni, two worked examples (a fact, a preference)** | **89%** | **7%** | **82%** |

Told in prose to call its tools, the model doesn't; shown one example, it does. The model with two
examples beats the external router, needs no rules, and its remaining "false alarms" are mostly
recommendations ("recommend a podcast"), where checking the user's taste is arguably right.

**Speech-to-speech.** NVIDIA's VoiceChat listens and speaks every 80 ms but rarely calls tools on its
own, so it still needs the external router; its full comparison is in progress (see What's next).

## How it's built

```
 browser (Meet) ──wss──▶ Caddy ──▶ LiveKit ──webhooks──▶ translator ──signals──▶ Temporal
                                     ▲                                             │
                                     │ audio                  roomSession ─▶ conversationTurn per exchange
                                     ▼                        reminders, email, memory indexing
                                   agent ──signals (turns)───────────────────────────┘
                          ┌──────────┼───────────────┬──────────────┐
                    mlx-audio   Qwen3-Omni / LLM      Laya       OpenSearch (memory)
                   (STT + TTS)                  (decision model)    + MiniLM re-ranker
```

**Three agents, one set of tools and memory:**

| agent | pipeline | voice to voice |
|---|---|---|
| `cascade` | Parakeet STT → Gemma 4 26B-A4B (LM Studio) → Kokoro TTS | 2–2.7 s |
| `omni` (best) | Qwen3-Omni 30B-A3B hears the audio and answers in text → Kokoro | 3.2 s with a memory lookup |
| `s2s` | NVIDIA NemotronLabs VoiceChat 11B, full-duplex speech-to-speech | in progress |

Turn-taking uses LiveKit's local audio turn detector, which judges from intonation whether a pause
ends the turn; silence thresholds alone either split sentences or slowed every reply.

**Temporal as the record.** LiveKit's webhooks arrive late, twice, or out of order, so a small
translator turns them into one Temporal workflow per room session. Each exchange becomes its own row
on the room's timeline (`💬 3 · 👤 "Remind me…" → 🤖 "Done…" · 🛠 set_reminder · ⏱ 1.3s`), and inside
it every step: how long the user spoke and how long recognition took, the memory search, each tool
call with its duration, and the agent's first token, first audio and voice-to-voice time. Reminders
are durable timers (cancelling is the undo); indexing an exchange into memory is a retried activity.

```
apps/
  agent/            LiveKit agents (cascade/, omni/, s2s/), the shared pipeline, tools, Temporal reporting
  translator/       LiveKit webhooks → Temporal
  worker/           Temporal worker; backfill-memory script
packages/
  memory/           OpenSearch store, recall, re-ranker, query expansion
  temporal/         workflows (room, turn, reminder, email), activities, client
  laya/ voicechat/  clients for the model sidecars
  config/ http/ text/  env, HTTP and string helpers
services/           Python model sidecars on MLX (laya, mlx-audio, omni, voicechat); one-off ONNX exports
benchmarks/         longmemeval, router, action-gate, voicechat, shared helpers
infra/ scripts/     container config; setup, start, stop
```

All application code is TypeScript; Python appears only where a model runs under MLX.

## Run it

Requires an Apple Silicon Mac with [Homebrew](https://brew.sh); 64 GB of memory is comfortable.

```sh
npm run setup     # tools, Node/Python dependencies, models, .env, containers, local CA trust
npm start         # pick an agent; a meeting opens in the browser and the agent joins it
npm stop          # native services (npm stop -- --all also stops containers and unloads the LLM)
```

`setup` is idempotent: it installs OrbStack, LM Studio, Node, Python 3.11 and the LiveKit CLI if
missing, creates `.env` with a generated LiveKit secret, builds each model sidecar's environment,
downloads the models you pick and asks for your password once, to trust Caddy's local CA. Firefox also
needs `security.enterprise_roots.enabled=true`. Use headphones: on laptop speakers the agent hears
itself.

Skip the menus with `AGENT=omni npm start` and `npm run setup -- --s2s --omni`.
`npm start -- <room> <identity>` picks the room and your name, and `NO_OPEN=1` prints the meeting
link instead of opening it. Only one agent runs at a time, and starting one frees the models the
others use. Logs are in `.run/`.

| service | address |
|---|---|
| LiveKit (browsers / backends) | `wss://livekit.localhost` / `ws://localhost:7880` |
| Temporal UI | http://localhost:8233 (`npm start` prints the room's link) |
| OpenSearch Dashboards | http://localhost:5602 (index `conversation-memory`) |

Everything is configured in `.env`, the only file with secrets.

**Reproduce the benchmarks** (LongMemEval data goes in `data/benchmarks/longmemeval/`, see the
[LongMemEval repo](https://github.com/xiaowu0162/LongMemEval)):

| benchmark | command |
|---|---|
| LongMemEval retrieval | `npm run bench -w @bench/longmemeval` |
| voice library for live testing | `npm run load-library -w @bench/longmemeval` |
| topic filters | `npm run topics -w @bench/longmemeval` |
| who decides when to search | `npm run bench\|gate\|combined\|omni -w @bench/router` |
| Laya as an action gate | `services/laya/.venv/bin/python services/laya/eval_gate.py` |
| speech-to-speech vs cascade | `npm run bench -w @bench/voicechat` |

## What's next

- **Recall with the model's own queries.** Use the 1–3 queries Qwen3-Omni writes in its recall call
  as expansion terms (benchmarked above), and drop the preference check that waits for the shared GPU.
- **Speech-to-speech.** VoiceChat's tool results already go in through one batched pass (~0.1 s instead
  of one 80 ms step per token); next is its full comparison with the cascade and omni agents.
- **Fine-tune Laya** on the routing and audit cases this project has collected in Temporal.
- **Deliver reminders** (a push, or a call back into the room) and send real email behind the
  existing approval step.
