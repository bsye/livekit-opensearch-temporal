# Local voice agent with long-term memory

A voice assistant that runs entirely on a laptop: self-hosted LiveKit for the call, local models on
the Apple GPU for speech and language, Temporal as the durable record of every session, and a
conversation memory the agent can search while it talks.

The project started from one assumption and was built step by step to test it. Every step below
comes with the benchmark that decided it.

## The assumption

> **A search engine over the text a conversation already produces (exact, timestamped exchanges,
> searched with BM25) retrieves past conversations about as well as embeddings do. No vector
> database is needed. A small, fast decision model (Laya, ~5 ms) still has a place next to it: it
> classifies and routes, while ranking stays with the search engine.**

Where it ended up, on LongMemEval_M (419 questions, ~2,400 past messages each):

| retrieval | recall@5 | top-1 | query time | needs |
|---|---|---|---|---|
| Contriever (the paper's dense baseline) | 0.456 | 0.372 | ~3 s | embeddings |
| Qwen3-Embedding-0.6B | 0.547 | 0.516 | 36 ms | embeddings |
| BM25 | 0.556 | 0.516 | ~30 ms | OpenSearch |
| **BM25 → MiniLM re-rank of top 20 (shipped)** | **0.652** | **0.652** | **~70 ms** | OpenSearch + a 22M model |
| Best measured: BM25 + LLM expansion + embeddings → 568M re-ranker, Laya-routed | 0.728 | 0.644 | ~2.3 s + LLM | everything |

Properly tokenised BM25 beats the paper's dense baseline. Embeddings add at most +0.03 on top of the
best embedding-free stack, and a small cross-encoder is worth more than any embedding model.

## How it was tested

### 1. A local agent to test with

LiveKit's self-hosted production setup runs on macOS through OrbStack, which gives Docker the
Linux-style host networking LiveKit needs. The agent is a LiveKit Agents worker (TypeScript) with
every model local:

```
Silero VAD → Parakeet STT → Gemma 4 26B-A4B (LM Studio) → Kokoro TTS
             └──────────── mlx-audio on the GPU ────────────┘
```

Docker on macOS has no Metal access, so the models run natively and everything else in containers.
Measured on an M5 Max:

| stage | model | latency |
|---|---|---|
| STT | parakeet-tdt-0.6b-v2 (whisper-large-v3-turbo: ~720 ms) | 60–110 ms |
| LLM first token | gemma-4-26b-a4b, `reasoning_effort: none` | 530–620 ms |
| TTS first audio | Kokoro-82M-8bit (bf16: ~390 ms) | ~200 ms |
| voice to voice | user stops talking → agent starts | 2–2.7 s |

Turn-taking needed the most tuning. Silence thresholds alone either split sentences at natural
pauses or slowed every reply. The fix was LiveKit's local audio turn detector (v1-mini), which judges
from intonation whether a pause ends the turn, plus barge-in that needs 600 ms and two words, so
"okay" or echo doesn't interrupt.

### 2. Temporal as the record of every session

The memory experiments needed a trustworthy record of what was said, by whom and when, and of
every action the agent took. Temporal provides that:

- **Webhooks are unreliable input.** LiveKit retries them, delivers duplicates and sends some after
  `room_finished`. A small translator verifies each webhook and signal-with-starts one workflow per
  room session (id = room sid). Whoever arrives first, webhook or agent, creates it; duplicates are
  dropped by event id; a grace period catches late events; a closed session can't be reopened.
- **One timeline per room.** Each participant gets a child workflow, so the Temporal UI shows a lane
  per actor. Signals are named after who did what (`👤 dalbi · track_published (AUDIO)`, `🤖 agent: "…"`,
  `⏱ voice-to-voice 1500ms`), so the history reads as the conversation itself. Live state is a query away.
- **Side effects are durable.** A reminder is a workflow with a durable timer, so it survives
  restarts and cancelling it is the undo. Indexing each exchange into memory is a retried activity
  whose failure never fails the call.

### 3. Laya as an action gate: rejected

The first job given to Laya was blocking wrong tool calls. On 24 labelled request/action pairs
([results](benchmarks/action-gate/RESULTS.md)) the best setting scores 21/24 at 3.9 ms, but it
approved "transfer 50 euros" → `amount=500` at P=1.00. It catches unrelated and negated actions,
not wrong numbers, and it is confident when it's wrong.

So consequential tools ask the user instead. The tool pauses inside its own `execute()` and runs a
confirmation task, so the LLM can't skip it or claim success early. Reversible tools (reminders) run
immediately and have an undo. Laya audits every action afterwards, off the critical path; its flags
land in Temporal for review and, over time, become labelled data for fine-tuning.

### 4. Memory without embeddings: the LongMemEval benchmark

Every exchange is stored in OpenSearch with exact times and Laya-assigned categories. The first
retriever was a parallel Laya scan of the whole history (~0.5 ms per message on the GPU). To know
whether that, or anything else, works, retrieval was measured on **LongMemEval** (Wu et al., ICLR
2025), with its official protocol ported line for line and every setting fixed before running
([methodology, retrievers and glossary](benchmarks/longmemeval/README.md),
[raw tables](benchmarks/longmemeval/RESULTS.md)).

| round | finding |
|---|---|
| BM25 vs the paper's BM25 | Tokenisation alone (lowercasing, stemming, stopwords) is worth +0.20 recall@5 on identical data, more than the gap the paper reports between dense retrievers and its BM25. |
| Laya as a ranker | Hurts: BM25 top 50 re-ordered by Laya drops recall@5 from 0.556 to 0.320. Laya classifies a text well and judges relevance between two texts poorly. |
| Contriever (dense baseline) | BM25 beats it by +0.10 recall@5 at ~65× lower latency. The harness reproduces the paper's Contriever-vs-BM25 gap (+0.100 vs +0.117). Contriever wins only on preference questions (paraphrase). |
| Closing the paraphrase gap without embeddings | LLM query expansion: 0.556 → 0.621, preference questions 0.33 → 0.63. Learned sparse: +0.017. Laya preference tags: hurt. |
| Modern embedders, hybrids, RM3, re-rankers | Qwen3-Embedding alone ≈ BM25. RRF hybrids add little over query expansion. RM3 hurts. **Cross-encoder re-ranking is the biggest single gain** (+0.07–0.10); the 22M MiniLM gets ~90% of the 568M bge's gain at 1/15 of the time. |
| Laya as a router | Re-rankers trained on web search hurt "what would I enjoy?" questions. Laya flags those questions (83% caught, 0–5% false alarms), and skipping the re-ranker for them lifts every metric. |

### 5. Into the agent

Recall uses the benchmarked pipeline:

```
question ─▶ Laya: asking for the user's preferences? (~5 ms)
   no  ─▶ BM25 over the user's words, time-range filter ─▶ top 20 ─▶ MiniLM re-rank ─▶ top 5     ~60 ms
   yes ─▶ LLM query expansion ─▶ BM25 (question + expansion) ─▶ top 5                          ~450 ms
```

To try it at scale by voice, a LongMemEval_M history is loaded as the user's past (632 sessions,
3,407 exchanges, dates shifted to end today) with 100 questions across all types. Live, the evidence
is in the top 5 for 86% of questions and ranked first for 79%, at 62 ms p50.

Voice testing on that library found three problems, all fixed:

- **Facts that changed.** Results go to the LLM oldest first, with the rule that the latest
  statement wins.
- **Memory feeding on itself.** Replies produced from memory are tagged through the transcript
  signal and never recalled later as if the user had said them.
- **Near misses blended into answers.** Hits the re-ranker scores below 0 are dropped, and "I don't
  remember" is preferred over a confident mix of unrelated messages.

### 6. When to search: Laya's real use case

With a cascade, the LLM decides when to call `recall`. A speech-to-speech model rarely does, so
something has to decide per turn ([results](benchmarks/router/RESULTS.md), 220 turns: 120 memory,
50 action, 50 chat):

| decision | memory turns caught | other turns searched | cost |
|---|---|---|---|
| Laya yes/no question | 1–13% | 0% | ~5 ms |
| Laya 3-way choice (past / action / chat) | 71% | 14% | ~5 ms |
| re-ranker relevance only | 47% | 23% | ~45 ms |
| **Laya OR (first person AND relevant match)** | **84%** | **21%** | ~70 ms, in parallel |

### Verdict

The assumption holds for retrieval: a search engine over well-structured text, plus a small
re-ranker, is within a few points of the best embedding stack at a fraction of the latency and
infrastructure. Laya's place is real but narrower than expected. It's poor at ranking and too
confident to block actions alone, but fast and accurate at fixed-label classification: routing
questions, routing turns, labelling exchanges, auditing actions.

## Architecture

```
 browser (Meet) ──wss──▶ Caddy ──▶ LiveKit ──webhooks──▶ translator ──signals──▶ Temporal
                                     ▲                                             │
                                     │ audio                     room / participant workflows,
                                     ▼                           reminders, email, memory indexing
                                   agent ──signals (transcript, metrics, actions)──┘
                          ┌──────────┼───────────────┬──────────────┐
                    mlx-audio     LM Studio         Laya       OpenSearch (memory)
                   (STT + TTS)     (LLM)      (decision model)    + MiniLM re-ranker
```

```
apps/
  agent/            LiveKit agent: cascade/ (STT → LLM → TTS), s2s/ (speech-to-speech), tools/
  translator/       LiveKit webhooks → Temporal
  worker/           Temporal worker; backfill-memory script
packages/
  temporal/         workflows, activities, client, signal labels
  memory/           OpenSearch store, recall pipeline, re-ranker, query expansion
  laya/             client for the Laya sidecar (ask, scan, route)
  voicechat/        client for the VoiceChat sidecar
  config/           env access and data paths
services/           Python model sidecars on MLX (laya, mlx-audio, voicechat, omni); one-off ONNX exports
benchmarks/         longmemeval, router, action-gate, voicechat
infra/              Caddy, Redis and Temporal config mounted into the containers
scripts/            setup, start, stop
```

All application code is TypeScript. Python appears only where a model runs under MLX.

## Run it

Requires an Apple Silicon Mac with [Homebrew](https://brew.sh); 64 GB of memory is comfortable.

```sh
npm run setup     # tools, Node/Python dependencies, models, .env, containers, local CA trust
npm start         # pick an agent, then a meeting opens in the browser and the agent joins it
npm stop          # native services (npm stop -- --all also stops containers and unloads the LLM)
```

`setup` is idempotent. It installs OrbStack, LM Studio, Node, Python 3.11 and the LiveKit CLI if
missing, creates `.env` from `.env.example` with a generated LiveKit secret, builds each sidecar's
venv, downloads the models and asks for your password once, to trust Caddy's local CA. Firefox also
needs `security.enterprise_roots.enabled=true`. Use headphones: on laptop speakers the agent hears
itself.

Setup asks which agents to fetch models for (multi-select; the cascade is always installed), and
`npm start` asks which agent to run:

| agent | pipeline |
|---|---|
| `cascade` (default) | Parakeet STT → Gemma 4 26B-A4B (LM Studio) → Kokoro TTS |
| `s2s` | NVIDIA NemotronLabs VoiceChat 11B, full-duplex speech-to-speech |
| `omni` | Qwen3-Omni 30B-A3B hears the audio and answers in text, Kokoro speaks (~24 GB) |

Skip the menus with `AGENT=omni npm start` and `npm run setup -- --s2s --omni`; outside a terminal
the cascade is used. Only one agent runs at a time: starting one stops the others. `npm start`
launches whatever isn't running, waits until each piece is ready and prints the room's Temporal
link. Logs are in `.run/`.

| service | address |
|---|---|
| LiveKit (browsers / backends) | `wss://livekit.localhost` / `ws://localhost:7880` |
| Temporal UI | http://localhost:8233 (filter: `RoomName="meet-…"`) |
| OpenSearch Dashboards | http://localhost:5602 (index `conversation-memory`) |
| LM Studio · mlx-audio · Laya | `:1234` · `:8000` · `:8100` |

Everything is configured in `.env`, the only file with secrets. Docker Compose renders the LiveKit,
egress and ingress configs from it.

## Benchmarks

| benchmark | question | command | results |
|---|---|---|---|
| LongMemEval | which retriever finds the right past message? | `npm run bench -w @bench/longmemeval` | [README](benchmarks/longmemeval/README.md) |
| action gate | can Laya block wrong tool calls? | `services/laya/.venv/bin/python services/laya/eval_gate.py` | [RESULTS](benchmarks/action-gate/RESULTS.md) |
| router | can Laya tell when to search memory? | `npm run bench\|gate\|combined -w @bench/router` | [RESULTS](benchmarks/router/RESULTS.md) |
| voicechat | speech-to-speech vs cascade on tools and memory | `npm run bench -w @bench/voicechat` | in progress |

The LongMemEval datasets go in `data/benchmarks/longmemeval/` (see the
[LongMemEval repo](https://github.com/xiaowu0162/LongMemEval)). `npm run load-library -w @bench/longmemeval`
loads the voice-test library and writes the question sheet the router and voicechat benchmarks use.

## What's next

- **Speech-to-speech agent** (`apps/agent/src/s2s`, `AGENT=s2s`). NVIDIA's NemotronLabs VoiceChat is
  full duplex: it listens and speaks every 80 ms and handles turn-taking and barge-in itself. Built
  so far: an MLX sidecar that injects a tool result in one batched pass (~0.1 s instead of one 80 ms
  step per token), the same tools and Temporal reporting as the cascade, and memory routing from
  step 6, which forces recall in and drops anything the model had started saying. On smoke runs a
  model step takes 77–86 ms p50, at the edge of real time. Next is the full comparison against the
  cascade (`benchmarks/voicechat`) on action, chat and memory turns.
- **Hybrid agent** (`AGENT=omni`, in progress). Qwen3-Omni hears the user's audio directly and
  streams text from ~130 ms; Kokoro speaks it. The goal is speech understanding without the
  transcription step, keeping the cascade's tools.
- **Fine-tune Laya on this domain.** The router and audit misses are narrow, fixed-label tasks,
  the kind Laya's authors report fine-tuning lifts from ~0.36 to ~0.77+. The audit flags already
  collected in Temporal are the start of that dataset.
- **Deliver reminders** (push or a call back into the room) and send real email behind the
  existing approval step.
