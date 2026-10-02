# Design notes

The code carries no comments; this is where the reasons live. Each note is something that looks
wrong or arbitrary in the code and isn't. Measurements behind them are in
[experiments.md](experiments.md).

## Agents (`apps/agent`)

**Turn-taking constants (cascade, omni).** VAD reports the end of speech after 300 ms of silence;
the audio turn detector (v1-mini) then decides from intonation whether the user is done, waiting
300–2500 ms. Silence alone split sentences at natural pauses or, tuned longer, slowed every reply.
v1-mini is pinned because dev mode defaults to v1, which only runs on LiveKit Cloud. Barge-in needs
600 ms and two words so "okay" or echo doesn't interrupt.

**Preemptive generation.** The cascade's LLM starts on the final transcript, but TTS waits for the
confirmed turn; otherwise discarded drafts get spoken and the agent seems to repeat itself. Omni has
it off entirely: its audio is attached when the turn completes, which would discard every draft.

**Voice-to-voice latency** adds the VAD's 300 ms back, since VAD reports the end of speech that late.

**Duplicate metrics.** The TTS StreamAdapter re-emits the wrapped TTS's metrics, so each request
arrives twice; they're deduplicated by request id.

**Gemma control tokens** (`cascade/llm-filter.ts`). After a tool call gemma-4 sometimes opens a
thought channel in plain content (`<|channel>thought … <channel|>`), which TTS would read out.

**Omni's audio marker.** agents-js's OpenAI format has no audio input, so the user's message carries
an `<audio:id>` marker and the sidecar swaps in the uploaded recording. Parakeet still transcribes
every turn: turn detection, history, Temporal and memory work on text. The recording starts 500 ms
before VAD's speech-start event, which arrives a little after speech actually starts.

**Omni's seeded examples and fixed greeting.** Told in prose to call `recall`, the model doesn't;
two worked examples placed before the conversation make it do so for 89% of memory questions. A
generated greeting continued from those examples ("I recommended The Rest Is History for you"), so
the greeting is fixed text.

**Omni's query expansion** uses the omni model itself, so no second LLM is loaded next to a 24 GB model.

**Speech-to-speech (s2s).** VoiceChat answers "where did Rachel move?" from its own knowledge
instead of calling recall, so memory is decided outside the model: Laya's route plus the recall
score, forced in as if the model had made the call. Its audio is held while that decision is made,
and dropped if memory gets forced in. The router waits 700 ms of quiet because transcript words
arrive more than 300 ms apart mid-sentence. Three tool calls per turn at most: the model can retry a
failing call forever. Its system message is the one NVIDIA trained tool calling with; a shorter one
of ours made it never use its function channel. Only the date changes in that prompt, because the
sidecar prefills each distinct prompt once (~1 min).

**Approval.** A tool that needs approval pauses inside its own `execute()` and runs a confirmation
task, so the LLM can't skip the question or claim success early. The question is spoken verbatim
(no LLM call, so its wording can't drift), and the task sees the conversation so far so it
understands "no, tell him Thursday".

**Audit.** Laya scores committed actions afterwards and flags low scores for review. It never blocks:
zero-shot it approved "50 euros" → `amount=500` with full confidence.

**Prefetch.** Recall starts on the user's words while the turn detector is still deciding (~0.7 s).
If the model then asks for a time range, or the early search found nothing (a follow-up it
rephrased), the tool searches again with the model's question.

**Tool timing** wraps each tool's `execute`, so an approval's wait for the user is included.

## Temporal (`packages/temporal`)

**Signal names.** The Temporal UI labels signals, children and activities only by name or summary,
so every one gets a readable label and workflows accept any signal name through a default handler.
A turn signal is labelled briefly because its child row follows with the details.

**`noteStep` does nothing.** It exists so each step of a turn (memory, a tool call, an approval) is
its own labelled row on the turn's timeline; the work happened in the agent.

**Workflow ids.** A room's id is its sid: names are reused, sids are unique per session. Webhooks and
the agent race to start it, so both use signal-with-start, and a closed session can't be reopened.
`room_finished` is followed by a one-minute grace period for late webhooks; 24 hours is the safety
net if it never arrives.

**Track type.** Protobuf JSON omits default enum values, and AUDIO is the default track type.

**Translator.** It answers 2xx only once Temporal has accepted the signal, so LiveKit retries
otherwise. LiveKit sends a participant's kind (AGENT, SIP…) on some events only, so it's remembered.

## Memory (`packages/memory`)

**Recall pipeline.** BM25 over the user's words, top 20 re-ranked by MiniLM, chosen on LongMemEval.
Re-rankers trained on web search hurt "what would I like?" questions, so Laya routes those to LLM
query expansion instead.

**Dropping hits the re-ranker scores below 0.** MiniLM logits below 0 mean "not relevant"; passing
them on let the LLM blend unrelated messages into answers.

**256 tokens** per passage: the same LongMemEval accuracy as 512, cheaper on long messages.

**"Nothing found" says *earlier* conversations.** The current conversation is in the model's history;
a bare "nothing found" made it deny what the user had said a minute before.

**`fromMemory`.** A reply produced from memory isn't new evidence and must never be recalled later as
if the user had said it. For the same reason the speech-to-speech summary keeps statements only: a
question asked in an earlier call came back as the answer to the same question.

**Document ids** derive from the room and the time of the reply, so indexing is idempotent.

## Sidecars (`services`)

Each is model-only Python (the models only run under MLX); everything around them is TypeScript.

**One GPU thread.** MLX isn't thread-safe, so every model call runs on one thread, one request at a
time. Laya, VoiceChat and Qwen3-Omni share the Mac's GPU, so a Laya call can wait seconds behind a
Qwen3-Omni generation.

**Omni: cancelled requests stop generating.** The agent cancels requests (an interrupted reply, a
discarded draft); without stopping, the abandoned reply holds the GPU thread and the next turn waits.
Replies are capped at 300 tokens for the same reason.

**Omni: only the last message is heard as audio.** Audio in the middle of a prompt breaks mlx-vlm's
prefix cache. After a tool call, the turn goes back to its transcript: the model already heard it.
If the prefix cache still fails, the request is retried uncached.

**Omni: text only.** Qwen3-Omni's own speech isn't used: mlx-vlm writes the whole reply before the
voice starts (4–5 s), while text streams from ~130 ms.

**Omni API.** `POST /v1/audio/turns?rate=N` takes raw mono PCM16 and returns an id;
`POST /v1/chat/completions` is OpenAI-compatible, and a user message containing `<audio:id>` is
heard as that recording. `OMNI_TRANSCRIPT_WITH_AUDIO` also gives the model the transcript.

**VoiceChat: tool results in one pass.** The model was trained to read tool results on its function
channel, one token per 80 ms frame, with its speech silent. Since every input token is known in
advance, they go through the language model in one batched pass (~0.1 s for 40 tokens instead of
40 steps). Each distinct system prompt is prefilled once and new sessions start from a copy.

**Laya.** `/v1/systemone` has TypeSafe Jev's request shape; `/v1/scan` scores one yes/no question
over many texts in GPU batches (~0.5 ms per text): 64 texts or 16k tokens per batch, with MLX's
buffer cache bounded because scans see many shapes. Common batch shapes are compiled at startup.
