"""NVIDIA NemotronLabs VoiceChat (full-duplex speech-to-speech) on the Apple GPU, as a websocket sidecar.

Model-only: everything around it (tools, Laya, memory, LiveKit, Temporal) is TypeScript.

Protocol: mlx-vlm's /v1/realtime VoiceChat events (session.update with a system_prompt,
input_audio_buffer.append of 16 kHz PCM16, response.text.delta / response.function.delta /
response.audio.delta at 22.05 kHz / conversation.item.input_audio_transcription.delta), plus what
mlx-vlm leaves out: returning a tool result to the model.

  client → {"type": "conversation.item.create", "item": {"type": "function_call_output", "output": "..."}}
           (optional "call": a tool call JSON to force before it, for calls made outside the model)

The model was trained to read tool results on its function channel, one token per 80 ms frame, as
`<TOOL_RESPONSE>[...]</TOOL_RESPONSE>` with its speech channel silent (NVIDIA NeMo,
streaming_s2s_pipeline.py "forced_function_tokens"). Here they go through the
language model in one batched pass (Session.inject), so a 40-token result costs ~0.1 s, not 40 steps.

  .venv/bin/python server.py   (see run.sh)
"""

from __future__ import annotations

import asyncio
import base64
import copy
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from mlx_vlm import load

MODEL = os.environ.get("VOICECHAT_MODEL", "mlx-community/NemotronLabs-VoiceChat-11B-8bit")
PORT = int(os.environ.get("VOICECHAT_PORT", "8200"))

# MLX isn't thread-safe: every model call runs on this one thread
gpu = ThreadPoolExecutor(max_workers=1)
model, processor = load(MODEL)
voicechat = model.create_session(processor)
tokenizer = voicechat.tokenizer
app = FastAPI()
busy = False

# Prefilling the system prompt runs one model step per token (~900 tokens with tools: over a minute),
# so each distinct prompt is prefilled once and new sessions start from a copy of that state. The
# copy shares the model (weights, modules, tokenizer) and duplicates only the session's caches.
prefilled: dict[tuple[str | None, int], object] = {}
shared = {id(m): m for m in model.modules()} | {id(o): o for o in (model, processor, voicechat, tokenizer, voicechat.model.config)}


def new_stream(system_prompt: str | None, seed: int):
    key = (system_prompt, seed)
    if key not in prefilled:
        prefilled[key] = voicechat.create_streaming_session(system_prompt=system_prompt, seed=seed, profile=True)
    return copy.deepcopy(prefilled[key], dict(shared))


class Session:
    """One streaming session with tool-result injection."""

    def __init__(self, system_prompt: str | None, seed: int):
        self.stream = new_stream(system_prompt, seed)
        self.pad = self.stream.config.pad_token_id
        self.frame_ms: list[float] = []

    def push(self, samples: np.ndarray) -> list[dict]:
        t = time.perf_counter()
        before = self.stream.frame_index
        events = self.stream.push_audio(samples, sample_rate=self.stream.input_sample_rate)
        frames = self.stream.frame_index - before
        if frames:
            self.frame_ms.append((time.perf_counter() - t) * 1000 / frames)
        return [serialize(e) for e in events]

    def inject(self, output: str, call: str | None = None) -> list[dict]:
        """Force a tool result (and optionally the call) onto the function channel in one batched pass.

        Frame by frame this costs a full model step per token (~80 ms: audio encoder 17, language model
        22, speech decoder 32, codec 4). But while a result goes in, the model hears silence and is
        silent itself (its speech channel is padded), and every input token is known in advance: so
        the language model takes all of them in one forward pass, like a prompt prefill, and the
        speech decoder and codec, which would only produce silence, are skipped.
        """
        st = self.stream
        text = (f"<TOOLCALL>[{call}]</TOOLCALL>" if call else "") + f"<TOOL_RESPONSE>[{output}]</TOOL_RESPONSE>"
        forced = tokenizer.encode(text, add_special_tokens=False)
        n = len(forced)
        t = time.perf_counter()
        # one silence frame through the audio encoder, its embedding reused for every injected step
        silence, _ = st._perception_step(mx.zeros((st.frame_samples,), dtype=mx.float32))
        # each step's input is the previous step's tokens: text padded, function channel = the forced tokens
        prev_text = [st._text_tokens[-1] if st._text_tokens else self.pad] + [self.pad] * (n - 1)
        prev_function = [st._function_tokens[-1] if st._function_tokens else self.pad] + forced[:-1]
        embed = st.model.stt_model.embed_tokens
        fused = (
            embed(mx.array([prev_text], dtype=mx.int32))
            + silence
            + st.config.function_channel_weight * embed(mx.array([prev_function], dtype=mx.int32))
        )
        out = st._language_step(fused)
        mx.eval(out.function_logits)
        st._text_tokens += [self.pad] * n
        st._function_tokens += forced
        st._timeline_index += n
        # the speech decoder skipped those (silent) steps: restart it from silence
        st._previous_code = mx.broadcast_to(
            st.model.tts_model.codec_silence_tokens[None, None, :], st._previous_code.shape
        )
        events: list[dict] = []
        for token in forced:
            update = st._function.append(token)
            if update is not None:
                events.append({"type": "response.function.delta", "delta": update[0], "text": update[1],
                               "frame_index": st.frame_index})
        events.append({"type": "conversation.item.injected", "tokens": n, "ms": round((time.perf_counter() - t) * 1000)})
        return events


def serialize(event) -> dict:
    common = {"frame_index": event.frame_index}
    if event.kind == "assistant_text_delta":
        return {"type": "response.text.delta", "delta": event.delta, "text": event.text, **common}
    if event.kind == "function_delta":
        return {"type": "response.function.delta", "delta": event.delta, "text": event.text, **common}
    if event.kind == "user_transcript_delta":
        return {"type": "conversation.item.input_audio_transcription.delta", "delta": event.delta, "text": event.text, **common}
    if event.kind == "audio":
        pcm = np.round(np.clip(np.asarray(event.samples, dtype=np.float32), -1, 1) * 32767).astype("<i2")
        return {"type": "response.audio.delta", "audio": base64.b64encode(pcm.tobytes()).decode(), "sample_rate": event.sample_rate, **common}
    return {"type": f"response.{event.kind}", **common}


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL, "busy": busy}


@app.websocket("/v1/realtime")
async def realtime(ws: WebSocket):
    global busy
    await ws.accept()
    loop = asyncio.get_running_loop()
    run = lambda fn, *a: loop.run_in_executor(gpu, fn, *a)  # noqa: E731

    async def send(payload: dict):
        await ws.send_text(json.dumps({"event_id": f"event_{uuid.uuid4().hex[:16]}", **payload}))

    if busy:
        await send({"type": "error", "error": {"code": "server_busy", "message": "one session at a time"}})
        await ws.close(code=1013)
        return
    busy = True
    session: Session | None = None
    try:
        await send({"type": "session.created"})
        while True:
            msg = json.loads(await ws.receive_text())
            kind = msg.get("type")
            if kind == "session.update":
                cfg = msg.get("session") or {}
                session = await run(Session, cfg.get("system_prompt"), int(cfg.get("seed", 0)))
                await send({
                    "type": "session.updated",
                    "session": {"model": MODEL, "input_sample_rate": session.stream.input_sample_rate,
                                "output_sample_rate": session.stream.output_sample_rate},
                })
            elif session is None:
                await send({"type": "error", "error": {"code": "not_configured", "message": "send session.update first"}})
            elif kind == "input_audio_buffer.append":
                pcm = np.frombuffer(base64.b64decode(msg["audio"]), dtype="<i2").astype(np.float32) / 32768.0
                for event in await run(session.push, pcm):
                    await send(event)
            elif kind == "conversation.item.create" and (msg.get("item") or {}).get("type") == "function_call_output":
                item = msg["item"]
                for event in await run(session.inject, str(item.get("output", "")), item.get("call")):
                    await send(event)
            elif kind == "session.stats":
                ms = sorted(session.frame_ms) or [0.0]
                await send({"type": "session.stats", "frames": session.stream.frame_index,
                            "frame_ms_p50": ms[len(ms) // 2], "frame_ms_p95": ms[int(len(ms) * 0.95)],
                            "profile": session.stream.profile.summary(drop_first=5) if session.stream.profile.frames else None})
            elif kind in ("input_audio_buffer.commit", "session.cancel"):
                break
            else:
                await send({"type": "error", "error": {"code": "unsupported", "message": f"unsupported event {kind!r}"}})
    except WebSocketDisconnect:
        pass
    finally:
        busy = False
        await run(mx.clear_cache)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
