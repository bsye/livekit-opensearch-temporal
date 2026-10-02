"""Qwen3-Omni-30B-A3B (MLX) as an OpenAI-compatible chat server that can hear the user's turn.

Model-only sidecar for apps/agent/src/omni: the agent (TypeScript) runs LiveKit turn-taking, Parakeet
transcripts for history, Kokoro for speech, tools, Laya and memory. Qwen3-Omni's own speech output
is not used: mlx-vlm writes the whole reply before its voice starts (first audio after 4-5 s), while
text streams from the first token after ~130 ms.

  POST /v1/audio/turns?rate=N    raw mono PCM16 of one user turn (any rate, resampled to 16 kHz) → {"id": "..."}
  POST /v1/chat/completions      OpenAI chat format (stream, tools). A user message containing
                                 <audio:ID> is heard as that audio instead of read: the latest one
                                 only; older turns keep their transcript (markers stripped).

  .venv/bin/python server.py   (see run.sh)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from mlx_vlm import apc, load, stream_generate

MODEL = os.environ.get("OMNI_MODEL", "mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit")
PORT = int(os.environ.get("OMNI_PORT", "8300"))
MAX_TURNS = 32  # recent turn audio kept for lookup
# Also give the model the turn's transcript after its audio (benchmarks/router "Letting the model
# decide": audio + transcript was the measured setting). A request's "transcript_with_audio" overrides.
TRANSCRIPT_WITH_AUDIO = os.environ.get("OMNI_TRANSCRIPT_WITH_AUDIO", "0").lower() in ("1", "true", "yes")
AUDIO_MARKER = re.compile(r"<audio:([0-9a-f]+)>")
MEMORY = re.compile(r"<memory>(.*?)</memory>", re.S)  # recall results the agent attached to a turn
TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)

model, processor = load(MODEL)
if hasattr(model, "disable_talker"):
    model.disable_talker()  # text only; Kokoro speaks
# Automatic prefix caching: the system prompt, tool definitions and earlier turns are processed once
# and reused by the next request; only the new turn (its audio) is computed. Without it every reply
# re-read the whole conversation first (time to first token 0.5-3 s instead of ~0.15 s).
apc_manager = apc.from_env(model_namespace="qwen3-omni", overrides={"enabled": True, "disk_enabled": False})
gpu = ThreadPoolExecutor(max_workers=1)  # MLX isn't thread-safe: one model call at a time
turns: OrderedDict[str, np.ndarray] = OrderedDict()
app = FastAPI()


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "local"}]}


def to_16k(pcm: np.ndarray, rate: int) -> np.ndarray:
    if rate == 16000:
        return pcm
    if rate % 16000 == 0:  # 48 kHz from WebRTC: average each group (a crude low-pass) and decimate
        k = rate // 16000
        return pcm[: len(pcm) // k * k].reshape(-1, k).mean(axis=1)
    return np.interp(np.arange(0, len(pcm), rate / 16000), np.arange(len(pcm)), pcm).astype(np.float32)


@app.post("/v1/audio/turns")
async def add_turn(request: Request):
    pcm = np.frombuffer(await request.body(), dtype="<i2").astype(np.float32) / 32768.0
    pcm = to_16k(pcm, int(request.query_params.get("rate", "16000")))
    turn_id = uuid.uuid4().hex[:12]
    turns[turn_id] = pcm
    while len(turns) > MAX_TURNS:
        turns.popitem(last=False)
    return {"id": turn_id, "seconds": round(pcm.shape[0] / 16000, 2)}


def text_of(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if part.get("type") == "text")


def to_conversation(messages: list[dict], transcript_with_audio: bool = False) -> tuple[list[dict], list[np.ndarray]]:
    """OpenAI messages → Qwen3-Omni conversation; the latest audio-marked user turn becomes audio.

    With transcript_with_audio, that turn also keeps its transcript as text after the audio (opt-in
    per request while it's measured: on audio alone the model often says it will act without calling
    the tool)."""
    # Only a turn the model hasn't answered yet is heard. Once a tool call/result follows it (the
    # follow-up request), it goes back to its transcript: the model already heard it and decided, and
    # audio in the middle of a prompt breaks mlx-vlm's prefix cache ("[broadcast_shapes] Shapes
    # (75776) and (0)"), which then costs an uncached retry on every answer after a tool call.
    heard = None
    last = messages[-1] if messages else None
    if last and last["role"] == "user":
        m = AUDIO_MARKER.search(text_of(last.get("content")))
        if m and m.group(1) in turns:
            heard = (len(messages) - 1, m.group(1))
    conversation, audios = [], []
    for i, msg in enumerate(messages):
        role = msg["role"]
        raw = AUDIO_MARKER.sub("", text_of(msg.get("content")))
        memory = MEMORY.findall(raw) if i == len(messages) - 1 or (heard and i == heard[0]) else []
        text = MEMORY.sub("", raw).strip()  # older turns: transcript only
        if heard and i == heard[0]:
            parts = [{"type": "audio", "audio": heard[1]}]
            if transcript_with_audio and text:
                parts.append({"type": "text", "text": f"Transcript (may contain errors): {text}"})
            parts += [{"type": "text", "text": f"<memory>{m}</memory>"} for m in memory]
            conversation.append({"role": "user", "content": parts})
            audios.append(turns[heard[1]])
        elif memory:
            conversation.append({"role": role, "content": text + "".join(f" <memory>{m}</memory>" for m in memory)})
        elif role == "assistant" and msg.get("tool_calls"):
            conversation.append({"role": "assistant", "content": text, "tool_calls": msg["tool_calls"]})
        elif role == "tool":
            conversation.append({"role": "tool", "content": text, "tool_call_id": msg.get("tool_call_id")})
        else:
            conversation.append({"role": "developer" if role == "developer" else role, "content": text})
    for c in conversation:
        if c["role"] == "developer":
            c["role"] = "system"
    return conversation, audios


def model_inputs(conversation: list[dict], audios: list[np.ndarray], tools):
    prompt = processor.apply_chat_template(conversation, tools=tools or None, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[prompt], audio=audios or None, padding=True)
    kw = {k: v for k, v in inputs.items() if isinstance(v, mx.array)}
    # as mlx_vlm's prepare_omni_inputs: the audio mask is per sample, the encoder counts mel frames
    if "feature_attention_mask" in kw and "audio_feature_lengths" not in kw:
        mask = kw["feature_attention_mask"]
        lengths = mask.sum(axis=1)
        mel_frames = kw["input_features"].shape[-1]
        if mask.shape[-1] > mel_frames:
            lengths = lengths // (mask.shape[-1] // mel_frames)
        kw["audio_feature_lengths"] = lengths.astype(mx.int32)
    input_ids = kw.pop("input_ids")
    kw.pop("attention_mask", None)
    return prompt, input_ids, kw


DEFAULT_MAX_TOKENS = 300  # a spoken reply; a runaway generation would hold the only GPU thread


class Cancelled(Exception):
    pass


def generate(body: dict, emit, cancel: threading.Event | None = None) -> dict:
    """Runs on the GPU thread; emit(text_delta) per token; stops when `cancel` is set. Returns usage."""
    if cancel is not None and cancel.is_set():
        raise Cancelled()  # cancelled while queued
    conversation, audios = to_conversation(body["messages"], bool(body.get("transcript_with_audio", TRANSCRIPT_WITH_AUDIO)))
    if os.environ.get("OMNI_DEBUG"):
        for c in conversation:
            content = c["content"] if isinstance(c["content"], str) else " ".join(
                p.get("text") or f"[{p['type']}]" for p in c["content"])
            print(f"  {c['role']:>9}: {content[:110]!r}" + (" +tool_calls" if c.get("tool_calls") else ""), flush=True)
    prompt, input_ids, kw = model_inputs(conversation, audios, body.get("tools"))
    max_tokens = int(body.get("max_completion_tokens") or body.get("max_tokens") or DEFAULT_MAX_TOKENS)
    temperature = float(body.get("temperature") if body.get("temperature") is not None else 0.0)
    completion = 0
    raw: list[str] = []
    started = time.perf_counter()
    first = None
    def run(cache):
        nonlocal first, completion
        for r in stream_generate(model, processor, prompt, input_ids=input_ids, max_tokens=max_tokens,
                                 temperature=temperature, apc_manager=cache, **kw):
            first = first or time.perf_counter() - started
            completion += 1
            if cancel is not None and cancel.is_set():
                print(f"cancelled by the client after {completion} tokens", flush=True)
                break
            if r.text:
                raw.append(r.text)
                emit(r.text)

    try:
        run(apc_manager)
    except ValueError as exc:
        # mlx-vlm's prefix cache can reuse an earlier turn's audio positions for a new turn
        # ("[broadcast_shapes] Shapes (1,649,2048) and (1,458,2048)"): redo this request uncached
        if completion:
            raise
        print(f"prefix cache failed ({exc}); retrying without it", flush=True)
        run(None)
    if os.environ.get("OMNI_DEBUG"):
        print("  raw:", repr("".join(raw)[:300]), flush=True)
    print(f"{int(input_ids.shape[-1])} prompt tokens{' + audio' if audios else ''}: first token "
          f"{(first or 0) * 1000:.0f} ms, {completion} tokens in {time.perf_counter() - started:.2f}s", flush=True)
    return {"prompt_tokens": int(input_ids.shape[-1]), "completion_tokens": completion,
            "total_tokens": int(input_ids.shape[-1]) + completion, "heard_audio": bool(audios)}


def tool_calls_in(text: str) -> list[dict]:
    calls = []
    for i, raw in enumerate(TOOL_CALL.findall(text)):
        try:
            c = json.loads(raw)
        except json.JSONDecodeError:
            continue
        args = c.get("arguments", {})
        calls.append({"index": i, "id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                      "function": {"name": c.get("name", ""),
                                   "arguments": args if isinstance(args, str) else json.dumps(args)}})
    return calls


class SpeechFilter:
    """Streams the reply's spoken text; from a <tool_call> on, nothing (the calls go out parsed)."""

    def __init__(self):
        self.text = ""  # everything generated
        self.sent = 0
        self.in_call = False

    def feed(self, delta: str) -> str:
        self.text += delta
        if self.in_call:
            return ""
        pending = self.text[self.sent:]
        start = pending.find("<tool_call>")
        if start >= 0:
            self.in_call = True
            self.sent = len(self.text)
            return pending[:start]
        lt = pending.rfind("<")  # hold back a possible partial "<tool_call>"
        out = pending[:lt] if lt >= 0 and "<tool_call>".startswith(pending[lt:]) else pending
        self.sent += len(out)
        return out

    def rest(self) -> str:
        return "" if self.in_call else self.text[self.sent:]


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    loop = asyncio.get_running_loop()
    rid = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())

    def chunk(delta: dict, finish=None, usage=None) -> str:
        payload = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": MODEL,
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        if usage is not None:
            payload["usage"] = usage
        return f"data: {json.dumps(payload)}\n\n"

    if not body.get("stream"):
        parts: list[str] = []
        usage = await loop.run_in_executor(gpu, generate, body, parts.append)
        text = "".join(parts)
        calls = tool_calls_in(text)
        message = {"role": "assistant", "content": TOOL_CALL.sub("", text).strip() or None}
        if calls:
            message["tool_calls"] = [{k: v for k, v in c.items() if k != "index"} for c in calls]
        return JSONResponse({"id": rid, "object": "chat.completion", "created": created, "model": MODEL,
                             "choices": [{"index": 0, "message": message,
                                          "finish_reason": "tool_calls" if calls else "stop"}],
                             "usage": usage})

    queue: asyncio.Queue = asyncio.Queue()
    done = object()
    # the agent cancels requests (an interrupted reply, a discarded draft): stop generating then,
    # or the abandoned reply keeps the only GPU thread busy and the next turn waits behind it
    cancel = threading.Event()

    def work():
        try:
            usage = generate(body, lambda d: loop.call_soon_threadsafe(queue.put_nowait, d), cancel)
            loop.call_soon_threadsafe(queue.put_nowait, ("usage", usage))
        except Cancelled:
            pass
        except Exception as exc:  # surfaced to the client as an error chunk
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))
        loop.call_soon_threadsafe(queue.put_nowait, done)

    gpu.submit(work)

    async def events():
        try:
            async for event in stream():
                yield event
        finally:  # client gone (or done): stop the generation
            cancel.set()

    async def stream():
        speech = SpeechFilter()
        usage = None
        yield chunk({"role": "assistant", "content": ""})
        while (item := await queue.get()) is not done:
            if isinstance(item, tuple):
                if item[0] == "error":
                    yield f"data: {json.dumps({'error': {'message': item[1]}})}\n\n"
                    return
                usage = item[1]
                continue
            out = speech.feed(item)
            if out:
                yield chunk({"content": out})
        tail = speech.rest().replace("<|im_end|>", "")
        if tail.strip():
            yield chunk({"content": tail})
        calls = tool_calls_in(speech.text)
        if calls:
            yield chunk({"tool_calls": calls})
        yield chunk({}, finish="tool_calls" if calls else "stop")
        if (body.get("stream_options") or {}).get("include_usage") and usage:
            yield f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'created': created, 'model': MODEL, 'choices': [], 'usage': {k: v for k, v in usage.items() if k != 'heard_audio'}})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
