"""Latency smoke test for Qwen3-Omni (thinker + talker) on MLX: spoken question in, text + speech out.

  .venv/bin/python omni_smoke.py <question.wav> [--tools]
"""

import json
import sys
import time

import mlx.core as mx
import numpy as np
from mlx_vlm import load
from mlx_vlm.utils import load_audio

MODEL = "mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit"
TOOLS = [{
    "type": "function",
    "function": {
        "name": "recall",
        "description": "Search past conversations with the user. Returns what the user said before, with dates.",
        "parameters": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]},
    },
}]

t = time.perf_counter()
model, processor = load(MODEL)
print(f"load {time.perf_counter() - t:.1f}s")
wav = load_audio(sys.argv[1], sr=processor.feature_extractor.sampling_rate)
conversation = [
    {"role": "system", "content": [{"type": "text", "text": "You are a helpful voice assistant. Keep replies to one or two short sentences."}]},
    {"role": "user", "content": [{"type": "audio", "audio": "x"}]},
]
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False,
                                     tools=TOOLS if "--tools" in sys.argv else None)
for run in range(2):  # the first run includes compilation
    t0 = time.perf_counter()
    inputs = processor(text=[text], audio=[wav], return_tensors="np", padding=True)
    kw = {k: mx.array(v) for k, v in inputs.items() if isinstance(v, np.ndarray)}
    if "feature_attention_mask" in kw:
        kw["audio_feature_lengths"] = kw["feature_attention_mask"].sum(axis=1).astype(mx.int32)
    input_ids = kw.pop("input_ids")
    kw.pop("attention_mask", None)
    first_audio = None
    samples = 0
    for kind, value in model.generate_stream(input_ids, speaker="Ethan", chunk_size=12, left_context_size=25,
                                             thinker_max_new_tokens=200, **kw):
        if kind == "text":
            t_text = time.perf_counter() - t0
            reply = processor.tokenizer.decode(value, skip_special_tokens=True)
            print(f"[run {run}] text after {t_text * 1000:.0f} ms ({len(value)} tokens): {reply!r}")
            if "<tool_call>" in processor.tokenizer.decode(value):
                print("  tool call:", processor.tokenizer.decode(value))
        else:
            if first_audio is None:
                first_audio = time.perf_counter() - t0
            samples += value.shape[-1]
    total = time.perf_counter() - t0
    if first_audio:
        print(f"[run {run}] first audio {first_audio * 1000:.0f} ms, {samples / 24000:.1f}s of speech in {total:.1f}s")
    else:
        print(f"[run {run}] no audio, total {total:.1f}s")
print(json.dumps({"peak_memory_gb": round(mx.get_peak_memory() / 1e9, 1)}))
