import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

import laya_mlx as laya
import mlx.core as mx
import numpy as np
from laya_mlx.agent import collate_items, temp_bucket

MODEL = os.environ.get("LAYA_MODEL", "aac6fef/laya-multilingual-mlx")
PORT = int(urlparse(os.environ.get("LAYA_BASE_URL", "http://localhost:8100")).port or 8100)

SCAN_BATCH = 64
SCAN_BATCH_TOKENS = 16384
mx.set_cache_limit(int(os.environ.get("LAYA_CACHE_LIMIT_GB", "2")) << 30)

agent = laya.load(MODEL)
agent.predict("warm up", {"q": {"type": "noul", "instructions": "Is this a test?"}})


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            return self.reply(200, {"status": "ok", "model": MODEL})
        self.reply(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/v1/scan":
            return self.scan()
        if self.path != "/v1/systemone":
            return self.reply(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
            state, questions = body["state"], body["questions"]
            if not isinstance(state, str):
                state = json.dumps(state)
        except (ValueError, KeyError, TypeError) as err:
            return self.reply(400, {"error": f"expected {{state, questions}}: {err}"})
        start = time.perf_counter()
        try:
            result = agent.predict(state, questions)
        except Exception as err:
            return self.reply(422, {"error": str(err)})
        latency = round((time.perf_counter() - start) * 1000, 1)
        self.reply(200, {"answers": result["answers"], "model": MODEL, "latency_ms": latency})

    def scan(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
            states, question = body["states"], body["question"]
            if not isinstance(states, list) or not isinstance(question, str):
                raise TypeError("states must be a list and question a string")
        except (ValueError, KeyError, TypeError) as err:
            return self.reply(400, {"error": f"expected {{states, question}}: {err}"})
        start = time.perf_counter()
        scores = scan(states, question)
        latency = round((time.perf_counter() - start) * 1000, 1)
        self.reply(200, {"scores": scores, "model": MODEL, "latency_ms": latency,
                         "active_mb": round(mx.get_active_memory() / 1e6), "peak_mb": round(mx.get_peak_memory() / 1e6)})

    def reply(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        print(f"{self.command} {self.path} {args[1] if len(args) > 1 else ''}", flush=True)


def scan(states, question):
    q = {"q": {"type": "noul", "instructions": question}}
    items = [agent.prepare(state if isinstance(state, str) else json.dumps(state), q)[0][0] for state in states]
    scores = []
    for chunk in batches(items):
        batch = collate_items(
            chunk, agent.tok.pad_token_id, pad_to_multiple=agent.pad_to_multiple, max_length=agent.cfg.get("max_len", 512)
        )
        logits, _ = agent.forward(batch)
        logits = np.asarray(logits)
        for row, item in enumerate(chunk):
            k, qt = len(item["markers"]), item["qtype"]
            scale = agent.temperature_by_options.get(temp_bucket(qt, k), agent.temperature[qt])
            z = logits[row, :k] / scale
            p = np.exp(z - z.max())
            p /= p.sum()
            scores.append(round(float(p[1]), 4))
    mx.clear_cache()
    return scores


def batches(items):
    batch, longest = [], 0
    for item in items:
        n = len(item["ids"])
        if batch and (len(batch) == SCAN_BATCH or max(longest, n) * (len(batch) + 1) > SCAN_BATCH_TOKENS):
            yield batch
            batch, longest = [], 0
        batch.append(item)
        longest = max(longest, n)
    if batch:
        yield batch


if __name__ == "__main__":
    for n in (1, SCAN_BATCH):
        scan(["User: warm up. Assistant: ok."] * n, "Is this a test?")
    print(f"laya ready on :{PORT} ({MODEL})", flush=True)
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
