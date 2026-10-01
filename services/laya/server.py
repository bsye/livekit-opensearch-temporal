"""Local Laya decision server (MLX, Apple GPU) with TypeSafe Jev's request/response shape.

    POST /v1/systemone  {"state": "...", "questions": {"name": {"type": "noul"|"choice"|"score", ...}}}
                     -> {"answers": {...}, "model": "...", "latency_ms": 4.1}
    GET  /health

Single-threaded on purpose: one model instance, requests take ~5ms, and MLX streams
aren't meant to be shared across threads.
"""
import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

import laya_mlx as laya

MODEL = os.environ.get("LAYA_MODEL", "aac6fef/laya-multilingual-mlx")
PORT = int(urlparse(os.environ.get("LAYA_BASE_URL", "http://localhost:8100")).port or 8100)

agent = laya.load(MODEL)
agent.predict("warm up", {"q": {"type": "noul", "instructions": "Is this a test?"}})


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            return self.reply(200, {"status": "ok", "model": MODEL})
        self.reply(404, {"error": "not found"})

    def do_POST(self):
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
        except Exception as err:  # bad question schema etc.
            return self.reply(422, {"error": str(err)})
        latency = round((time.perf_counter() - start) * 1000, 1)
        self.reply(200, {"answers": result["answers"], "model": MODEL, "latency_ms": latency})

    def reply(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):  # one compact line per request
        print(f"{self.command} {self.path} {args[1] if len(args) > 1 else ''}", flush=True)


if __name__ == "__main__":
    print(f"laya ready on :{PORT} ({MODEL})", flush=True)
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
