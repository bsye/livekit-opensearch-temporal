#!/usr/bin/env bash
# NVIDIA NemotronLabs VoiceChat (speech-to-speech) on the Apple GPU (MLX), websocket on VOICECHAT_URL.
# Python because the model only runs under MLX; everything that talks to it is TypeScript.
set -euo pipefail
cd "$(dirname "$0")"
set -a; . ../../.env; set +a
url="${VOICECHAT_URL:-ws://localhost:8200}"; port="${url##*:}"; port="${port%%/*}"

if [ ! -x .venv/bin/python ]; then
  /opt/homebrew/bin/python3.11 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi
VOICECHAT_PORT="$port" exec "$PWD/.venv/bin/python" "$PWD/server.py"
