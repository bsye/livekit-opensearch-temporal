#!/usr/bin/env bash
# Native speech server on the Apple GPU (MLX): OpenAI-compatible STT + TTS.
# Runs outside Docker because containers on macOS can't use Metal.
set -euo pipefail
cd "$(dirname "$0")"
set -a; . ../../.env; set +a
port="${SPEECH_BASE_URL##*:}"; port="${port%%/*}"

if [ ! -x .venv/bin/mlx_audio.server ]; then
  /opt/homebrew/bin/python3.11 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi

.venv/bin/mlx_audio.server --host 127.0.0.1 --port "$port" &
server=$!
trap 'kill $server 2>/dev/null' EXIT INT TERM
until curl -sf "$SPEECH_BASE_URL/models" >/dev/null; do sleep 0.5; done

# Models load on first use; warm both up now so the first user turn isn't slow
warmup=$(mktemp -t mlx-warmup).wav
curl -sf -o "$warmup" "$SPEECH_BASE_URL/audio/speech" -H 'content-type: application/json' \
  -d "{\"model\":\"$TTS_MODEL\",\"voice\":\"$TTS_VOICE\",\"input\":\"Warming up.\",\"response_format\":\"wav\"}"
curl -sf -o /dev/null "$SPEECH_BASE_URL/audio/transcriptions" -F "file=@$warmup" -F "model=$STT_MODEL"
rm -f "$warmup"
echo "mlx-audio ready on $SPEECH_BASE_URL (STT $STT_MODEL, TTS $TTS_MODEL)"
wait $server
