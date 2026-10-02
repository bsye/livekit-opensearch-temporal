#!/usr/bin/env bash
cd "$(dirname "$0")" && . ../lib.sh
ensure_venv

.venv/bin/mlx_audio.server --host 127.0.0.1 --port "$(port_of "$SPEECH_BASE_URL")" &
server=$!
trap 'kill $server 2>/dev/null' EXIT INT TERM
until curl -sf "$SPEECH_BASE_URL/models" >/dev/null; do sleep 0.5; done

warmup=$(mktemp -t mlx-warmup).wav
curl -sf -o "$warmup" "$SPEECH_BASE_URL/audio/speech" -H 'content-type: application/json' \
  -d "{\"model\":\"$TTS_MODEL\",\"voice\":\"$TTS_VOICE\",\"input\":\"Warming up.\",\"response_format\":\"wav\"}"
curl -sf -o /dev/null "$SPEECH_BASE_URL/audio/transcriptions" -F "file=@$warmup" -F "model=$STT_MODEL"
rm -f "$warmup"
echo "mlx-audio ready on $SPEECH_BASE_URL (STT $STT_MODEL, TTS $TTS_MODEL)"
wait $server
