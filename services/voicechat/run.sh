#!/usr/bin/env bash
# NVIDIA VoiceChat (speech-to-speech) on the Apple GPU (MLX), websocket on VOICECHAT_URL.
cd "$(dirname "$0")" && . ../lib.sh
ensure_venv
VOICECHAT_PORT="$(port_of "$VOICECHAT_URL")" exec "$PWD/.venv/bin/python" "$PWD/server.py"
