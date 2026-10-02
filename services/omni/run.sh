#!/usr/bin/env bash
# Qwen3-Omni (MLX): OpenAI-compatible chat that hears the user's turn, on OMNI_BASE_URL.
cd "$(dirname "$0")" && . ../lib.sh
ensure_venv
OMNI_PORT="$(port_of "$OMNI_BASE_URL")" exec "$PWD/.venv/bin/python" "$PWD/server.py"
