#!/usr/bin/env bash
cd "$(dirname "$0")" && . ../lib.sh
ensure_venv
VOICECHAT_PORT="$(port_of "$VOICECHAT_URL")" exec "$PWD/.venv/bin/python" "$PWD/server.py"
