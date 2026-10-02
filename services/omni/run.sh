#!/usr/bin/env bash
cd "$(dirname "$0")" && . ../lib.sh
ensure_venv
OMNI_PORT="$(port_of "$OMNI_BASE_URL")" exec "$PWD/.venv/bin/python" "$PWD/server.py"
