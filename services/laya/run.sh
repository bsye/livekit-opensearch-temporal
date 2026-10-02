#!/usr/bin/env bash
cd "$(dirname "$0")" && . ../lib.sh
ensure_venv
exec "$PWD/.venv/bin/python" "$PWD/server.py"
