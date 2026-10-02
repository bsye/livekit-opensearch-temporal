#!/usr/bin/env bash
# Laya decision model on the Apple GPU (MLX).
cd "$(dirname "$0")" && . ../lib.sh
ensure_venv
exec "$PWD/.venv/bin/python" "$PWD/server.py"
