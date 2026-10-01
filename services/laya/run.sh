#!/usr/bin/env bash
# Local Laya decision server on the Apple GPU (MLX); Jev-compatible POST /v1/systemone.
set -euo pipefail
cd "$(dirname "$0")"
set -a; . ../../.env; set +a
if [ ! -x .venv/bin/python ]; then
  /opt/homebrew/bin/python3.11 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi
exec "$PWD/.venv/bin/python" "$PWD/server.py"
