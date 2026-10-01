#!/usr/bin/env bash
# One-time ONNX export of OpenSearch's doc-only neural sparse encoder (benchmark only).
set -euo pipefail
cd "$(dirname "$0")"
OUT=../../data/models/sparse-doc-v3-distill
if [ -f "$OUT/onnx/model.onnx" ]; then echo "already exported: $OUT"; exit 0; fi
# reuse the export toolchain from services/contriever
PY=../contriever/.venv/bin/python
[ -x "$PY" ] || { /opt/homebrew/bin/python3.11 -m venv ../contriever/.venv; ../contriever/.venv/bin/pip install -q torch "transformers<5" onnx; }
"$PY" export.py "$OUT"
