#!/usr/bin/env bash
cd "$(dirname "$0")" && . ../lib.sh
case "${1:-}" in
  contriever) script=contriever.py; out=contriever ;;
  sparse) script=sparse_encoder.py; out=sparse-doc-v3-distill ;;
  *) echo "usage: $0 contriever|sparse" >&2; exit 1 ;;
esac
OUT="$ROOT/data/models/$out"
if [ -f "$OUT/onnx/model.onnx" ]; then echo "already exported: $OUT"; exit 0; fi
ensure_venv
.venv/bin/python "$script" "$OUT"
