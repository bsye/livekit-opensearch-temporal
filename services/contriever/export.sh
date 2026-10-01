#!/usr/bin/env bash
# One-time conversion of facebook/contriever (PyTorch weights only on the Hub) to ONNX, laid out
# for @huggingface/transformers (onnx/model.onnx + tokenizer). Used only as a benchmark baseline
# (benchmarks/longmemeval, the dense retriever from the LongMemEval paper); never in the product.
set -euo pipefail
cd "$(dirname "$0")"
OUT=../../data/models/contriever
if [ -f "$OUT/onnx/model.onnx" ]; then echo "already exported: $OUT"; exit 0; fi
[ -x .venv/bin/python ] || /opt/homebrew/bin/python3.11 -m venv .venv
.venv/bin/pip install -q torch "transformers<5" onnx
.venv/bin/python export.py "$OUT"
