#!/usr/bin/env bash
# Run the efficiency benchmark sweep (throughput / memory / prefill / decode) across all
# archs, sequence lengths, and modes. Each cell runs in an isolated subprocess; results are
# appended idempotently so the sweep is resumable.
set -euo pipefail
cd "$(dirname "$0")/.."

ARCHS=${ARCHS:-"transformer,mamba,jamba"}
LENS=${LENS:-"512,1024,2048,4096,8192,16384,32768"}
MODES=${MODES:-"train,prefill,decode"}
BATCH=${BATCH:-1}
OUT=${OUT:-"results/summaries/efficiency.jsonl"}

python -m ssm_bench.eval.efficiency \
  --archs "$ARCHS" --lens "$LENS" --modes "$MODES" --batch "$BATCH" --out "$OUT"
echo "Efficiency sweep complete -> $OUT"
