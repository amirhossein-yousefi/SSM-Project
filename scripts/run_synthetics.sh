#!/usr/bin/env bash
# Run the synthetic mechanistic-task sweep across all archs and tasks. Each (arch, task)
# invocation runs an internal cell sweep and writes idempotent JSONL records, so a
# disconnected run resumes by just re-launching (completed cells are skipped).
set -euo pipefail
cd "$(dirname "$0")/.."

ARCHS=${ARCHS:-"transformer mamba jamba"}
TASKS=${TASKS:-"mqar induction selective_copy"}
STEPS=${STEPS:-3000}
OUT=${OUT:-"results/summaries/synthetic.jsonl"}

for task in $TASKS; do
  for arch in $ARCHS; do
    echo "=== SYNTHETIC $task / $arch ==="
    python -m ssm_bench.eval.synthetic_train \
      --arch "$arch" --task "$task" --steps "$STEPS" --out "$OUT"
  done
done
echo "Synthetic sweep complete -> $OUT"
