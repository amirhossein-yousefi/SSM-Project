#!/usr/bin/env bash
# Train all architectures (x seeds), resumable. Skips any run with a DONE marker and
# auto-resumes the rest from their last checkpoint. Re-running after a Colab disconnect
# simply continues. Override any variable from the environment.
set -euo pipefail
cd "$(dirname "$0")/.."

ARCHS=${ARCHS:-"transformer mamba jamba"}
SEEDS=${SEEDS:-"1337"}
DATA_DIR=${DATA_DIR:-"data_cache/fineweb_edu_gpt2"}
RUNS_DIR=${RUNS_DIR:-"runs"}
TOTAL_STEPS=${TOTAL_STEPS:-6000}
MICRO_BATCH=${MICRO_BATCH:-16}
GRAD_ACCUM=${GRAD_ACCUM:-32}
STAGE_DIR=${STAGE_DIR:-}        # e.g. /content on Colab for fast atomic staging
EXTRA=${EXTRA:-}                # any extra flags, e.g. "--lr 3e-4"

stage_flag=""
[ -n "$STAGE_DIR" ] && stage_flag="--stage_dir $STAGE_DIR"

for arch in $ARCHS; do
  for seed in $SEEDS; do
    run="$RUNS_DIR/${arch}_seed${seed}"
    if [ -f "$run/DONE" ]; then
      echo "SKIP $arch/seed$seed (DONE)"
      continue
    fi
    echo "=== TRAIN $arch seed=$seed -> $run ==="
    python scripts/train.py \
      --arch "$arch" --seed "$seed" \
      --data_dir "$DATA_DIR" --output_dir "$run" \
      --total_steps "$TOTAL_STEPS" \
      --micro_batch_size "$MICRO_BATCH" --grad_accum "$GRAD_ACCUM" \
      $stage_flag $EXTRA
  done
done
echo "All runs complete."
