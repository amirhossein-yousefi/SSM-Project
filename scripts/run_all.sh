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
# Jamba's Mamba layers run on the memory-heavy torch path -> smaller micro-batch, more
# accumulation. micro*accum stays 512 for every arm, so the effective batch is identical.
JAMBA_MICRO_BATCH=${JAMBA_MICRO_BATCH:-8}
JAMBA_GRAD_ACCUM=${JAMBA_GRAD_ACCUM:-64}
STAGE_DIR=${STAGE_DIR:-}        # e.g. /content on Colab for fast atomic staging
EXTRA=${EXTRA:-}                # any extra flags, e.g. "--lr 3e-4"

# reduce allocator fragmentation (helps the large torch-path Mamba activations)
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

stage_flag=""
[ -n "$STAGE_DIR" ] && stage_flag="--stage_dir $STAGE_DIR"

for arch in $ARCHS; do
  for seed in $SEEDS; do
    run="$RUNS_DIR/${arch}_seed${seed}"
    if [ -f "$run/DONE" ]; then
      echo "SKIP $arch/seed$seed (DONE)"
      continue
    fi
    arch_flags=""
    if [ "$arch" = "jamba" ]; then
      mb="$JAMBA_MICRO_BATCH"; ga="$JAMBA_GRAD_ACCUM"
      # fast Mamba kernel with autocast OFF (fp32) -> avoids the kernel/bf16 dtype bug
      # and the slow torch path. Drop --force_kernels to fall back to the (slow) torch path.
      arch_flags="--force_kernels --no_autocast"
    else
      mb="$MICRO_BATCH"; ga="$GRAD_ACCUM"
    fi
    echo "=== TRAIN $arch seed=$seed (micro_batch=$mb grad_accum=$ga) -> $run ==="
    python scripts/train.py \
      --arch "$arch" --seed "$seed" \
      --data_dir "$DATA_DIR" --output_dir "$run" \
      --total_steps "$TOTAL_STEPS" \
      --micro_batch_size "$mb" --grad_accum "$ga" \
      $arch_flags $stage_flag $EXTRA
  done
done
echo "All runs complete."
