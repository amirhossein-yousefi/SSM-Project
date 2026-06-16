# ssm-bench — SSM vs Jamba vs Transformer

A reproducible benchmark that **fairly compares three sequence-model families** at a matched
~125M parameter budget:

| Arm | Architecture | HF model |
|-----|--------------|----------|
| **Transformer** | Llama-style decoder (RoPE + SwiGLU + RMSNorm) | `LlamaForCausalLM` |
| **SSM** | Mamba-2 selective state-space model | `Mamba2ForCausalLM` |
| **Jamba** | Hybrid Mamba + attention + MoE | `JambaForCausalLM` |

All three share one tokenizer, one data stream, one training loop, and one eval harness, so
differences reflect the **architecture**, not the setup. The project is **Colab-first and
resumable**: training survives the 12 h session cap / idle disconnects by checkpointing exact
state to Google Drive and auto-resuming.

> Targets a Google Colab **A100 (40 GB)**. Also runs locally on any CUDA GPU. The Mamba/Jamba
> CUDA kernels are optional — without them, HuggingFace's correct (slower) torch path is used.

## What gets compared (the "Everything" suite)

1. **LM quality** — train each model from scratch on a ~1.5 B-token slice of FineWeb-Edu;
   report validation **perplexity / bits-per-token**.
2. **Synthetic mechanistic tasks** — where the architectures genuinely diverge:
   - **MQAR** (multi-query associative recall): attention/Jamba stay ~100 %, pure Mamba
     degrades once recall load exceeds its fixed state.
   - **Induction / in-context copy**: train short, **extrapolate** long — Mamba generalizes far
     past its training length, attention does not.
   - **Selective copy**: a selectivity sanity task (all solve it).
3. **Efficiency benchmark** (the headline) — throughput, peak memory, prefill latency, and
   **autoregressive decode latency** vs sequence length. Transformer decode latency + KV-cache
   memory grow with length; Mamba stays flat; Jamba sits between.

## Project layout

```
configs/            transformer.yaml  mamba.yaml  jamba.yaml  data.yaml   (param-matched specs)
src/ssm_bench/
  models/           registry.py  param_utils.py  mamba_torch.py
  data/             prepare_fineweb.py  packed_loader.py  synthetics.py
  train/            trainer.py  checkpoint.py  schedule.py
  eval/             lm_quality.py  synthetic_train.py  efficiency.py
  utils/            logging.py  seed.py
scripts/            train.py  run_all.sh  run_synthetics.sh  run_efficiency.sh
                    aggregate.py  plots.py
notebooks/          colab_runner.ipynb        (the resumable Colab driver)
results/            summaries/*.csv   figures/*.png
runs/               <arch>_seed<seed>/{config.json, log.jsonl, checkpoints/, DONE}   (gitignored)
tests/              test_loader.py  test_synthetics.py  test_param_match.py  test_resume.py
```

## Quickstart (Colab — recommended)

Open `notebooks/colab_runner.ipynb` on an A100 runtime and run top-to-bottom. It mounts Drive,
syncs this repo, installs deps (with a kernel-or-fallback probe + wheel cache), pre-tokenizes
the data **once**, then trains all three arms, runs the synthetic + efficiency sweeps, and
renders the figures. **If Colab disconnects, just re-run the notebook** — finished runs are
skipped and in-progress runs resume from their last checkpoint (exact step / LR / data position
/ RNG). Download `results/summaries/*.csv` + `results/figures/*.png` back into this repo to
commit.

## Quickstart (local)

```bash
# 0. install torch (CUDA build for your platform) first, then:
make install                 # core deps + editable install
make install-kernels         # OPTIONAL fast Mamba kernels (may fail -> eager fallback)

# 1. verify the three models are parameter-matched (~125M ± 5%)
make check-params

# 2. one-time: pre-tokenize FineWeb-Edu to uint16 shards
make data DATA_DIR=data_cache/fineweb_edu_gpt2

# 3. train all arms (resumable; skips DONE, auto-resumes the rest)
make train-all

# 4. evaluations
make synthetics
make efficiency

# 5. collate + plot
make aggregate
make plots
```

`make smoke` runs a tiny end-to-end training step (random data, tiny model) to sanity-check the
harness without any downloads or GPU.

## How resume works (the core design)

- **Checkpoints** (`src/ssm_bench/train/checkpoint.py`) save model + optimizer + LR scheduler +
  `global_step` + the **data cursor** + **all RNG states**. Writes are atomic (tmp +
  `os.replace`); on Colab they stage to fast local disk then atomic-rename onto Drive.
- **Data** (`src/ssm_bench/data/packed_loader.py`) is read from flat token shards via an integer
  cursor; resume state is just `{epoch, cursor, seed}`, so the exact next batch is reproduced —
  no re-tokenization, bit-exact, identical token order across all three models.
- A **`DONE`** marker ends a run; orchestration skips runs that have it. Save cadence (~250
  steps) + a SIGTERM flush mean a disconnect loses at most a few minutes.

## Fairness notes

- Same GPT-2 BPE tokenizer (`vocab_size=50304`) and **tied embeddings** across all arms, so the
  comparison rests on the backbone, not the vocab head.
- Models are matched on **total** params; Jamba's MoE means total ≠ active, so both numbers are
  reported (`config.json` / the results table). Set `num_experts: 1` in `configs/jamba.yaml` for
  a clean dense Mamba+attention hybrid.
- The committed configs are **starting points**; `make check-params` finalizes the per-arm layer
  counts to within 5 % on the actual machine (closed-form param math for Mamba2/Jamba is
  unreliable).
- **Jamba precision.** Jamba's Mamba-1-style CUDA kernel mismatches dtypes under bf16 autocast,
  so the orchestration trains Jamba with the **kernel on but autocast off** (`--force_kernels
  --no_autocast`, i.e. fp32) — fast like the Mamba-2 arm and numerically stable, just fp32 vs the
  others' bf16 autocast. fp32 logits use more memory, so Jamba uses `8×64` vs the others' `16×32`
  (still `micro_batch × grad_accum = 512`, so identical effective batch). Drop `--force_kernels`
  to fall back to the (correct but very slow) torch path. Time-based checkpointing
  (`--ckpt_seconds`, default 600) guarantees a resumable checkpoint within ~10 min regardless of
  per-step speed.

## Results

After a run, figures land in `results/figures/`:

| Figure | Shows |
|--------|-------|
| `fig_ppl_curve.png` | val bits/token vs training step, per arch |
| `fig_mqar.png` / `fig_induction.png` / `fig_selcopy.png` | accuracy vs sequence length |
| `fig_latency.png` | decode + prefill latency vs sequence length (log-log) |
| `fig_memory.png` | peak GPU memory vs sequence length (× marks OOM) |
| `fig_throughput.png` | training throughput vs sequence length |

_(Populate the summary tables here from `results/summaries/*.csv` once you've run the sweeps.)_

## Tests

```bash
make test     # numpy tests (loader, synthetics) run anywhere;
              # param-match + resume tests run where torch+transformers are installed
```

## License

MIT — see [LICENSE](LICENSE).
