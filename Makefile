# Convenience targets for the ssm-benchmark project.
# Most real runs happen on Colab via notebooks/colab_runner.ipynb; these mirror them locally.

PYTHON ?= python
DATA_DIR ?= data_cache/fineweb_edu_gpt2
RUNS_DIR ?= runs
SEED ?= 1337
TOTAL_STEPS ?= 6000

.PHONY: help install install-kernels data check-params train-all train-transformer train-mamba train-jamba \
        synthetics efficiency aggregate plots test smoke clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:           ## Install core python deps (torch must already be present)
	$(PYTHON) -m pip install -r requirements.txt && $(PYTHON) -m pip install -e .

install-kernels:   ## Build optional fast Mamba/Jamba CUDA kernels (slow, may fail -> eager fallback)
	MAX_JOBS=4 $(PYTHON) -m pip install -r requirements-kernels.txt --no-build-isolation

data:              ## One-time: pre-tokenize FineWeb-Edu to uint16 shards (DATA_DIR=...)
	$(PYTHON) -m ssm_bench.data.prepare_fineweb --out $(DATA_DIR)

check-params:      ## Verify all three models land within 5% of ~125M params
	$(PYTHON) -m ssm_bench.models.param_utils --check

train-all:         ## Resumable: train all three archs, skip DONE, auto-resume
	bash scripts/run_all.sh

train-transformer: ## Train just the transformer arm
	$(PYTHON) scripts/train.py --arch transformer --seed $(SEED) --data_dir $(DATA_DIR) \
	  --output_dir $(RUNS_DIR)/transformer_seed$(SEED) --total_steps $(TOTAL_STEPS)

train-mamba:       ## Train just the mamba arm
	$(PYTHON) scripts/train.py --arch mamba --seed $(SEED) --data_dir $(DATA_DIR) \
	  --output_dir $(RUNS_DIR)/mamba_seed$(SEED) --total_steps $(TOTAL_STEPS)

train-jamba:       ## Train just the jamba arm
	$(PYTHON) scripts/train.py --arch jamba --seed $(SEED) --data_dir $(DATA_DIR) \
	  --output_dir $(RUNS_DIR)/jamba_seed$(SEED) --total_steps $(TOTAL_STEPS)

synthetics:        ## Run the synthetic mechanistic-task sweep (MQAR / induction / selective-copy)
	bash scripts/run_synthetics.sh

efficiency:        ## Run the throughput / memory / latency benchmark sweep
	bash scripts/run_efficiency.sh

aggregate:         ## Collate runs/*/log.jsonl into results/summaries/*.csv
	$(PYTHON) scripts/aggregate.py

plots:             ## Render figures from results/summaries/*.csv into results/figures/
	$(PYTHON) scripts/plots.py

test:              ## Run the test suite (heavy torch tests skip if torch absent)
	$(PYTHON) -m pytest

smoke:             ## Tiny end-to-end training run to sanity-check the harness
	$(PYTHON) scripts/train.py --arch transformer --seed 0 --smoke \
	  --output_dir $(RUNS_DIR)/_smoke --total_steps 20

clean:             ## Remove smoke/test scratch runs
	rm -rf $(RUNS_DIR)/_smoke $(RUNS_DIR)/_test* .pytest_cache
