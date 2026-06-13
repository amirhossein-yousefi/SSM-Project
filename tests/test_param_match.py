"""Parameter-matching across the three architectures (requires torch + transformers).

Skipped locally if torch/transformers are absent; run on Colab. Tolerance is generous here
because the committed configs are starting points finalized by `param_utils --check`.
"""
import os

import pytest
import yaml

pytest.importorskip("torch")
pytest.importorskip("transformers")

from ssm_bench.models.param_utils import assert_matched, count_params  # noqa: E402
from ssm_bench.models.registry import build_model  # noqa: E402

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")


def _model_cfg(arch):
    with open(os.path.join(CONFIG_DIR, f"{arch}.yaml")) as f:
        return dict(yaml.safe_load(f)["model"])


@pytest.mark.parametrize("arch", ["transformer", "mamba", "jamba"])
def test_builds_and_has_params(arch):
    model = build_model(arch, _model_cfg(arch))
    assert count_params(model) > 50_000_000


def test_archs_param_matched():
    counts = {a: count_params(build_model(a, _model_cfg(a)))
              for a in ["transformer", "mamba", "jamba"]}
    # 8% tolerance: starting configs; tighten after `param_utils --check` on Colab.
    assert_matched(counts, tol=0.08)
