"""Exact-resume checkpoint round-trip (requires torch + transformers).

Trains a tiny model a few steps, checkpoints, then rebuilds from scratch and resumes —
asserting the restored model parameters match and the NEXT data batch is identical (data
cursor + RNG restored). Skipped locally if torch/transformers are absent.
"""
import os

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch  # noqa: E402

from ssm_bench.data.packed_loader import RandomTokenLoader  # noqa: E402
from ssm_bench.models.registry import build_model  # noqa: E402
from ssm_bench.train import checkpoint as ckpt  # noqa: E402
from ssm_bench.train.schedule import cosine_with_warmup  # noqa: E402
from ssm_bench.utils.seed import set_all_seeds  # noqa: E402

TINY = dict(vocab_size=128, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
            num_key_value_heads=2, intermediate_size=64, max_position_embeddings=64,
            tie_word_embeddings=True, hidden_act="silu")


def _make(seed=0):
    set_all_seeds(seed)
    model = build_model("transformer", dict(TINY))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = cosine_with_warmup(opt, warmup_steps=2, total_steps=20)
    loader = RandomTokenLoader(128, block_size=16, batch_size=2, seed=seed)
    return model, opt, sched, loader


def _train_steps(model, opt, sched, loader, n):
    import torch.nn.functional as F

    model.train()
    for _ in range(n):
        x, y = loader.next_batch()
        logits = model(input_ids=x).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)


def test_checkpoint_roundtrip(tmp_path):
    ckpt_dir = str(tmp_path / "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    model, opt, sched, loader = _make(seed=0)
    _train_steps(model, opt, sched, loader, 3)
    cfg = {"arch": "transformer", "seed": 0, "model": TINY}
    state = ckpt.build_state(model, opt, sched, global_step=3, epoch=loader.epoch,
                             best_val=1.23, loader=loader, config=cfg)
    ckpt.save_checkpoint(state, ckpt_dir, "last.pt")

    # the batch the original loader would produce next
    x_ref, y_ref = loader.next_batch()
    ref_params = [p.detach().clone() for p in model.parameters()]

    # fresh objects, resume
    model2, opt2, sched2, loader2 = _make(seed=999)  # different seed on purpose
    step, best = ckpt.maybe_resume(ckpt_dir, model2, opt2, sched2, loader2,
                                   expected_arch="transformer")
    assert step == 3 and abs(best - 1.23) < 1e-6

    # next batch must match (data cursor + seed restored)
    x2, y2 = loader2.next_batch()
    assert torch.equal(x2.cpu(), x_ref.cpu()) and torch.equal(y2.cpu(), y_ref.cpu())

    # restored params match
    for a, b in zip(ref_params, model2.parameters()):
        assert torch.allclose(a, b)


def test_resume_arch_mismatch_raises(tmp_path):
    ckpt_dir = str(tmp_path / "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    model, opt, sched, loader = _make(seed=0)
    state = ckpt.build_state(model, opt, sched, 1, 0, 9.0, loader,
                             {"arch": "transformer"})
    ckpt.save_checkpoint(state, ckpt_dir, "last.pt")
    m2, o2, s2, l2 = _make(seed=0)
    with pytest.raises(ValueError):
        ckpt.maybe_resume(ckpt_dir, m2, o2, s2, l2, expected_arch="mamba")
