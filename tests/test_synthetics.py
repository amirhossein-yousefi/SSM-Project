"""Synthetic generators: shapes + label masking. Pure-numpy (no torch needed)."""
import numpy as np

from ssm_bench.data.synthetics import (IGNORE, make_induction, make_mqar,
                                       make_selective_copy)


def test_mqar_structure():
    B, L, kv, V = 8, 64, 4, 128
    x, y = make_mqar(B, L, kv, V, seed=0)
    assert x.shape == (B, L) and y.shape == (B, L)
    # exactly kv answer positions per row
    assert (y != IGNORE).sum() == B * kv
    # values come from the upper half of the vocab
    vals = y[y != IGNORE]
    assert (vals >= V // 2).all()
    # keys in context come from the lower half
    ctx_keys = x[:, 0:2 * kv:2]
    assert (ctx_keys < V // 2).all()


def test_induction_structure():
    B, L, V = 8, 32, 16
    x, y = make_induction(B, L, V, seed=0)
    assert x.shape == (B, L)
    # one answer per row, at the last position
    assert (y != IGNORE).sum() == B
    for b in range(B):
        assert y[b, -1] == x[b, -1]      # answer equals final token
        assert x[b, -2] == x[b, L - 2]   # cue repeated before the answer


def test_selective_copy_structure():
    B, L, nt, V = 8, 64, 16, 32
    x, y = make_selective_copy(B, L, nt, V, seed=0)
    total = L + 1 + nt
    assert x.shape == (B, total)
    # nt answer positions per row, all in the answer region
    assert (y != IGNORE).sum() == B * nt
    assert (y[:, : L + 1] == IGNORE).all()        # nothing scored before the delimiter
    assert (x[:, L] == V - 1).all()               # delimiter token present
    # teacher-forced answer region equals the labels
    assert np.array_equal(x[:, L + 1:], y[:, L + 1:])


def test_generators_are_deterministic():
    a = make_mqar(4, 64, 4, 128, seed=5)
    b = make_mqar(4, 64, 4, 128, seed=5)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
