"""Resumable loader: cursor round-trip + x/y shift. Pure-numpy (no torch needed)."""
import numpy as np

from ssm_bench.data.packed_loader import PackedTokenLoader, RandomTokenLoader


def _make_shards(tmp_path, n_shards=2, shard_len=5000):
    d = tmp_path / "data"
    d.mkdir()
    rng = np.random.default_rng(0)
    for i in range(n_shards):
        arr = rng.integers(0, 1000, size=shard_len).astype(np.uint16)
        np.save(d / f"train_{i:03d}.npy", arr)
    return str(d)


def test_xy_is_shifted_by_one(tmp_path):
    d = _make_shards(tmp_path)
    loader = PackedTokenLoader(d, "train", block_size=16, batch_size=4,
                              as_torch=False, seed=1)
    x, y = loader.next_batch()
    assert x.shape == (4, 16) and y.shape == (4, 16)
    # y[t] == x[t+1]  ->  y[:, :-1] == x[:, 1:]
    assert np.array_equal(y[:, :-1], x[:, 1:])


def test_cursor_roundtrip_reproduces_next_batch(tmp_path):
    d = _make_shards(tmp_path)
    a = PackedTokenLoader(d, "train", block_size=16, batch_size=4, as_torch=False, seed=42)
    for _ in range(3):
        a.next_batch()
    state = a.state_dict()
    x1, y1 = a.next_batch()

    b = PackedTokenLoader(d, "train", block_size=16, batch_size=4, as_torch=False, seed=42)
    b.load_state_dict(state)
    x2, y2 = b.next_batch()
    assert np.array_equal(x1, x2) and np.array_equal(y1, y2)


def test_epoch_rollover_is_deterministic(tmp_path):
    # force many batches so the epoch rolls over, then check resume still matches
    d = _make_shards(tmp_path, n_shards=1, shard_len=600)
    a = PackedTokenLoader(d, "train", block_size=16, batch_size=4, as_torch=False, seed=7)
    for _ in range(20):
        a.next_batch()
    state = a.state_dict()
    x1, _ = a.next_batch()
    b = PackedTokenLoader(d, "train", block_size=16, batch_size=4, as_torch=False, seed=7)
    b.load_state_dict(state)
    x2, _ = b.next_batch()
    assert np.array_equal(x1, x2)
    assert state["epoch"] >= 1  # we wrapped at least once


def test_random_loader_resumes(tmp_path):
    a = RandomTokenLoader(vocab_size=100, block_size=8, batch_size=2, as_torch=False, seed=3)
    for _ in range(5):
        a.next_batch()
    st = a.state_dict()
    x1, _ = a.next_batch()
    b = RandomTokenLoader(vocab_size=100, block_size=8, batch_size=2, as_torch=False, seed=3)
    b.load_state_dict(st)
    x2, _ = b.next_batch()
    assert np.array_equal(x1, x2)
