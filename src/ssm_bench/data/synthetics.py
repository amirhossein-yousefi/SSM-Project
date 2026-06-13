"""Synthetic mechanistic-task generators (MQAR / induction / selective-copy).

Each generator returns `(inputs, labels)` as int64 numpy arrays of shape [B, L], with
`labels == -100` everywhere except the answer positions. This matches the HuggingFace
`labels=` loss contract (the model shifts internally; loss is computed only on positions
!= -100), and lets us measure accuracy only on the answer positions.

These are pure-numpy and torch-free so they can be unit-tested anywhere; synthetic_train.py
converts to torch tensors.

Task definitions follow the public references:
  * MQAR            — HazyResearch/zoology, "Zoology" (arXiv:2312.04927)
  * induction heads — Mamba paper (arXiv:2312.00752)
  * selective copy  — Mamba paper (arXiv:2312.00752)
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

IGNORE = -100


# ----------------------------------------------------------------- MQAR --------

def make_mqar(
    batch_size: int,
    seq_len: int,
    num_kv_pairs: int,
    vocab_size: int,
    power_a: float = 0.01,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Multi-Query Associative Recall.

    The vocab is split in half: keys from the lower half, values from the upper half.
    The context is an interleaved key->value dictionary (k v k v ...). After the context,
    queries (re-used keys) appear at power-law-spaced positions; the model must emit each
    key's value at the position right after the query.

    Exposes the SSM recall limit: attention recalls any pair exactly at any length, while
    a fixed-size SSM state degrades once the recall load exceeds its capacity.
    """
    rng = np.random.default_rng(seed)
    assert vocab_size % 2 == 0, "vocab_size must be even (split into keys/values)"
    kv = vocab_size // 2
    keys_pool = np.arange(1, kv)
    values_pool = np.arange(kv, vocab_size)
    ctx = 2 * num_kv_pairs
    assert ctx + num_kv_pairs + 1 <= seq_len, "seq_len too short for context + queries"

    inputs = np.zeros((batch_size, seq_len), dtype=np.int64)
    labels = np.full((batch_size, seq_len), IGNORE, dtype=np.int64)

    space = seq_len - ctx - 1
    p = power_a * np.arange(1, space + 1) ** (power_a - 1)
    p /= p.sum()

    for b in range(batch_size):
        keys = rng.choice(keys_pool, num_kv_pairs, replace=False)
        vals = rng.choice(values_pool, num_kv_pairs, replace=False)
        inputs[b, 0:ctx:2] = keys
        inputs[b, 1:ctx:2] = vals

        offs = np.sort(rng.choice(space, num_kv_pairs, replace=False, p=p))
        qpos = ctx + offs                     # positions of the query (= the key)
        inputs[b, qpos] = keys
        labels[b, qpos + 1] = vals            # predict the value one step after the query
    return inputs, labels


# ------------------------------------------------------------ induction --------

def make_induction(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Induction / in-context copy: '... [cue] X ... [cue]' -> predict X.

    Trained at a short length; the headline result is length EXTRAPOLATION (Mamba
    generalizes to ~1M length; attention fails past ~2x train length).
    """
    rng = np.random.default_rng(seed)
    inputs = rng.integers(0, vocab_size, size=(batch_size, seq_len), dtype=np.int64)
    labels = np.full((batch_size, seq_len), IGNORE, dtype=np.int64)

    for b in range(batch_size):
        cue = int(rng.integers(0, vocab_size))
        target = int(rng.integers(0, vocab_size))
        # cue->target bigram somewhere in the early/middle region
        q = int(rng.integers(0, seq_len - 3))
        inputs[b, q] = cue
        inputs[b, q + 1] = target
        # repeat the cue at the penultimate position; model predicts target at the last
        inputs[b, seq_len - 2] = cue
        inputs[b, seq_len - 1] = target
        labels[b, seq_len - 1] = target
    return inputs, labels


# --------------------------------------------------------- selective copy ------

def make_selective_copy(
    batch_size: int,
    seq_len: int,
    num_tokens: int,
    vocab_size: int,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Selective copy: data tokens scattered among blanks, recalled in order after a delimiter.

    Layout (length = seq_len + 1 + num_tokens):
      [ filler region of length seq_len with num_tokens data tokens at random positions ]
      [ DELIM ]
      [ answer region: the data tokens in original order (teacher-forced) ]
    Random spacing defeats non-selective (LTI) models; selection (Mamba) and attention solve it.

    Token ids: 0 = blank, 1..num_data data tokens, last id = DELIM.
    """
    rng = np.random.default_rng(seed)
    blank = 0
    delim = vocab_size - 1
    data_lo, data_hi = 1, vocab_size - 1          # data tokens in [1, vocab_size-1)
    total_len = seq_len + 1 + num_tokens
    assert num_tokens <= seq_len, "more data tokens than filler slots"

    inputs = np.full((batch_size, total_len), blank, dtype=np.int64)
    labels = np.full((batch_size, total_len), IGNORE, dtype=np.int64)

    for b in range(batch_size):
        positions = np.sort(rng.choice(seq_len, num_tokens, replace=False))
        data = rng.integers(data_lo, data_hi, size=num_tokens)
        inputs[b, positions] = data
        inputs[b, seq_len] = delim
        ans = slice(seq_len + 1, seq_len + 1 + num_tokens)
        inputs[b, ans] = data                     # teacher forcing
        labels[b, ans] = data                     # predict the data tokens in order
    return inputs, labels


GENERATORS = {
    "mqar": make_mqar,
    "induction": make_induction,
    "selective_copy": make_selective_copy,
}
