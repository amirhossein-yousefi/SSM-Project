"""Resumable, deterministic token loader over flat uint16 shards.

Resume state is just `{epoch, cursor, seed}`:
  * the per-epoch window order is a deterministic permutation seeded by `seed + epoch`,
  * `cursor` is an integer index into that order.
Restoring `(epoch, cursor)` reproduces the exact next batch — no re-tokenization, no
shard replay, bit-exact across all three architectures. This is simpler and more correct
than HuggingFace's `IterableDataset.state_dict()` (which replays shards and has had
off-by-N resume bugs).

torch is imported lazily so the indexing logic is testable without a torch install.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


class PackedTokenLoader:
    def __init__(
        self,
        data_dir: str,
        split: str,
        block_size: int,
        batch_size: int,
        device: str = "cpu",
        seed: int = 1337,
        as_torch: bool = True,
    ):
        self.data_dir = data_dir
        self.split = split
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        self.as_torch = as_torch

        self.shards: List[np.memmap] = self._open_shards(data_dir, split)
        if not self.shards:
            raise FileNotFoundError(
                f"no '{split}_*.npy' shards in {data_dir}. Run prepare_fineweb.py first."
            )
        lengths = [len(s) for s in self.shards]
        # cum[i] = global start offset of shard i; cum[-1] = total tokens
        self.cum = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        self.total_tokens = int(self.cum[-1])
        self.window = block_size + 1
        self.num_windows = self.total_tokens // self.window
        if self.num_windows < batch_size:
            raise ValueError(
                f"split '{split}' has only {self.num_windows} windows < batch_size {batch_size}"
            )

        self.epoch = 0
        self.cursor = 0
        self._build_epoch(self.epoch)

    # -------------------------------------------------------------- shards -----
    @staticmethod
    def _open_shards(data_dir: str, split: str) -> List[np.memmap]:
        paths = sorted(glob.glob(os.path.join(data_dir, f"{split}_*.npy")))
        return [np.load(p, mmap_mode="r") for p in paths]

    def _build_epoch(self, epoch: int) -> None:
        rng = np.random.default_rng(self.seed + epoch)
        self.window_order = rng.permutation(self.num_windows)

    def _read(self, start: int, length: int) -> np.ndarray:
        """Read `length` tokens beginning at global index `start`, crossing shards."""
        out = np.empty(length, dtype=np.int64)
        filled = 0
        si = int(np.searchsorted(self.cum, start, side="right") - 1)
        local = start - int(self.cum[si])
        while filled < length:
            shard = self.shards[si]
            take = min(length - filled, len(shard) - local)
            out[filled:filled + take] = shard[local:local + take]
            filled += take
            si += 1
            local = 0
        return out

    # -------------------------------------------------------------- batches ----
    def next_batch(self) -> Tuple:
        """Return (x, y) of shape [batch, block_size]; y is x shifted by one."""
        if self.cursor + self.batch_size > self.num_windows:
            # advance to the next epoch (drop the partial tail for determinism)
            self.epoch += 1
            self._build_epoch(self.epoch)
            self.cursor = 0

        widx = self.window_order[self.cursor:self.cursor + self.batch_size]
        x = np.empty((self.batch_size, self.block_size), dtype=np.int64)
        y = np.empty((self.batch_size, self.block_size), dtype=np.int64)
        for i, w in enumerate(widx):
            span = self._read(int(w) * self.window, self.window)
            x[i] = span[:-1]
            y[i] = span[1:]
        self.cursor += self.batch_size

        if not self.as_torch:
            return x, y
        import torch

        xt = torch.from_numpy(x).to(self.device, non_blocking=True)
        yt = torch.from_numpy(y).to(self.device, non_blocking=True)
        return xt, yt

    @property
    def tokens_per_batch(self) -> int:
        return self.batch_size * self.block_size

    # -------------------------------------------------------------- resume -----
    def state_dict(self) -> Dict[str, int]:
        return {"epoch": self.epoch, "cursor": self.cursor, "seed": self.seed}

    def load_state_dict(self, state: Dict[str, int]) -> None:
        self.seed = state["seed"]
        self.epoch = state["epoch"]
        self._build_epoch(self.epoch)
        self.cursor = state["cursor"]


def load_manifest(data_dir: str) -> Optional[dict]:
    path = os.path.join(data_dir, "manifest.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


class RandomTokenLoader:
    """Deterministic random-token loader with the PackedTokenLoader interface.

    Used for smoke runs / CI when no tokenized data is present. Batch contents are a pure
    function of (seed, step) so resume reproduces the exact stream.
    """

    def __init__(self, vocab_size: int, block_size: int, batch_size: int,
                 device: str = "cpu", seed: int = 1337, as_torch: bool = True):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        self.as_torch = as_torch
        self.epoch = 0
        self.cursor = 0  # step index

    def next_batch(self):
        rng = np.random.default_rng(self.seed * 1_000_003 + self.cursor)
        x = rng.integers(0, self.vocab_size,
                         size=(self.batch_size, self.block_size), dtype=np.int64)
        y = np.roll(x, -1, axis=1)
        self.cursor += 1
        if not self.as_torch:
            return x, y
        import torch

        return (torch.from_numpy(x).to(self.device),
                torch.from_numpy(y).to(self.device))

    @property
    def tokens_per_batch(self) -> int:
        return self.batch_size * self.block_size

    def state_dict(self):
        return {"epoch": self.epoch, "cursor": self.cursor, "seed": self.seed}

    def load_state_dict(self, state):
        self.seed = state["seed"]
        self.epoch = state["epoch"]
        self.cursor = state["cursor"]
