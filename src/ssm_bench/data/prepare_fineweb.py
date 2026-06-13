"""One-time pre-tokenization: FineWeb-Edu -> flat uint16 token shards.

Streams the dataset, tokenizes with the GPT-2 BPE, inserts an end-of-text token between
documents, and packs the token stream into fixed-size shards. Shard 0 is held out as the
validation split. Writes a manifest.json describing the cache.

Run this ONCE and cache the output on Google Drive so every Colab session reuses it:

    python -m ssm_bench.data.prepare_fineweb --out /content/drive/MyDrive/ssm_data/fineweb_edu_gpt2

If interrupted, just re-run (it's one-time; the stream restarts from the beginning).
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List

import numpy as np

try:  # nice progress bar if available
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x=None, **k):
        return x if x is not None else iter(())


def _atomic_save_npy(path: str, arr: np.ndarray) -> None:
    tmp = f"{path}.tmp"
    np.save(tmp, arr)
    # np.save appends .npy; normalize the temp name then atomic-rename
    tmp_npy = tmp + ".npy" if not tmp.endswith(".npy") else tmp
    os.replace(tmp_npy, path)


def prepare(
    out_dir: str,
    tokenizer_name: str = "gpt2",
    dataset: str = "HuggingFaceFW/fineweb-edu",
    dataset_name: str = "sample-10BT",
    eot_token: int = 50256,
    shard_tokens: int = 100_000_000,
    target_train_tokens: int = 1_500_000_000,
    val_shards: int = 1,
    batch_docs: int = 1024,
) -> None:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    os.makedirs(out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(tokenizer_name)

    ds = load_dataset(dataset, name=dataset_name, split="train", streaming=True)

    total_target = target_train_tokens + val_shards * shard_tokens
    buf = np.empty(shard_tokens, dtype=np.uint16)
    fill = 0
    shard_idx = 0
    written: List[dict] = []
    total_written = 0

    def shard_name(idx: int) -> str:
        # first `val_shards` shards are the validation split
        if idx < val_shards:
            return f"val_{idx:03d}.npy"
        return f"train_{idx - val_shards:03d}.npy"

    def flush(n: int) -> None:
        nonlocal shard_idx, total_written
        path = os.path.join(out_dir, shard_name(shard_idx))
        _atomic_save_npy(path, buf[:n].copy())
        written.append({"file": os.path.basename(path), "tokens": int(n)})
        total_written += n
        print(f"  wrote {os.path.basename(path)}  ({n:,} tokens, {total_written:,} total)")
        shard_idx += 1

    pbar = tqdm(total=total_target, unit="tok", unit_scale=True, desc="tokenizing")
    text_batch: List[str] = []

    def encode_and_pack(texts: List[str]) -> None:
        nonlocal fill
        enc = tok(texts)["input_ids"]
        for ids in enc:
            ids = ids + [eot_token]
            i = 0
            while i < len(ids):
                take = min(len(ids) - i, shard_tokens - fill)
                buf[fill:fill + take] = ids[i:i + take]
                fill += take
                i += take
                if fill == shard_tokens:
                    flush(shard_tokens)
                    fill = 0
            pbar.update(len(ids))

    done = False
    for doc in ds:
        text = doc.get("text", "")
        if not text:
            continue
        text_batch.append(text)
        if len(text_batch) >= batch_docs:
            encode_and_pack(text_batch)
            text_batch = []
            if total_written + fill >= total_target:
                done = True
                break
    if not done and text_batch:
        encode_and_pack(text_batch)

    if fill > 0:  # final partial shard
        flush(fill)
    pbar.close()

    manifest = {
        "tokenizer": tokenizer_name,
        "dataset": dataset,
        "dataset_name": dataset_name,
        "eot_token": eot_token,
        "shard_tokens": shard_tokens,
        "val_shards": val_shards,
        "shards": written,
        "total_tokens": total_written,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDone: {total_written:,} tokens across {len(written)} shards -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-tokenize FineWeb-Edu to uint16 shards.")
    ap.add_argument("--out", required=True, help="output dir (cache on Drive for Colab)")
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--dataset_name", default="sample-10BT")
    ap.add_argument("--eot_token", type=int, default=50256)
    ap.add_argument("--shard_tokens", type=int, default=100_000_000)
    ap.add_argument("--target_train_tokens", type=int, default=1_500_000_000)
    ap.add_argument("--val_shards", type=int, default=1)
    args = ap.parse_args()
    prepare(
        out_dir=args.out,
        tokenizer_name=args.tokenizer,
        dataset=args.dataset,
        dataset_name=args.dataset_name,
        eot_token=args.eot_token,
        shard_tokens=args.shard_tokens,
        target_train_tokens=args.target_train_tokens,
        val_shards=args.val_shards,
    )


if __name__ == "__main__":
    main()
