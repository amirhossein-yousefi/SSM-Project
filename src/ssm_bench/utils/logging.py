"""Lightweight JSONL logging + run-id helpers (no wandb/tensorboard).

One JSON object per line, appended and fsync'd so a mid-run kill never corrupts
prior records. Matches the project convention of `log.jsonl` per run.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Iterator, List


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    """Append one record as a line to `path`, flushing to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    line = json.dumps(record, default=_json_default)
    with open(path, "a") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read all records from a JSONL file (skips blank/corrupt trailing lines)."""
    out: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # a torn last line from a hard kill — ignore it
                continue
    return out


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    for rec in read_jsonl(path):
        yield rec


def dump_config(path: str, config: Dict[str, Any]) -> None:
    """Write `config.json` for a run (idempotent; warns on drift)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if os.path.exists(path):
        try:
            existing = json.load(open(path))
        except Exception:
            existing = None
        if existing is not None and existing != config:
            print(f"[warn] {path} exists with different config; keeping the original "
                  f"(resume uses the saved checkpoint config).")
        return
    with open(path, "w") as f:
        json.dump(config, f, indent=2, default=_json_default, sort_keys=True)


def run_id(**fields: Any) -> str:
    """Deterministic short id from keyword fields (for idempotent eval cells)."""
    blob = json.dumps(fields, sort_keys=True, default=_json_default)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def _json_default(o: Any) -> Any:
    # make numpy scalars / arrays JSON-serializable
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return str(o)
