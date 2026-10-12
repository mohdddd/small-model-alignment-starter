"""Small shared helpers: seeding, JSON(L) IO, metric logging, Kaggle input lookup."""

from __future__ import annotations

import glob
import json
import os
import random
import time
from pathlib import Path


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def append_jsonl(path, row) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


class MetricLogger:
    """Appends one JSON object per event to metrics.jsonl (survives disconnects)."""

    def __init__(self, run_dir):
        self.path = Path(run_dir) / "metrics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.start = time.time()

    def log(self, **kwargs):
        kwargs.setdefault("elapsed_min", round((time.time() - self.start) / 60, 2))
        append_jsonl(self.path, kwargs)
        print(json.dumps(kwargs), flush=True)


def find_one(pattern: str, root: str = "/kaggle/input") -> str:
    """Locate a file/dir inside attached Kaggle inputs regardless of mount layout."""
    hits = sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))
    if not hits:
        raise FileNotFoundError(f"No match for {pattern!r} under {root} - is the right notebook output attached?")
    if len(hits) > 1:
        print(f"[find_one] {len(hits)} matches for {pattern!r}; using {hits[0]}")
    return hits[0]


def peak_gpu_memory_gb():
    import torch

    if not torch.cuda.is_available():
        return {}
    return {f"cuda:{i}": round(torch.cuda.max_memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())}
