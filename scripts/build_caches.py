#!/usr/bin/env python3
"""Build the memory-mapped sequence caches the trainer reads.

Each cache is cache/<name>/{X.npy (n, 20000, 112) float32, Y.npy (n, 20000, 2) float32,
M.npy (n, 20000) bool is_scored (valid only), manifest.json}. Row groups of the parquet
files are sequences, so a cache is defined by its source file and a list of row groups.

    python scripts/build_caches.py                 # all five caches
    python scripts/build_caches.py train_full eval # a subset

Caches and their size on disk:
    train_full     all 10,607 train sequences                     ~100 GB
    valid_all      all 1,873 valid sequences                      ~17 GB
    valid_retrain  1,073 valid sequences (configs/valid_split.json "retrain")
    eval           300 valid sequences ("dev"), used by the trainer for in-run evaluation
    holdout        500 valid sequences ("holdout"), never trained on in screen runs

Needs datasets/train.parquet, datasets/valid.parquet and the starter-pack utils.py at the repo root.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import pyarrow.parquet as pq  # noqa: E402
from data import build_cache  # noqa: E402

split = json.load(open(ROOT / "configs" / "valid_split.json"))


def n_groups(source):
    return pq.ParquetFile(ROOT / "datasets" / f"{source}.parquet").metadata.num_row_groups


CACHES = {
    "train_full":    ("train", lambda: list(range(n_groups("train")))),
    "valid_all":     ("valid", lambda: list(range(n_groups("valid")))),
    "valid_retrain": ("valid", lambda: sorted(split["retrain"])),
    "eval":          ("valid", lambda: sorted(split["dev"])),
    "holdout":       ("valid", lambda: sorted(split["holdout"])),
}

if __name__ == "__main__":
    names = sys.argv[1:] or list(CACHES)
    for name in names:
        source, groups = CACHES[name]
        rg = groups()
        build_cache(source, ROOT / "cache" / name, len(rg), row_groups=rg)
    print("done")
