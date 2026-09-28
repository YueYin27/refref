#!/usr/bin/env python
"""Add explicit train/val/test split lists to a nerfstudio transforms.json.

nerfstudio-data normally derives its split from `train_split_fraction`, which yields
only two sets -- and `val` and `test` both resolve to the same held-out frames, so
there is no independent test set. But if transforms.json carries `train_filenames`,
`val_filenames` and `test_filenames`, the dataparser uses those verbatim
(nerfstudio_dataparser.py:194-204) and the fraction logic is bypassed entirely.

This script writes those three lists. Frames whose basename starts with --test-prefix
become the test set (the separate capture trajectory); the rest are the training orbit
and get divided train/val using the same rule nerfstudio's fraction split uses, so the
result matches what you would have got by default.

    python scripts/inject_splits.py real_captures/courtyard/bulb
    python scripts/inject_splits.py real_captures/courtyard/bulb --dry-run
    python scripts/inject_splits.py real_captures/*/*/          # every scene

The original file is kept as transforms.json.bak on first run. Re-running recomputes
the lists from scratch, so it is safe to run repeatedly.
"""
import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np


def compute_splits(file_paths, test_prefix: str, train_fraction: float):
    """Partition file_paths into (train, val, test) lists of the same strings."""
    test = [p for p in file_paths if Path(p).name.startswith(test_prefix)]
    orbit = [p for p in file_paths if not Path(p).name.startswith(test_prefix)]

    # Same rule as nerfstudio's get_train_eval_split_fraction: equally spaced train
    # indices spanning the sequence, val is whatever is left over.
    n = len(orbit)
    n_train = math.ceil(n * train_fraction)
    i_train = np.linspace(0, n - 1, n_train, dtype=int)
    i_val = np.setdiff1d(np.arange(n), i_train)

    train = [orbit[i] for i in i_train]
    val = [orbit[i] for i in i_val]
    return train, val, test


def process(scene_dir: Path, test_prefix: str, train_fraction: float, dry_run: bool) -> bool:
    tj = scene_dir / "transforms.json"
    if not tj.is_file():
        print(f"[skip] {scene_dir} - no transforms.json")
        return False

    meta = json.loads(tj.read_text())
    file_paths = [f["file_path"] for f in meta["frames"]]

    train, val, test = compute_splits(file_paths, test_prefix, train_fraction)

    if not test:
        print(
            f"[FAIL] {scene_dir} - no frames start with '{test_prefix}'. "
            f"The test trajectory must be in this transforms.json too, with that prefix."
        )
        return False

    print(f"[ok]   {scene_dir} - train {len(train)} / val {len(val)} / test {len(test)}")
    if dry_run:
        return True

    # Keys the dataparser looks for. Written before "frames" purely for readability.
    meta.pop("train_filenames", None)
    meta.pop("val_filenames", None)
    meta.pop("test_filenames", None)
    frames = meta.pop("frames")
    meta["train_filenames"] = train
    meta["val_filenames"] = val
    meta["test_filenames"] = test
    meta["frames"] = frames

    backup = tj.with_suffix(".json.bak")
    if not backup.exists():
        shutil.copy2(tj, backup)
    tj.write_text(json.dumps(meta, indent=2))
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene_dirs", nargs="+", help="Scene folder(s) containing transforms.json")
    parser.add_argument(
        "--test-prefix",
        default="test_",
        help="Frames whose filename starts with this are the test trajectory (default: test_)",
    )
    parser.add_argument(
        "--train-split-fraction",
        type=float,
        default=0.9,
        help="Fraction of the training orbit used for train, remainder is val (default: 0.9)",
    )
    parser.add_argument("--dry-run", "-n", action="store_true", help="Report the split without writing")

    args = parser.parse_args()

    ok = sum(
        process(Path(d), args.test_prefix, args.train_split_fraction, args.dry_run) for d in args.scene_dirs
    )
    print(f"----\n{ok}/{len(args.scene_dirs)} scene(s) {'checked' if args.dry_run else 'updated'}")


if __name__ == "__main__":
    main()
