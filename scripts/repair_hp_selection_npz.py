"""Remove legacy scalar-None gamma placeholders from our validation archives."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from evaluate_hp_selection_validation import DEFAULT_OUTPUT


def repair(root: Path) -> int:
    count = 0
    for path in root.rglob("*.npz"):
        with np.load(path, allow_pickle=True) as archive:
            arrays = {key: archive[key] for key in archive.files}
        gamma = arrays.get("gamma")
        if gamma is None or gamma.dtype != object or gamma.size != 1 or gamma.item() is not None:
            continue
        arrays.pop("gamma")
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
        count += 1
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(f"Repaired {repair(args.output)} archives")
