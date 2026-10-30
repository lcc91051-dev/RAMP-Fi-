"""Construct CSI features and the fixed data split from raw Widar recordings."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample

from data import ROOT, sha, write_json
from intel_csi import IntelCSI
from split import ensure


PARTS = ("labeled_train", "validation", "unlabeled_train", "target_test")


def stability_scores(h):
    amplitude = np.abs(h)
    return np.mean(np.mean(amplitude, axis=2) /
                   (np.var(amplitude, axis=2) + 1e-8), axis=1)


def normalize(td):
    amplitude = np.abs(td)
    phase = np.angle(td)
    low, high = float(amplitude.min()), float(amplitude.max())
    amplitude = (amplitude - low) / (high - low + 1e-8) * np.pi
    return np.stack((amplitude, phase), axis=1).astype(np.float32)


def process_file(raw_path):
    """Return two ratio streams using the highest-scoring receiver as reference."""
    parser = IntelCSI(Path(raw_path))
    parser.read()
    h = parser.get_scaled_csi().squeeze(-1).transpose(2, 1, 0)
    t = parser.timestamp_low
    if len(t) < 2 or not np.all(np.diff(t) >= 0) or not np.isfinite(h).all():
        raise ValueError(f"Invalid CSI or timestamps: {raw_path}")
    future = np.searchsorted(t, t + 80000, side="left")
    now = np.flatnonzero(future < len(t))
    future = future[now]
    if not len(now):
        raise ValueError(f"Recording shorter than 80 ms: {raw_path}")
    scores = stability_scores(h)
    reference = int(scores.argmax())
    numerators = sorted((index for index in range(3) if index != reference),
                        key=lambda index: (-scores[index], index))
    ratio = h[numerators] / (h[reference] + 1e-8)
    td = ratio[:, :, future] - ratio[:, :, now]
    representation = normalize(resample(td, 800, axis=-1))
    if representation.shape != (2, 2, 30, 800) or not np.isfinite(representation).all():
        raise ValueError(f"Unexpected representation: {raw_path}")
    return representation


def _job(args):
    index, row, raw_root = args
    relative = Path(row["raw"].replace("\\", "/"))
    return index, process_file(raw_root / relative)


def prepare(part, raw_root, cache_dir, workers=2):
    if part not in PARTS:
        raise ValueError(part)
    raw_root, cache_dir = Path(raw_root), Path(cache_dir)
    split_path = ensure(raw_root, cache_dir)
    rows = json.loads(split_path.read_text(encoding="utf-8"))["room2_labeled"][part]
    cache_dir.mkdir(parents=True, exist_ok=True)
    array_path = cache_dir / f"{part}.npy"
    done_path = cache_dir / f"{part}.done.npy"
    meta_path = cache_dir / f"{part}.json"
    expected = {"split_sha256": sha(split_path), "partition": part,
                "count": len(rows), "representation": "stable_reference_two_ratios_amp_phase_30x800"}
    if meta_path.exists():
        actual = json.loads(meta_path.read_text(encoding="utf-8"))
        if actual["expected"] != expected or sha(array_path) != actual["array_sha256"]:
            raise ValueError(f"Cache metadata mismatch: {part}")
        print(f"Prepared {part}: already complete", flush=True)
        return
    if array_path.exists():
        if not done_path.exists():
            raise FileExistsError(f"Incomplete cache lacks progress file: {array_path}")
        array = np.lib.format.open_memmap(array_path, mode="r+")
        done = np.lib.format.open_memmap(done_path, mode="r+")
        if array.shape != (len(rows), 2, 2, 30, 800) or done.shape != (len(rows),):
            raise ValueError(f"Partial cache has unexpected shape: {part}")
    else:
        array = np.lib.format.open_memmap(array_path, mode="w+", dtype=np.float32,
                                          shape=(len(rows), 2, 2, 30, 800))
        done = np.lib.format.open_memmap(done_path, mode="w+", dtype=np.bool_,
                                         shape=(len(rows),))
        done[:] = False
    pending = ((index, rows[index], raw_root) for index in np.flatnonzero(~done))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for count, (index, value) in enumerate(executor.map(_job, pending, chunksize=2), start=1):
            array[index] = value
            done[index] = True
            if count % 100 == 0:
                array.flush(); done.flush()
                print(f"{part}: {int(done.sum())}/{len(done)}", flush=True)
    if not bool(done.all()):
        raise RuntimeError(f"Preprocessing incomplete: {part}")
    array.flush(); done.flush()
    del array, done
    write_json(meta_path, {"expected": expected, "array_sha256": sha(array_path)})
    done_path.unlink()
    print(f"Prepared {part}: {len(rows)} recordings", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True,
                        help="Root directory of the original raw CSI dataset")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "cache")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--parts", nargs="+", choices=PARTS, default=PARTS)
    args = parser.parse_args()
    for part in args.parts:
        prepare(part, args.raw_root, args.cache_dir, args.workers)


if __name__ == "__main__":
    main()
