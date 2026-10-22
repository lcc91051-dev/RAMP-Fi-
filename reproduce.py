"""One-command preprocessing, seed-43 training, and held-out evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from data import ROOT
from preprocess import PARTS, prepare
from train import SCENARIOS, fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True,
                        help="Root directory of the separately downloaded raw CSI dataset")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "cache")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--scenario", choices=(*SCENARIOS, "both"), default="both")
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    for part in PARTS:
        if part == "target_test":
            continue
        prepare(part, args.raw_root, args.cache_dir, args.workers)
    scenarios = SCENARIOS if args.scenario == "both" else (args.scenario,)
    checkpoints = {}
    for scenario in scenarios:
        checkpoints[scenario] = fit(scenario, args.seed, args.cache_dir, args.output_dir)
    # Target recordings and their encoded labels are accessed only after all fits.
    prepare("target_test", args.raw_root, args.cache_dir, args.workers)
    for scenario in scenarios:
        checkpoint = checkpoints[scenario]
        subprocess.run([sys.executable, str(ROOT / "predict.py"),
                        "--scenario", scenario, "--weights", str(checkpoint),
                        "--partition", "target_test", "--evaluate",
                        "--cache-dir", str(args.cache_dir)], check=True)


if __name__ == "__main__":
    main()
