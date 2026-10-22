"""Train RAMP-Fi from the fixed cross-room split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data import ROOT, Data, sha
from stages import train_online, train_phase


SCENARIOS = ("room2_labeled", "room3_labeled")
PHASES = ("semantic_learning", "paired_view_learning")


def finished(folder: Path, phase: str, initial: Path | None = None) -> bool:
    marker = folder / "complete.json"
    if not marker.exists():
        return False
    record = json.loads(marker.read_text(encoding="utf-8"))
    if record["reference"] != 1 or sha(folder / "best_model.pth") != record["checkpoint_sha256"]:
        raise RuntimeError(f"Invalid completed stage: {folder}")
    config_path = folder / "config.json"
    if not config_path.exists():
        raise RuntimeError(f"Missing configuration for completed stage: {folder}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if phase == "semantic_learning":
        if (config.get("pseudo_label_rule") != "direct_weak_view" or
                record.get("pseudo_label_rule") != "direct_weak_view"):
            raise RuntimeError(f"Stage A uses a different pseudo-label rule: {folder}")
    elif phase == "paired_view_learning":
        if initial is None or config.get("initial_semantic_sha256") != sha(initial / "best_model.pth"):
            raise RuntimeError(f"Stage B does not match its Stage A checkpoint: {folder}")
    else:
        raise ValueError(phase)
    return True


def fit(scenario: str, seed: int, cache_dir: Path, output_dir: Path) -> Path:
    if scenario not in SCENARIOS:
        raise ValueError(scenario)
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires an NVIDIA CUDA GPU, matching the original runs")
    base = output_dir / scenario / f"seed_{seed}"
    for phase in PHASES:
        folder = base / phase
        initial = base / "semantic_learning"
        if finished(folder, phase, initial):
            print(f"{scenario} seed={seed} {phase}: already complete", flush=True)
            continue
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "best_model.pth").exists():
            raise RuntimeError(f"Partial {phase} fit has no optimizer resume state: {folder}")
        if phase == "paired_view_learning" and not finished(initial, "semantic_learning"):
            raise RuntimeError(f"Semantic learning is incomplete for {scenario} seed={seed}")
        print(f"{scenario} seed={seed} {phase}: training", flush=True)
        data = Data(scenario, cache_dir=cache_dir)
        try:
            if phase == "semantic_learning":
                train_online(data, folder, seed, ref=1, epochs=80)
            elif phase == "paired_view_learning":
                train_phase(data, folder, seed, base / "semantic_learning", ref=1,
                            epochs=40)
        finally:
            del data
            torch.cuda.empty_cache()
        if not finished(folder, phase, initial):
            raise RuntimeError(f"Training phase did not complete: {folder}")
    return base / "paired_view_learning" / "best_model.pth"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=(*SCENARIOS, "both"), default="both")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "cache")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    scenarios = SCENARIOS if args.scenario == "both" else (args.scenario,)
    for scenario in scenarios:
        print(f"Final checkpoint: {fit(scenario, args.seed, args.cache_dir, args.output_dir)}",
              flush=True)


if __name__ == "__main__":
    main()
