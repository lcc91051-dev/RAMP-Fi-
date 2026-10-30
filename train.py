"""Train the complete RAMP-Fi model on the fixed cross-room split."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from data import ROOT, Data, sha
from training import train_ramp_fi

SCENARIOS = ("room2_labeled", "room3_labeled")

def finished(folder: Path, seed: int, split_hash: str) -> bool:
    marker = folder / "complete.json"
    if not marker.exists():
        return False
    record = json.loads(marker.read_text(encoding="utf-8"))
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    expected = {"model": "RAMP-Fi", "seed": seed, "epochs": 80, "reference": 1,
                "distribution_alignment": False, "pseudo_label_rule": "direct_weak_view",
                "split_sha256": split_hash}
    if (any(config.get(key) != value for key, value in expected.items()) or
            record.get("model_type") != "RAMP-Fi" or
            record.get("checkpoint_sha256") != sha(folder / "best_model.pth")):
        raise RuntimeError(f"Completed model does not match the requested configuration: {folder}")
    return True

def fit(scenario: str, seed: int, cache_dir: Path, output_dir: Path) -> Path:
    if scenario not in SCENARIOS:
        raise ValueError(scenario)
    split_hash = sha(cache_dir / "split.json")
    folder = output_dir / scenario / f"seed_{seed}" / "ramp_fi"
    if finished(folder, seed, split_hash):
        print(f"RAMP-Fi {scenario} seed={seed}: already complete", flush=True)
        return folder / "best_model.pth"
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires an NVIDIA CUDA GPU, matching the reference runs")
    if folder.exists() and any(folder.iterdir()):
        raise RuntimeError(f"Incomplete training has no optimizer resume state; preserve it and use a new output directory: {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    print(f"RAMP-Fi {scenario} seed={seed}: training for 80 epochs", flush=True)
    data = Data(scenario, cache_dir=cache_dir)
    try:
        train_ramp_fi(data, folder, seed, ref=1, epochs=80)
    finally:
        del data
        torch.cuda.empty_cache()
    if not finished(folder, seed, split_hash):
        raise RuntimeError(f"Training did not complete: {folder}")
    return folder / "best_model.pth"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=(*SCENARIOS, "both"), default="both")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "cache")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    scenarios = SCENARIOS if args.scenario == "both" else (args.scenario,)
    for scenario in scenarios:
        for seed in dict.fromkeys(args.seeds):
            print(f"Checkpoint: {fit(scenario, seed, args.cache_dir, args.output_dir)}", flush=True)

if __name__ == "__main__":
    main()
