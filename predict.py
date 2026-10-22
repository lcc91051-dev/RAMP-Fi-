"""RAMP-Fi inference from a raw .dat file or a preprocessed partition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import ROOT, CLASSES, Data
from model import PhaseNetwork, transform
from preprocess import process_file


SCENARIOS = ("room2_labeled", "room3_labeled")
NAMES = {number: name for name, number in CLASSES.items()}


def probabilities(model, array, device):
    """Average class probabilities of the two ratio streams once."""
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(array), 64):
            x = torch.from_numpy(np.asarray(array[start:start + 64], dtype=np.float32)).to(device)
            _, logits = model(transform(x, "phase_augment"))
            output.append(logits.softmax(-1).mean(1).cpu().numpy())
    return np.concatenate(output)


def load_model(path, device):
    model = PhaseNetwork().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
    model.eval()
    return model


def score(y_true, y_pred):
    accuracy = float((y_true == y_pred).mean())
    f1s = []
    for category in range(len(CLASSES)):
        tp = int(((y_true == category) & (y_pred == category)).sum())
        fp = int(((y_true != category) & (y_pred == category)).sum())
        fn = int(((y_true == category) & (y_pred != category)).sum())
        denominator = 2 * tp + fp + fn
        f1s.append(2 * tp / denominator if denominator else 0.0)
    return accuracy, float(np.mean(f1s))


def audit_label(row):
    """Decode held-out truth only after inference; never used during training."""
    if row["label"] is not None:
        return int(row["label"])
    gesture = row["file"].split("_room_", 1)[1].split("_", 1)[1].split("_user_", 1)[0]
    return CLASSES[gesture]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--raw-file", type=Path,
                        help="Predict one original Intel 5300 .dat recording")
    parser.add_argument("--partition", choices=("validation", "unlabeled_train", "target_test"),
                        help="Predict one cache partition from the frozen split")
    parser.add_argument("--evaluate", action="store_true",
                        help="Print accuracy/macro-F1 when labels are available")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "cache")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if (args.raw_file is None) == (args.partition is None):
        parser.error("Specify exactly one of --raw-file or --partition")
    weight = args.weights
    if weight is None:
        weight = ROOT / "weights" / ("room2_labeled_seed43.pth" if args.scenario == SCENARIOS[0]
                                   else "room3_labeled_seed43.pth")
    model = load_model(weight, args.device)
    if args.raw_file is not None:
        x = process_file(args.raw_file)[None]
        probability = probabilities(model, x, args.device)[0]
        label = int(probability.argmax())
        print(json.dumps({"predicted_class": NAMES[label], "class_id": label,
                          "probabilities": {NAMES[i]: float(p) for i, p in enumerate(probability)}},
                         ensure_ascii=False, indent=2))
        return
    data = Data(args.scenario, allow_target=args.partition == "target_test",
                cache_dir=args.cache_dir, device=args.device)
    predictions = []
    for start in range(0, len(data.parts[args.partition]), 64):
        ids = range(start, min(start + 64, len(data.parts[args.partition])))
        predictions.append(probabilities(model, data.numpy(args.partition, ids, ref=1),
                                         args.device))
    probability = np.concatenate(predictions)
    chosen = probability.argmax(axis=1)
    result = {"scenario": args.scenario, "partition": args.partition,
              "n": len(chosen), "reference": "ranked_stable_antenna", "streams": 2,
              "weights": str(weight)}
    if args.evaluate:
        labels = np.asarray([audit_label(row) for row in data.parts[args.partition]],
                            dtype=np.int64)
        accuracy, macro_f1 = score(labels, chosen)
        result.update(accuracy_pct=100 * accuracy, macro_f1_pct=100 * macro_f1)
    else:
        result["predicted_class_ids"] = chosen.tolist()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
