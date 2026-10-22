"""Dataset access for the generated cross-room split and CSI cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
CLASSES = {
    "Push&Pull": 0,
    "Sweep": 1,
    "Clap": 2,
    "Slide": 3,
    "DrawO(Horizontal)": 4,
    "DrawZigzag(Horizontal)": 5,
}


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Data:
    def __init__(self, scenario, allow_target=False, cache_dir=None, device=None):
        self.cache_dir = Path(cache_dir) if cache_dir is not None else ROOT / "data" / "cache"
        self.path = self.cache_dir / "split.json"
        self.manifest = json.loads(self.path.read_text(encoding="utf-8"))[scenario]
        self.fingerprint = sha(self.path)
        self.scenario = scenario
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.keys = ("labeled_train", "validation", "unlabeled_train") + (("target_test",) if allow_target else ())
        self.parts = {key: self.manifest[key] for key in self.keys}
        self.maps = {}
        for rows in self.parts.values():
            for row in rows:
                name = row["cache_partition"]
                if name not in self.maps:
                    array = np.load(self.cache_dir / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                    if array.shape[1:] not in ((2, 2, 30, 800), (3, 2, 2, 30, 800)):
                        raise ValueError(f"Unexpected cache shape for {name}: {array.shape}")
                    self.maps[name] = array
        self.nl = len(self.parts["labeled_train"])
        self.nu = len(self.parts["unlabeled_train"])
        assert all(row["label"] is None for row in self.parts["unlabeled_train"])
        self.y_l = torch.tensor([row["label"] for row in self.parts["labeled_train"]], device=self.device)
        self.y_v = torch.tensor([row["label"] for row in self.parts["validation"]], device=self.device)
        self.y_u = torch.full((self.nu,), -1, device=self.device, dtype=torch.long)
        self.users_l = [row["user"] for row in self.parts["labeled_train"]]

    def numpy(self, part, ids=None, ref=None):
        if ref is not None and ref != 1:
            raise ValueError("This release contains only the ranked stable reference")
        rows = self.parts[part]
        if ids is not None:
            rows = [rows[int(index)] for index in ids]
        output = np.empty((len(rows), 2, 2, 30, 800), dtype=np.float32)
        for index, row in enumerate(rows):
            value = self.maps[row["cache_partition"]][row["cache_index"]]
            output[index] = value[1] if value.ndim == 5 else value
        return output

    def load_single(self, ref=1):
        if ref != 1:
            raise ValueError("This release contains only the ranked stable reference")
        self.ref = ref
        self.x_l = torch.from_numpy(self.numpy("labeled_train")).to(self.device)
        self.x_v = torch.from_numpy(self.numpy("validation")).to(self.device)
        self.x_u = torch.from_numpy(self.numpy("unlabeled_train")).to(self.device)
        return self
