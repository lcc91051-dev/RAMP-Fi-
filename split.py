"""Build the fixed two-room protocol directly from raw Widar CSI filenames.

The output is generated locally under data/cache and is not distributed with
the model. Only recording names from the labeled room are interpreted as
training labels; the other source room and Room 1 remain label-masked.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np


DATE_TO_ROOM = {"20181130": "room1", "20181204": "room2",
                "20181209": "room2", "20181211": "room3"}
GESTURES = ("Push&Pull", "Sweep", "Clap", "Slide",
            "DrawO(Horizontal)", "DrawZigzag(Horizontal)")
NAME = re.compile(r"^(user\d+)-(\d+)-(\d+)-(\d+)-(\d+)-r2\.dat$")
EXCLUDED = {
    # These two recordings were absent from the original valid CSI pool.
    "20181211/20181211/user7/user7-1-5-4-4-r2.dat",
    "20181211/20181211/user8/user8-3-3-3-5-r2.dat",
}
COUNTS = {"labeled_train": 1300, "validation": 325,
          "unlabeled_train": 2998, "target_test": 6750}


def recording_rows(raw_root: Path) -> list[dict]:
    raw_root = raw_root.resolve()
    if not raw_root.is_dir():
        raise NotADirectoryError(raw_root)
    rows = []
    for path in raw_root.rglob("*-r2.dat"):
        rel = path.relative_to(raw_root).as_posix()
        if rel in EXCLUDED:
            continue
        date = rel[:8]
        if date not in DATE_TO_ROOM:
            continue
        match = NAME.fullmatch(path.name)
        if match is None:
            continue
        user, gesture_id, location, orientation, repetition = match.groups()
        gesture_number = int(gesture_id)
        if not 1 <= gesture_number <= 6:
            continue
        room = DATE_TO_ROOM[date]
        name = (f"{date}_room_{room}_{GESTURES[gesture_number - 1]}_user_{user}"
                f"_location_{location}_orientation_{orientation}"
                f"_repetition_{repetition}_receiver_r2.npy")
        rows.append(dict(file=name, raw=rel, label=gesture_number - 1,
                         room=room, user=user))
    rows.sort(key=lambda row: row["file"])
    if len(rows) != 11373 or len({row["file"] for row in rows}) != len(rows):
        raise ValueError(f"Expected 11,373 unique recordings; found {len(rows)}")
    return rows


def _room2_split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    groups: dict[int, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["label"], []).append(row)
    rng = random.Random(42)
    train, validation = [], []
    for group in groups.values():
        indices = list(range(len(group)))
        rng.shuffle(indices)
        selected = set(indices[:round(0.2 * len(indices))])
        for index, row in enumerate(group):
            (validation if index in selected else train).append(row)
    return (sorted(train, key=lambda row: row["file"]),
            sorted(validation, key=lambda row: row["file"]))


def _room3_split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(42)
    train, validation = [], []
    for category in range(6):
        group = [row for row in rows if row["label"] == category]
        order = rng.permutation(len(group))
        count = round(0.2 * len(group))
        validation.extend(group[int(index)] for index in order[:count])
        train.extend(group[int(index)] for index in order[count:])
    return train, validation


def build(raw_root: Path) -> dict:
    rows = recording_rows(Path(raw_root))
    room2 = [row for row in rows if row["room"] == "room2"]
    room3 = [row for row in rows if row["room"] == "room3"]
    room1 = [row for row in rows if row["room"] == "room1"]
    train2, validation2 = _room2_split(room2)
    parts = {"labeled_train": train2, "validation": validation2,
             "unlabeled_train": [{**row, "label": None} for row in room3],
             "target_test": [{**row, "label": None} for row in room1]}
    if {key: len(value) for key, value in parts.items()} != COUNTS:
        raise ValueError("Dataset partition counts differ from the published protocol")
    indexed = {key: [{**row, "cache_partition": key, "cache_index": index}
                     for index, row in enumerate(value)] for key, value in parts.items()}
    source2 = indexed["labeled_train"] + indexed["validation"]
    source3 = [{**row, "label": GESTURES.index(row["file"].split("_room_", 1)[1]
                     .split("_", 1)[1].split("_user_", 1)[0])}
               for row in indexed["unlabeled_train"]]
    train3, validation3 = _room3_split(source3)
    common = {"split_seed": 42, "receiver": "r2", "target_room": "room1"}
    return {
        "room2_labeled": {**common, "labeled_room": "room2", "unlabeled_room": "room3",
                          **indexed},
        "room3_labeled": {**common, "labeled_room": "room3", "unlabeled_room": "room2",
                          "labeled_train": train3, "validation": validation3,
                          "unlabeled_train": [{**row, "label": None} for row in source2],
                          "target_test": indexed["target_test"]},
    }


def ensure(raw_root: Path, cache_dir: Path) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "split.json"
    result = build(raw_root)
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != result:
            raise ValueError(f"Existing split differs from the raw dataset: {path}")
    else:
        path.write_text(encoded, encoding="utf-8")
    return path
