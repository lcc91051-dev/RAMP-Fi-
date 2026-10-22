# RAMP-Fi

RAMP-Fi recognizes WiFi CSI gestures in an unseen room using one labeled
source room and a second, unlabeled source room. This repository provides the
preprocessing code, model, training procedure, and seed-43 checkpoints. See
the paper for the full analysis.

## Method

Each recording is converted into two synchronized antenna-ratio streams. We
rank the three receive antennas by an amplitude-stability score, use the
highest-ranked antenna as the ratio denominator, take an 80-ms temporal
difference, resample to 800 frames, and extract amplitude and phase from 30
subcarriers. A shared ResNet-18 processes the two streams.

Training has two stages. Stage A runs 80 epochs of labeled-room classification
and online pseudo-label learning on the unlabeled source room. Pseudo-labels
come directly from weak-view class probabilities and pass a relative confidence
filter. Stage B starts
from the selected Stage A weights and runs 40 epochs of labeled classification
and within-recording feature pairing across the two ratio streams, using both
original and phase-perturbed inputs. The pair loss uses labeled and unlabeled
source recordings. The labeled-room validation set selects each stage's
checkpoint. The final model is the selected Stage B checkpoint. At inference,
the two streams' class probabilities are averaged. The unseen room is used
only for evaluation.

## Dataset

Download the original [Widar3.0 dataset](https://tns.thss.tsinghua.edu.cn/widar3.0/).
The experiment uses the Widar3.0-G6D subset from these four CSI archives:

```
CSI/CSI_20181130.zip
CSI/CSI_20181204.zip
CSI/CSI_20181209.zip
CSI/CSI_20181211.zip
```

If needed, the dataset FTP endpoint is `166.111.80.127:40121` (username
`widarftp`, password `widar2019`).

Extract all four into one directory, for example `data/widar3g6d`. Keep the
date and user subdirectories created by the archives. The code reads the
original Intel 5300 `*-r2.dat` recordings directly. It selects the six
gestures, generates the fixed seed-42 room split from filenames, and writes
the processed CSI cache to `data/cache/` on the first run. The generated split
contains 1,300/325 labeled training/validation recordings from Room 2,
2,998 unlabeled Room 3 recordings, and 6,750 Room 1 evaluation recordings.
For the reverse direction, Room 3 has 2,398/600 labeled
training/validation recordings and Room 2 has 1,625 unlabeled recordings.
Two recordings omitted by the fixed experimental protocol are listed in
`split.py`.

Some Widar3.0-G6D repositories provide `utils/createWidar3g6d.py` to create
`widar3-g6_csi_domain_train_cache.pkl` and
`widar3-g6_csi_domain_test_cache.pkl`. Those pickle caches are **not inputs to
this implementation**; use the extracted `.dat` recordings instead.

## Environment and experiment settings

Use Python 3.12 and an NVIDIA GPU with CUDA-enabled PyTorch. The released
checkpoints were trained with PyTorch 2.14.0+cu130, NumPy 2.5.2, and SciPy
1.18.1. Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

The two supported settings are Room 2 labeled / Room 3 unlabeled / Room 1
evaluation and Room 3 labeled / Room 2 unlabeled / Room 1 evaluation. The
released checkpoints use seed 43. Data splitting uses seed 42, with 20% of
each labeled-room class reserved for validation. Training uses 80 + 40 epochs;
the exact optimizer, batch, augmentation, and loss settings are in
`stages.py` and `training_utils.py`.

## Run

Train both settings and evaluate on Room 1:

```bash
python reproduce.py --raw-root data/widar3g6d
```

To train only one setting, add `--scenario room2_labeled` or
`--scenario room3_labeled`. To prepare the data without training:

```bash
python preprocess.py --raw-root data/widar3g6d
```

To classify one raw recording with the released Room 3 labeled checkpoint:

```bash
python predict.py --scenario room3_labeled --raw-file path/to/recording-r2.dat
```

Use `--scenario room2_labeled` for the other checkpoint. After preprocessing,
evaluate either checkpoint on the held-out room with:

```bash
python predict.py --scenario room3_labeled --partition target_test --evaluate
```

The generated cache and training outputs are excluded from Git. Dataset
files must be obtained under the dataset provider's terms. Code is released
under the MIT license; the pretrained weights are supplied in `weights/`.

The Intel 5300 CSI parser and TD-CSI preprocessing build on the publicly
released [UniCrossFi/ARC-Fi implementation](https://github.com/KaoruMiyazono/UnicrossFi).
The Widar3.0 recordings are not redistributed in this repository.
