# RAMP-Fi

RAMP-Fi recognizes Wi-Fi CSI gestures using a labeled source room and an
unlabeled source room, with evaluation on a separate target room. This release
contains the complete model, preprocessing, training, and inference code.
The default reproduction command runs five random seeds for each of the two
source-room settings. Pretrained and verification checkpoints are not included.

## Model and training

For each recording, the temporal mean amplitude divided by its temporal
variance (with a numerical stabilizer) is averaged across subcarriers. The
highest-scoring receive antenna becomes the shared denominator. The other
two antennas are ordered by decreasing score and form two ratio streams.
An 80 ms temporal difference is followed by Fourier resampling to 800 frames
and amplitude/phase extraction over 30 subcarriers.

RAMP-Fi trains for 80 epochs. Each ratio stream is processed independently
with shared ResNet-18 parameters. Labeled recordings supervise both weak and
strong augmentations. On unlabeled recordings, each weak prediction supplies
a hard pseudo-label and a confidence mask; the corresponding strong
prediction receives the masked classification loss. The threshold is 0.9
times the mean maximum probability of the labeled weak predictions.
Probabilities are used directly, without distribution alignment. The masked
loss is averaged over all unlabeled stream inputs, including rejected inputs
with zero contribution. Its cosine ramp reaches weight 1 halfway through
training.

The optimizer is SGD with momentum 0.9, initial learning rate 0.03, and weight
decay 0.001 on matrix/tensor parameters. The learning rate follows
`0.03 * cos(acos(0.25) * progress)`. Each update draws 32 labeled and 96
unlabeled recordings; each recording contributes two streams. Weak
augmentation uses edge-padded time shifts and amplitude gain. Strong
augmentation additionally uses noise and time/subcarrier masks. The code
retains the labeled/mixed-batch logit interpolation and BatchNorm buffer
restoration used in the reference training implementation.

Labeled-source validation accuracy selects the checkpoint (the latest epoch
wins ties). At inference, the two streams' class probabilities are averaged.
Target-room data are not used for training or checkpoint selection.

## Data

Obtain the original [Widar3.0 dataset](https://tns.thss.tsinghua.edu.cn/widar3.0/)
under its provider's terms. Extract these four CSI archives into a common root,
keeping their date/user subdirectories:

```
CSI/CSI_20181130.zip
CSI/CSI_20181204.zip
CSI/CSI_20181209.zip
CSI/CSI_20181211.zip
```

The loader reads original Intel 5300 `*-r2.dat` recordings, not third-party
pickle feature caches. `split.py` generates the frozen seed-42 protocol and
lists the two excluded recordings. The split is checked against the fingerprints
in `weights/split_fingerprints.json`.

| Setting | Labeled training | Labeled validation | Unlabeled source | Target evaluation |
|---|---:|---:|---:|---:|
| Room 2 labeled | Room 2: 1,300 | Room 2: 325 | Room 3: 2,998 | Room 1: 6,750 |
| Room 3 labeled | Room 3: 2,398 | Room 3: 600 | Room 2: 1,625 | Room 1: 6,750 |

Unlabeled and target labels are masked in the training loader. Both settings
use Room 1 as the target; that room has been used in prior method development,
so these evaluations are not a new blind test or evidence covering every room.

## Environment

The reference environment is Python 3.12.14, PyTorch 2.14.0+cu130, NumPy 2.5.2,
and SciPy 1.18.1. Install CUDA-enabled PyTorch appropriate for your NVIDIA GPU,
then the dependencies:

```bash
python -m pip install -r requirements.txt
```

Training requires an NVIDIA CUDA GPU. Inference supports CPU and CUDA.
Seeds and deterministic cuDNN settings are fixed in code; bitwise-identical
retraining across different hardware or library versions is not guaranteed.
Training saves checkpoints and their SHA-256 hashes locally for subsequent
inference and integrity checks.

## Reproduce

Preprocess, train both settings for all five seeds, and evaluate the target
room after the requested fits finish:

```bash
python reproduce.py --raw-root data/widar3g6d
```

For one setting and seed:

```bash
python reproduce.py --raw-root data/widar3g6d --scenario room2_labeled --seeds 43
```

Prepare features or train from an existing cache separately:

```bash
python preprocess.py --raw-root data/widar3g6d
python train.py --scenario both
```

Outputs are saved under `outputs/<setting>/seed_<seed>/ramp_fi/`. Completed
fits are verified and skipped. Interrupted training without optimizer state
is preserved; use a new output directory to restart that fit.

## Inference with a trained checkpoint

Checkpoints are generated locally by training and are not distributed in this
repository. `weights/split_fingerprints.json` contains only the frozen dataset
split fingerprints; it is not a model checkpoint. Pass the desired
`outputs/<setting>/seed_<seed>/ramp_fi/best_model.pth` explicitly with `--weights`.

Predict one raw recording:

```bash
python predict.py --scenario room3_labeled --seed 43 --weights outputs/room3_labeled/seed_43/ramp_fi/best_model.pth --raw-file path/to/recording-r2.dat
```

Evaluate a locally trained checkpoint after preprocessing:

```bash
python predict.py --scenario room2_labeled --seed 42 --weights outputs/room2_labeled/seed_42/ramp_fi/best_model.pth --partition target_test --evaluate
```

Use `--weights path/to/best_model.pth` for a newly trained checkpoint and
`--device cpu` for CPU inference. Class IDs follow the six-class mapping in
`data.py`.

## Implementation notes and attribution

The Intel 5300 parser is adapted from
[UniCrossFi](https://github.com/KaoruMiyazono/UnicrossFi).
The relative-confidence filtering and labeled/mixed-batch logit interpolation
follow ideas from [AdaMatch](https://arxiv.org/abs/2106.04732), adapted here to
CSI ratio streams and CSI augmentations. The residual encoder follows
[ResNet](https://arxiv.org/abs/1512.03385).

Code is supplied under the MIT license. Dataset recordings, generated caches,
training logs, exploratory analysis files, and model checkpoints are not
distributed.
