# Conditional WGAN-GP for Synthetic Ictal EEG Generation

Code accompanying the paper *"Conditional WGAN-GP with Temporal Smoothness
Regularisation for Synthetic EEG Generation in Epileptic Seizure Detection:
Internal Benchmark and CHB-MIT External Validation."*

## What this does

Trains a class-conditional WGAN-GP (with temporal smoothness + spectral
losses, projection conditioning, and EMA weight averaging) to synthesize
ictal EEG segments, then benchmarks classical ML and deep learning seizure
classifiers with/without this synthetic augmentation, on:

1. **Bonn University Epileptic Seizure Recognition dataset** (internal, 80/20 split)
2. **CHB-MIT Scalp EEG database** (external validation, 5 patients)

## Feature space design (read this before modifying anything)

Every downstream model — classical ML *and* deep learning — is trained and
evaluated in a single consistent representation: the `MinMaxScaler` fit
exclusively on Bonn **training** data, range `[-1, 1]`. CHB-MIT windows are
extracted in raw µV amplitude and transformed with that same fitted scaler
(`transform` only, never re-fit). This is documented at the top of
`cwgan_gp_eeg_pipeline.py` and must match the paper's Section III.A.2 —
if you change one, change the other in the same commit.

## Data

Neither dataset is redistributed in this repository.

- **Bonn dataset**: public. See Andrzejak et al. (2001), *Phys. Rev. E* 64,
  061907. Available via the UCI Machine Learning Repository.
- **CHB-MIT Scalp EEG**: public. See Shoeb & Guttag (2010). Available via
  [PhysioNet](https://physionet.org/content/chbmit/).

Download both and point the environment variables below at your local
copies before running:

```bash
export EEG_DATA_DIR=/path/to/bonn         # must contain data.csv
export EEG_CHBMIT_DIR=/path/to/chbmit     # must contain chb01_03.edf, chb03_01.edf, etc.
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python cwgan_gp_eeg_pipeline.py
```

This single run does everything: data loading → GAN training → classical
ML + deep learning benchmarking → all evaluation metrics → all ~10 figures
→ prints the full deep learning results table (Precision/Recall/F1/AUC on
both Bonn and CHB-MIT) → copies figures and the fitted scaler to
`EEG_RESULTS_DIR` if set (Colab: defaults to
`/content/drive/MyDrive/eeg_project/results` when Drive is mounted; safe
no-op otherwise). There is no separate figure-generation step to remember
to run afterward.

Random seed is fixed (`SEED = 42`) for the GAN, train/test split, and all
classical ML models. Deep learning results in the current script are
single-run; if you rerun with different seeds for the camera-ready version,
report mean ± std across seeds rather than a single run (see paper
Discussion/Limitations).

## Repository structure

```
cwgan_gp_eeg_pipeline.py   # full pipeline: data loading -> GAN training -> classical ML + DL benchmarking -> evaluation
requirements.txt
README.md
```

## Runtime

GAN training: up to 10,000 steps with MMD-based early stopping (~6,000 steps
typical). Deep learning benchmark: 3 architectures × 2 conditions
(baseline/augmented), ~100 epochs each with early stopping on validation
recall. Expect a single-GPU run to take several hours end-to-end.

## License / citation

MIT License
