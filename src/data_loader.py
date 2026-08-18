"""
Data loading for the UCI HAR dataset.

Handles two representations of the same recordings:
  1. Engineered features  -> (n_windows, 561) precomputed feature vectors
  2. Raw inertial signals -> (n_windows, 128, 9) raw sensor windows

The train/test split shipped with the dataset is subject-independent:
21 subjects appear only in train, 9 appear only in test. We keep that split
intact so the model is always evaluated on people it has never seen.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import requests

# Outer download. Note the archive ships a zip-inside-a-zip.
DATA_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)

# Resolve paths relative to the repo root, not the caller's working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DATASET_DIR = DATA_DIR / "UCI HAR Dataset"

# The 9 raw inertial signal files, in a fixed channel order.
SIGNALS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]

# Integer label -> human-readable activity name.
ACTIVITY_LABELS = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}


def download_data(force: bool = False) -> Path:
    """Download and extract the dataset if it is not already present.

    Returns the path to the extracted 'UCI HAR Dataset' directory.
    """
    if DATASET_DIR.exists() and not force:
        return DATASET_DIR

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading UCI HAR dataset (~59 MB)...")
    resp = requests.get(DATA_URL, timeout=120)
    resp.raise_for_status()

    # The outer zip contains an inner 'UCI HAR Dataset.zip'; extract both.
    with zipfile.ZipFile(io.BytesIO(resp.content)) as outer:
        inner_name = next(
            n for n in outer.namelist() if n.endswith("UCI HAR Dataset.zip")
        )
        with outer.open(inner_name) as inner_file:
            with zipfile.ZipFile(io.BytesIO(inner_file.read())) as inner:
                inner.extractall(DATA_DIR)

    if not DATASET_DIR.exists():
        raise RuntimeError(
            f"Extraction finished but {DATASET_DIR} was not found. "
            "The archive layout may have changed."
        )
    print(f"Ready at {DATASET_DIR}")
    return DATASET_DIR


def load_features(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the precomputed 561-feature representation for one split.

    split: "train" or "test".
    Returns (X, y, subjects):
        X        -> (n_windows, 561) float
        y        -> (n_windows,) int in 1..6
        subjects -> (n_windows,) int subject id
    """
    _check_split(split)
    base = download_data() / split
    X = np.loadtxt(base / f"X_{split}.txt")
    y = np.loadtxt(base / f"y_{split}.txt").astype(int)
    subjects = np.loadtxt(base / f"subject_{split}.txt").astype(int)
    return X, y, subjects


def load_raw_signals(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the raw inertial signals for one split.

    split: "train" or "test".
    Returns (X, y, subjects):
        X        -> (n_windows, 128, 9) float, channel order = SIGNALS
        y        -> (n_windows,) int in 1..6
        subjects -> (n_windows,) int subject id
    """
    _check_split(split)
    base = download_data() / split
    signals_dir = base / "Inertial Signals"

    channels = []
    for name in SIGNALS:
        arr = np.loadtxt(signals_dir / f"{name}_{split}.txt")  # (n_windows, 128)
        channels.append(arr)
    # Stack along a new last axis -> (n_windows, 128, 9)
    X = np.stack(channels, axis=-1)

    y = np.loadtxt(base / f"y_{split}.txt").astype(int)
    subjects = np.loadtxt(base / f"subject_{split}.txt").astype(int)
    return X, y, subjects


def load_feature_names() -> list[str]:
    """Return the 561 feature names from features.txt, in order."""
    path = download_data() / "features.txt"
    names = []
    with open(path) as f:
        for line in f:
            # Each line is "<index> <name>"; keep the name.
            names.append(line.strip().split(" ", 1)[1])
    return names


def _check_split(split: str) -> None:
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")


if __name__ == "__main__":
    # Smoke test: download, load both representations, print a summary.
    Xf_tr, yf_tr, s_tr = load_features("train")
    Xf_te, yf_te, s_te = load_features("test")
    Xr_tr, _, _ = load_raw_signals("train")

    print("Feature representation:")
    print(f"  X_train {Xf_tr.shape}   X_test {Xf_te.shape}")
    print(f"  classes present: {sorted(set(yf_tr))}")
    print(f"  train subjects ({len(set(s_tr))}): {sorted(set(s_tr))}")
    print(f"  test  subjects ({len(set(s_te))}): {sorted(set(s_te))}")
    overlap = set(s_tr) & set(s_te)
    print(f"  subject overlap between train/test: {overlap or 'none'}")
    print("Raw-signal representation:")
    print(f"  X_train {Xr_tr.shape}  (windows, timesteps, channels)")
