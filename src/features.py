"""
Feature extraction from raw inertial signal windows.

The UCI HAR dataset ships 561 precomputed features. This module builds a smaller,
transparent feature set from the raw (n_windows, 128, 9) signals so we understand
where such features come from and can compare a hand-built set against the
official one.

Two families of features are computed per channel:

  Time domain      - mean, std, min, max, median, range, energy, rms, iqr,
                     mean absolute deviation, mean-crossing rate, skewness,
                     kurtosis
  Frequency domain - dominant frequency, spectral energy, spectral entropy,
                     spectral centroid

Plus cross-axis correlations between the three axes of each sensor triad.

The sampling rate is 50 Hz and each window is 128 samples (2.56 s).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew

from .data_loader import SIGNALS

SAMPLING_RATE = 50  # Hz


def _time_domain(x: np.ndarray) -> dict[str, float]:
    """Time-domain features for a single 1-D signal window."""
    n = len(x)
    mean = float(np.mean(x))
    energy = float(np.sum(x ** 2) / n)
    # Mean-crossing rate: how often the signal crosses its own mean.
    centered = x - mean
    crossings = np.sum(np.diff(np.sign(centered)) != 0)
    return {
        "mean": mean,
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "median": float(np.median(x)),
        "range": float(np.max(x) - np.min(x)),
        "energy": energy,
        "rms": float(np.sqrt(np.mean(x ** 2))),
        "iqr": float(np.percentile(x, 75) - np.percentile(x, 25)),
        "mad": float(np.mean(np.abs(x - mean))),
        "mcr": float(crossings / n),
        "skew": float(skew(x)),
        "kurtosis": float(kurtosis(x)),
    }


def _freq_domain(x: np.ndarray) -> dict[str, float]:
    """Frequency-domain features via the real FFT of a single window."""
    n = len(x)
    fft_mag = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLING_RATE)

    # Ignore the DC component (index 0) for dominant-frequency style features.
    mag = fft_mag[1:]
    f = freqs[1:]
    total = np.sum(mag) + 1e-12

    dominant = float(f[np.argmax(mag)]) if len(mag) else 0.0
    # Spectral entropy: entropy of the normalised magnitude spectrum.
    p = mag / total
    spectral_entropy = float(-np.sum(p * np.log2(p + 1e-12)))
    # Spectral centroid: magnitude-weighted mean frequency.
    centroid = float(np.sum(f * mag) / total)

    return {
        "dom_freq": dominant,
        "spec_energy": float(np.sum(mag ** 2) / n),
        "spec_entropy": spectral_entropy,
        "spec_centroid": centroid,
    }


def extract_window_features(window: np.ndarray) -> dict[str, float]:
    """Extract all features for one window.

    window: (128, 9) array, channel order = data_loader.SIGNALS.
    Returns an ordered dict of {feature_name: value}.
    """
    feats: dict[str, float] = {}

    # Per-channel time and frequency features.
    for ch_idx, ch_name in enumerate(SIGNALS):
        signal = window[:, ch_idx]
        for fname, fval in _time_domain(signal).items():
            feats[f"{ch_name}_{fname}"] = fval
        for fname, fval in _freq_domain(signal).items():
            feats[f"{ch_name}_{fname}"] = fval

    # Cross-axis correlations within each sensor triad.
    triads = {
        "body_acc": ("body_acc_x", "body_acc_y", "body_acc_z"),
        "body_gyro": ("body_gyro_x", "body_gyro_y", "body_gyro_z"),
        "total_acc": ("total_acc_x", "total_acc_y", "total_acc_z"),
    }
    idx = {name: i for i, name in enumerate(SIGNALS)}
    for triad, (ax, ay, az) in triads.items():
        x, y, z = window[:, idx[ax]], window[:, idx[ay]], window[:, idx[az]]
        feats[f"{triad}_corr_xy"] = _safe_corr(x, y)
        feats[f"{triad}_corr_xz"] = _safe_corr(x, z)
        feats[f"{triad}_corr_yz"] = _safe_corr(y, z)

    return feats


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation that returns 0 for a flat (zero-variance) signal."""
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def extract_features_batch(
    X: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Extract features for a batch of windows.

    X: (n_windows, 128, 9) raw signals.
    Returns (features, feature_names):
        features      -> (n_windows, n_features)
        feature_names -> list of length n_features
    """
    rows = []
    feature_names: list[str] | None = None
    for i in range(X.shape[0]):
        feats = extract_window_features(X[i])
        if feature_names is None:
            feature_names = list(feats.keys())
        rows.append(list(feats.values()))
    return np.asarray(rows, dtype=float), feature_names


if __name__ == "__main__":
    # Smoke test on a handful of windows.
    from .data_loader import load_raw_signals

    X, y, _ = load_raw_signals("train")
    feats, names = extract_features_batch(X[:50])
    print(f"Extracted {feats.shape[1]} features from {feats.shape[0]} windows")
    print(f"First 8 feature names: {names[:8]}")
    print(f"Any NaNs: {np.isnan(feats).any()}")
