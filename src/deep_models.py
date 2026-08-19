"""
Deep learning models for HAR on raw inertial signals.

Kept separate from src/models.py so the classical track never needs to import
torch. Two architectures learn features directly from the raw (128, 9) windows,
with no hand-engineered features:

  CNN1D  - three 1D-convolution blocks over the time axis, then global pooling.
  BiLSTM - a bidirectional LSTM over the 128 timesteps.

Both are trained with a subject-independent validation split carved out of the
training subjects, and early stopping on validation macro-F1.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class CNN1D(nn.Module):
    """1D CNN over the time axis. Expects input (batch, channels=9, length=128)."""

    def __init__(self, n_channels: int = 9, n_classes: int = 6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),      # 128 -> 64
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),     # 64 -> 32
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),                             # global pool
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class BiLSTM(nn.Module):
    """Bidirectional LSTM. Expects input (batch, length=128, features=9)."""

    def __init__(self, n_features: int = 9, hidden: int = 64, n_classes: int = 6):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden, num_layers=1,
            batch_first=True, bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x)          # (batch, length, hidden*2)
        last = out[:, -1, :]           # final timestep, both directions
        return self.classifier(last)


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
def standardize_signals(X_train, X_test):
    """Per-channel standardisation, using training statistics only.

    X shape: (n_windows, 128, 9). Mean/std computed over windows and timesteps
    for each of the 9 channels, then applied to both splits.
    """
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std


def subject_val_split(X, y, subjects, val_subjects):
    """Split a training set into train/val by subject id (no subject overlap)."""
    val_mask = np.isin(subjects, val_subjects)
    return X[~val_mask], y[~val_mask], X[val_mask], y[val_mask]


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _loaders(X_tr, y_tr, X_val, y_val, layout, batch_size=64):
    """Build train/val DataLoaders. layout='cnn' -> (N,9,128); 'lstm' -> (N,128,9)."""
    def to_tensor(X):
        t = torch.tensor(X, dtype=torch.float32)
        return t.permute(0, 2, 1) if layout == "cnn" else t

    # Labels 1..6 -> 0..5 for cross-entropy.
    train_ds = TensorDataset(to_tensor(X_tr), torch.tensor(y_tr - 1, dtype=torch.long))
    val_ds = TensorDataset(to_tensor(X_val), torch.tensor(y_val - 1, dtype=torch.long))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=256, shuffle=False),
    )


def train_model(
    model, X_tr, y_tr, X_val, y_val, layout,
    epochs=60, lr=1e-3, patience=8, verbose=True,
):
    """Train with Adam and early stopping on validation macro-F1.

    Returns (model_with_best_weights, history) where history is a list of
    per-epoch (train_loss, val_f1).
    """
    train_loader, val_loader = _loaders(X_tr, y_tr, X_val, y_val, layout)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_f1, best_state, wait, history = -1.0, None, 0, []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
        train_loss = total_loss / len(train_loader.dataset)

        val_f1 = _macro_f1(model, val_loader)
        history.append((train_loss, val_f1))
        if verbose:
            print(f"  epoch {epoch+1:2d}  train_loss={train_loss:.3f}  val_F1={val_f1:.3f}")

        if val_f1 > best_f1:
            best_f1, best_state, wait = val_f1, _clone_state(model), 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"  early stop at epoch {epoch+1} (best val_F1={best_f1:.3f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predict(model, X, layout, batch_size=256):
    """Return integer predictions in 1..6 for input windows X."""
    model.eval()
    t = torch.tensor(X, dtype=torch.float32)
    if layout == "cnn":
        t = t.permute(0, 2, 1)
    preds = []
    for i in range(0, len(t), batch_size):
        logits = model(t[i:i + batch_size])
        preds.append(logits.argmax(dim=1).numpy())
    return np.concatenate(preds) + 1  # back to 1..6


@torch.no_grad()
def _macro_f1(model, loader):
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        ps.append(model(xb).argmax(dim=1).numpy())
        ys.append(yb.numpy())
    return f1_score(np.concatenate(ys), np.concatenate(ps), average="macro")


def _clone_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}
