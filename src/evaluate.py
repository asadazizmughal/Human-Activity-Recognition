"""
Evaluation utilities: fit/time a model, compute metrics, plot a confusion matrix.

Everything here reports on the subject-independent test set, so all numbers
reflect generalisation to people the model never trained on.
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)


def evaluate_model(model, X_train, y_train, X_test, y_test) -> dict:
    """Fit a model, time the fit, and score it on the test set.

    Returns a dict with accuracy, macro-F1, fit time (seconds), the fitted
    model, and its test-set predictions.
    """
    start = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - start

    preds = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "macro_f1": f1_score(y_test, preds, average="macro"),
        "fit_time_s": fit_time,
        "model": model,
        "preds": preds,
    }


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names: list[str],
    title: str = "Confusion matrix",
    normalize: bool = True,
    ax=None,
    save_path: str | None = None,
):
    """Plot a confusion matrix as an annotated heatmap.

    normalize=True shows row-normalised rates (recall per true class), which is
    the more readable view when classes are imbalanced.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # Annotate each cell; pick text colour for contrast against the fill.
    thresh = (cm.max() + cm.min()) / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            text = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(
                j, i, text, ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=9,
            )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    return ax
