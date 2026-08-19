"""
Classical model definitions for the HAR comparison.

Linear and kernel models (Logistic Regression, SVM) are wrapped in a pipeline
with StandardScaler, since they are sensitive to feature scale. Tree ensembles
(Random Forest, Gradient Boosting) are scale-invariant and used as-is.

Scalers are fit only on the training data inside each pipeline, so no test
information leaks into training.
"""

from __future__ import annotations

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def get_classical_models() -> dict:
    """Return the four classical models, keyed by display name.

    Random state is fixed everywhere for reproducibility.
    """
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=0),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=0, n_jobs=-1
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            random_state=0
        ),
        "SVM (RBF)": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=10, random_state=0),
        ),
    }
