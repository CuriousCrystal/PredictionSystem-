"""
Model evaluation: accuracy, classification reports, confusion matrix,
and cross-validation.
"""

import logging
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

from src.config import CV_FOLDS

logger = logging.getLogger(__name__)


def evaluate_model(
    name: str,
    model: Any,
    X_test,
    y_test,
    class_names: list[str] | None = None,
) -> tuple[np.ndarray, float]:
    """
    Evaluate a trained model/pipeline and print metrics.

    Args:
        name: Display name of the model.
        model: Trained sklearn estimator or Pipeline.
        X_test: Test features (raw DataFrame for Pipelines).
        y_test: Encoded test labels.
        class_names: Optional list of class label names for the report.

    Returns:
        Tuple of (predictions array, accuracy float).
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n{'-' * 50}")
    print(f"  {name}")
    print(f"{'-' * 50}")
    print(f"  Accuracy: {acc:.4f} ({acc * 100:.1f}%)")
    print(f"\n{classification_report(y_test, y_pred, target_names=class_names)}")

    return y_pred, acc


def compute_confusion_matrix(
    y_test: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """
    Compute the confusion matrix from pre-computed predictions.

    Unlike the previous ``get_confusion_matrix`` this accepts ``y_pred``
    directly, avoiding a redundant ``model.predict()`` call.

    Args:
        y_test: True labels.
        y_pred: Predicted labels.

    Returns:
        Confusion matrix as a numpy array.
    """
    return confusion_matrix(y_test, y_pred)


def cross_validate_model(
    name: str,
    model: Any,
    X,
    y,
    cv: int = CV_FOLDS,
) -> np.ndarray:
    """
    Perform k-fold cross-validation and print results.

    When ``model`` is a Pipeline, preprocessing is re-fit inside each fold
    so there is no data leakage.

    Args:
        name: Display name of the model.
        model: An sklearn estimator or Pipeline (cloned internally).
        X: Full feature set (raw DataFrame for Pipelines).
        y: Full encoded label set.
        cv: Number of folds.

    Returns:
        Array of per-fold accuracy scores.
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(
        f"  {name:25s} | CV Mean: {scores.mean():.4f} +/- {scores.std():.4f}  "
        f"| Folds: {np.round(scores, 3)}"
    )
    return scores
