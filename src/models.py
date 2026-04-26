"""
Model definitions, training, and hyperparameter tuning.

Every model is wrapped in an ``sklearn.pipeline.Pipeline`` that chains
``ColumnTransformer`` (preprocessing) → classifier.  This ensures:
  - No data leakage during cross-validation
  - Consistent preprocessing for training, evaluation, and prediction
"""

import logging
from typing import Any

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import RANDOM_STATE, CV_FOLDS
from src.preprocessing import build_preprocessor

logger = logging.getLogger(__name__)


def _make_pipeline(classifier: Any) -> Pipeline:
    """Wrap a classifier in a preprocessing Pipeline."""
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", classifier),
    ])


def get_pipelines() -> dict[str, Pipeline]:
    """
    Return a dictionary of classifier name → untrained Pipeline.

    Each Pipeline includes a fresh ColumnTransformer (preprocessor) so that
    models can be trained and cross-validated independently.

    Returns:
        dict mapping model names to sklearn Pipeline instances.
    """
    return {
        "Decision Tree": _make_pipeline(
            DecisionTreeClassifier(random_state=RANDOM_STATE)
        ),
        "Random Forest": _make_pipeline(
            RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100)
        ),
        "Gradient Boosting": _make_pipeline(
            GradientBoostingClassifier(random_state=RANDOM_STATE, n_estimators=100)
        ),
        "Logistic Regression": _make_pipeline(
            LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
        ),
        "Perceptron": _make_pipeline(
            Perceptron(random_state=RANDOM_STATE, max_iter=1000)
        ),
        "MLP Neural Network": _make_pipeline(
            MLPClassifier(
                activation="logistic",
                random_state=RANDOM_STATE,
                max_iter=1000,
                hidden_layer_sizes=(100, 50),
            )
        ),
        "SVM": _make_pipeline(
            SVC(random_state=RANDOM_STATE, kernel="rbf")
        ),
        "KNN": _make_pipeline(
            KNeighborsClassifier(n_neighbors=5)
        ),
    }


def train_model(pipeline: Pipeline, X_train, y_train) -> Pipeline:
    """
    Fit a pipeline (preprocessor + classifier) on training data.

    Args:
        pipeline: An untrained sklearn Pipeline.
        X_train: Raw training features (DataFrame with mixed types).
        y_train: Encoded training labels.

    Returns:
        The fitted Pipeline.
    """
    pipeline.fit(X_train, y_train)
    return pipeline


def tune_random_forest(X_train, y_train) -> tuple[Pipeline, dict, float]:
    """
    Run GridSearchCV on a Random Forest pipeline.

    The search grid includes ``n_estimators``, ``max_depth``,
    ``min_samples_split``, and ``max_features``.

    Args:
        X_train: Raw training features.
        y_train: Encoded training labels.

    Returns:
        Tuple of (best Pipeline, best params dict, best CV score).
    """
    pipeline = _make_pipeline(
        RandomForestClassifier(random_state=RANDOM_STATE)
    )

    param_grid = {
        "classifier__n_estimators": [50, 100, 200, 300],
        "classifier__max_depth": [3, 5, 10, 15, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__max_features": ["sqrt", "log2", None],
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=CV_FOLDS,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)

    print(f"\n  Best Random Forest Params: {grid.best_params_}")
    print(f"  Best CV Score: {grid.best_score_:.4f}")

    return grid.best_estimator_, grid.best_params_, grid.best_score_


def tune_gradient_boosting(X_train, y_train) -> tuple[Pipeline, dict, float]:
    """
    Run GridSearchCV on a Gradient Boosting pipeline.

    Args:
        X_train: Raw training features.
        y_train: Encoded training labels.

    Returns:
        Tuple of (best Pipeline, best params dict, best CV score).
    """
    pipeline = _make_pipeline(
        GradientBoostingClassifier(random_state=RANDOM_STATE)
    )

    param_grid = {
        "classifier__n_estimators": [50, 100, 200, 300],
        "classifier__max_depth": [3, 5, 7, 10],
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "classifier__subsample": [0.8, 1.0],
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=CV_FOLDS,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)

    print(f"\n  Best Gradient Boosting Params: {grid.best_params_}")
    print(f"  Best CV Score: {grid.best_score_:.4f}")

    return grid.best_estimator_, grid.best_params_, grid.best_score_
