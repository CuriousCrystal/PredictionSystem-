"""
Data preprocessing: ColumnTransformer-based feature pipeline, target encoding,
and stratified train/test split.

Key design decisions:
  - OneHotEncoder for nominal features (no false ordinal relationships)
  - OrdinalEncoder for truly ordinal features (StageID, GradeID)
  - StandardScaler for numeric features (required by SVM, KNN, MLP, etc.)
  - Target encoding via LabelEncoder (safe to fit on full y)
  - Feature preprocessing is embedded inside each model's Pipeline so it is
    automatically applied per-fold during cross-validation (no data leakage).
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from src.config import (
    GRADE_ORDER,
    NOMINAL_FEATURES,
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
    RANDOM_STATE,
    STAGE_ORDER,
    TARGET_COL,
    TEST_SIZE,
)

logger = logging.getLogger(__name__)


def build_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer that handles all feature types.

    - Numeric features → StandardScaler
    - Nominal categorical features → OneHotEncoder
    - Ordinal categorical features → OrdinalEncoder (with explicit ordering)

    Returns:
        Configured (unfitted) ColumnTransformer.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "nom",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop="if_binary",
                ),
                NOMINAL_FEATURES,
            ),
            (
                "ord",
                OrdinalEncoder(
                    categories=[STAGE_ORDER, GRADE_ORDER],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                ORDINAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def encode_target(y: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    """
    Encode the target column with LabelEncoder.

    Safe to fit on the full target series because label encoding the target
    does not leak any feature information.

    Args:
        y: Series of raw class labels (e.g. "H", "L", "M").

    Returns:
        Tuple of (encoded integer array, fitted LabelEncoder).
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    logger.info("Target classes: %s", list(le.classes_))
    return y_encoded, le


def split_data(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Stratified train/test split on raw (unprocessed) features.

    Feature preprocessing is handled inside each model's Pipeline so that
    encoders/scalers are fit only on the training fold.

    Args:
        X: Raw feature DataFrame (string + numeric columns).
        y: Encoded target array.
        test_size: Fraction reserved for testing.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info("Train set: %d samples", X_train.shape[0])
    logger.info("Test set:  %d samples", X_test.shape[0])
    logger.info("Features:  %d (%s)", X_train.shape[1], list(X.columns))

    return X_train, X_test, y_train, y_test


def get_feature_names(fitted_pipeline: Any) -> list[str]:
    """
    Extract human-readable feature names from a fitted Pipeline.

    Strips the ColumnTransformer prefix (e.g. ``num__``, ``nom__``) for
    cleaner display in plots.

    Args:
        fitted_pipeline: A fitted sklearn Pipeline whose first step is a
                         ColumnTransformer.

    Returns:
        List of cleaned feature name strings.
    """
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    raw_names = preprocessor.get_feature_names_out()
    return [name.split("__", 1)[-1] for name in raw_names]
