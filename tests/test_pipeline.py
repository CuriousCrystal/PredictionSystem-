"""
Unit tests for the Student Performance Prediction pipeline.

Run with:  pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.config import (
    NOMINAL_FEATURES,
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
    TARGET_COL,
)
from src.data_loader import load_data
from src.preprocessing import build_preprocessor, encode_target, split_data
from src.models import get_pipelines, train_model
from src.evaluation import evaluate_model, compute_confusion_matrix


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_data():
    """Load the dataset once for all tests."""
    return load_data()


@pytest.fixture(scope="module")
def prepared_data(raw_data):
    """Prepare X, y, and split once for all tests."""
    X = raw_data.drop(TARGET_COL, axis=1)
    y = raw_data[TARGET_COL]
    y_enc, le = encode_target(y)
    X_train, X_test, y_train, y_test = split_data(X, y_enc)
    return X_train, X_test, y_train, y_test, le


# ── Data Loading Tests ────────────────────────────────────────────────────────

class TestDataLoader:
    def test_load_returns_dataframe(self, raw_data):
        assert isinstance(raw_data, pd.DataFrame)

    def test_expected_shape(self, raw_data):
        assert raw_data.shape[1] == 17  # 16 features + 1 target

    def test_target_column_exists(self, raw_data):
        assert TARGET_COL in raw_data.columns

    def test_no_missing_values(self, raw_data):
        assert raw_data.isnull().sum().sum() == 0

    def test_all_feature_columns_present(self, raw_data):
        expected = set(NUMERIC_FEATURES + NOMINAL_FEATURES + ORDINAL_FEATURES + [TARGET_COL])
        assert expected == set(raw_data.columns)

    def test_load_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_data("nonexistent_file.csv")


# ── Preprocessing Tests ──────────────────────────────────────────────────────

class TestPreprocessing:
    def test_encode_target_returns_integers(self, raw_data):
        y_enc, le = encode_target(raw_data[TARGET_COL])
        assert y_enc.dtype in (np.int32, np.int64)

    def test_encode_target_classes(self, raw_data):
        _, le = encode_target(raw_data[TARGET_COL])
        assert set(le.classes_) == {"H", "L", "M"}

    def test_split_sizes(self, prepared_data):
        X_train, X_test, y_train, y_test, _ = prepared_data
        total = len(X_train) + len(X_test)
        assert 0.28 <= len(X_test) / total <= 0.32  # ~30%

    def test_split_preserves_features(self, prepared_data):
        X_train, X_test, _, _, _ = prepared_data
        assert list(X_train.columns) == list(X_test.columns)

    def test_preprocessor_transforms(self, prepared_data):
        X_train, X_test, _, _, _ = prepared_data
        preprocessor = build_preprocessor()
        X_transformed = preprocessor.fit_transform(X_train)
        assert X_transformed.shape[0] == X_train.shape[0]
        assert X_transformed.shape[1] > X_train.shape[1]  # one-hot expands


# ── Model Tests ───────────────────────────────────────────────────────────────

class TestModels:
    def test_get_pipelines_returns_dict(self):
        pipelines = get_pipelines()
        assert isinstance(pipelines, dict)
        assert len(pipelines) == 8

    def test_all_are_pipelines(self):
        for name, pipe in get_pipelines().items():
            assert isinstance(pipe, Pipeline), f"{name} is not a Pipeline"

    def test_pipeline_has_preprocessor_and_classifier(self):
        for name, pipe in get_pipelines().items():
            assert "preprocessor" in pipe.named_steps, f"{name} missing preprocessor"
            assert "classifier" in pipe.named_steps, f"{name} missing classifier"

    def test_train_and_predict(self, prepared_data):
        X_train, X_test, y_train, y_test, _ = prepared_data
        pipelines = get_pipelines()
        pipe = pipelines["Decision Tree"]
        fitted = train_model(pipe, X_train, y_train)
        preds = fitted.predict(X_test)
        assert len(preds) == len(y_test)
        assert set(preds).issubset(set(y_train))


# ── Evaluation Tests ──────────────────────────────────────────────────────────

class TestEvaluation:
    def test_confusion_matrix_shape(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 1, 1, 0])
        cm = compute_confusion_matrix(y_true, y_pred)
        assert cm.shape == (3, 3)

    def test_evaluate_model_returns_tuple(self, prepared_data):
        X_train, X_test, y_train, y_test, le = prepared_data
        pipe = get_pipelines()["Decision Tree"]
        train_model(pipe, X_train, y_train)
        y_pred, acc = evaluate_model(
            "Decision Tree", pipe, X_test, y_test,
            class_names=le.classes_.tolist(),
        )
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0
        assert len(y_pred) == len(y_test)
