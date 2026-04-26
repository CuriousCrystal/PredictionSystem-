"""
Student Performance Prediction — Main Entry Point

Orchestrates data loading, EDA, preprocessing, model training,
evaluation, hyperparameter tuning, and visualization.

Usage:
    python main.py                # Full pipeline (EDA + train + evaluate)
    python main.py --skip-eda     # Skip EDA visualizations
    python main.py --tune         # Include hyperparameter tuning
"""

import logging
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

from src.config import (
    ALL_FEATURE_COLS,
    INPUT_PROMPTS,
    NOMINAL_FEATURES,
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
    RESULTS_DIR,
    TARGET_COL,
)
from src.data_loader import load_data, show_data_summary
from src.evaluation import compute_confusion_matrix, cross_validate_model, evaluate_model
from src.models import get_pipelines, train_model, tune_gradient_boosting, tune_random_forest
from src.preprocessing import encode_target, get_feature_names, split_data
from src.visualizations import (
    plot_class_distribution,
    plot_confusion_matrices,
    plot_correlation_heatmap,
    plot_cross_validation_results,
    plot_eda_grid,
    plot_feature_importance,
    plot_model_comparison,
)

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress only specific noisy warnings (convergence, etc.)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning)


def run_eda(data: pd.DataFrame) -> None:
    """Run exploratory data analysis with visualizations."""
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    show_data_summary(data)
    plot_class_distribution(data)
    plot_correlation_heatmap(data)
    plot_eda_grid(data)


def run_training(pipelines, X_train, X_test, y_train, y_test, class_names):
    """
    Train all models, evaluate them, and return results.

    Returns:
        Tuple of (trained_pipelines dict, accuracy dict, confusion_matrix dict).
    """
    print("\n" + "=" * 60)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 60)

    trained = {}
    accuracies = {}
    cm_dict = {}

    for name, pipeline in pipelines.items():
        fitted = train_model(pipeline, X_train, y_train)
        trained[name] = fitted

        y_pred, acc = evaluate_model(name, fitted, X_test, y_test,
                                     class_names=class_names)
        accuracies[name] = acc
        # Reuse y_pred — no redundant predict() call
        cm_dict[name] = compute_confusion_matrix(y_test, y_pred)

    return trained, accuracies, cm_dict


def run_cross_validation(X, y):
    """Run k-fold cross-validation for all models (fresh pipeline instances)."""
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION (5-Fold)")
    print("=" * 60)

    # Fresh pipelines so CV doesn't interfere with trained models
    fresh_pipelines = get_pipelines()
    cv_results = {}
    for name, pipeline in fresh_pipelines.items():
        scores = cross_validate_model(name, pipeline, X, y)
        cv_results[name] = scores

    return cv_results


def run_hyperparameter_tuning(X_train, y_train):
    """Run GridSearchCV for Random Forest and Gradient Boosting."""
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING")
    print("=" * 60)

    print("\n[...] Tuning Random Forest...")
    rf_best, rf_params, rf_score = tune_random_forest(X_train, y_train)

    print("\n[...] Tuning Gradient Boosting...")
    gb_best, gb_params, gb_score = tune_gradient_boosting(X_train, y_train)

    return {"Random Forest (Tuned)": rf_best, "Gradient Boosting (Tuned)": gb_best}


def save_best_model(trained_models, accuracies, label_encoder):
    """Save the best performing model and label encoder to disk with joblib."""
    best_name = max(accuracies, key=accuracies.get)
    best_model = trained_models[best_name]
    best_acc = accuracies[best_name]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_path = os.path.join(RESULTS_DIR, "best_model.joblib")
    le_path = os.path.join(RESULTS_DIR, "label_encoder.joblib")
    joblib.dump(best_model, model_path)
    joblib.dump(label_encoder, le_path)
    print(f"\n[OK] Best model saved: {best_name} (accuracy: {best_acc:.4f}) -> {model_path}")
    return best_name


def predict_single_input(trained_pipelines, label_encoder):
    """
    Interactive prediction for a single student.

    Collects user input for all features and runs prediction through all
    trained Pipelines. No manual encoding is needed — the Pipeline handles
    all preprocessing automatically.
    """
    print("\n" + "=" * 60)
    print("PREDICT FOR A SINGLE STUDENT")
    print("=" * 60)

    inputs = {}
    for col in ALL_FEATURE_COLS:
        prompt, valid = INPUT_PROMPTS.get(col, (f"{col}: ", None))
        while True:
            val = input(f"  {prompt}")
            if valid == "int":
                try:
                    inputs[col] = int(val)
                    break
                except ValueError:
                    print("    [!] Please enter a number.")
            elif valid is not None and val not in valid:
                print(f"    [!] Choose from: {valid}")
            else:
                inputs[col] = val
                break

    # Build a raw DataFrame — the Pipeline preprocessor handles encoding
    input_df = pd.DataFrame([inputs])

    print(f"\n  {'Model':30s} | Prediction")
    print(f"  {'-' * 45}")

    best_pipeline = trained_pipelines.get("Random Forest (Tuned)", trained_pipelines.get("Random Forest"))
    
    for name, pipeline in trained_pipelines.items():
        pred = pipeline.predict(input_df)
        label = label_encoder.inverse_transform(pred)[0]
        print(f"  {name:30s} | {label}")
        if pipeline == best_pipeline:
            final_pred = label

    from src.analyzer import generate_student_analysis
    analysis_text = generate_student_analysis(inputs, final_pred)
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    print(analysis_text)


def main():
    """
    Main ML pipeline.

    Steps:
        1. Load raw data
        2. Exploratory Data Analysis (optional, skip with --skip-eda)
        3. Encode target & split into train/test (features stay raw)
        4. Train all models (Pipeline handles preprocessing per-model)
        5. Cross-validation (fresh pipelines, full dataset)
        6. Hyperparameter tuning (optional, enable with --tune)
        7. Generate visualizations
        8. Save best model
        9. Print results summary
       10. Interactive single-student prediction (optional)
    """
    args = set(sys.argv[1:])
    skip_eda = "--skip-eda" in args
    do_tune = "--tune" in args

    print("+" + "=" * 58 + "+")
    print("|     Student Performance Prediction -- ML Pipeline       |")
    print("+" + "=" * 58 + "+")

    # 1. Load data
    data = load_data()

    # 2. EDA (on raw data — before any transformations)
    if not skip_eda:
        run_eda(data)

    # 3. Encode target & split
    X_all = data.drop(TARGET_COL, axis=1)
    y_all = data[TARGET_COL]

    y_all_enc, label_encoder = encode_target(y_all)
    class_names = label_encoder.classes_.tolist()
    label_map = {i: c for i, c in enumerate(class_names)}

    X_train, X_test, y_train, y_test = split_data(X_all, y_all_enc)
    print(f"\n  Class mapping: {label_map}")

    # 4. Train & evaluate
    pipelines = get_pipelines()
    trained, accuracies, cm_dict = run_training(
        pipelines, X_train, X_test, y_train, y_test, class_names
    )

    # 5. Cross-validation (full dataset, fresh pipelines)
    cv_results = run_cross_validation(X_all, y_all_enc)

    # 6. Hyperparameter tuning (optional)
    if do_tune:
        tuned_models = run_hyperparameter_tuning(X_train, y_train)
        for name, model in tuned_models.items():
            trained[name] = model
            y_pred, acc = evaluate_model(name, model, X_test, y_test,
                                         class_names=class_names)
            accuracies[name] = acc
            cm_dict[name] = compute_confusion_matrix(y_test, y_pred)

    # 7. Visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    plot_model_comparison(accuracies)
    plot_confusion_matrices(cm_dict, class_names)
    plot_cross_validation_results(cv_results)

    # Feature importance (from Random Forest pipeline)
    rf_pipeline = trained.get("Random Forest (Tuned)", trained.get("Random Forest"))
    if rf_pipeline is not None:
        rf_clf = rf_pipeline.named_steps["classifier"]
        if hasattr(rf_clf, "feature_importances_"):
            feat_names = get_feature_names(rf_pipeline)
            plot_feature_importance(rf_clf, feat_names, "Random Forest")

    # 8. Save best model and label encoder
    best_name = save_best_model(trained, accuracies, label_encoder)

    # 9. Summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    for name, acc in sorted(accuracies.items(), key=lambda x: x[1], reverse=True):
        marker = " *BEST*" if name == best_name else ""
        print(f"  {name:35s} | {acc:.4f} ({acc*100:.1f}%){marker}")
    print("=" * 60)

    # 10. Interactive prediction
    choice = input("\nDo you want to predict for a specific student? (y/n): ")
    if choice.strip().lower() == "y":
        predict_single_input(trained, label_encoder)

    print("\n[OK] Pipeline complete. Results saved to 'results/' directory.")


if __name__ == "__main__":
    main()
