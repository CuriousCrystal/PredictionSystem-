"""
Visualization utilities: EDA charts, model comparison, confusion matrices,
feature importance, and cross-validation box plots.
"""

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.config import RESULTS_DIR

# ── Style configuration ──────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0",
    "text.color": "#e0e0e0",
    "xtick.color": "#e0e0e0",
    "ytick.color": "#e0e0e0",
    "figure.dpi": 120,
    "font.size": 10,
})

PALETTE = ["#e94560", "#0f3460", "#53d8a8"]   # L=red, M=blue, H=green
CLASS_ORDER = ["L", "M", "H"]


def _save(fig: plt.Figure, filename: str) -> None:
    """Save figure to the results directory."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  -> Saved: {path}")


# ── EDA Plots ─────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(data) -> None:
    """Plot a correlation heatmap of all numeric features."""
    fig, ax = plt.subplots(figsize=(12, 8))
    corr = data.corr(numeric_only=True)
    sns.heatmap(
        corr, annot=True, cmap="coolwarm", center=0,
        linewidths=0.5, fmt=".2f", ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "correlation_heatmap.png")
    plt.show()


def plot_class_distribution(data) -> None:
    """Plot overall class distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x="Class", data=data, order=CLASS_ORDER, palette=PALETTE, ax=ax)
    ax.set_title("Student Performance Class Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")

    # Add value labels on bars
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}",
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    _save(fig, "class_distribution.png")
    plt.show()


def plot_eda_grid(data) -> None:
    """Plot a 2×3 grid of countplots for key categorical features."""
    categories = [
        ("gender", "Gender"),
        ("Semester", "Semester"),
        ("NationalITy", "Nationality"),
        ("StageID", "Stage"),
        ("StudentAbsenceDays", "Absence Days"),
        ("ParentAnsweringSurvey", "Parent Survey"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (col, title) in enumerate(categories):
        sns.countplot(
            x=col, hue="Class", data=data,
            hue_order=CLASS_ORDER, palette=PALETTE, ax=axes[idx],
        )
        axes[idx].set_title(f"Class by {title}", fontsize=11, fontweight="bold")
        axes[idx].tick_params(axis="x", rotation=45)
        axes[idx].legend(title="Class", fontsize=8)

    plt.suptitle("Exploratory Data Analysis — Key Features",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, "eda_grid.png")
    plt.show()


# ── Model Evaluation Plots ───────────────────────────────────────────────────

def plot_model_comparison(results: dict[str, float]) -> None:
    """
    Bar chart comparing test accuracy of all models.

    Args:
        results: dict mapping model name → accuracy (float).
    """
    names = list(results.keys())
    accs = [results[n] for n in names]
    colors = sns.color_palette("viridis", len(names))

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(names, accs, color=colors, edgecolor="#e0e0e0", linewidth=0.5)

    # Add value labels
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{acc:.3f}", va="center", fontweight="bold", fontsize=10)

    ax.set_xlim(0, max(accs) + 0.08)
    ax.set_xlabel("Test Accuracy", fontsize=12)
    ax.set_title("Model Comparison — Test Accuracy", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    _save(fig, "model_comparison.png")
    plt.show()


def plot_confusion_matrices(
    cm_dict: dict[str, np.ndarray],
    class_names: list[str],
) -> None:
    """
    Plot confusion matrices for multiple models in a grid.

    Args:
        cm_dict: dict mapping model name → confusion matrix (numpy array).
        class_names: list of class label strings.
    """
    n = len(cm_dict)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))

    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (name, cm) in enumerate(cm_dict.items()):
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="YlOrRd",
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[idx], linewidths=0.5,
        )
        axes[idx].set_title(name, fontsize=10, fontweight="bold")
        axes[idx].set_ylabel("Actual")
        axes[idx].set_xlabel("Predicted")

    # Hide unused axes
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Confusion Matrices", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, "confusion_matrices.png")
    plt.show()


def plot_feature_importance(
    model: Any,
    feature_names: list[str],
    model_name: str = "Random Forest",
) -> None:
    """
    Plot feature importance from a tree-based model.

    Args:
        model: A fitted tree-based sklearn estimator with feature_importances_.
        feature_names: List of feature name strings (already cleaned).
        model_name: Display name of the model.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)

    # Show top 20 features to keep the chart readable after one-hot expansion
    if len(indices) > 20:
        indices = indices[-20:]

    fig, ax = plt.subplots(figsize=(10, max(6, len(indices) * 0.4)))
    colors = sns.color_palette("viridis", len(indices))
    ax.barh(range(len(indices)), importances[indices],
            color=colors, edgecolor="#e0e0e0")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Feature Importance — {model_name}", fontsize=14, fontweight="bold")

    # Add value labels
    for i, v in enumerate(importances[indices]):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    _save(fig, "feature_importance.png")
    plt.show()


def plot_cross_validation_results(cv_results: dict[str, np.ndarray]) -> None:
    """
    Box plot of cross-validation scores per model.

    Args:
        cv_results: dict mapping model name → array of per-fold scores.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    names = list(cv_results.keys())
    data = [cv_results[n] for n in names]

    bp = ax.boxplot(data, labels=names, patch_artist=True, vert=True)
    colors = sns.color_palette("viridis", len(names))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Cross-Validation Scores (5-Fold)", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    _save(fig, "cross_validation.png")
    plt.show()
