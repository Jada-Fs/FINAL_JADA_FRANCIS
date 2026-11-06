"""
evaluation.py
=============
Comprehensive evaluation utilities for the Uganda Smart Agriculture GNN project.

Provides:
  - Metrics computation for classification and regression tasks
  - Comparison tables between Random Forest and GNN
  - Visualization utilities (confusion matrix, feature importance, ROC/PR curves,
    actual vs predicted, training loss curves)
  - Report generation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

LOGGER = logging.getLogger(__name__)

# Shared style
PALETTE = {
    "RF": "#4C72B0",
    "GNN": "#55A868",
    "positive": "#C44E52",
    "negative": "#64B5CD",
    "neutral": "#8172B2",
}


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float]
    pr_auc: Optional[float]
    confusion_mat: np.ndarray
    n_test: int
    model_name: str
    task: str


@dataclass
class RegressionMetrics:
    mae: float
    rmse: float
    r2: float
    n_test: int
    model_name: str
    task: str


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    model_name: str,
    task: str,
    average: str = "binary",
) -> ClassificationMetrics:
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average=average, zero_division=0))
    rec = float(recall_score(y_true, y_pred, average=average, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average=average, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)

    roc_auc = None
    pr_auc = None
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            score_col = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
            roc_auc = float(roc_auc_score(y_true, score_col))
            pr_auc = float(average_precision_score(y_true, score_col))
        except Exception:
            pass

    return ClassificationMetrics(
        accuracy=acc, precision=prec, recall=rec, f1=f1,
        roc_auc=roc_auc, pr_auc=pr_auc,
        confusion_mat=cm, n_test=len(y_true),
        model_name=model_name, task=task,
    )


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    task: str,
) -> RegressionMetrics:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return RegressionMetrics(mae=mae, rmse=rmse, r2=r2, n_test=len(y_true),
                              model_name=model_name, task=task)


# ---------------------------------------------------------------------------
# Comparison tables
# ---------------------------------------------------------------------------

def classification_comparison_table(rf_metrics: ClassificationMetrics, gnn_metrics: ClassificationMetrics) -> str:
    """Return a markdown-formatted comparison table."""
    rows = [
        ("Accuracy", f"{rf_metrics.accuracy:.4f}", f"{gnn_metrics.accuracy:.4f}"),
        ("Precision", f"{rf_metrics.precision:.4f}", f"{gnn_metrics.precision:.4f}"),
        ("Recall", f"{rf_metrics.recall:.4f}", f"{gnn_metrics.recall:.4f}"),
        ("F1-score", f"{rf_metrics.f1:.4f}", f"{gnn_metrics.f1:.4f}"),
        ("ROC-AUC", f"{rf_metrics.roc_auc:.4f}" if rf_metrics.roc_auc else "N/A",
                    f"{gnn_metrics.roc_auc:.4f}" if gnn_metrics.roc_auc else "N/A"),
        ("PR-AUC", f"{rf_metrics.pr_auc:.4f}" if rf_metrics.pr_auc else "N/A",
                   f"{gnn_metrics.pr_auc:.4f}" if gnn_metrics.pr_auc else "N/A"),
        ("Test samples", str(rf_metrics.n_test), str(gnn_metrics.n_test)),
    ]
    header = f"| Metric | Random Forest | GNN (GraphSAGE) |\n|--------|:---:|:---:|\n"
    body = "\n".join(f"| {m} | {r} | {g} |" for m, r, g in rows)
    return header + body


def regression_comparison_table(rf_metrics: RegressionMetrics, gnn_metrics: RegressionMetrics) -> str:
    rows = [
        ("MAE", f"{rf_metrics.mae:.4f}", f"{gnn_metrics.mae:.4f}"),
        ("RMSE", f"{rf_metrics.rmse:.4f}", f"{gnn_metrics.rmse:.4f}"),
        ("R²", f"{rf_metrics.r2:.4f}", f"{gnn_metrics.r2:.4f}"),
        ("Test samples", str(rf_metrics.n_test), str(gnn_metrics.n_test)),
    ]
    header = "| Metric | Random Forest | GNN (GraphSAGE) |\n|--------|:---:|:---:|\n"
    body = "\n".join(f"| {m} | {r} | {g} |" for m, r, g in rows)
    return header + body


# ---------------------------------------------------------------------------
# Visualization functions (return Figure objects for Streamlit display)
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    title: str = "Confusion Matrix",
    class_names: Optional[list[str]] = None,
    cmap: str = "Blues",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = class_names or [str(i) for i in range(cm.shape[0])]
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap,
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_feature_importance(
    importances: dict[str, float],
    title: str = "Feature Importance",
    top_n: int = 10,
    color: str = PALETTE["RF"],
) -> plt.Figure:
    items = list(importances.items())[:top_n]
    names = [k for k, _ in items]
    vals = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(7, max(3, len(names) * 0.45)))
    bars = ax.barh(names[::-1], vals[::-1], color=color, edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, fmt="{:.4f}", padding=3, fontsize=9)
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(vals) * 1.2)
    fig.tight_layout()
    return fig


def plot_roc_curve(
    rf_true: np.ndarray, rf_proba: np.ndarray,
    gnn_true: np.ndarray, gnn_proba: np.ndarray,
    task: str = "Task",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    for true, proba, name, color in [
        (rf_true, rf_proba, "Random Forest", PALETTE["RF"]),
        (gnn_true, gnn_proba, "GNN (GraphSAGE)", PALETTE["GNN"]),
    ]:
        try:
            score_col = proba[:, 1] if proba.ndim > 1 else proba
            fpr, tpr, _ = roc_curve(true, score_col)
            auc = roc_auc_score(true, score_col)
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.3f})")
        except Exception:
            pass
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"ROC Curve – {task}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_pr_curve(
    rf_true: np.ndarray, rf_proba: np.ndarray,
    gnn_true: np.ndarray, gnn_proba: np.ndarray,
    task: str = "Task",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    for true, proba, name, color in [
        (rf_true, rf_proba, "Random Forest", PALETTE["RF"]),
        (gnn_true, gnn_proba, "GNN (GraphSAGE)", PALETTE["GNN"]),
    ]:
        try:
            score_col = proba[:, 1] if proba.ndim > 1 else proba
            precision, recall, _ = precision_recall_curve(true, score_col)
            ap = average_precision_score(true, score_col)
            ax.plot(recall, precision, color=color, lw=2, label=f"{name} (AP={ap:.3f})")
        except Exception:
            pass
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(f"Precision-Recall Curve – {task}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_actual_vs_predicted(
    rf_true: np.ndarray, rf_pred: np.ndarray,
    gnn_true: np.ndarray, gnn_pred: np.ndarray,
    task: str = "Yield Forecasting",
    ylabel: str = "Yield (t/ha)",
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, true, pred, name, color in [
        (axes[0], rf_true, rf_pred, "Random Forest", PALETTE["RF"]),
        (axes[1], gnn_true, gnn_pred, "GNN (GraphSAGE)", PALETTE["GNN"]),
    ]:
        ax.scatter(true, pred, alpha=0.7, color=color, edgecolors="white", s=60)
        lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
        ax.plot(lims, lims, "k--", lw=1.5, label="Perfect prediction")
        r2 = r2_score(true, pred)
        rmse = np.sqrt(mean_squared_error(true, pred))
        ax.set_xlabel(f"Actual {ylabel}", fontsize=10)
        ax.set_ylabel(f"Predicted {ylabel}", fontsize=10)
        ax.set_title(f"{name}\nR²={r2:.3f}  RMSE={rmse:.3f}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
    fig.suptitle(f"Actual vs Predicted – {task}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_metric_comparison_bar(
    rf_metrics: dict,
    gnn_metrics: dict,
    metric_names: list[str],
    title: str = "Model Comparison",
    higher_is_better: Optional[list[bool]] = None,
) -> plt.Figure:
    """Bar chart comparing RF vs GNN for a set of metrics."""
    x = np.arange(len(metric_names))
    width = 0.35
    rf_vals = [rf_metrics.get(m, 0.0) or 0.0 for m in metric_names]
    gnn_vals = [gnn_metrics.get(m, 0.0) or 0.0 for m in metric_names]

    fig, ax = plt.subplots(figsize=(max(7, len(metric_names) * 1.5), 5))
    bars1 = ax.bar(x - width / 2, rf_vals, width, label="Random Forest", color=PALETTE["RF"], alpha=0.85)
    bars2 = ax.bar(x + width / 2, gnn_vals, width, label="GNN (GraphSAGE)", color=PALETTE["GNN"], alpha=0.85)
    ax.bar_label(bars1, fmt="%.3f", fontsize=8, padding=2)
    ax.bar_label(bars2, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_ylabel("Metric Value", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(max(rf_vals), max(gnn_vals)) * 1.25)
    fig.tight_layout()
    return fig


def plot_training_loss(train_losses: list[float], title: str = "GNN Training Loss") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_losses, color=PALETTE["GNN"], lw=2, label="Training Loss")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Comprehensive evaluation summary
# ---------------------------------------------------------------------------

def evaluate_all_models(results: dict) -> dict:
    """
    Compute summary metrics from a results dict returned by run_experiments().
    Returns a human-readable comparison dict.
    """
    summary = {}
    for task, task_results in results.items():
        rf_r = task_results.get("rf")
        gnn_r = task_results.get("gnn")
        if rf_r is None:
            continue

        if hasattr(rf_r, "accuracy"):
            # Classification
            entry = {
                "type": "classification",
                "rf": {
                    "accuracy": round(rf_r.accuracy, 4),
                    "f1": round(rf_r.f1, 4),
                    "recall": round(rf_r.recall, 4),
                    "roc_auc": round(rf_r.roc_auc, 4) if rf_r.roc_auc else None,
                    "pr_auc": round(rf_r.pr_auc, 4) if rf_r.pr_auc else None,
                },
            }
            if gnn_r:
                gnn_metrics = compute_classification_metrics(
                    gnn_r.true_labels, gnn_r.predictions, gnn_r.probabilities,
                    "GNN", task,
                )
                entry["gnn"] = {
                    "accuracy": round(gnn_metrics.accuracy, 4),
                    "f1": round(gnn_metrics.f1, 4),
                    "recall": round(gnn_metrics.recall, 4),
                    "roc_auc": round(gnn_metrics.roc_auc, 4) if gnn_metrics.roc_auc else None,
                    "pr_auc": round(gnn_metrics.pr_auc, 4) if gnn_metrics.pr_auc else None,
                }
                winner = "GNN" if (entry["gnn"]["f1"] > entry["rf"]["f1"]) else "Random Forest"
                entry["best_model"] = winner
        else:
            # Regression
            entry = {
                "type": "regression",
                "rf": {
                    "mae": round(rf_r.mae, 4),
                    "rmse": round(rf_r.rmse, 4),
                    "r2": round(rf_r.r2, 4),
                },
            }
            if gnn_r:
                gnn_metrics = compute_regression_metrics(
                    gnn_r.true_labels, gnn_r.predictions, "GNN", task,
                )
                entry["gnn"] = {
                    "mae": round(gnn_metrics.mae, 4),
                    "rmse": round(gnn_metrics.rmse, 4),
                    "r2": round(gnn_metrics.r2, 4),
                }
                winner = "GNN" if (entry["gnn"]["r2"] > entry["rf"]["r2"]) else "Random Forest"
                entry["best_model"] = winner

        summary[task] = entry
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Evaluation module loaded. Import and use in app.py or notebooks.")
