"""
random_forest_models.py
=======================
Random Forest baselines for the three prediction tasks.

Each baseline:
  - Uses the same core node features as the GNN (no graph connectivity)
  - Is evaluated on the same held-out test split as the GNN
  - Produces feature importance rankings
  - Supports probability calibration for classification tasks

Tasks:
  1. Crop suitability classification
  2. Disease risk classification
  3. Crop yield regression
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

LOGGER = logging.getLogger(__name__)


@dataclass
class RFClassificationResult:
    model_name: str
    task: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float]
    pr_auc: Optional[float]
    confusion_matrix: np.ndarray
    feature_importances: dict[str, float]
    predictions: np.ndarray
    true_labels: np.ndarray
    probabilities: Optional[np.ndarray]
    n_train: int
    n_test: int


@dataclass
class RFRegressionResult:
    model_name: str
    task: str
    mae: float
    rmse: float
    r2: float
    feature_importances: dict[str, float]
    predictions: np.ndarray
    true_labels: np.ndarray
    n_train: int
    n_test: int


def _feature_importance_dict(model, feature_names: list[str]) -> dict[str, float]:
    """Return a sorted dict of feature importances."""
    importances = model.feature_importances_
    pairs = sorted(zip(feature_names, importances), key=lambda x: -x[1])
    return {name: round(float(imp), 6) for name, imp in pairs}


def train_rf_suitability_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    n_estimators: int = 200,
    random_state: int = 42,
    class_weight: str = "balanced",
    calibrate: bool = True,
    save_dir: Optional[Path] = None,
) -> RFClassificationResult:
    """
    Train and evaluate a Random Forest for crop suitability classification.

    Binary classification: suitable (1) vs not suitable (0) for the leading crop.
    For multi-class crop type prediction, the same function works by changing y labels.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=-1,
        max_features="sqrt",
    )

    if calibrate and len(np.unique(y_train)) == 2:
        model = CalibratedClassifierCV(rf, method="isotonic", cv=3)
        model.fit(X_train, y_train)
        raw_rf = model.calibrated_classifiers_[0].estimator
    else:
        rf.fit(X_train, y_train)
        model = rf
        raw_rf = rf

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    n_classes = len(np.unique(y_train))
    avg = "binary" if n_classes == 2 else "weighted"

    roc_auc = None
    pr_auc = None
    if n_classes == 2:
        try:
            roc_auc = float(roc_auc_score(y_test, y_proba[:, 1]))
            pr_auc = float(average_precision_score(y_test, y_proba[:, 1]))
        except Exception:
            pass

    feat_imp = _feature_importance_dict(raw_rf, feature_names)

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_dir / "rf_suitability.joblib")
        LOGGER.info("Saved suitability RF to %s", save_dir / "rf_suitability.joblib")

    result = RFClassificationResult(
        model_name="Random Forest",
        task="Crop Suitability",
        accuracy=float(accuracy_score(y_test, y_pred)),
        precision=float(precision_score(y_test, y_pred, average=avg, zero_division=0)),
        recall=float(recall_score(y_test, y_pred, average=avg, zero_division=0)),
        f1=float(f1_score(y_test, y_pred, average=avg, zero_division=0)),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        confusion_matrix=confusion_matrix(y_test, y_pred),
        feature_importances=feat_imp,
        predictions=y_pred,
        true_labels=y_test,
        probabilities=y_proba,
        n_train=len(X_train),
        n_test=len(X_test),
    )
    LOGGER.info(
        "[RF Suitability] Acc=%.3f | F1=%.3f | ROC-AUC=%s",
        result.accuracy, result.f1, f"{roc_auc:.3f}" if roc_auc else "N/A",
    )
    return result


def train_rf_disease_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    n_estimators: int = 200,
    random_state: int = 42,
    class_weight: str = "balanced",
    calibrate: bool = True,
    save_dir: Optional[Path] = None,
) -> RFClassificationResult:
    """
    Train and evaluate a Random Forest for disease risk classification.

    Emphasizes recall for the positive (high-risk) class to minimise
    missed disease alerts.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=-1,
        min_samples_leaf=2,
    )

    if calibrate:
        try:
            model = CalibratedClassifierCV(rf, method="isotonic", cv=3)
            model.fit(X_train, y_train)
            raw_rf = model.calibrated_classifiers_[0].estimator
        except Exception:
            rf.fit(X_train, y_train)
            model = rf
            raw_rf = rf
    else:
        rf.fit(X_train, y_train)
        model = rf
        raw_rf = rf

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    roc_auc = None
    pr_auc = None
    try:
        roc_auc = float(roc_auc_score(y_test, y_proba[:, 1]))
        pr_auc = float(average_precision_score(y_test, y_proba[:, 1]))
    except Exception:
        pass

    feat_imp = _feature_importance_dict(raw_rf, feature_names)

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_dir / "rf_disease.joblib")

    result = RFClassificationResult(
        model_name="Random Forest",
        task="Disease Risk",
        accuracy=float(accuracy_score(y_test, y_pred)),
        precision=float(precision_score(y_test, y_pred, zero_division=0)),
        recall=float(recall_score(y_test, y_pred, zero_division=0)),
        f1=float(f1_score(y_test, y_pred, zero_division=0)),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        confusion_matrix=confusion_matrix(y_test, y_pred),
        feature_importances=feat_imp,
        predictions=y_pred,
        true_labels=y_test,
        probabilities=y_proba,
        n_train=len(X_train),
        n_test=len(X_test),
    )
    LOGGER.info(
        "[RF Disease] Acc=%.3f | Recall=%.3f | ROC-AUC=%s",
        result.accuracy, result.recall, f"{roc_auc:.3f}" if roc_auc else "N/A",
    )
    return result


def train_rf_yield_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    n_estimators: int = 200,
    random_state: int = 42,
    save_dir: Optional[Path] = None,
) -> RFRegressionResult:
    """Train and evaluate a Random Forest regressor for crop yield forecasting."""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        max_features=0.7,
        min_samples_leaf=1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    feat_imp = _feature_importance_dict(model, feature_names)

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_dir / "rf_yield.joblib")

    result = RFRegressionResult(
        model_name="Random Forest",
        task="Crop Yield Forecasting",
        mae=mae,
        rmse=rmse,
        r2=r2,
        feature_importances=feat_imp,
        predictions=y_pred,
        true_labels=y_test,
        n_train=len(X_train),
        n_test=len(X_test),
    )
    LOGGER.info("[RF Yield] MAE=%.4f | RMSE=%.4f | R²=%.4f", mae, rmse, r2)
    return result


def cross_validate_rf(
    X: np.ndarray,
    y: np.ndarray,
    task: str = "classification",
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """Run stratified k-fold cross-validation and return mean ± std metrics."""
    if task == "classification":
        model = RandomForestClassifier(
            n_estimators=100, random_state=random_state, class_weight="balanced", n_jobs=-1
        )
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scores = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted")
        metric = "f1_weighted"
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
        cv = n_splits
        scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
        scores = -scores  # flip sign
        metric = "rmse"

    return {
        "metric": metric,
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "scores": scores.tolist(),
        "n_splits": n_splits,
    }


def predict_suitability(
    model,
    X: np.ndarray,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (predicted_labels, probabilities) from a trained suitability model.
    """
    proba = model.predict_proba(X)
    if proba.shape[1] == 2:
        labels = (proba[:, 1] >= threshold).astype(int)
        return labels, proba
    else:
        labels = proba.argmax(axis=1)
        return labels, proba


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_all_datasets
    from preprocessing import (
        preprocess_for_disease,
        preprocess_for_forecasting,
        preprocess_for_suitability,
    )

    datasets = load_all_datasets()

    print("=== Crop Suitability ===")
    split_s = preprocess_for_suitability(datasets["suitability"])
    result_s = train_rf_suitability_classifier(
        split_s.X_train, split_s.y_train,
        split_s.X_test, split_s.y_test,
        split_s.feature_names,
    )
    print(f"Accuracy: {result_s.accuracy:.3f} | F1: {result_s.f1:.3f}")

    print("\n=== Disease Risk ===")
    split_d = preprocess_for_disease(datasets["disease"])
    result_d = train_rf_disease_classifier(
        split_d.X_train, split_d.y_train,
        split_d.X_test, split_d.y_test,
        split_d.feature_names,
    )
    print(f"Accuracy: {result_d.accuracy:.3f} | Recall: {result_d.recall:.3f}")

    print("\n=== Yield Forecasting ===")
    split_f = preprocess_for_forecasting(datasets["forecasting"])
    result_f = train_rf_yield_regressor(
        split_f.X_train, split_f.y_train,
        split_f.X_test, split_f.y_test,
        split_f.feature_names,
    )
    print(f"MAE: {result_f.mae:.4f} | RMSE: {result_f.rmse:.4f} | R²: {result_f.r2:.4f}")
