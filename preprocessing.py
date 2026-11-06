"""
preprocessing.py
================
Reusable data preprocessing pipeline for the Uganda Smart Agriculture GNN project.

Pipeline steps:
  1. Inspect and report data types and missing values
  2. Remove duplicate and invalid records
  3. Impute missing values (median for numeric, mode for categorical)
  4. Detect and cap outliers using IQR method
  5. Encode categorical variables
  6. Scale numerical features (StandardScaler, stored for reuse)
  7. Align datasets by location key
  8. Report class imbalance for classification tasks
  9. Prevent data leakage through split-aware processing

All scalers are fitted on training data only and applied to test data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core feature definitions
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "longitude", "latitude", "temperature", "rainfall", "humidity",
    "soil_ph", "soil_organic_carbon", "soil_moisture", "soil_nitrogen",
]

SUITABILITY_FEATURES = NUMERIC_FEATURES + ["suitability_score"]
DISEASE_FEATURES = NUMERIC_FEATURES
FORECASTING_FEATURES = NUMERIC_FEATURES


@dataclass
class PreprocessedSplit:
    """Holds train/test splits with fitted transformers."""
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    target_name: str
    scaler: StandardScaler
    label_encoder: Optional[LabelEncoder] = None
    train_indices: Optional[np.ndarray] = None
    test_indices: Optional[np.ndarray] = None
    class_weights: Optional[dict] = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Inspection utilities
# ---------------------------------------------------------------------------

def inspect_dataframe(df: pd.DataFrame, name: str = "DataFrame") -> dict:
    """Return a summary report of the dataset."""
    report = {
        "name": name,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "numeric_summary": {},
    }
    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        desc = numeric.describe().T
        report["numeric_summary"] = desc.to_dict(orient="index")
    LOGGER.info(
        "%s: %d rows, %d cols, %d missing cells, %d duplicates",
        name, report["n_rows"], report["n_cols"],
        sum(report["missing"].values()), report["duplicates"],
    )
    return report


def detect_outliers_iqr(df: pd.DataFrame, columns: list[str], factor: float = 1.5) -> pd.DataFrame:
    """
    Return a boolean DataFrame marking outliers per column using the IQR method.
    factor=1.5 is the standard Tukey fence; use 3.0 for extreme outliers only.
    """
    result = pd.DataFrame(False, index=df.index, columns=columns)
    for col in columns:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        result[col] = (df[col] < lower) | (df[col] > upper)
    return result


def check_class_imbalance(y: pd.Series | np.ndarray, target_name: str = "target") -> dict:
    """Report class distribution and imbalance ratio."""
    series = pd.Series(y) if not isinstance(y, pd.Series) else y
    counts = series.value_counts()
    proportions = (counts / len(series) * 100).round(2)
    imbalance_ratio = float(counts.max() / counts.min()) if len(counts) > 1 else 1.0
    LOGGER.info(
        "Class distribution for '%s': %s | imbalance ratio: %.2f",
        target_name, dict(counts), imbalance_ratio,
    )
    return {
        "counts": counts.to_dict(),
        "proportions": proportions.to_dict(),
        "imbalance_ratio": imbalance_ratio,
        "n_classes": len(counts),
    }


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

def remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with obviously invalid values for geographical or physical columns."""
    df = df.copy()
    before = len(df)
    # Uganda bounding box (generous)
    if "longitude" in df.columns:
        df = df[df["longitude"].between(29.0, 35.5)]
    if "latitude" in df.columns:
        df = df[df["latitude"].between(-2.0, 5.0)]
    if "soil_ph" in df.columns:
        df = df[df["soil_ph"].between(3.0, 9.0)]
    if "temperature" in df.columns:
        df = df[df["temperature"].between(5.0, 45.0)]
    if "rainfall" in df.columns:
        df = df[df["rainfall"] >= 0]
    if "humidity" in df.columns:
        df = df[df["humidity"].between(0, 100)]
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed:
        LOGGER.info("Removed %d invalid or duplicate rows.", removed)
    return df.reset_index(drop=True)


def impute_missing(df: pd.DataFrame, numeric_strategy: str = "median") -> pd.DataFrame:
    """Impute missing values. Numeric: median or mean. Categorical: mode."""
    df = df.copy()
    for col in df.columns:
        if not df[col].isna().any():
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            fill = df[col].median() if numeric_strategy == "median" else df[col].mean()
            df[col] = df[col].fillna(fill)
            LOGGER.debug("Imputed '%s' with %s=%.4f", col, numeric_strategy, fill)
        else:
            modes = df[col].mode(dropna=True)
            if not modes.empty:
                df[col] = df[col].fillna(modes.iloc[0])
    return df


def cap_outliers(df: pd.DataFrame, columns: list[str], factor: float = 3.0) -> pd.DataFrame:
    """Cap outliers at the Tukey fence (factor=3 for extreme outliers)."""
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        n_capped = int(((df[col] < lower) | (df[col] > upper)).sum())
        if n_capped:
            df[col] = df[col].clip(lower, upper)
            LOGGER.debug("Capped %d outliers in '%s'", n_capped, col)
    return df


def encode_categoricals(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Label-encode categorical columns. Returns (encoded_df, encoder_dict)."""
    df = df.copy()
    encoders: dict[str, LabelEncoder] = {}
    for col in columns:
        if col not in df.columns:
            continue
        le = LabelEncoder()
        df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        LOGGER.debug("Encoded '%s' -> %d classes", col, len(le.classes_))
    return df, encoders


# ---------------------------------------------------------------------------
# Split-aware preprocessing
# ---------------------------------------------------------------------------

def preprocess_for_task(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    test_size: float = 0.25,
    random_state: int = 42,
    task: str = "regression",  # "regression" | "classification"
    stratify: bool = True,
) -> PreprocessedSplit:
    """
    Full preprocessing pipeline for a single prediction task.

    1. Validate inputs
    2. Clean data
    3. Impute missing values
    4. Split into train / test (before fitting scaler to prevent leakage)
    5. Fit StandardScaler on train, apply to test
    6. Encode classification targets if needed

    Parameters
    ----------
    df: Input dataframe
    feature_columns: List of feature column names
    target_column: Target column name
    test_size: Fraction for test split
    random_state: Reproducibility seed
    task: 'regression' or 'classification'
    stratify: Whether to stratify classification splits

    Returns
    -------
    PreprocessedSplit with fitted transformers
    """
    available_features = [c for c in feature_columns if c in df.columns]
    if not available_features:
        raise ValueError(f"None of the requested features found in dataframe. Requested: {feature_columns}")
    if target_column not in df.columns:
        raise ValueError(f"Target '{target_column}' not found in dataframe.")

    # Clean
    df_clean = remove_invalid_rows(df)
    df_clean = impute_missing(df_clean)
    numeric_feats = [c for c in available_features if pd.api.types.is_numeric_dtype(df_clean[c])]
    df_clean = cap_outliers(df_clean, numeric_feats)

    X = df_clean[available_features].values
    y_raw = df_clean[target_column].values
    indices = df_clean.index.to_numpy()

    # Encode target for classification
    le = None
    if task == "classification" and not pd.api.types.is_float_dtype(y_raw):
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
    else:
        y = y_raw.astype(float)

    # Stratify only for classification
    strat = y if (task == "classification" and stratify) else None

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )

    # Fit scaler on training data only (prevent leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Compute class weights for imbalanced classification
    class_weights = None
    if task == "classification":
        unique, counts = np.unique(y_train, return_counts=True)
        total = len(y_train)
        class_weights = {int(cls): float(total / (len(unique) * cnt))
                         for cls, cnt in zip(unique, counts)}

    metadata = {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": len(available_features),
        "task": task,
        "target": target_column,
    }

    if task == "classification":
        imbalance = check_class_imbalance(y_train, target_column)
        metadata["class_imbalance"] = imbalance

    return PreprocessedSplit(
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        feature_names=available_features,
        target_name=target_column,
        scaler=scaler,
        label_encoder=le,
        train_indices=idx_train,
        test_indices=idx_test,
        class_weights=class_weights,
        metadata=metadata,
    )


def preprocess_for_suitability(df: pd.DataFrame, **kwargs) -> PreprocessedSplit:
    """Preprocess for crop suitability classification task."""
    features = [c for c in SUITABILITY_FEATURES if c in df.columns]
    return preprocess_for_task(df, features, "suitable", task="classification", **kwargs)


def preprocess_for_disease(df: pd.DataFrame, **kwargs) -> PreprocessedSplit:
    """Preprocess for disease risk classification task."""
    features = [c for c in DISEASE_FEATURES if c in df.columns]
    return preprocess_for_task(df, features, "disease_risk", task="classification", **kwargs)


def preprocess_for_forecasting(df: pd.DataFrame, **kwargs) -> PreprocessedSplit:
    """Preprocess for crop yield regression task."""
    features = [c for c in FORECASTING_FEATURES if c in df.columns]
    return preprocess_for_task(df, features, "crop_yield", task="regression", **kwargs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_all_datasets
    datasets = load_all_datasets()

    print("\n--- Inspecting base dataset ---")
    report = inspect_dataframe(datasets["base"], "Uganda Base Dataset")
    print(f"Rows: {report['n_rows']}, Cols: {report['n_cols']}")
    print(f"Duplicates: {report['duplicates']}")

    print("\n--- Suitability preprocessing ---")
    split = preprocess_for_suitability(datasets["suitability"])
    print(f"Train: {split.X_train.shape}, Test: {split.X_test.shape}")

    print("\n--- Disease preprocessing ---")
    split_d = preprocess_for_disease(datasets["disease"])
    print(f"Train: {split_d.X_train.shape}")

    print("\n--- Forecasting preprocessing ---")
    split_f = preprocess_for_forecasting(datasets["forecasting"])
    print(f"Train: {split_f.X_train.shape}")
