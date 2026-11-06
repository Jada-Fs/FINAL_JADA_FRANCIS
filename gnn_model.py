"""
gnn_model.py
============
Graph Neural Network models for the Uganda Smart Agriculture project.

Implements:
  1. GraphSAGE-based classifier for crop suitability and disease risk
  2. GraphSAGE-based regressor for crop yield forecasting
  3. GCN-based alternatives for comparison
  4. Training and evaluation routines with early stopping

Architecture:
  - Two GraphSAGE convolution layers with ReLU and dropout
  - Task-specific output head (classification or regression)
  - Same core node features as the corresponding Random Forest baseline
  - Additional information: graph connectivity and neighbour aggregation

PyTorch Geometric is required.  If not installed, a graceful fallback is used.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

LOGGER = logging.getLogger(__name__)


class TorchUnavailableError(RuntimeError):
    """Raised when the optional PyTorch / PyG stack cannot be imported."""


def _try_import_torch():
    """Attempt to import PyTorch and PyG, returning (torch, F, SAGEConv, GCNConv) or None."""
    try:
        import torch
        import torch.nn.functional as F
        from torch_geometric.nn import GCNConv, SAGEConv
        return torch, F, SAGEConv, GCNConv
    except Exception as exc:
        LOGGER.warning("PyTorch / PyG not available: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_sage_classifier(in_channels: int, hidden_channels: int = 32, num_classes: int = 2, dropout: float = 0.2):
    """Build a Residual GraphSAGE classifier with LayerNorm and dense feature skip."""
    result = _try_import_torch()
    if result is None:
        raise TorchUnavailableError("torch_geometric required for GNN training.")
    torch, F, SAGEConv, _ = result

    class SAGEClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin0 = torch.nn.Linear(in_channels, hidden_channels)
            self.conv1 = SAGEConv(hidden_channels, hidden_channels)
            self.norm1 = torch.nn.LayerNorm(hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
            self.norm2 = torch.nn.LayerNorm(hidden_channels)
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels + in_channels, hidden_channels // 2),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_channels // 2, num_classes),
            )

        def forward(self, x, edge_index):
            h0 = F.gelu(self.lin0(x))
            m1 = self.conv1(h0, edge_index)
            h1 = F.gelu(self.norm1(m1 + h0))
            m2 = self.conv2(h1, edge_index)
            h2 = F.gelu(self.norm2(m2 + h1))
            return self.fc(torch.cat([h2, x], dim=-1))

        def predict_proba(self, x, edge_index):
            logits = self.forward(x, edge_index)
            return F.softmax(logits, dim=-1)

    return SAGEClassifier()


def build_sage_regressor(in_channels: int, hidden_channels: int = 32, dropout: float = 0.2):
    """Build a Residual GraphSAGE regressor with LayerNorm and dense feature skip."""
    result = _try_import_torch()
    if result is None:
        raise TorchUnavailableError("torch_geometric required for GNN training.")
    torch, F, SAGEConv, _ = result

    class SAGERegressor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin0 = torch.nn.Linear(in_channels, hidden_channels)
            self.conv1 = SAGEConv(hidden_channels, hidden_channels)
            self.norm1 = torch.nn.LayerNorm(hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
            self.norm2 = torch.nn.LayerNorm(hidden_channels)
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels + in_channels, hidden_channels // 2),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_channels // 2, 1),
            )

        def forward(self, x, edge_index):
            h0 = F.gelu(self.lin0(x))
            m1 = self.conv1(h0, edge_index)
            h1 = F.gelu(self.norm1(m1 + h0))
            m2 = self.conv2(h1, edge_index)
            h2 = F.gelu(self.norm2(m2 + h1))
            return self.fc(torch.cat([h2, x], dim=-1)).squeeze(-1)

    return SAGERegressor()


# ---------------------------------------------------------------------------
# Training results
# ---------------------------------------------------------------------------

@dataclass
class GNNTrainResult:
    task: str
    train_losses: list[float]
    val_metric: float          # accuracy for classification, RMSE for regression
    best_epoch: int
    predictions: np.ndarray    # on test nodes
    true_labels: np.ndarray
    probabilities: Optional[np.ndarray] = None  # for classification


# ---------------------------------------------------------------------------
# Training routine
# ---------------------------------------------------------------------------

def train_gnn(
    node_features: np.ndarray,
    targets: np.ndarray,
    edge_index: np.ndarray,
    task: str = "regression",  # "regression" | "classification"
    num_classes: int = 2,
    hidden_channels: int = 64,
    epochs: int = 300,
    lr: float = 0.01,
    patience: int = 25,
    min_epochs: int = 35,
    dropout: float = 0.2,
    random_state: int = 42,
    save_path: Optional[Path] = None,
    test_mask: Optional[np.ndarray] = None,
) -> GNNTrainResult:
    """
    Train a GraphSAGE model on the full graph with node-level masks.

    Transductive training: the model sees all nodes during training but
    is only evaluated on held-out test nodes.

    Parameters
    ----------
    node_features : scaled node feature matrix (N, F)
    targets : target array (N,)
    edge_index : edge connectivity (2, E)
    task : 'regression' or 'classification'
    num_classes : number of classes for classification
    hidden_channels : GNN hidden layer size
    epochs : maximum training epochs
    lr : Adam learning rate
    patience : early stopping patience on validation metric
    dropout : dropout rate
    random_state : seed for train/val/test split
    save_path : optional path to save best model weights
    test_mask : optional pre-defined boolean test mask; if None, 20% of nodes held out

    Returns
    -------
    GNNTrainResult
    """
    result = _try_import_torch()
    if result is None:
        raise TorchUnavailableError("Install torch and torch_geometric to use GNN training.")
    torch, F, SAGEConv, _ = result

    from sklearn.metrics import accuracy_score, mean_squared_error

    n = len(targets)
    if test_mask is not None:
        test_idx = np.where(test_mask)[0]
        train_val_idx = np.where(~test_mask)[0]
        val_split = max(1, int(0.2 * len(train_val_idx)))
        train_idx = train_val_idx[:-val_split]
        val_idx = train_val_idx[-val_split:]
    else:
        if task == "classification" and len(np.unique(targets)) > 1:
            from sklearn.model_selection import train_test_split
            train_val_idx, test_idx = train_test_split(
                np.arange(n), test_size=0.2, random_state=random_state, stratify=targets
            )
            val_size = max(1, int(0.25 * len(train_val_idx)))
            train_idx, val_idx = train_test_split(
                train_val_idx, test_size=val_size, random_state=random_state,
                stratify=targets[train_val_idx] if len(np.unique(targets[train_val_idx])) > 1 else None
            )
        else:
            rng = np.random.default_rng(random_state)
            perm = rng.permutation(n)
            split = int(0.8 * n)
            train_val_idx = perm[:split]
            test_idx = perm[split:]
            val_split = max(1, int(0.2 * len(train_val_idx)))
            train_idx = train_val_idx[:len(train_val_idx) - val_split]
            val_idx = train_val_idx[len(train_val_idx) - val_split:]

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask_t = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask_t[test_idx] = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(node_features, dtype=torch.float32).to(device)
    edge_idx = torch.tensor(edge_index, dtype=torch.long).to(device)

    scaler_y = None
    if task == "classification":
        y = torch.tensor(targets, dtype=torch.long).to(device)
        model = build_sage_classifier(x.shape[1], hidden_channels, num_classes, dropout).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        def score_fn(pred, true_idx):
            return accuracy_score(targets[true_idx], pred)
        higher_is_better = True
    else:
        from sklearn.preprocessing import StandardScaler
        scaler_y = StandardScaler()
        y_scaled_tr = scaler_y.fit_transform(targets[train_idx].reshape(-1, 1)).ravel()
        y_scaled_all = scaler_y.transform(targets.reshape(-1, 1)).ravel()
        y = torch.tensor(y_scaled_all, dtype=torch.float32).to(device)
        model = build_sage_regressor(x.shape[1], hidden_channels, dropout).to(device)
        criterion = torch.nn.MSELoss()
        def score_fn(pred, true_idx):
            pred_orig = scaler_y.inverse_transform(pred.reshape(-1, 1)).ravel()
            return float(np.sqrt(mean_squared_error(targets[true_idx], pred_orig)))
        higher_is_better = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_metric = -np.inf if higher_is_better else np.inf
    best_val_loss = np.inf
    best_epoch = 0
    patience_counter = 0
    best_state = None
    train_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_idx)
        loss = criterion(out[train_mask], y[train_mask])

        loss.backward()
        optimizer.step()
        scheduler.step()
        train_losses.append(float(loss.item()))

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(x, edge_idx)
            val_loss = float(criterion(val_out[val_mask], y[val_mask]).item())
            if task == "classification":
                val_pred = val_out[val_mask].argmax(dim=-1).cpu().numpy()
            else:
                val_pred = val_out[val_mask].cpu().numpy()

        metric = score_fn(val_pred, val_idx)

        # Track best model based on validation loss (with minimum warmup epochs)
        improved = (val_loss < best_val_loss)
        if improved and epoch >= min_epochs:
            best_val_loss = val_loss
            best_metric = metric
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif epoch >= min_epochs:
            patience_counter += 1
            if patience_counter >= patience:
                LOGGER.info("Early stopping at epoch %d (best: epoch %d with val_loss=%.4f)", epoch, best_epoch, best_val_loss)
                break

        if epoch % 50 == 0:
            LOGGER.info("Epoch %3d | loss=%.4f | val_metric=%.4f", epoch, loss.item(), metric)

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, save_path)
        LOGGER.info("Best GNN weights saved to %s", save_path)
        if scaler_y is not None:
            import joblib
            scaler_path = save_path.parent / (save_path.stem + "_scaler_y.joblib")
            joblib.dump(scaler_y, scaler_path)

    # Test evaluation
    model.eval()
    with torch.no_grad():
        test_out = model(x, edge_idx)
        if task == "classification":
            test_proba = model.predict_proba(x, edge_idx)[test_mask_t].cpu().numpy()
            test_pred = test_out[test_mask_t].argmax(dim=-1).cpu().numpy()
            test_true = targets[test_idx]
        else:
            test_pred_scaled = test_out[test_mask_t].cpu().numpy()
            test_pred = scaler_y.inverse_transform(test_pred_scaled.reshape(-1, 1)).ravel()
            test_proba = None
            test_true = targets[test_idx]

    return GNNTrainResult(
        task=task,
        train_losses=train_losses,
        val_metric=best_metric,
        best_epoch=best_epoch,
        predictions=test_pred,
        true_labels=test_true,
        probabilities=test_proba,
    )


def predict_gnn(
    node_features: np.ndarray,
    edge_index: np.ndarray,
    model_path: Path,
    task: str = "regression",
    num_classes: int = 2,
    hidden_channels: int = 32,
) -> np.ndarray:
    """
    Load saved GNN weights and produce predictions for all nodes.

    Returns probability array (classification) or value array (regression).
    """
    result = _try_import_torch()
    if result is None:
        raise TorchUnavailableError("torch_geometric required for inference.")
    torch, F, _, _ = result

    device = torch.device("cpu")
    x = torch.tensor(node_features, dtype=torch.float32).to(device)
    edge_idx = torch.tensor(edge_index, dtype=torch.long).to(device)
    in_channels = node_features.shape[1]

    if task == "classification":
        model = build_sage_classifier(in_channels, hidden_channels, num_classes).to(device)
    else:
        model = build_sage_regressor(in_channels, hidden_channels).to(device)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        out = model(x, edge_idx)
        if task == "classification":
            return F.softmax(out, dim=-1).cpu().numpy()
        else:
            import joblib
            scaler_path = model_path.parent / (model_path.stem + "_scaler_y.joblib")
            raw_vals = out.cpu().numpy()
            if scaler_path.is_file():
                scaler_y = joblib.load(scaler_path)
                return scaler_y.inverse_transform(raw_vals.reshape(-1, 1)).ravel()
            return raw_vals


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))

    result = _try_import_torch()
    if result is None:
        print("PyTorch not available – skipping GNN test.")
    else:
        from data_loader import load_all_datasets
        from graph_builder import build_graph_from_dataframe
        from preprocessing import FORECASTING_FEATURES

        datasets = load_all_datasets()
        graph = build_graph_from_dataframe(
            datasets["forecasting"],
            feature_columns=FORECASTING_FEATURES,
            target_column="crop_yield",
        )
        train_result = train_gnn(
            graph.node_features, graph.targets, graph.edge_index,
            task="regression", epochs=100, patience=15,
        )
        print(f"Best epoch: {train_result.best_epoch}")
        print(f"Val RMSE: {train_result.val_metric:.4f}")
        print(f"Predictions: {train_result.predictions[:5]}")
