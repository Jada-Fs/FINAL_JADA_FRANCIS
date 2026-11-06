"""
graph_builder.py
================
Constructs a spatial k-NN graph from Uganda district data for use by the GNN.

Nodes  : districts (or agricultural locations)
Edges  : k-nearest neighbours by geographic distance (Haversine)
Weights: inverse distance (optional)

The graph allows the GNN to aggregate information from connected neighbouring
districts.  The same core features are available to Random Forest without the
graph connectivity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

LOGGER = logging.getLogger(__name__)

# Haversine earth radius (km)
EARTH_RADIUS_KM = 6371.0


@dataclass
class AgriculturalGraph:
    """Undirected k-NN graph stored as numpy arrays."""
    node_features: np.ndarray        # shape (N, F)
    targets: np.ndarray              # shape (N,)
    edge_index: np.ndarray           # shape (2, E)  – undirected (both directions)
    edge_weights: np.ndarray         # shape (E,)    – inverse distance
    coords: np.ndarray               # shape (N, 2)  – (lat, lon) for visualisation
    node_ids: list[str]              # district names
    feature_names: list[str]         # feature column names
    target_name: str
    n_neighbors: int

    @property
    def num_nodes(self) -> int:
        return int(self.node_features.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def num_features(self) -> int:
        return int(self.node_features.shape[1])

    def summary(self) -> dict:
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "num_features": self.num_features,
            "avg_degree": self.num_edges / self.num_nodes if self.num_nodes > 0 else 0,
            "n_neighbors": self.n_neighbors,
            "target": self.target_name,
            "features": self.feature_names,
        }


def haversine_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Haversine distances (km) between lat/lon points.

    Parameters
    ----------
    coords : ndarray of shape (N, 2) – columns are [latitude, longitude] in degrees.

    Returns
    -------
    dist_matrix : ndarray of shape (N, N) in kilometres.
    """
    lat = np.radians(coords[:, 0])
    lon = np.radians(coords[:, 1])

    # Broadcasting
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]

    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return EARTH_RADIUS_KM * c


def build_knn_edges(
    coords: np.ndarray,
    n_neighbors: int = 5,
    use_haversine: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build undirected k-NN edges from geographic coordinates.

    Returns
    -------
    edge_index : ndarray of shape (2, 2*E)
    edge_weights : ndarray of shape (2*E,) – inverse distance weights
    """
    n = len(coords)
    if n < 2:
        raise ValueError("At least two nodes are required to build a graph.")
    k = min(n_neighbors + 1, n)  # +1 because kNN includes self

    if use_haversine:
        dist_matrix = haversine_matrix(coords)
        # Use sklearn with precomputed distances
        nbrs = NearestNeighbors(n_neighbors=k, metric="precomputed")
        nbrs.fit(dist_matrix)
        distances, indices = nbrs.kneighbors(dist_matrix)
    else:
        nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean")
        nbrs.fit(coords)
        distances, indices = nbrs.kneighbors(coords)

    # Skip self (index 0 from kneighbors includes self at distance 0)
    src_list, dst_list, w_list = [], [], []
    for i in range(n):
        for j_idx in range(1, k):
            j = int(indices[i, j_idx])
            d = float(distances[i, j_idx])
            w = 1.0 / (d + 1e-6)  # avoid zero division
            src_list.append(i)
            dst_list.append(j)
            w_list.append(w)

    src = np.array(src_list, dtype=np.int64)
    dst = np.array(dst_list, dtype=np.int64)
    w = np.array(w_list, dtype=np.float32)

    # Make undirected: add reversed edges
    edge_index = np.vstack([
        np.concatenate([src, dst]),
        np.concatenate([dst, src])
    ])
    edge_weights = np.concatenate([w, w])

    return edge_index, edge_weights


def build_graph_from_dataframe(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    id_column: str = "district",
    n_neighbors: int = 5,
    scale_features: bool = True,
) -> AgriculturalGraph:
    """
    Build an AgriculturalGraph from a district-level dataframe.

    Parameters
    ----------
    df : DataFrame with district data
    feature_columns : Columns to use as node features
    target_column : Column to use as prediction target
    id_column : Column containing node identifiers (district names)
    n_neighbors : Number of spatial neighbours per node
    scale_features : Whether to z-score node features

    Returns
    -------
    AgriculturalGraph
    """
    df = df.copy().reset_index(drop=True)

    # Ensure geographic columns
    if "latitude" not in df.columns or "longitude" not in df.columns:
        raise ValueError("DataFrame must contain 'latitude' and 'longitude' columns.")

    # Aggregate to district level if duplicates (e.g. multi-crop df)
    if df[id_column].duplicated().any():
        agg_cols = {col: "mean" for col in feature_columns if col in df.columns and col != id_column}
        agg_cols[target_column] = "mean"
        agg_cols["latitude"] = "first"
        agg_cols["longitude"] = "first"
        df = df.groupby(id_column, as_index=False).agg(agg_cols)
        LOGGER.info("Aggregated to %d unique nodes by '%s'.", len(df), id_column)

    node_ids = df[id_column].tolist()
    coords = df[["latitude", "longitude"]].values.astype(np.float64)

    available_features = [c for c in feature_columns if c in df.columns]
    X = df[available_features].values.astype(np.float32)

    if scale_features:
        means = X.mean(axis=0, keepdims=True)
        stds = X.std(axis=0, keepdims=True) + 1e-8
        X = (X - means) / stds

    y = df[target_column].values.astype(np.float32)

    edge_index, edge_weights = build_knn_edges(coords, n_neighbors=n_neighbors)

    graph = AgriculturalGraph(
        node_features=X,
        targets=y,
        edge_index=edge_index,
        edge_weights=edge_weights,
        coords=coords,
        node_ids=node_ids,
        feature_names=available_features,
        target_name=target_column,
        n_neighbors=n_neighbors,
    )

    LOGGER.info(
        "Graph built: %d nodes, %d edges (avg degree %.1f), %d features",
        graph.num_nodes, graph.num_edges,
        graph.num_edges / graph.num_nodes,
        graph.num_features,
    )
    return graph


def get_neighbour_info(graph: AgriculturalGraph, node_idx: int) -> dict:
    """Return neighbour indices and distances for a single node."""
    edge_src = graph.edge_index[0]
    edge_dst = graph.edge_index[1]
    neighbour_mask = edge_src == node_idx
    neighbours = edge_dst[neighbour_mask]
    weights = graph.edge_weights[neighbour_mask]
    distances_km = 1.0 / (weights + 1e-12)
    return {
        "node": graph.node_ids[node_idx],
        "neighbours": [graph.node_ids[i] for i in neighbours],
        "distances_km": distances_km.tolist(),
    }


def validate_graph(graph: AgriculturalGraph) -> list[str]:
    """
    Run basic graph validation checks.

    Returns a list of warning strings (empty if all checks pass).
    """
    warnings = []
    if graph.num_nodes < 5:
        warnings.append(f"Very small graph: only {graph.num_nodes} nodes.")
    if graph.num_edges == 0:
        warnings.append("Graph has no edges – GNN message passing will be trivial.")
    if np.isnan(graph.node_features).any():
        warnings.append("Node features contain NaN values.")
    if np.isnan(graph.targets).any():
        warnings.append("Target values contain NaN values.")
    degrees = np.bincount(graph.edge_index[0], minlength=graph.num_nodes)
    isolated = int((degrees == 0).sum())
    if isolated > 0:
        warnings.append(f"{isolated} isolated node(s) with no edges.")
    if warnings:
        for w in warnings:
            LOGGER.warning("Graph validation: %s", w)
    else:
        LOGGER.info("Graph validation passed.")
    return warnings


def graph_to_pyg(graph: AgriculturalGraph):
    """
    Convert AgriculturalGraph to a PyTorch Geometric Data object.

    Returns None if torch_geometric is not installed.
    """
    try:
        import torch
        from torch_geometric.data import Data
        return Data(
            x=torch.tensor(graph.node_features, dtype=torch.float32),
            edge_index=torch.tensor(graph.edge_index, dtype=torch.long),
            edge_attr=torch.tensor(graph.edge_weights, dtype=torch.float32),
            y=torch.tensor(graph.targets, dtype=torch.float32),
        )
    except ImportError:
        LOGGER.warning("torch_geometric not available – returning None.")
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_all_datasets
    from preprocessing import FORECASTING_FEATURES

    datasets = load_all_datasets()
    df = datasets["forecasting"]

    graph = build_graph_from_dataframe(
        df,
        feature_columns=FORECASTING_FEATURES,
        target_column="crop_yield",
        n_neighbors=5,
    )
    print(graph.summary())

    info = get_neighbour_info(graph, 0)
    print(f"\nNode '{info['node']}' neighbours: {info['neighbours']}")

    warnings = validate_graph(graph)
    if not warnings:
        print("\nGraph validation: PASSED")
