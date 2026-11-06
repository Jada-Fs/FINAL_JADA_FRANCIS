# Uganda Smart Agriculture GNN System

> **Using Graph Neural Networks to Model Interdependence Among Neighbouring Farms for Climate-Resilient Crop Forecasting**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-orange.svg)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/streamlit-1.32+-red.svg)](https://streamlit.io)

---

## 🌾 Project Overview

A complete AI-driven smart agriculture system focused on Uganda that uses graph neural networks to model spatial interdependence among neighbouring districts for:

1. **Crop Suitability Prediction** – which crops best match local soil and climate conditions
2. **Disease Risk Assessment** – environmental conditions conducive to crop disease outbreaks
3. **Climate-Resilient Yield Forecasting** – district-level yield estimation under current and future climate scenarios

The system compares two models:
- **Random Forest** – individual-location prediction without graph connectivity
- **GNN (GraphSAGE)** – learns from both local features and information aggregated from spatial neighbours

---

## 🏗️ Project Structure

Project_B/
├── datasets/                   # 📁 Centralized repository of all datasets used
│   ├── README.md               # Detailed schema, physical units, and data dictionary
│   ├── raw/                    # Authentic source data (UNMA, NASA POWER, SoilGrids, UBOS)
│   │   ├── uganda_agri_dataset.csv       # Primary 30-district dataset across all 10 AEZs
│   │   ├── uganda_climate_monthly.csv    # 360-row 12-month climatology & water balance
│   │   ├── uganda_climate_timeseries.csv # 1991–2024 warming trend & drought record
│   │   ├── soil_data.geojson             # ISRIC SoilGrids v2.0 spatial vector features
│   │   ├── fao_yield.csv                 # FAOSTAT historical yield statistics
│   │   └── world_bank_climate.csv        # World Bank CCKP historical climate data
│   ├── processed/              # Feature tables & JSON payloads for web & ML pipeline
│   │   ├── clean_data.parquet            # Validated parquet feature store
│   │   ├── districts_enriched.json       # Enriched district profiles & Haversine weights
│   │   └── clean_data_metadata.json      # Preprocessing logs & schema metadata
│   └── graph/                  # Spatial graph topology & adjacency matrices
│       ├── graph.npz                     # NumPy archive (X, A, edge_index, y)
│       └── graph.pt                      # PyTorch Geometric Data object (30 nodes, 300 edges)
├── dashboard.html              # Standalone Executive Web Dashboard (High-end Glassmorphism)
├── app.py                      # Streamlit interactive application (8 pages)
├── data_loader.py              # Dataset loader (supports datasets/ and smart_agri_gnn/data/)
├── preprocessing.py            # Data cleaning, imputation, scaling
├── eda.py                      # Exploratory data analysis visualizations
├── graph_builder.py            # Spatial k-NN graph construction
├── random_forest_models.py     # RF baselines for all three tasks
├── gnn_model.py                # Residual GraphSAGE classifier and regressor
├── crop_suitability.py         # Crop suitability module (FAO EcoCrop)
├── disease_risk.py             # Disease risk module & contagion simulation
├── crop_forecasting.py         # Yield forecasting module & IPCC scenarios
├── evaluation.py               # Metrics, comparison tables, plots
├── run_experiments.py          # End-to-end unified experiment runner
├── models/                     # Trained model weights (RF + GraphSAGE checkpoints)
└── results/                    # Quantitative experiment results JSON
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
# Activate virtual environment
.venv\Scripts\activate   # Windows
# OR
source .venv/bin/activate  # Linux/Mac

# Install requirements
pip install -r requirements.txt
```

### 2. (Optional) Install PyTorch for GNN training

```bash
pip install torch>=2.2 torch-geometric>=2.5
```

### 3. Train models

```bash
# Train Random Forest only (fast, no PyTorch required)
python run_experiments.py --no-gnn

# Train both RF and GNN (requires PyTorch + PyG)
python run_experiments.py
```

### 4. Launch the dashboard

```bash
streamlit run app.py
```

---

## 📊 Data Sources

| Source | Variables | Coverage |
|--------|-----------|----------|
| Uganda Agricultural Dataset | Temperature, rainfall, humidity, soil pH, SOC, moisture, N, yield | 30 Ugandan districts |
| FAO EcoCrop Database | Crop environmental thresholds | 10 Uganda crops |
| NASA POWER | Annual climate data | Point/district level |
| SoilGrids (ISRIC) | Soil physical/chemical properties | 250m resolution |
| FAOSTAT | Crop production statistics | National/regional |

---

## 🔬 Research Design

### Graph Construction
- **Nodes**: 30 Ugandan districts (or agricultural locations)
- **Node features**: Temperature, rainfall, humidity, soil pH, SOC, moisture, nitrogen, lat/lon
- **Edges**: k-nearest neighbours (k=5) using Haversine geographic distance
- **Edge weights**: Inverse geographic distance

### Model Architecture
- **Random Forest**: 200 trees, no graph connectivity
- **GNN (GraphSAGE)**: 2 message-passing layers (hidden=64), dropout=0.3

### Evaluation
- Crop suitability: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
- Disease risk: Precision, Recall, F1, ROC-AUC (emphasis on Recall for rare events)
- Yield forecasting: MAE, RMSE, R²

---

## ⚠️ Important Disclaimers

1. **Disease risk labels are PROTOTYPE only** – derived from environmental triggers in plant-pathology literature, NOT from verified field outbreak records.
2. **District-level predictions** – not individual farm-level predictions.
3. **No temporal dynamics** – cross-sectional data without seasonal time series.
4. **Model outputs are NOT ground truth** – all predictions must be confirmed by qualified agricultural extension officers.

---

## 📋 Module Reference

| Module | Purpose |
|--------|---------|
| `data_loader.py` | Load Uganda data, generate FAO EcoCrop suitability labels, prototype disease labels |
| `preprocessing.py` | Split-aware cleaning, imputation, outlier capping, scaling, class imbalance reporting |
| `eda.py` | Distribution plots, correlation heatmaps, geographic scatters, bivariate charts |
| `graph_builder.py` | Haversine k-NN graph, edge validation, PyG conversion |
| `random_forest_models.py` | RF classifiers and regressor with calibration, feature importance, CV |
| `gnn_model.py` | GraphSAGE with transductive training, early stopping, graceful fallback |
| `crop_suitability.py` | Rule-based scoring, ML prediction wrapper, ranked recommendations |
| `disease_risk.py` | Environmental trigger scoring, neighbour outbreak context |
| `crop_forecasting.py` | Yield prediction, feature engineering, climate scenario simulation |
| `evaluation.py` | Metrics, comparison tables, all visualization functions |
| `run_experiments.py` | End-to-end pipeline, saves models and results |
| `app.py` | 8-page Streamlit dashboard |

---

## 📄 Citation

```
Uganda Smart Agriculture GNN System (2024)
Research prototype for GNN-based agricultural prediction.
Data: FAOSTAT, NASA POWER, SoilGrids, FAO EcoCrop.
```
