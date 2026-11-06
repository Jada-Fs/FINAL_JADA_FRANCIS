"""
data_loader.py
==============
Loads and integrates Uganda agricultural datasets from multiple sources.

Genuine data sources documented:
  - uganda_agri_dataset.csv  : District-level climate, soil and yield
                               observations compiled from publicly available
                               FAO / FAOSTAT, NASA POWER, and SoilGrids values
                               for 30 Ugandan districts (see README for citations).
  - Crop suitability labels  : Derived from FAO EcoCrop thresholds for the crops
                               most commonly grown in Uganda.
  - Disease risk records      : Simulated from documented environmental triggers
                               (temperature × humidity windows) published in
                               plant-pathology literature.  Clearly labelled as
                               prototype.

Where farm-level data are unavailable, districts are used as spatial nodes.
Predictions are at the district level, not verified individual-farm measurements.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants – Uganda district metadata
# ---------------------------------------------------------------------------

# Uganda NARO/CIMMYT Maize Varieties & Hybrids with FAO EcoCrop-derived suitability thresholds
# (temperature range degC, rainfall mm/yr, pH range, humidity range)
CROP_THRESHOLDS: dict[str, dict] = {
    "Bazooka Hybrid": {
        "temp_min": 19, "temp_max": 33,
        "rain_min": 600, "rain_max": 1600,
        "ph_min": 5.5, "ph_max": 7.5,
        "humidity_min": 45, "humidity_max": 85,
    },
    "Longe 5 Dent": {
        "temp_min": 17, "temp_max": 28,
        "rain_min": 800, "rain_max": 1900,
        "ph_min": 5.5, "ph_max": 7.0,
        "humidity_min": 60, "humidity_max": 90,
    },
    "Longe 10H Commercial": {
        "temp_min": 18, "temp_max": 31,
        "rain_min": 750, "rain_max": 1700,
        "ph_min": 5.5, "ph_max": 7.2,
        "humidity_min": 50, "humidity_max": 88,
    },
    "Longe 7H Basin Hybrid": {
        "temp_min": 20, "temp_max": 32,
        "rain_min": 900, "rain_max": 2000,
        "ph_min": 5.2, "ph_max": 7.0,
        "humidity_min": 65, "humidity_max": 92,
    },
    "UH5051 Savannah OPV": {
        "temp_min": 21, "temp_max": 35,
        "rain_min": 500, "rain_max": 1400,
        "ph_min": 5.5, "ph_max": 7.5,
        "humidity_min": 40, "humidity_max": 80,
    },
    "Longe 4 Drought-Escape": {
        "temp_min": 21, "temp_max": 35,
        "rain_min": 450, "rain_max": 1300,
        "ph_min": 5.5, "ph_max": 7.5,
        "humidity_min": 40, "humidity_max": 78,
    },
    "NARO Maize 56D (QPM)": {
        "temp_min": 19, "temp_max": 31,
        "rain_min": 700, "rain_max": 1600,
        "ph_min": 5.5, "ph_max": 7.2,
        "humidity_min": 50, "humidity_max": 85,
    },
    "SC627 Commercial Grain": {
        "temp_min": 18, "temp_max": 30,
        "rain_min": 700, "rain_max": 1600,
        "ph_min": 5.5, "ph_max": 7.2,
        "humidity_min": 50, "humidity_max": 88,
    },
    "WEMA HB-1 Climate": {
        "temp_min": 20, "temp_max": 34,
        "rain_min": 500, "rain_max": 1500,
        "ph_min": 5.3, "ph_max": 7.4,
        "humidity_min": 42, "humidity_max": 82,
    },
    "Panner 67 Highland": {
        "temp_min": 15, "temp_max": 26,
        "rain_min": 950, "rain_max": 2200,
        "ph_min": 5.2, "ph_max": 6.8,
        "humidity_min": 65, "humidity_max": 95,
    },
}

# Maize foliar disease environmental triggers (based on plant-pathology literature)
DISEASE_TRIGGERS: dict[str, dict] = {
    "Maize Gray Leaf Spot": {
        "crop": "Maize", "temp_min": 20, "temp_max": 30,
        "humidity_min": 70, "risk_threshold": 0.55,
    },
    "Northern Corn Leaf Blight": {
        "crop": "Maize", "temp_min": 18, "temp_max": 27,
        "humidity_min": 75, "risk_threshold": 0.50,
    },
    "Maize Common Rust": {
        "crop": "Maize", "temp_min": 16, "temp_max": 25,
        "humidity_min": 65, "risk_threshold": 0.52,
    },
    "Maize Streak Virus": {
        "crop": "Maize", "temp_min": 22, "temp_max": 33,
        "humidity_min": 55, "risk_threshold": 0.45,
    },
    "Maize Head Smut": {
        "crop": "Maize", "temp_min": 21, "temp_max": 32,
        "humidity_min": 60, "risk_threshold": 0.48,
    },
}

FEATURE_COLUMNS = [
    "longitude", "latitude", "temperature", "rainfall", "humidity",
    "soil_ph", "soil_organic_carbon", "soil_moisture", "soil_nitrogen",
]

DATA_ROOT_PRIMARY = Path(__file__).parent / "datasets" / "raw"
DATA_ROOT_FALLBACK = Path(__file__).parent / "smart_agri_gnn" / "data" / "raw"


def _raw_path(filename: str, data_root: Path | None = None) -> Path:
    if data_root is not None:
        return Path(data_root) / filename
    if (DATA_ROOT_PRIMARY / filename).is_file():
        return DATA_ROOT_PRIMARY / filename
    return DATA_ROOT_FALLBACK / filename


def filter_uganda(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce strict filtering to Uganda (country_code='UGA' or country='Uganda')."""
    if "country_code" in df.columns:
        return df[df["country_code"].str.upper() == "UGA"].copy()
    if "country" in df.columns:
        return df[df["country"].str.lower() == "uganda"].copy()
    return df


def load_base_dataset(data_root: Path | None = None) -> pd.DataFrame:
    """Load the Uganda district agricultural dataset, strictly filtered to Uganda."""
    path = _raw_path("uganda_agri_dataset.csv", data_root)
    if not path.is_file():
        LOGGER.warning("Base dataset not found at %s - generating synthetic data.", path)
        return _generate_synthetic_base()
    df = pd.read_csv(path)
    df = filter_uganda(df)
    LOGGER.info("Loaded base dataset (strictly filtered to Uganda): %d rows, %d columns", len(df), len(df.columns))
    return df


def _generate_synthetic_base(n: int = 30, seed: int = 42) -> pd.DataFrame:
    """Generate a plausible synthetic Uganda dataset for development use."""
    rng = np.random.default_rng(seed)
    districts = [
        "Kampala", "Wakiso", "Mukono", "Jinja", "Iganga", "Mbale", "Sironko",
        "Kapchorwa", "Tororo", "Busia", "Lira", "Gulu", "Kitgum", "Arua",
        "Moyo", "Mbarara", "Kabale", "Bushenyi", "Kasese", "Bundibugyo",
        "Masaka", "Rakai", "Sembabule", "Soroti", "Kumi", "Katakwi",
        "Nebbi", "Hoima", "Kiboga", "Nakasongola",
    ][:n]
    return pd.DataFrame({
        "district": districts,
        "longitude": 32.58 + rng.uniform(-2.0, 2.5, n),
        "latitude": 0.35 + rng.uniform(-1.5, 3.0, n),
        "temperature": rng.uniform(18, 26, n).round(2),
        "rainfall": rng.uniform(900, 1700, n).round(1),
        "humidity": rng.uniform(62, 80, n).round(1),
        "soil_ph": rng.uniform(4.8, 7.0, n).round(2),
        "soil_organic_carbon": rng.uniform(0.5, 2.3, n).round(2),
        "soil_moisture": rng.uniform(0.22, 0.38, n).round(3),
        "soil_nitrogen": rng.uniform(1.2, 2.7, n).round(2),
        "crop_yield": rng.uniform(1.8, 3.8, n).round(3),
    })


def assign_crop_suitability_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign the best-matching crop label to each district using FAO EcoCrop
    thresholds.  Rule-based label used as supervised target for classification.

    Source: FAO EcoCrop database thresholds for Uganda-relevant crops.
    """
    df = df.copy()
    best_crop = []
    suitability_scores = []

    for _, row in df.iterrows():
        scores: dict[str, float] = {}
        for crop, th in CROP_THRESHOLDS.items():
            score = 0.0
            if th["temp_min"] <= row["temperature"] <= th["temp_max"]:
                score += 1.0
            if th["rain_min"] <= row["rainfall"] <= th["rain_max"]:
                score += 1.0
            if th["ph_min"] <= row["soil_ph"] <= th["ph_max"]:
                score += 1.0
            if "humidity" in row.index and th["humidity_min"] <= row["humidity"] <= th["humidity_max"]:
                score += 1.0
            scores[crop] = score
        top = max(scores, key=lambda k: scores[k])
        best_crop.append(top)
        suitability_scores.append(scores)

    df["suitable_crop"] = best_crop
    score_df = pd.DataFrame(suitability_scores, index=df.index)
    score_df.columns = [f"score_{c.lower().replace(' ', '_')}" for c in score_df.columns]
    df = pd.concat([df, score_df], axis=1)
    return df


def assign_disease_risk_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign binary disease-risk labels and risk probabilities.

    IMPORTANT: These labels are PROTOTYPE only. Derived from documented
    environmental-trigger ranges in plant-pathology literature, not from verified
    field disease outbreak records.
    """
    df = df.copy()
    risk_levels = []
    risk_probs = []

    for _, row in df.iterrows():
        temp = float(row.get("temperature", 22.0))
        humidity = float(row.get("humidity", 70.0))
        rainfall = float(row.get("rainfall", 1200.0))

        # Temperature suitability for pathogen incubation (spores favor 21-25 degC)
        temp_score = 1.0 if 21.0 <= temp <= 24.5 else max(0.0, 1.0 - 0.25 * abs(temp - 22.7))
        # Relative humidity drives canopy leaf wetness duration
        hum_score = 1.0 if humidity >= 73.5 else max(0.0, (humidity - 66.0) / 7.5)
        # Rainfall enhances fungal spore dispersal and bacterial splash
        rain_score = min(1.0, max(0.0, (rainfall - 1100.0) / 450.0))

        # Composite environmental disease risk index
        composite_risk = 0.45 * hum_score + 0.35 * rain_score + 0.20 * temp_score

        # Deterministic, reproducible regional variance
        seed_val = int(abs(float(row.get("longitude", 32.0))) * 1000) % (2**31 - 1)
        rng = np.random.default_rng(seed_val)
        prob = float(np.clip(composite_risk + rng.normal(0, 0.03), 0.05, 0.95))

        # High risk threshold at 0.65 (districts with combined high humidity and heavy rainfall)
        risk_level = 1 if prob >= 0.65 else 0
        risk_levels.append(risk_level)
        risk_probs.append(round(prob, 3))

    df["disease_risk"] = risk_levels
    df["disease_risk_prob"] = risk_probs
    return df


def build_multicrop_dataset(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Expand the district-level dataset by pairing each district with multiple
    crop observations.  Returns a row per (district, crop) combination.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _, base in df.iterrows():
        for crop, th in CROP_THRESHOLDS.items():
            score = 0.0
            if th["temp_min"] <= base["temperature"] <= th["temp_max"]:
                score += 0.35
            if th["rain_min"] <= base["rainfall"] <= th["rain_max"]:
                score += 0.35
            if th["ph_min"] <= base["soil_ph"] <= th["ph_max"]:
                score += 0.20
            if th["humidity_min"] <= base.get("humidity", 70) <= th["humidity_max"]:
                score += 0.10
            noise = rng.normal(0, 0.15)
            yield_est = base["crop_yield"] * (0.6 + 0.8 * score) + noise
            row = {
                "district": base["district"],
                "crop": crop,
                "longitude": base["longitude"],
                "latitude": base["latitude"],
                "temperature": base["temperature"],
                "rainfall": base["rainfall"],
                "humidity": base["humidity"],
                "soil_ph": base["soil_ph"],
                "soil_organic_carbon": base["soil_organic_carbon"],
                "soil_moisture": base["soil_moisture"],
                "soil_nitrogen": base["soil_nitrogen"],
                "suitability_score": round(score, 3),
                "suitable": int(score >= 0.55),
                "yield_estimate": round(max(0.3, yield_est), 3),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def load_climate_monthly(data_root: Path | None = None) -> pd.DataFrame:
    """Load the authentic 12-month district climatology dataset, strictly filtered to Uganda."""
    path = _raw_path("uganda_climate_monthly.csv", data_root)
    if path.is_file():
        df = pd.read_csv(path)
        return filter_uganda(df)
    return pd.DataFrame()


def load_climate_timeseries(data_root: Path | None = None) -> pd.DataFrame:
    """Load the 1991-2024 historical warming and precipitation timeseries, strictly filtered to Uganda."""
    path = _raw_path("uganda_climate_timeseries.csv", data_root)
    if path.is_file():
        df = pd.read_csv(path)
        return filter_uganda(df)
    return pd.DataFrame()


def load_spatiotemporal_dataset(data_root: Path | None = None) -> pd.DataFrame:
    """Load the authentic multi-year (2018-2023) ECMWF ERA5 spatio-temporal dataset, strictly filtered to Uganda."""
    path = _raw_path("uganda_spatiotemporal_dataset.csv", data_root)
    if path.is_file():
        df = pd.read_csv(path)
        return filter_uganda(df)
    return pd.DataFrame()


def load_all_datasets(data_root: Path | None = None) -> dict:
    """
    Load and return all datasets used by the pipeline.

    Returns a dict with keys:
      'base'              : District environmental and yield observations
      'suitability'       : Multi-crop suitability frame (classification target)
      'disease'           : District-level disease risk frame (prototype labels)
      'forecasting'       : Frame for yield regression
      'spatiotemporal'    : Multi-year 2018-2023 ECMWF ERA5 spatio-temporal dataset
      'climate_monthly'   : 12-month agro-climatology (precipitation, temperature, ET0)
      'climate_timeseries': 1991-2024 warming and drought anomaly record
      'label_encoder'     : Fitted LabelEncoder for crop names
    """
    base = load_base_dataset(data_root)
    base = assign_crop_suitability_labels(base)
    base = assign_disease_risk_labels(base)

    suitability = build_multicrop_dataset(base)
    le = LabelEncoder()
    suitability["crop_encoded"] = le.fit_transform(suitability["crop"])

    monthly = load_climate_monthly(data_root)
    timeseries = load_climate_timeseries(data_root)
    spatiotemporal = load_spatiotemporal_dataset(data_root)

    LOGGER.info(
        "Datasets ready - base: %d rows | suitability: %d rows | spatiotemporal: %d rows | monthly: %d rows | timeseries: %d rows",
        len(base), len(suitability), len(spatiotemporal), len(monthly), len(timeseries),
    )
    return {
        "base": base,
        "suitability": suitability,
        "disease": base[
            ["district", "longitude", "latitude", "temperature", "rainfall",
             "humidity", "soil_ph", "soil_organic_carbon", "soil_moisture",
             "soil_nitrogen", "disease_risk", "disease_risk_prob"]
        ].copy(),
        "forecasting": base[
            ["district", "longitude", "latitude", "temperature", "rainfall",
             "humidity", "soil_ph", "soil_organic_carbon", "soil_moisture",
             "soil_nitrogen", "crop_yield"]
        ].copy(),
        "spatiotemporal": spatiotemporal,
        "climate_monthly": monthly,
        "climate_timeseries": timeseries,
        "label_encoder": le,
        "crop_thresholds": CROP_THRESHOLDS,
        "disease_triggers": DISEASE_TRIGGERS,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    datasets = load_all_datasets()
    for name, dataset in datasets.items():
        if isinstance(dataset, pd.DataFrame):
            print(f"\n--- {name} ({len(dataset)} rows) ---")
            print(dataset.head(3))
