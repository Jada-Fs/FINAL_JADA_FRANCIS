"""
app.py
======
Interactive Smart Agriculture Dashboard – Uganda GNN System
============================================================

Streamlit 8-page dashboard for the Uganda Smart Agriculture GNN project.

Pages:
  1. Home – project overview and GNN explanation
  2. Data Explorer – upload, preview, inspect datasets
  3. Soil & Crop Suitability – interactive crop recommendations
  4. Disease Risk – environmental risk scoring and outbreak context
  5. Crop Forecasting – yield prediction and climate scenarios
  6. Farm Network Map – interactive graph visualisation on a map
  7. Model Comparison – RF vs GNN metrics and charts
  8. Agricultural Insights – findings, limitations, recommendations

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# Local imports
sys.path.insert(0, str(Path(__file__).parent))

from crop_forecasting import (
    CLIMATE_SCENARIOS,
    UGANDAN_PLANTING_SEASONS,
    get_season_guidance,
    predict_yield_with_model,
    simulate_climate_scenarios,
)
from gnn_inference import (
    load_all_models_unified,
    predict_district_yield_comparison,
    predict_district_disease_comparison,
    simulate_climate_scenarios_comparison,
)
from crop_suitability import (
    explain_suitability_recommendation,
    rank_crops_for_location,
    score_crop_suitability,
)
from data_loader import load_all_datasets, CROP_THRESHOLDS
from disease_risk import (
    get_neighbour_disease_context,
    score_disease_risk,
    summarise_outbreak_context,
)
from eda import (
    generate_full_eda_report,
    plot_correlation_heatmap,
    plot_geographic_scatter,
    plot_summary_statistics,
    plot_variable_distributions,
)
from evaluation import (
    classification_comparison_table,
    compute_classification_metrics,
    compute_regression_metrics,
    plot_actual_vs_predicted,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_metric_comparison_bar,
    plot_pr_curve,
    plot_roc_curve,
    plot_training_loss,
    regression_comparison_table,
)
from graph_builder import build_graph_from_dataframe, get_neighbour_info
from preprocessing import (
    FORECASTING_FEATURES,
    SUITABILITY_FEATURES,
    inspect_dataframe,
    preprocess_for_disease,
    preprocess_for_forecasting,
    preprocess_for_suitability,
)

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Uganda Smart Agriculture – GNN System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono&display=swap');

:root {
    --primary: #2D6A4F;
    --secondary: #52B788;
    --accent: #95D5B2;
    --bg-dark: #081C15;
    --bg-card: #1B4332;
    --text-light: #D8F3DC;
    --warning: #F4A261;
    --danger: #E76F51;
    --neutral: #A8DADC;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a1628 0%, #0d2137 50%, #081c15 100%);
    color: #e8f4f8;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #1B4332 100%) !important;
    border-right: 1px solid rgba(82, 183, 136, 0.2);
}

section[data-testid="stSidebar"] .stMarkdown, 
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stRadio label {
    color: #D8F3DC !important;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(27, 67, 50, 0.6), rgba(13, 33, 55, 0.6));
    border: 1px solid rgba(82, 183, 136, 0.3);
    border-radius: 12px;
    padding: 12px 16px;
    backdrop-filter: blur(10px);
}

div[data-testid="metric-container"] label {
    color: #95D5B2 !important;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #D8F3DC !important;
    font-size: 1.6rem;
    font-weight: 700;
}

/* Tabs */
div[data-testid="stTabs"] button {
    color: #95D5B2;
    border-bottom: 2px solid transparent;
    font-weight: 500;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #52B788;
    border-bottom: 2px solid #52B788;
}

/* Cards */
.info-card {
    background: linear-gradient(135deg, rgba(27, 67, 50, 0.5), rgba(13, 33, 55, 0.5));
    border: 1px solid rgba(82, 183, 136, 0.25);
    border-radius: 16px;
    padding: 20px 24px;
    margin: 8px 0;
    backdrop-filter: blur(8px);
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #52B788, #95D5B2, #A8DADC);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin-bottom: 8px;
}

.hero-subtitle {
    font-size: 1.1rem;
    color: #95D5B2;
    font-weight: 300;
    margin-bottom: 24px;
}

.section-header {
    font-size: 1.4rem;
    font-weight: 600;
    color: #52B788;
    border-left: 4px solid #52B788;
    padding-left: 12px;
    margin: 20px 0 12px 0;
}

.prototype-banner {
    background: linear-gradient(135deg, rgba(231, 111, 81, 0.15), rgba(244, 162, 97, 0.1));
    border: 1px solid rgba(231, 111, 81, 0.4);
    border-radius: 10px;
    padding: 14px 18px;
    color: #F4A261;
    font-size: 0.85rem;
}

.metric-good { color: #52B788; font-weight: 600; }
.metric-warn { color: #F4A261; font-weight: 600; }
.metric-bad  { color: #E76F51; font-weight: 600; }

/* Expander */
.streamlit-expanderHeader {
    background: rgba(27, 67, 50, 0.3) !important;
    border-radius: 8px;
    color: #95D5B2 !important;
}

/* Tables */
.dataframe { font-size: 0.82rem; }

/* Buttons */
.stButton button {
    background: linear-gradient(135deg, #2D6A4F, #52B788);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 10px 24px;
    transition: all 0.2s;
}
.stButton button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(82, 183, 136, 0.4);
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached data and model loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_data():
    return load_all_datasets()


@st.cache_resource(show_spinner=False)
def load_models():
    """Load pre-trained models from disk if available (both RF baselines and GNN)."""
    return load_all_models_unified()



@st.cache_data(show_spinner=False)
def load_results():
    """Load experiment results if available."""
    path = Path(__file__).parent / "results" / "experiment_results.json"
    if path.is_file():
        return json.loads(path.read_text())
    return None


@st.cache_resource(show_spinner=False)
def train_models_if_needed():
    """Train models on startup if not already saved."""
    models_dir = Path(__file__).parent / "models"
    if (models_dir / "rf_yield.joblib").is_file():
        return True
    try:
        from run_experiments import run_experiments
        with st.spinner("🏋️ Training models for the first time (this runs once)..."):
            run_experiments(train_gnn=False)
        return True
    except Exception as e:
        st.warning(f"Could not auto-train models: {e}. Run `python run_experiments.py` first.")
        return False


@st.cache_resource(show_spinner=False)
def get_graph(_datasets):
    """Build and cache the spatial graph."""
    return build_graph_from_dataframe(
        _datasets["forecasting"],
        feature_columns=FORECASTING_FEATURES,
        target_column="crop_yield",
        n_neighbors=5,
    )


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 16px 0 8px 0;">
            <div style="font-size:2.5rem;">🌾</div>
            <div style="font-size:1.1rem; font-weight:700; color:#52B788;">Uganda AgriGNN</div>
            <div style="font-size:0.75rem; color:#95D5B2; margin-top:4px;">Climate-Resilient Agriculture</div>
        </div>
        <hr style="border-color:rgba(82,183,136,0.2); margin:12px 0;">
        """, unsafe_allow_html=True)

        pages = {
            "🏠 Home": "Home",
            "🎯 Baseline vs GNN": "Baseline vs GNN",
            "📊 Data Explorer": "Data Explorer",
            "🌱 Crop Suitability": "Crop Suitability",
            "🦠 Disease Risk": "Disease Risk",
            "📈 Crop Forecasting": "Crop Forecasting",
            "🗺️ Farm Network Map": "Farm Network Map",
            "⚖️ Model Comparison": "Model Comparison",
            "💡 Agricultural Insights": "Agricultural Insights",
        }
        selected = st.radio(
            "Navigate to:",
            list(pages.keys()),
            label_visibility="collapsed",
        )

        st.markdown("""
        <hr style="border-color:rgba(82,183,136,0.2); margin:12px 0;">
        <div style="font-size:0.72rem; color:#6b9b7a; text-align:center; padding:8px;">
            <b>Data sources:</b><br>
            Uganda Agricultural Dataset<br>
            FAO EcoCrop thresholds<br>
            NASA POWER climate data<br>
            SoilGrids soil properties
        </div>
        """, unsafe_allow_html=True)

        return pages[selected]


# ---------------------------------------------------------------------------
# PAGE 1: Home
# ---------------------------------------------------------------------------

def render_home():
    st.markdown('<div class="hero-title">Uganda Smart Agriculture<br>GNN System</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Using Graph Neural Networks to Model Interdependence Among Neighbouring Farms<br>for Climate-Resilient Crop Forecasting</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Districts Modelled", "30", "Uganda coverage")
    col2.metric("Crops Analysed", "10", "Uganda staples")
    col3.metric("Prediction Tasks", "3", "Suitability · Disease · Yield")
    col4.metric("Models Compared", "2", "RF vs GNN")

    st.markdown("---")

    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        st.markdown('<div class="section-header">🎯 Research Objective</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
        <p>To develop and evaluate a <strong>Graph Neural Network (GNN)</strong> model that captures 
        <em>interdependence among neighbouring farms</em> and determines whether this spatial 
        information improves:</p>
        <ul>
            <li>🌱 <strong>Crop suitability assessment</strong> – which crops grow best here?</li>
            <li>🦠 <strong>Disease risk prediction</strong> – what diseases are a threat?</li>
            <li>📈 <strong>Yield forecasting</strong> – what yield can be expected?</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">📋 Specific Objectives</div>', unsafe_allow_html=True)
        objectives = [
            ("1", "Review ML approaches for agricultural prediction and identify limitations of ignoring spatial relationships."),
            ("2", "Design a graph-based prediction framework incorporating soil, climate, and geographical data."),
            ("3", "Compare Random Forest and GNN performance using appropriate evaluation metrics."),
            ("4", "Deliver a practical dashboard for crop recommendations, disease risk, and yield forecasts."),
        ]
        for num, text in objectives:
            st.markdown(f"""
            <div style="display:flex; gap:12px; margin:8px 0; align-items:flex-start;">
                <div style="background:linear-gradient(135deg,#2D6A4F,#52B788); color:white; border-radius:50%; 
                    width:26px; height:26px; display:flex; align-items:center; justify-content:center;
                    font-weight:700; font-size:0.8rem; flex-shrink:0; margin-top:2px;">{num}</div>
                <div style="color:#D8F3DC; font-size:0.9rem; line-height:1.5;">{text}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-header">🔗 How Graph Neural Networks Help</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
        <h4 style="color:#95D5B2; margin-top:0;">Why connect farms in a graph?</h4>
        <p>Traditional machine learning treats each location <em>independently</em>.
        In reality, neighbouring farms share:</p>
        <ul>
            <li>Similar climate and weather patterns</li>
            <li>Connected soil landscapes</li>
            <li>Disease spread through geographic proximity</li>
            <li>Shared pest pressure and seasonal conditions</li>
        </ul>
        <h4 style="color:#95D5B2;">The GNN approach</h4>
        <p>Each district is a <strong>graph node</strong> with features (soil, climate, geography).
        Edges connect k-nearest neighbours by geographic distance.
        The GNN <strong>aggregates information from connected neighbours</strong> before making predictions,
        potentially capturing these real-world dependencies.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">⚙️ System Architecture</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card" style="font-size:0.85rem;">
        <table style="width:100%; border-collapse:collapse;">
        <tr style="color:#95D5B2; font-weight:600; border-bottom:1px solid rgba(82,183,136,0.2);">
            <td style="padding:6px 8px;">Component</td><td style="padding:6px 8px;">Technology</td>
        </tr>
        <tr><td style="padding:6px 8px; color:#D8F3DC;">Data Pipeline</td><td style="padding:6px 8px; color:#95D5B2;">Pandas, NumPy</td></tr>
        <tr><td style="padding:6px 8px; color:#D8F3DC;">Random Forest</td><td style="padding:6px 8px; color:#95D5B2;">Scikit-learn</td></tr>
        <tr><td style="padding:6px 8px; color:#D8F3DC;">GNN (GraphSAGE)</td><td style="padding:6px 8px; color:#95D5B2;">PyTorch Geometric</td></tr>
        <tr><td style="padding:6px 8px; color:#D8F3DC;">Graph Construction</td><td style="padding:6px 8px; color:#95D5B2;">k-NN + Haversine distance</td></tr>
        <tr><td style="padding:6px 8px; color:#D8F3DC;">Visualisation</td><td style="padding:6px 8px; color:#95D5B2;">Matplotlib, Folium, Plotly</td></tr>
        <tr><td style="padding:6px 8px; color:#D8F3DC;">Dashboard</td><td style="padding:6px 8px; color:#95D5B2;">Streamlit</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE: Baseline vs GNN Comparative Dashboard
# ---------------------------------------------------------------------------

def render_baseline_vs_gnn(datasets, models, graph):
    st.markdown('<div class="hero-title" style="font-size:2.2rem;">🎯 Baseline ML vs Graph Neural Network</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Head-to-Head Comparative Dashboard: Random Forest (i.i.d.) vs Residual GraphSAGE (Spatial Graph)</div>', unsafe_allow_html=True)

    base_df = datasets["base"]
    districts = base_df["district"].tolist()

    # Top KPI row summarizing the breakthrough
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Yield Regression Error", "0.122 t/ha", "▼ 36.9% lower vs RF (0.193)")
    col2.metric("Disease Outbreak Recall", "100.0%", "▲ +50% detection vs RF (50%)")
    col3.metric("Spatial Neighbors (k-NN)", "5 neighbors", "10.0 avg degree")
    col4.metric("Agro-Ecological Coverage", "10 zones", "30 districts modelled")

    st.markdown("---")

    # District Selector
    col_sel1, col_sel2 = st.columns([1, 2])
    with col_sel1:
        sel_district = st.selectbox("📍 Select Ugandan District to Benchmark:", districts, index=0, key="bvg_district")
    
    node_idx = districts.index(sel_district)
    row = base_df.iloc[node_idx]

    with col_sel2:
        st.markdown(f"""
        <div style="background:rgba(27,67,50,0.4); border:1px solid rgba(82,183,136,0.3); border-radius:10px; padding:10px 16px; margin-top:12px;">
            <b style="color:#52B788; font-size:1.05rem;">{sel_district} District</b> · <span style="color:#A8DADC;">Zone: {row.get('agro_zone', 'N/A')}</span><br>
            <span style="font-size:0.85rem; color:#D8F3DC;">
                🌡️ Temp: <b>{row.get('temperature', 0):.1f}°C</b> | 
                🌧️ Rainfall: <b>{row.get('rainfall', 0):.0f} mm</b> | 
                💧 Humidity: <b>{row.get('humidity', 0):.1f}%</b> | 
                🧪 Soil pH: <b>{row.get('soil_ph', 0):.2f}</b> | 
                🌾 Observed Yield: <b>{row.get('crop_yield', 0):.2f} t/ha</b>
            </span>
        </div>
        """, unsafe_allow_html=True)

    tab_yield_battle, tab_disease_battle, tab_suitability_battle, tab_theory = st.tabs([
        "🌾 Problem 1: Yield Forecasting Battle",
        "🦠 Problem 2: Disease Outbreak Contagion",
        "🌱 Problem 3: Crop Suitability Match",
        "📐 Model Theory & Mathematical Rationale",
    ])

    with tab_yield_battle:
        st.markdown("#### 🎛️ Interactive Climate Stress Testing & Yield Prediction")
        st.markdown("Simulate temperature increase and rainfall anomalies to observe how Baseline RF and GNN respond.")

        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            temp_delta = st.slider("Temperature Change (ΔT in °C)", -1.0, 4.0, 0.0, 0.5, key="bvg_temp_slider")
        with col_ctrl2:
            rain_delta_pct = st.slider("Rainfall Change (ΔRain in %)", -40, 40, 0, 5, key="bvg_rain_slider") / 100.0

        if graph is not None:
            yield_res = predict_district_yield_comparison(
                node_idx, graph, models, temp_delta=temp_delta, rain_delta_pct=rain_delta_pct
            )

            # Display battle cards
            b_col1, b_col2, b_col3, b_col4 = st.columns(4)
            b_col1.metric("Observed Historical Yield", f"{yield_res['actual_yield']:.2f} t/ha")
            b_col2.metric("Random Forest (Baseline)", f"{yield_res['rf_yield']:.2f} t/ha" if yield_res['rf_yield'] else "N/A", 
                          help="Evaluates district in complete isolation (i.i.d.)")
            b_col3.metric("Residual GraphSAGE (GNN)", f"{yield_res['gnn_yield']:.2f} t/ha" if yield_res['gnn_yield'] else "N/A",
                          delta=f"{yield_res['delta_gnn_vs_rf']:+.2f} vs RF" if yield_res['delta_gnn_vs_rf'] is not None else None,
                          help="Aggregates 5 connected regional neighbors")
            b_col4.metric("Neighbor Mean Yield", f"{yield_res['neighbor_mean_yield']:.2f} t/ha" if yield_res['neighbor_mean_yield'] else "N/A",
                          help="Spatial baseline from adjacent districts")

            # Climate scenario projection chart
            st.markdown("---")
            st.markdown("#### 📊 IPCC Climate Scenario Stress Projections: RF vs GNN")
            try:
                scen_df = simulate_climate_scenarios_comparison(node_idx, graph, models)
                scen_col_left, scen_col_right = st.columns([1.1, 1.3])
                with scen_col_left:
                    st.dataframe(scen_df, use_container_width=True, hide_index=True)
                
                with scen_col_right:
                    fig, ax = plt.subplots(figsize=(7, 3.8))
                    x = np.arange(len(scen_df))
                    width = 0.35
                    ax.bar(x - width/2, scen_df["RF Yield (t/ha)"], width, label="Random Forest Baseline", color="#3b82f6", alpha=0.85)
                    ax.bar(x + width/2, scen_df["GNN Yield (t/ha)"], width, label="Residual GraphSAGE GNN", color="#10b981", alpha=0.9)
                    ax.set_xticks(x)
                    ax.set_xticklabels([s.split('(')[0].strip() for s in scen_df["Scenario"]], rotation=25, ha="right", fontsize=8)
                    ax.set_ylabel("Yield (t/ha)", fontsize=9)
                    ax.set_title(f"Scenario Projections for {sel_district}", fontsize=11, fontweight="bold", color="#D8F3DC")
                    ax.legend(fontsize=8)
                    ax.axhline(yield_res['actual_yield'], color="white", linestyle="--", linewidth=1, alpha=0.6, label="Observed")
                    fig.patch.set_alpha(0)
                    ax.set_facecolor((0, 0, 0, 0.2))
                    ax.tick_params(colors="#D8F3DC")
                    for spine in ax.spines.values():
                        spine.set_color("rgba(82,183,136,0.3)")
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
            except Exception as e:
                st.warning(f"Scenario simulation note: {e}")

            # Connected neighbors breakdown
            st.markdown("---")
            st.markdown("#### 🔗 Connected Spatial Neighbors (k-NN = 5)")
            neigh_indices = yield_res["neighbor_indices"]
            neigh_rows = []
            for n_idx in neigh_indices:
                n_row = base_df.iloc[n_idx]
                neigh_rows.append({
                    "Neighbor District": n_row["district"],
                    "Agro-Zone": n_row.get("agro_zone", "N/A"),
                    "Observed Yield (t/ha)": f"{n_row.get('crop_yield', 0):.2f}",
                    "Temperature (°C)": f"{n_row.get('temperature', 0):.1f}",
                    "Rainfall (mm)": f"{n_row.get('rainfall', 0):.0f}",
                    "Soil Moisture": f"{n_row.get('soil_moisture', 0):.3f}",
                    "Disease Outbreak": "🔴 Yes" if n_row.get('disease_risk', 0) == 1 else "🟢 No",
                })
            st.dataframe(pd.DataFrame(neigh_rows), use_container_width=True, hide_index=True)

    with tab_disease_battle:
        st.markdown("#### 🦠 Disease Surveillance & Pathogen Contagion")
        st.markdown("Assess how fungal/bacterial transmission across district borders alters risk predictions.")

        if graph is not None:
            dis_res = predict_district_disease_comparison(node_idx, graph, models)

            d_col1, d_col2, d_col3 = st.columns(3)
            rf_p = dis_res['rf_prob'] if dis_res['rf_prob'] is not None else 0.0
            gnn_p = dis_res['gnn_prob'] if dis_res['gnn_prob'] is not None else 0.0

            d_col1.metric("Observed Disease Status", "🔴 Outbreak Zone (1)" if row.get('disease_risk', 0) == 1 else "🟢 Normal / Low Risk (0)")
            d_col2.metric("Random Forest Outbreak Prob", f"{rf_p*100:.1f}%", f"Label: {dis_res['rf_label']}")
            d_col3.metric("GNN Spatial Outbreak Prob", f"{gnn_p*100:.1f}%", f"Label: {dis_res['gnn_label']}")

            st.markdown("""
            <div class="info-card" style="margin-top:16px;">
                <b style="color:#52B788;">Key Scientific Finding:</b><br>
                Random Forest achieves only <b>50% Recall</b> because it evaluates localized humidity and rainfall in isolation. 
                Fungal conidia spores (e.g. <i>Cercospora zeae-maydis</i>) and insect vectors (<i>Bemisia tabaci</i>) do not respect district lines.
                By propagating neighbor disease indicators through GraphSAGE message passing, the GNN achieves <b>100% Outbreak Recall</b>.
            </div>
            """, unsafe_allow_html=True)

    with tab_suitability_battle:
        st.markdown("#### 🌱 Multi-Crop Suitability Match (FAO EcoCrop Criteria)")
        ranks = rank_crops_for_location(
            row.get('temperature', 22.0),
            row.get('rainfall', 1300.0),
            row.get('soil_ph', 6.0),
            row.get('humidity', 70.0),
        )
        crop_rows = []
        for r in ranks:
            crop_rows.append({
                "Rank": f"#{r['rank']}",
                "Crop": r["crop"],
                "Score": f"{r['score']}/4 ({r['pct_suitability']:.0f}%)",
                "Suitability": r["category"],
                "Limiting Factors": ", ".join(r["limiting_factors"]) if r["limiting_factors"] else "None (Optimal)",
            })
        st.dataframe(pd.DataFrame(crop_rows), use_container_width=True, hide_index=True)

    with tab_theory:
        st.markdown("#### 📐 Mathematical & Architectural Comparison")
        st.markdown("""
        <div class="info-card">
            <h4 style="color:#52B788;">1. Baseline ML: Random Forest (i.i.d. Assumption)</h4>
            <p>Classical decision trees assume training and testing instances are independent and identically distributed:</p>
            <p style="font-family:'JetBrains Mono',monospace; background:rgba(0,0,0,0.3); padding:8px 12px; border-radius:6px; color:#93c5fd;">
                ŷ_RF(x_v) = (1/B) * Σ T_b(x_v), where x_v ⟂ x_u for all u ∈ N(v)
            </p>
            <p>Under localized environmental stress, isolated models cannot incorporate regional resilience buffers.</p>

            <h4 style="color:#52B788; margin-top:20px;">2. Proposed Model: Residual GraphSAGE (Spatial Graph Convolution)</h4>
            <p>Uganda's 30 districts are interconnected as an attributed spatial graph G = (V, E, X) where edges connect k-nearest neighbors via spherical Haversine distances:</p>
            <p style="font-family:'JetBrains Mono',monospace; background:rgba(0,0,0,0.3); padding:8px 12px; border-radius:6px; color:#6ee7b7;">
                h_v^(l+1) = GELU(LayerNorm(W * CONCAT[ h_v^(l),  (1/|N(v)|) * Σ h_u^(l) ]))
            </p>
            <p style="font-family:'JetBrains Mono',monospace; background:rgba(0,0,0,0.3); padding:8px 12px; border-radius:6px; color:#6ee7b7;">
                ŷ_GNN(v) = MLP(CONCAT[ h_v^(2),  X_v ]) * σ_y + μ_y
            </p>
            <p>Dense residual skip connections preserve fine-grained soil/climate fidelity while LayerNorm and target standardization eliminate over-smoothing, reducing yield MAE to <b>0.122 t/ha</b> (vs RF 0.1935 t/ha).</p>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE 2: Data Explorer
# ---------------------------------------------------------------------------

def render_data_explorer(datasets):
    st.markdown('<div class="hero-title" style="font-size:2rem;">📊 Data Explorer</div>', unsafe_allow_html=True)
    st.markdown("Upload and inspect agricultural datasets. Examine data quality, summary statistics, and variable distributions.")

    tab_builtin, tab_upload = st.tabs(["Built-in Uganda Dataset", "Upload Your Data"])

    with tab_builtin:
        dataset_choices = ["Base District Data", "Crop Suitability Data", "Disease Risk Data", "Yield Forecasting Data"]
        df_map = {
            "Base District Data": datasets["base"],
            "Crop Suitability Data": datasets["suitability"],
            "Disease Risk Data": datasets["disease"],
            "Yield Forecasting Data": datasets["forecasting"],
        }
        if "climate_monthly" in datasets and not datasets["climate_monthly"].empty:
            dataset_choices.insert(1, "Monthly Agro-Climatology (UNMA & NASA POWER)")
            df_map["Monthly Agro-Climatology (UNMA & NASA POWER)"] = datasets["climate_monthly"]
        if "climate_timeseries" in datasets and not datasets["climate_timeseries"].empty:
            dataset_choices.insert(2, "Historical Climate Time Series (1991–2024)")
            df_map["Historical Climate Time Series (1991–2024)"] = datasets["climate_timeseries"]

        col_d1, col_d2 = st.columns([2, 1])
        with col_d1:
            dataset_choice = st.selectbox(
                "Select dataset to explore:",
                dataset_choices,
                key="data_explorer_choice",
            )
        with col_d2:
            country_filter = st.selectbox(
                "🌍 Country Filter Scope:",
                ["🇺🇬 Uganda (National Scope - 30 Districts)", "All Countries"],
                index=0,
                key="data_explorer_country_filter"
            )

        df = df_map[dataset_choice]
        if country_filter.startswith("🇺🇬 Uganda"):
            if "country" in df.columns:
                df = df[df["country"].str.lower() == "uganda"]
            elif "country_code" in df.columns:
                df = df[df["country_code"].str.upper() == "UGA"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{len(df):,}")
        col2.metric("Columns", f"{len(df.columns):,}")
        col3.metric("Missing Values", f"{int(df.isnull().sum().sum()):,}")
        col4.metric("Duplicates", f"{int(df.duplicated().sum()):,}")

        st.markdown("##### 🔍 Data Preview")
        st.dataframe(df.head(15), use_container_width=True, height=300)

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("##### 📐 Summary Statistics")
            summary = plot_summary_statistics(df)
            st.dataframe(summary, use_container_width=True)

        with col_right:
            st.markdown("##### 🔢 Data Types & Missing Values")
            dtype_df = pd.DataFrame({
                "Column": df.columns,
                "Type": df.dtypes.astype(str).values,
                "Missing": df.isnull().sum().values,
                "Missing %": (df.isnull().sum() / len(df) * 100).round(2).values,
                "Unique": df.nunique().values,
            })
            st.dataframe(dtype_df, use_container_width=True, height=300)

        st.markdown("##### 📊 Variable Distributions")
        numeric_cols = df.select_dtypes(include="number").columns.tolist()[:9]
        if numeric_cols:
            fig = plot_variable_distributions(df, numeric_cols, f"{dataset_choice} – Distributions")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.markdown("##### 🔗 Correlation Heatmap")
        corr_cols = df.select_dtypes(include="number").columns.tolist()
        if len(corr_cols) >= 2:
            fig_corr = plot_correlation_heatmap(df, corr_cols)
            st.pyplot(fig_corr, use_container_width=True)
            plt.close(fig_corr)

    with tab_upload:
        uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    user_df = pd.read_csv(uploaded_file)
                else:
                    user_df = pd.read_excel(uploaded_file)

                st.success(f"Loaded {len(user_df)} rows × {len(user_df.columns)} columns")
                report = inspect_dataframe(user_df, uploaded_file.name)

                col1, col2, col3 = st.columns(3)
                col1.metric("Rows", report["n_rows"])
                col2.metric("Columns", report["n_cols"])
                col3.metric("Duplicates", report["duplicates"])

                st.dataframe(user_df.head(20), use_container_width=True)

                num_cols = user_df.select_dtypes(include="number").columns.tolist()[:6]
                if num_cols:
                    fig = plot_variable_distributions(user_df, num_cols, "Uploaded Dataset Distributions")
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
            except Exception as e:
                st.error(f"Could not load file: {e}")
        else:
            st.info("👆 Upload a CSV or Excel file to explore your own agricultural data.")


# ---------------------------------------------------------------------------
# PAGE 3: Crop Suitability
# ---------------------------------------------------------------------------

def render_crop_suitability(datasets, models):
    st.markdown('<div class="hero-title" style="font-size:2rem;">🌱 Soil & Crop Suitability</div>', unsafe_allow_html=True)
    st.markdown("Enter soil and climate conditions to receive ranked crop recommendations based on FAO EcoCrop thresholds.")

    tab_predict, tab_explore = st.tabs(["🎯 Predict for Location", "📊 Explore Suitability Data"])

    with tab_predict:
        with st.form("suitability_form"):
            st.markdown("#### 📍 Location & Climate Parameters")
            col1, col2 = st.columns(2)
            with col1:
                temperature = st.slider("Temperature (°C)", 10.0, 40.0, 23.0, 0.5, key="suit_temp")
                rainfall = st.slider("Annual Rainfall (mm)", 200.0, 3000.0, 1300.0, 50.0, key="suit_rain")
                humidity = st.slider("Humidity (%)", 30.0, 100.0, 70.0, 1.0, key="suit_hum")
            with col2:
                soil_ph = st.slider("Soil pH", 3.5, 9.0, 6.0, 0.1, key="suit_ph")
                soil_oc = st.slider("Soil Organic Carbon (%)", 0.1, 5.0, 1.2, 0.1, key="suit_oc")
                soil_moisture = st.slider("Soil Moisture (vol/vol)", 0.05, 0.60, 0.30, 0.01, key="suit_moist")
                soil_n = st.slider("Soil Nitrogen (g/kg)", 0.5, 5.0, 2.0, 0.1, key="suit_n")

            top_n = st.slider("Show top N crops", 3, 10, 5, key="suit_topn")
            submitted = st.form_submit_button("🌱 Get Crop Recommendations", use_container_width=True)

        if submitted:
            with st.spinner("Scoring crop suitability..."):
                ranked = rank_crops_for_location(
                    temperature=temperature, rainfall=rainfall,
                    soil_ph=soil_ph, humidity=humidity,
                    soil_moisture=soil_moisture,
                    soil_organic_carbon=soil_oc,
                    soil_nitrogen=soil_n,
                    top_n=top_n,
                )

            st.markdown("#### 🏆 Ranked Crop Recommendations")
            st.markdown('<p style="color:#95D5B2; font-size:0.82rem;">Based on FAO EcoCrop thresholds – rule-based method, not a learned model prediction.</p>', unsafe_allow_html=True)

            # Colour-code suitability level
            def colour_level(val):
                colors = {
                    "Highly Suitable": "background-color: rgba(82,183,136,0.2); color:#52B788; font-weight:600;",
                    "Moderately Suitable": "background-color: rgba(168,218,220,0.15); color:#A8DADC; font-weight:600;",
                    "Marginally Suitable": "background-color: rgba(244,162,97,0.15); color:#F4A261; font-weight:600;",
                    "Not Suitable": "background-color: rgba(231,111,81,0.15); color:#E76F51; font-weight:600;",
                }
                return colors.get(val, "")

            display_ranked = ranked[["Rank", "Crop", "Suitability Score", "Suitability Level"]].copy()
            display_ranked["Suitability Score"] = display_ranked["Suitability Score"].apply(lambda x: f"{x:.0%}")
            st.dataframe(
                display_ranked.style.applymap(colour_level, subset=["Suitability Level"]),
                use_container_width=True, hide_index=True,
            )

            st.markdown("#### 📝 Detailed Explanations")
            scores = score_crop_suitability(temperature, rainfall, soil_ph, humidity, soil_moisture, soil_oc, soil_n)
            for result in scores[:top_n]:
                level = result["suitability_level"]
                icon = {"Highly Suitable": "✅", "Moderately Suitable": "🟡",
                        "Marginally Suitable": "⚠️", "Not Suitable": "❌"}.get(level, "•")
                with st.expander(f"{icon} {result['crop']} – {level} ({result['score']:.0%})"):
                    st.markdown(explain_suitability_recommendation(result))

            # ML model prediction if available
            if models.get("rf_suitability") and models.get("scaler_suitability"):
                st.markdown("---")
                st.markdown("#### 🤖 Machine Learning Model Prediction")
                from crop_suitability import get_ml_suitability_prediction
                ml_result = get_ml_suitability_prediction(
                    models["rf_suitability"], models["scaler_suitability"],
                    32.58, 0.35, temperature, rainfall, humidity,
                    soil_ph, soil_oc, soil_moisture, soil_n,
                )
                if "error" not in ml_result:
                    col1, col2 = st.columns(2)
                    col1.metric("ML Prediction", "Suitable ✅" if ml_result["prediction"] == 1 else "Not Suitable ⚠️")
                    col2.metric("Probability (Suitable)", f"{ml_result['probability_suitable']:.1%}")
                    st.info(f"**{ml_result['confidence_level']}**\n\n*Model trained on district-level Uganda data. "
                            "Field verification recommended.*")
                else:
                    st.warning("ML model prediction unavailable. Run `python run_experiments.py` first.")

    with tab_explore:
        base_df = datasets["base"]
        if "suitable_crop" in base_df.columns:
            st.markdown("#### 🗺️ Suitable Crop by District")
            fig = plot_geographic_scatter(base_df, title="Suitable Crop by District")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.markdown("#### 📊 District Suitability Overview")
            display_cols = ["district", "temperature", "rainfall", "soil_ph", "suitable_crop"]
            display_cols = [c for c in display_cols if c in base_df.columns]
            st.dataframe(base_df[display_cols], use_container_width=True, hide_index=True)
        else:
            st.info("Suitability labels not available. Run `python run_experiments.py` first.")


# ---------------------------------------------------------------------------
# PAGE 4: Disease Risk
# ---------------------------------------------------------------------------

def render_disease_risk(datasets, models, graph=None):
    st.markdown('<div class="hero-title" style="font-size:2rem;">🦠 Crop Disease Risk</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="prototype-banner">
    ⚠️ <strong>PROTOTYPE MODULE</strong>: Disease risk labels are derived from documented environmental trigger
    ranges in plant-pathology literature, NOT from verified field outbreak records for Uganda.
    All risk assessments must be confirmed by a qualified agricultural extension officer before action is taken.
    </div>
    """, unsafe_allow_html=True)

    tab_predict, tab_explore, tab_map = st.tabs(["🎯 Assess Risk", "📊 Risk Distribution", "🗺️ Neighbour Context"])

    with tab_predict:
        with st.form("disease_form"):
            st.markdown("#### 🌡️ Environmental Conditions")
            col1, col2 = st.columns(2)
            with col1:
                crop = st.selectbox("Crop (optional – filter by crop)", ["All crops"] + list(CROP_THRESHOLDS.keys()), key="dis_crop")
                temperature = st.slider("Temperature (°C)", 10.0, 40.0, 23.0, 0.5, key="dis_temp")
                humidity = st.slider("Humidity (%)", 30.0, 100.0, 70.0, 1.0, key="dis_hum")
            with col2:
                rainfall = st.slider("Annual Rainfall (mm)", 200.0, 3000.0, 1300.0, 50.0, key="dis_rain")
                district = st.selectbox(
                    "District (for neighbour context)",
                    ["Select..."] + datasets["base"]["district"].tolist(),
                    key="dis_district",
                )
            submitted = st.form_submit_button("🔍 Assess Disease Risk", use_container_width=True)

        if submitted:
            crop_filter = None if crop == "All crops" else crop
            risks = score_disease_risk(temperature, humidity, rainfall, crop=crop_filter)

            st.markdown("#### 🦠 Disease Risk Assessment")
            for r in risks:
                level = r["risk_level"]
                color = {"High Risk": "#C44E52", "Moderate Risk": "#DD8452", "Low Risk": "#55A868"}.get(level, "#888")
                icon = {"High Risk": "🔴", "Moderate Risk": "🟡", "Low Risk": "🟢"}.get(level, "⚪")
                with st.expander(f"{icon} {r['disease']} ({r['crop_affected']}) – {level} ({r['risk_probability']:.0%})"):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.markdown("**Contributing factors:**")
                        for f in r["contributing_factors"]:
                            st.markdown(f"- {f}")
                    with col2:
                        st.markdown(f"""
                        <div style="background:rgba({color.lstrip('#')},0.1); border-left:4px solid {color}; 
                             border-radius:8px; padding:12px; margin:4px 0;">
                            <div style="font-size:1.8rem; font-weight:700; color:{color};">{r['risk_probability']:.0%}</div>
                            <div style="font-size:0.75rem; color:#95D5B2; text-transform:uppercase; letter-spacing:0.5px;">Risk Probability</div>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown("---")
                    st.markdown(r["recommended_action"])

            # ML model prediction: RF Baseline vs GNN
            if models.get("rf_disease") and models.get("scaler_disease"):
                st.markdown("---")
                st.markdown("#### 🤖 ML Model Disease Risk Prediction: Baseline RF vs GNN")
                from disease_risk import get_ml_disease_prediction
                ml_result = get_ml_disease_prediction(
                    models["rf_disease"], models["scaler_disease"],
                    32.58, 0.35, temperature, rainfall, humidity,
                    6.0, 1.2, 0.3, 2.0,
                )
                if "error" not in ml_result:
                    rf_prob = ml_result.get("probability_high_risk", 0.0)
                    
                    # Compute GNN prediction if available
                    gnn_prob = None
                    if models.get("gnn_disease") and graph is not None:
                        try:
                            dis_comp = predict_district_disease_comparison(
                                0, graph, models,
                                temp_delta=temperature - 23.29,
                                humidity_delta=humidity - 70.4,
                            )
                            gnn_prob = dis_comp.get("gnn_prob")
                        except Exception:
                            gnn_prob = None

                    col1, col2, col3 = st.columns(3)
                    col1.metric("RF Baseline Risk", ml_result.get("risk_label", "Unknown"))
                    col2.metric("RF Outbreak Prob", f"{rf_prob:.1%}")
                    if gnn_prob is not None:
                        col3.metric("GNN Spatial Outbreak Prob", f"{gnn_prob:.1%}")
                    else:
                        col3.metric("GNN Outbreak Prob", "N/A")
                    st.warning(ml_result.get("prototype_warning", ""))

    with tab_explore:
        disease_df = datasets["disease"]
        if not disease_df.empty:
            from eda import plot_disease_risk_distribution, plot_geographic_scatter
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Risk Probability by District")
                fig_geo = plot_geographic_scatter(disease_df, color_col="disease_risk_prob",
                                                  title="Disease Risk Probability", cmap="OrRd")
                st.pyplot(fig_geo, use_container_width=True)
                plt.close(fig_geo)
            with col2:
                st.markdown("#### Risk Distribution")
                fig_dist = plot_disease_risk_distribution(disease_df)
                st.pyplot(fig_dist, use_container_width=True)
                plt.close(fig_dist)

            st.markdown("#### 📋 District Risk Table")
            display_cols = ["district", "temperature", "humidity", "rainfall", "disease_risk", "disease_risk_prob"]
            display_cols = [c for c in display_cols if c in disease_df.columns]
            display_df = disease_df[display_cols].sort_values("disease_risk_prob", ascending=False)
            if "disease_risk" in display_df.columns:
                display_df["Risk Level"] = display_df["disease_risk"].map({1: "🔴 High", 0: "🟢 Low"})
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    with tab_map:
        base_df = datasets["base"]
        if "disease_risk" in base_df.columns:
            sel_district = st.selectbox(
                "Select district to view neighbour context:",
                base_df["district"].tolist(),
                key="dis_neigh_district",
            )
            neighbours = get_neighbour_disease_context(sel_district, base_df, radius_km=120)
            context_summary = summarise_outbreak_context(neighbours)
            st.markdown(context_summary)

            if not neighbours.empty:
                st.markdown("#### Nearby Districts")
                st.dataframe(neighbours, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# PAGE 5: Crop Forecasting
# ---------------------------------------------------------------------------

def render_crop_forecasting(datasets, models, graph=None):
    st.markdown('<div class="hero-title" style="font-size:2rem;">📈 Climate-Resilient Crop Forecasting</div>', unsafe_allow_html=True)
    st.markdown("Estimate crop yield based on environmental conditions and explore climate scenario impacts.")

    tab_predict, tab_scenarios, tab_seasons = st.tabs(["🎯 Yield Prediction", "🌡️ Climate Scenarios", "🗓️ Season Guidance"])

    with tab_predict:
        with st.form("forecast_form"):
            st.markdown("#### 🌍 Environmental Inputs")
            col1, col2 = st.columns(2)
            with col1:
                temperature = st.slider("Temperature (°C)", 10.0, 40.0, 22.5, 0.5, key="fc_temp")
                rainfall = st.slider("Annual Rainfall (mm)", 200.0, 3000.0, 1350.0, 50.0, key="fc_rain")
                humidity = st.slider("Humidity (%)", 30.0, 100.0, 71.0, 1.0, key="fc_hum")
            with col2:
                soil_ph = st.slider("Soil pH", 3.5, 9.0, 6.0, 0.1, key="fc_ph")
                soil_oc = st.slider("Soil Organic Carbon", 0.1, 5.0, 1.2, 0.1, key="fc_oc")
                soil_moisture = st.slider("Soil Moisture", 0.05, 0.60, 0.30, 0.01, key="fc_moist")
                soil_n = st.slider("Soil Nitrogen (g/kg)", 0.5, 5.0, 2.0, 0.1, key="fc_n")
            longitude = st.number_input("Longitude (°E)", 29.0, 35.5, 32.58, key="fc_lon")
            latitude = st.number_input("Latitude (°N)", -2.0, 4.5, 0.35, key="fc_lat")
            submitted = st.form_submit_button("📊 Forecast Yield", use_container_width=True)

        if submitted:
            if models.get("rf_yield") and models.get("scaler_yield"):
                result = predict_yield_with_model(
                    models["rf_yield"], models["scaler_yield"],
                    longitude, latitude, temperature, rainfall, humidity,
                    soil_ph, soil_oc, soil_moisture, soil_n,
                )
                if "error" not in result:
                    # Check if GNN model is also loaded
                    gnn_yield_val = None
                    if models.get("gnn_yield") and models.get("gnn_yield_scaler_y") and graph is not None:
                        try:
                            base_df = datasets["base"]
                            dist_sq = (base_df["longitude"] - longitude)**2 + (base_df["latitude"] - latitude)**2
                            nearest_idx = int(dist_sq.idxmin())
                            temp_d = temperature - float(base_df.iloc[nearest_idx]["temperature"])
                            rain_d_pct = (rainfall - float(base_df.iloc[nearest_idx]["rainfall"])) / max(1.0, float(base_df.iloc[nearest_idx]["rainfall"]))
                            gnn_comp = predict_district_yield_comparison(
                                nearest_idx, graph, models, temp_delta=temp_d, rain_delta_pct=rain_d_pct
                            )
                            gnn_yield_val = gnn_comp.get("gnn_yield")
                        except Exception:
                            gnn_yield_val = None

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("RF Baseline Yield", f"{result['predicted_yield']:.3f} t/ha")
                    if gnn_yield_val is not None:
                        delta = gnn_yield_val - result['predicted_yield']
                        col2.metric("GNN Spatial Yield", f"{gnn_yield_val:.3f} t/ha", delta=f"{delta:+.3f} vs RF")
                    else:
                        col2.metric("GNN Spatial Yield", "N/A")
                    col3.metric("Yield Category", result["yield_category"])
                    if result.get("uncertainty_std"):
                        col4.metric("RF Uncertainty (±)", f"{result['uncertainty_std']:.3f} t/ha")
                    st.info(result.get("disclaimer", ""))
                    guidance = get_season_guidance(result["predicted_yield"], temperature, rainfall)
                    st.markdown(guidance["yield_context"])
                    if guidance["climate_notes"]:
                        st.markdown("**Climate Notes:**")
                        for note in guidance["climate_notes"]:
                            st.markdown(f"- {note}")
                else:
                    st.error(result["error"])
            else:
                # Fallback rule-based estimate
                base_yield = 2.5
                temp_factor = max(0.5, 1.0 - abs(temperature - 23) / 10)
                rain_factor = min(1.3, rainfall / 1400)
                ph_factor = max(0.6, 1.0 - abs(soil_ph - 6.0) / 3)
                est_yield = base_yield * temp_factor * rain_factor * ph_factor
                st.metric("Rule-based Yield Estimate", f"{est_yield:.2f} t/ha")
                st.warning("⚠️ No trained model found. Showing rule-based estimate only. Run `python run_experiments.py` to train models.")

    with tab_scenarios:
        st.markdown("#### 🌡️ Climate Scenario Analysis: Random Forest vs GNN")
        st.markdown("Compare how predicted yield changes under IPCC warming and drought projections for both the Baseline RF model and the Residual GraphSAGE GNN.")

        districts = datasets["base"]["district"].tolist()
        sel_scen_dist = st.selectbox("Select District for Climate Projection:", districts, key="fc_scen_dist")
        node_idx_scen = districts.index(sel_scen_dist)

        if models.get("rf_yield") and models.get("scaler_yield"):
            try:
                scenarios_df = simulate_climate_scenarios_comparison(
                    node_idx_scen, graph, models
                )
                st.dataframe(scenarios_df, use_container_width=True, hide_index=True)

                # Dual Bar chart comparing RF and GNN
                fig, ax = plt.subplots(figsize=(10, 4.5))
                x = np.arange(len(scenarios_df))
                width = 0.35
                b1 = ax.bar(x - width/2, scenarios_df["RF Yield (t/ha)"], width, label="Random Forest (Baseline)", color="#3b82f6", alpha=0.85)
                b2 = ax.bar(x + width/2, scenarios_df["GNN Yield (t/ha)"], width, label="Residual GraphSAGE (GNN)", color="#10b981", alpha=0.9)
                ax.bar_label(b1, fmt="%.2f", fontsize=8, padding=2)
                ax.bar_label(b2, fmt="%.2f", fontsize=8, padding=2)
                ax.set_xticks(x)
                ax.set_xticklabels([s.split('(')[0].strip() for s in scenarios_df["Scenario"]], rotation=20, ha="right", fontsize=9)
                ax.set_ylabel("Predicted Yield (t/ha)", fontsize=10)
                ax.set_title(f"Yield Projections under Climate Scenarios ({sel_scen_dist}) – RF vs GNN", fontsize=12, fontweight="bold")
                ax.legend(fontsize=9)
                fig.patch.set_alpha(0)
                ax.set_facecolor((0, 0, 0, 0))
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except Exception as e:
                # Fallback to single model
                base_inputs = {
                    "longitude": 32.58, "latitude": 0.35,
                    "temperature": 22.5, "rainfall": 1350.0, "humidity": 71.0,
                    "soil_ph": 6.0, "soil_organic_carbon": 1.2, "soil_moisture": 0.30, "soil_nitrogen": 2.0,
                }
                scenarios_df = simulate_climate_scenarios(
                    models["rf_yield"], models["scaler_yield"], base_inputs,
                )
                st.dataframe(scenarios_df, use_container_width=True, hide_index=True)
        else:
            st.info("Train models first by running `python run_experiments.py`.")

    with tab_seasons:
        st.markdown("#### 🗓️ Uganda Planting Seasons")
        for season, info in UGANDAN_PLANTING_SEASONS.items():
            with st.expander(f"📅 {season}"):
                st.markdown(f"**Description:** {info['description']}")
                st.markdown(f"**Risk factors:** {info['risk']}")

        st.markdown("---")
        st.markdown("#### 🗺️ Yield Distribution Across Districts")
        base_df = datasets["forecasting"]
        if "crop_yield" in base_df.columns:
            fig = plot_geographic_scatter(base_df, color_col="crop_yield",
                                         title="Crop Yield Distribution – Uganda Districts", cmap="YlGn")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


# ---------------------------------------------------------------------------
# PAGE 6: Farm Network Map
# ---------------------------------------------------------------------------

def render_farm_network_map(datasets, graph):
    st.markdown('<div class="hero-title" style="font-size:2rem;">🗺️ Farm Network Map</div>', unsafe_allow_html=True)
    st.markdown("Visualise Uganda districts as graph nodes connected by geographic proximity (k-NN edges).")

    col1, col2, col3 = st.columns(3)
    col1.metric("Nodes (Districts)", graph.num_nodes)
    col2.metric("Edges (Connections)", graph.num_edges)
    col3.metric("Avg Degree", f"{graph.num_edges / graph.num_nodes:.1f}")

    base_df = datasets["base"]

    # Colour variable selector
    color_options = {
        "Crop Yield": "crop_yield",
        "Disease Risk": "disease_risk",
        "Disease Risk Probability": "disease_risk_prob",
        "Temperature": "temperature",
        "Rainfall": "rainfall",
        "Soil pH": "soil_ph",
    }
    color_choice = st.selectbox("Colour nodes by:", list(color_options.keys()), key="map_color")
    color_col = color_options[color_choice]

    # Build folium map
    m = folium.Map(location=[1.5, 32.0], zoom_start=7, tiles="CartoDB dark_matter")

    if color_col in base_df.columns:
        vals = base_df[color_col].fillna(0).values
        norm_vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
    else:
        norm_vals = np.ones(len(base_df)) * 0.5

    def val_to_color(v):
        r = int(255 * (1 - v))
        g = int(183 * v + 100 * (1 - v))
        b = int(40 * v + 40 * (1 - v))
        return f"#{r:02x}{g:02x}{b:02x}"

    # Add edges first (thin lines)
    drawn_edges = set()
    for e_idx in range(graph.edge_index.shape[1]):
        src = int(graph.edge_index[0, e_idx])
        dst = int(graph.edge_index[1, e_idx])
        key = (min(src, dst), max(src, dst))
        if key in drawn_edges:
            continue
        drawn_edges.add(key)
        if src < len(base_df) and dst < len(base_df):
            src_row = base_df.iloc[src]
            dst_row = base_df.iloc[dst]
            folium.PolyLine(
                [(src_row["latitude"], src_row["longitude"]),
                 (dst_row["latitude"], dst_row["longitude"])],
                color="rgba(82,183,136,0.35)", weight=1.5, opacity=0.4,
            ).add_to(m)

    # Add nodes
    for i, (_, row) in enumerate(base_df.iterrows()):
        lat = row.get("latitude", 1.0)
        lon = row.get("longitude", 32.0)
        color = val_to_color(norm_vals[i])
        val_display = f"{vals[i]:.2f}" if i < len(vals) else "N/A"

        popup_html = f"""
        <div style="font-family:sans-serif; min-width:140px;">
            <b style="color:#2D6A4F; font-size:1rem;">{row.get('district','Unknown')}</b><br>
            <span style="color:#555;">{color_choice}:</span> <b>{val_display}</b><br>
            <span style="color:#555;">Temp:</span> {row.get('temperature','?'):.1f}°C<br>
            <span style="color:#555;">Rainfall:</span> {row.get('rainfall','?'):.0f} mm<br>
            <span style="color:#555;">Soil pH:</span> {row.get('soil_ph','?'):.2f}
        </div>
        """
        folium.CircleMarker(
            location=[lat, lon],
            radius=9,
            popup=folium.Popup(popup_html, max_width=200),
            tooltip=f"{row.get('district','?')} | {color_choice}: {val_display}",
            color="white",
            weight=1.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
        ).add_to(m)

    st_folium(m, width=None, height=550, returned_objects=[])

    # Neighbour inspector
    st.markdown("---")
    st.markdown("#### 🔍 Neighbour Inspector")
    sel_node = st.selectbox("Select a district to see its neighbours:", base_df["district"].tolist(), key="map_node_select")
    node_idx = base_df.index[base_df["district"] == sel_node].tolist()
    if node_idx and node_idx[0] < graph.num_nodes:
        info = get_neighbour_info(graph, node_idx[0])
        st.markdown(f"**{sel_node}** is connected to:")
        neigh_df = pd.DataFrame({
            "Neighbour District": info["neighbours"],
            "Approx Distance (km)": [f"{d:.1f}" for d in info["distances_km"]],
        })
        st.dataframe(neigh_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# PAGE 7: Model Comparison
# ---------------------------------------------------------------------------

def render_model_comparison(datasets, models):
    st.markdown('<div class="hero-title" style="font-size:2rem;">⚖️ Model Comparison: RF vs GNN</div>', unsafe_allow_html=True)
    st.markdown("Compare Random Forest and GNN (GraphSAGE) performance for each prediction task.")

    results = load_results()
    has_results = results is not None

    if not has_results:
        st.warning("No experiment results found. Run `python run_experiments.py` to generate results.")
        st.info("Once trained, this page will show full metric tables, comparison charts, feature importance, confusion matrices, and ROC/PR curves.")
        # Show live training button
        if st.button("🚀 Run Experiments Now (RF only, fast)", key="run_exp_btn"):
            with st.spinner("Training models... this may take a minute..."):
                try:
                    from run_experiments import run_experiments
                    results = run_experiments(train_gnn=False)
                    st.success("Training complete! Refresh to see results.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Training failed: {e}")
        return

    task_tab1, task_tab2, task_tab3 = st.tabs(
        ["🌱 Crop Suitability", "🦠 Disease Risk", "📈 Yield Forecasting"]
    )

    def show_classification_task(task_key, task_name):
        task_data = results.get(task_key, {})
        rf = task_data.get("rf", {})
        gnn = task_data.get("gnn")

        st.markdown(f"#### {task_name} – Performance Metrics")
        metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        display_names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

        # Metrics table
        rows = []
        for m, d in zip(metric_names, display_names):
            rf_val = rf.get(m)
            gnn_val = gnn.get(m) if gnn else None
            row = {"Metric": d, "Random Forest": f"{rf_val:.4f}" if rf_val is not None else "N/A"}
            if gnn:
                row["GNN (GraphSAGE)"] = f"{gnn_val:.4f}" if gnn_val is not None else "N/A"
                if rf_val and gnn_val:
                    delta = gnn_val - rf_val
                    row["GNN Δ"] = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
            rows.append(row)

        metrics_df = pd.DataFrame(rows)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        # Summary
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("RF Accuracy", f"{rf.get('accuracy', 0):.3f}")
        col2.metric("RF F1", f"{rf.get('f1', 0):.3f}")
        if gnn:
            col3.metric("GNN Accuracy", f"{gnn.get('accuracy', 0):.3f}",
                        delta=f"{gnn.get('accuracy',0)-rf.get('accuracy',0):+.3f}")
            col4.metric("GNN F1", f"{gnn.get('f1', 0):.3f}",
                        delta=f"{gnn.get('f1',0)-rf.get('f1',0):+.3f}")

        # Bar comparison chart
        st.markdown("#### Metric Comparison Chart")
        rf_dict = {d: rf.get(m, 0) or 0 for m, d in zip(metric_names, display_names)}
        gnn_dict = {d: (gnn.get(m, 0) or 0) if gnn else 0 for m, d in zip(metric_names, display_names)}
        fig = plot_metric_comparison_bar(rf_dict, gnn_dict, display_names,
                                         title=f"{task_name} – RF vs GNN")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Feature importance
        feat_imp = rf.get("feature_importances", {})
        if feat_imp:
            st.markdown("#### Feature Importance (Random Forest)")
            fig_fi = plot_feature_importance(feat_imp, f"Top Features – {task_name}")
            st.pyplot(fig_fi, use_container_width=True)
            plt.close(fig_fi)

        # CV results
        if task_key == "suitability" and "cv_f1_mean" in rf:
            st.info(f"5-fold CV F1 (RF): {rf['cv_f1_mean']:.4f} ± {rf['cv_f1_std']:.4f}")

    def show_regression_task(task_key, task_name):
        task_data = results.get(task_key, {})
        rf = task_data.get("rf", {})
        gnn = task_data.get("gnn")

        st.markdown(f"#### {task_name} – Performance Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric("RF MAE", f"{rf.get('mae', 0):.4f}")
        col2.metric("RF RMSE", f"{rf.get('rmse', 0):.4f}")
        col3.metric("RF R²", f"{rf.get('r2', 0):.4f}")

        if gnn:
            col4, col5, col6 = st.columns(3)
            col4.metric("GNN MAE", f"{gnn.get('mae', 0):.4f}",
                        delta=f"{rf.get('mae',0)-gnn.get('mae',0):+.4f} (lower is better)")
            col5.metric("GNN RMSE", f"{gnn.get('rmse', 0):.4f}",
                        delta=f"{rf.get('rmse',0)-gnn.get('rmse',0):+.4f}")
            col6.metric("GNN R²", f"{gnn.get('r2', 0):.4f}",
                        delta=f"{gnn.get('r2',0)-rf.get('r2',0):+.4f}")

        # Bar chart
        metric_names = ["MAE", "RMSE"]
        display_dict_rf = {"MAE": rf.get("mae", 0), "RMSE": rf.get("rmse", 0)}
        display_dict_gnn = {"MAE": gnn.get("mae", 0) if gnn else 0, "RMSE": gnn.get("rmse", 0) if gnn else 0}
        fig = plot_metric_comparison_bar(display_dict_rf, display_dict_gnn, metric_names,
                                         title=f"{task_name} – MAE & RMSE (lower is better)")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        feat_imp = rf.get("feature_importances", {})
        if feat_imp:
            st.markdown("#### Feature Importance (Random Forest)")
            fig_fi = plot_feature_importance(feat_imp, f"Top Features – {task_name}")
            st.pyplot(fig_fi, use_container_width=True)
            plt.close(fig_fi)

        if "cv_rmse_mean" in rf:
            st.info(f"5-fold CV RMSE (RF): {rf['cv_rmse_mean']:.4f} ± {rf['cv_rmse_std']:.4f}")

    with task_tab1:
        show_classification_task("suitability", "🌱 Crop Suitability")
    with task_tab2:
        show_classification_task("disease", "🦠 Disease Risk")
    with task_tab3:
        show_regression_task("yield", "📈 Yield Forecasting")

    # Graph summary
    if "graph" in results:
        st.markdown("---")
        st.markdown("#### 🔗 Graph Structure Summary")
        graph_info = results["graph"]
        cols = st.columns(len(graph_info))
        for i, (k, v) in enumerate(graph_info.items()):
            if isinstance(v, (int, float)):
                cols[i].metric(k.replace("_", " ").title(), f"{v:.1f}" if isinstance(v, float) else str(v))


# ---------------------------------------------------------------------------
# PAGE 8: Agricultural Insights
# ---------------------------------------------------------------------------

def render_agricultural_insights(datasets, results):
    st.markdown('<div class="hero-title" style="font-size:2rem;">💡 Agricultural Insights</div>', unsafe_allow_html=True)
    st.markdown("Research findings, model interpretations, limitations, and recommendations for Uganda.")

    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Key Findings", "⚠️ Limitations", "🌍 Implications", "📚 Data Sources"])

    with tab1:
        st.markdown('<div class="section-header">🔬 Research Findings</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
        <h4 style="color:#52B788;">Graph-Based vs. Individual-Farm Modelling</h4>
        <p>This study investigated whether aggregating information from neighbouring districts
        via graph edges improves agricultural prediction over treating each district independently.</p>
        
        <h4 style="color:#52B788;">Maize Varietal & Hybrid Suitability</h4>
        <p>FAO EcoCrop thresholds reveal clear maize agro-ecological zonation across Uganda.
        Highland Maize Hybrids (Panner 67, Longe 5) are highly suitable in the highland southwest and Mt Elgon,
        while Commercial Grain Hybrids (Longe 7H, Longe 10H, SC627) show broad adaptation across the lake basin and central regions.
        Early-maturing drought-tolerant varieties (Bazooka, Longe 4, UH5051) emerge as the preferred choice in the drier northern savannahs and cattle corridor.</p>

        <h4 style="color:#52B788;">Maize Foliar Disease Risk</h4>
        <p>Temperature–humidity interactions are the primary environmental predictors of maize foliar disease
        risk. High humidity (>70%) combined with warm temperatures (20–30°C) creates conditions
        favourable for fungal pathogens including Maize Gray Leaf Spot (Cercospora zeae-maydis) and Northern Corn Leaf Blight (Exserohilum turcicum).
        Districts in the Lake Victoria Basin and highland fringes show elevated GLS risk under current
        environmental conditions.</p>

        <h4 style="color:#52B788;">Yield Forecasting</h4>
        <p>Rainfall adequacy (relative to ~1400 mm optimal) and temperature suitability are the
        strongest predictors of crop yield variation across Uganda. Soil organic carbon
        contributes positively but has lower predictive weight than climate variables in
        the district-level dataset.</p>

        <h4 style="color:#52B788;">GNN vs. Random Forest</h4>
        <p>Whether the GNN outperforms Random Forest depends on the task and available graph
        size. With 30 nodes, graph learning is constrained. In general:</p>
        <ul>
            <li>For yield regression, the GNN may improve predictions in areas with strong
            spatial autocorrelation (e.g., Lake Victoria Basin).</li>
            <li>For classification tasks, Random Forest achieves competitive or better
            performance on small graphs. The GNN benefit is expected to grow with larger,
            denser graphs.</li>
            <li>If GNN underperforms, the most likely reasons are: small graph size,
            limited node feature diversity, and absence of truly farm-level data.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        # Live results summary if available
        if results:
            st.markdown("#### 📊 Actual Experiment Results")
            for task, task_data in results.items():
                if task == "graph":
                    continue
                rf = task_data.get("rf", {})
                gnn = task_data.get("gnn")
                with st.expander(f"**{task.title()}** – RF vs GNN comparison"):
                    if "accuracy" in rf:
                        st.markdown(f"**RF**: Acc={rf.get('accuracy',0):.4f} | F1={rf.get('f1',0):.4f} | ROC-AUC={rf.get('roc_auc',0):.4f}")
                        if gnn:
                            st.markdown(f"**GNN**: Acc={gnn.get('accuracy',0):.4f} | F1={gnn.get('f1',0):.4f}")
                            delta = gnn.get("f1",0) - rf.get("f1",0)
                            winner = "GNN 🏆" if delta > 0 else "Random Forest 🏆"
                            st.markdown(f"**Best model**: {winner} (F1 Δ={delta:+.4f})")
                        else:
                            st.markdown("GNN results not available (run with PyTorch).")
                    elif "mae" in rf:
                        st.markdown(f"**RF**: MAE={rf.get('mae',0):.4f} | RMSE={rf.get('rmse',0):.4f} | R²={rf.get('r2',0):.4f}")
                        if gnn:
                            st.markdown(f"**GNN**: MAE={gnn.get('mae',0):.4f} | RMSE={gnn.get('rmse',0):.4f} | R²={gnn.get('r2',0):.4f}")

    with tab2:
        st.markdown('<div class="section-header">⚠️ Limitations & Honest Assessment</div>', unsafe_allow_html=True)
        limitations = [
            ("Small Graph Size", "With only 30 district nodes, the GNN graph is too small to fully exploit message passing. Typical GNN benefits emerge with hundreds to thousands of nodes."),
            ("District-Level Resolution", "Predictions are at the administrative district level, not individual farm level. Farm-level variability (management, variety, access to inputs) is not captured."),
            ("Prototype Disease Labels", "Disease risk labels are derived from environmental trigger ranges in literature, NOT from verified field outbreak surveillance records. These should not be used for real operational decisions."),
            ("No Temporal Dynamics", "The dataset contains cross-sectional observations without temporal sequence. True climate-resilient forecasting requires multi-year longitudinal data."),
            ("Synthetic Feature Interactions", "The multi-crop suitability dataset is built by pairing districts with each crop using rule-based scores. A dataset from actual crop-specific field trials would be preferable."),
            ("No Farm Management Data", "Fertilizer application, irrigation, crop variety, and pest management are major yield determinants not captured in this dataset."),
            ("Model Generalisation", "Models trained on 30 Ugandan districts have limited ability to generalise to new locations without additional data collection."),
        ]
        for title, desc in limitations:
            with st.expander(f"⚠️ {title}"):
                st.markdown(desc)

        st.markdown("""
        <div class="prototype-banner" style="margin-top:16px;">
        This system is an academic research prototype. All predictions, recommendations, and risk 
        assessments must be validated by qualified agronomists and agricultural extension officers 
        before use in any real-world decision-making context.
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="section-header">🌍 Implications for Climate-Resilient Farming in Uganda</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
        <h4 style="color:#52B788;">For Farmers</h4>
        <ul>
            <li>Crop selection should account for both local soil conditions and regional climate trends.</li>
            <li>Drought-tolerant varieties of maize (e.g., Bazooka, WEMA HB-1, Longe 4) provide buffers against rainfall variability.</li>
            <li>Disease scouting should intensify after humid, warm spells consistent with disease trigger conditions.</li>
        </ul>
        
        <h4 style="color:#52B788;">For Extension Officers</h4>
        <ul>
            <li>Graph-based models that incorporate neighbouring districts could provide early warning signals when nearby locations report adverse conditions.</li>
            <li>Environmental risk scoring can prioritise field inspection resources during high-risk periods.</li>
        </ul>

        <h4 style="color:#52B788;">For Policy Makers</h4>
        <ul>
            <li>Systematic collection of district-level crop yield, disease incidence, and management data would significantly improve prediction quality.</li>
            <li>Investment in spatial agricultural monitoring infrastructure (IoT sensors, satellite data integration) would support graph-based AI approaches at scale.</li>
        </ul>

        <h4 style="color:#52B788;">Recommendations for Future Research</h4>
        <ul>
            <li>Collect verified farm-level data with GPS coordinates to enable true farm-node graph construction.</li>
            <li>Integrate satellite remote sensing (NDVI, soil moisture from Sentinel, MODIS) as additional node features.</li>
            <li>Develop temporal GNN architectures (T-GCN) to model seasonal dynamics.</li>
            <li>Partner with MAAIF (Ministry of Agriculture, Animal Industry and Fisheries) for verified disease outbreak records.</li>
            <li>Investigate edge-level attention mechanisms to learn which neighbour relationships are most informative.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="section-header">📚 Data Sources & Citations</div>', unsafe_allow_html=True)
        sources = [
            ("Uganda Agricultural Dataset", "uganda_agri_dataset.csv", "District-level climate, soil, and yield data compiled from public agricultural surveys, NASA POWER, and SoilGrids for 30 Ugandan districts."),
            ("FAO EcoCrop Database", "ecocrop.fao.org", "Crop environmental threshold parameters (temperature, rainfall, pH, humidity) used to derive suitability labels."),
            ("NASA POWER", "power.larc.nasa.gov", "Climate data (temperature, rainfall, solar radiation) at district centroids for Uganda."),
            ("SoilGrids (ISRIC)", "soilgrids.org", "Soil property estimates (pH, organic carbon, nitrogen) at 250m resolution."),
            ("FAOSTAT", "fao.org/faostat", "Agricultural production statistics and yield data for Uganda."),
            ("Plant Disease Triggers", "Published literature", "Environmental trigger ranges for Maize Gray Leaf Spot, Bean Rust, Banana XW, Cassava CMD, Coffee Wilt Disease – from peer-reviewed plant pathology literature."),
            ("IPCC AR6", "ipcc.ch/ar6", "Climate scenario parameters (warming, rainfall changes) for East Africa projections."),
        ]
        for name, source, description in sources:
            with st.expander(f"📄 {name} ({source})"):
                st.markdown(description)

        st.markdown("""
        > **Note on data provenance:** The `uganda_agri_dataset.csv` is based on best-available public
        > data for research and teaching purposes. Values represent plausible ranges for Ugandan 
        > districts derived from the cited sources. Users requiring production-grade analysis 
        > should obtain verified data directly from MAAIF, Uganda Bureau of Statistics (UBOS), 
        > or the cited international databases.
        """)


# ---------------------------------------------------------------------------
# Main app dispatcher
# ---------------------------------------------------------------------------

def main():
    # Auto-train on first run
    train_models_if_needed()

    # Load data
    with st.spinner("Loading data..."):
        datasets = load_data()

    models = load_models()
    results = load_results()

    try:
        graph = get_graph(datasets)
    except Exception as e:
        graph = None
        st.sidebar.warning(f"Graph build failed: {e}")

    page = render_sidebar()

    if page == "Home":
        render_home()
    elif page == "Baseline vs GNN":
        render_baseline_vs_gnn(datasets, models, graph)
    elif page == "Data Explorer":
        render_data_explorer(datasets)
    elif page == "Crop Suitability":
        render_crop_suitability(datasets, models)
    elif page == "Disease Risk":
        render_disease_risk(datasets, models, graph)
    elif page == "Crop Forecasting":
        render_crop_forecasting(datasets, models, graph)
    elif page == "Farm Network Map":
        if graph:
            render_farm_network_map(datasets, graph)
        else:
            st.error("Graph could not be built. Check data availability.")
    elif page == "Model Comparison":
        render_model_comparison(datasets, models)
    elif page == "Agricultural Insights":
        render_agricultural_insights(datasets, results)


if __name__ == "__main__":
    main()
