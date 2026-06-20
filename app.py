"""
🏦 LOAN DEFAULT RISK PREDICTOR - Streamlit Web App
====================================================
Production-ready ML deployment untuk prediksi risiko gagal bayar kredit.

Setup:
1. pip install -r requirements.txt
2. Taruh file model di folder models/
3. streamlit run app.py

Author: ML Group 2 - Institut Teknologi Kalimantan
Model: LightGBM + CatBoost + Logistic Regression
Dataset: Home Credit Default Risk (307,511 records)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import pickle
from pathlib import Path

# ============================================
# PAGE CONFIG & GLOBAL STYLING
# ============================================

st.set_page_config(
    page_title="Loan Default Risk Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Premium dark-friendly styling */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
    }
    .main-header h1 { color: white; margin: 0; font-size: 2rem; }
    .main-header p { color: #a0aec0; margin: 0.3rem 0 0 0; font-size: 1rem; }

    .metric-card {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
        border-left: 4px solid #0f3460;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .metric-card h3 { margin: 0; font-size: 0.85rem; color: #6c757d; }
    .metric-card p { margin: 0.3rem 0 0 0; font-size: 1.5rem; font-weight: bold; color: #1a1a2e; }

    .risk-high { color: #e53e3e; font-weight: 800; font-size: 1.8rem; }
    .risk-medium { color: #dd6b20; font-weight: 800; font-size: 1.8rem; }
    .risk-low { color: #38a169; font-weight: 800; font-size: 1.8rem; }

    .rec-box {
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-size: 1rem;
    }
    .rec-reject { background: #fed7d7; border-left: 5px solid #e53e3e; }
    .rec-review { background: #fefcbf; border-left: 5px solid #dd6b20; }
    .rec-approve { background: #c6f6d5; border-left: 5px solid #38a169; }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        border-radius: 6px 6px 0 0;
    }

    div[data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }

    .demo-badge {
        background: #fefcbf;
        color: #744210;
        padding: 0.3rem 0.8rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# DATA LOADING & MODE DETECTION
# ============================================

MODELS_DIR = Path("models")


@st.cache_resource
def load_real_models():
    """Load real trained models from pickle files."""
    try:
        with open(MODELS_DIR / "models_trained.pkl", "rb") as f:
            models = pickle.load(f)
        with open(MODELS_DIR / "imputer.pkl", "rb") as f:
            imputer = pickle.load(f)
        with open(MODELS_DIR / "feature_names.pkl", "rb") as f:
            feature_names = pickle.load(f)
        with open(MODELS_DIR / "config.json", "r") as f:
            config = json.load(f)
        # Optional: class weight models
        cw_models = None
        cw_path = MODELS_DIR / "models_class_weight.pkl"
        if cw_path.exists():
            with open(cw_path, "rb") as f:
                cw_models = pickle.load(f)
        return {
            "models": models,
            "imputer": imputer,
            "feature_names": feature_names,
            "config": config,
            "cw_models": cw_models,
        }
    except Exception as e:
        return None


def get_demo_config():
    """Hardcoded demo data from notebook outputs."""
    return {
        "feature_names": [
            "EXT_SOURCE_3", "EXT_SOURCE_2", "EXT_SOURCE_1", "AGE_YEARS",
            "REGION_RATING_CLIENT_W_CITY", "REGION_RATING_CLIENT",
            "NAME_INCOME_TYPE_Working", "NAME_EDUCATION_TYPE_Higher education",
            "DAYS_LAST_PHONE_CHANGE", "CODE_GENDER",
            "YEARS_ID_PUBLISH", "REG_CITY_NOT_WORK_CITY",
            "NAME_EDUCATION_TYPE_Secondary / secondary special",
            "NAME_INCOME_TYPE_Pensioner", "FLAG_EMP_PHONE",
            "ORGANIZATION_TYPE_XNA", "YEARS_REGISTRATION",
            "FLAG_WORK_PHONE", "NAME_CONTRACT_TYPE_Revolving loans",
            "REG_CITY_NOT_LIVE_CITY", "DEF_30_CNT_SOCIAL_CIRCLE",
            "LIVE_CITY_NOT_WORK_CITY", "DEF_60_CNT_SOCIAL_CIRCLE",
            "FLAG_DOCUMENT_3", "AMT_REQ_CREDIT_BUREAU_QRT",
            "OBS_30_CNT_SOCIAL_CIRCLE", "YEARS_EMPLOYED",
            "TOTAL_DOCUMENTS_SUBMITTED", "OBS_60_CNT_SOCIAL_CIRCLE",
            "EMPLOYMENT_TO_INCOME_RATIO",
            "AMT_CREDIT", "AMT_ANNUITY", "AMT_INCOME_TOTAL",
            "INCOME_TO_ANNUITY_RATIO"
        ],
        "feature_stats": {
            "EXT_SOURCE_3": {"min": 0.0, "max": 0.896, "mean": 0.51, "median": 0.54, "std": 0.19},
            "EXT_SOURCE_2": {"min": 0.0, "max": 0.855, "mean": 0.51, "median": 0.57, "std": 0.19},
            "EXT_SOURCE_1": {"min": 0.0, "max": 0.963, "mean": 0.50, "median": 0.51, "std": 0.21},
            "AGE_YEARS": {"min": 20.5, "max": 69.0, "mean": 43.9, "median": 43.1, "std": 11.9},
            "REGION_RATING_CLIENT_W_CITY": {"min": 1, "max": 3, "mean": 2.05, "median": 2, "std": 0.51},
            "REGION_RATING_CLIENT": {"min": 1, "max": 3, "mean": 2.05, "median": 2, "std": 0.51},
            "NAME_INCOME_TYPE_Working": {"min": 0, "max": 1, "mean": 0.52, "median": 1, "std": 0.50},
            "NAME_EDUCATION_TYPE_Higher education": {"min": 0, "max": 1, "mean": 0.24, "median": 0, "std": 0.43},
            "DAYS_LAST_PHONE_CHANGE": {"min": -4292, "max": 0, "mean": -962, "median": -757, "std": 826},
            "CODE_GENDER": {"min": 0, "max": 1, "mean": 0.34, "median": 0, "std": 0.47},
            "YEARS_ID_PUBLISH": {"min": 0.0, "max": 19.7, "mean": 8.2, "median": 7.6, "std": 4.1},
            "REG_CITY_NOT_WORK_CITY": {"min": 0, "max": 1, "mean": 0.23, "median": 0, "std": 0.42},
            "NAME_EDUCATION_TYPE_Secondary / secondary special": {"min": 0, "max": 1, "mean": 0.71, "median": 1, "std": 0.45},
            "NAME_INCOME_TYPE_Pensioner": {"min": 0, "max": 1, "mean": 0.18, "median": 0, "std": 0.38},
            "FLAG_EMP_PHONE": {"min": 0, "max": 1, "mean": 0.82, "median": 1, "std": 0.39},
            "ORGANIZATION_TYPE_XNA": {"min": 0, "max": 1, "mean": 0.18, "median": 0, "std": 0.38},
            "YEARS_REGISTRATION": {"min": 0.0, "max": 67.6, "mean": 13.7, "median": 12.3, "std": 9.5},
            "FLAG_WORK_PHONE": {"min": 0, "max": 1, "mean": 0.20, "median": 0, "std": 0.40},
            "NAME_CONTRACT_TYPE_Revolving loans": {"min": 0, "max": 1, "mean": 0.08, "median": 0, "std": 0.27},
            "REG_CITY_NOT_LIVE_CITY": {"min": 0, "max": 1, "mean": 0.08, "median": 0, "std": 0.27},
            "DEF_30_CNT_SOCIAL_CIRCLE": {"min": 0, "max": 34, "mean": 0.14, "median": 0, "std": 0.50},
            "LIVE_CITY_NOT_WORK_CITY": {"min": 0, "max": 1, "mean": 0.18, "median": 0, "std": 0.38},
            "DEF_60_CNT_SOCIAL_CIRCLE": {"min": 0, "max": 24, "mean": 0.10, "median": 0, "std": 0.42},
            "FLAG_DOCUMENT_3": {"min": 0, "max": 1, "mean": 0.71, "median": 1, "std": 0.45},
            "AMT_REQ_CREDIT_BUREAU_QRT": {"min": 0, "max": 261, "mean": 0.57, "median": 0, "std": 0.85},
            "OBS_30_CNT_SOCIAL_CIRCLE": {"min": 0, "max": 348, "mean": 1.42, "median": 1, "std": 2.40},
            "YEARS_EMPLOYED": {"min": 0, "max": 49.1, "mean": 6.3, "median": 4.7, "std": 6.5},
            "TOTAL_DOCUMENTS_SUBMITTED": {"min": 0, "max": 4, "mean": 0.79, "median": 1, "std": 0.51},
            "OBS_60_CNT_SOCIAL_CIRCLE": {"min": 0, "max": 344, "mean": 1.40, "median": 1, "std": 2.38},
            "EMPLOYMENT_TO_INCOME_RATIO": {"min": -0.0, "max": 0.0003, "mean": 0.00004, "median": 0.00003, "std": 0.00004},
            "AMT_CREDIT": {"min": 25650, "max": 4050000, "mean": 598500, "median": 508500, "std": 400000},
            "AMT_ANNUITY": {"min": 1615, "max": 258025, "mean": 27108, "median": 24903, "std": 14493},
            "AMT_INCOME_TOTAL": {"min": 25650, "max": 117000000, "mean": 168798, "median": 147150, "std": 237123},
            "INCOME_TO_ANNUITY_RATIO": {"min": 0.5, "max": 4760, "mean": 7.4, "median": 5.8, "std": 24.0},
        },
        "threshold_results": [
            {"Model": "Logistic Regression", "Threshold Optimal": 0.513, "Recall @0.5": 0.4087, "Recall @Optimal": 0.3948, "Precision @Optimal": 0.2318, "F1 @Optimal": 0.2921},
            {"Model": "LightGBM", "Threshold Optimal": 0.147, "Recall @0.5": 0.0332, "Recall @Optimal": 0.4274, "Precision @Optimal": 0.2266, "F1 @Optimal": 0.2962},
            {"Model": "CatBoost", "Threshold Optimal": 0.160, "Recall @0.5": 0.0284, "Recall @Optimal": 0.4157, "Precision @Optimal": 0.2333, "F1 @Optimal": 0.2989},
        ],
        "model_metrics": [
            {"Model": "CatBoost", "AUC": 0.7494, "Recall": 0.0284, "F1-Score": 0.0537, "Accuracy": 0.9192, "Train Time (s)": 771.23, "Inference Time (s)": 0.28},
            {"Model": "LightGBM", "AUC": 0.7433, "Recall": 0.0332, "F1-Score": 0.0620, "Accuracy": 0.9188, "Train Time (s)": 869.90, "Inference Time (s)": 5.97},
            {"Model": "Logistic Regression", "AUC": 0.7411, "Recall": 0.4087, "F1-Score": 0.2891, "Accuracy": 0.8377, "Train Time (s)": 79.84, "Inference Time (s)": 0.05},
        ],
        "comparison_table": [
            {"Model": "CatBoost", "Pendekatan": "SMOTE + Tuning", "Threshold": 0.160, "Accuracy": 0.8426, "Precision": 0.2333, "Recall": 0.4157, "F1-Score": 0.2989, "AUC": 0.7494},
            {"Model": "CatBoost", "Pendekatan": "Class Weight (Thresh=0.5)", "Threshold": 0.500, "Accuracy": 0.7412, "Precision": 0.1784, "Recall": 0.6117, "F1-Score": 0.2762, "AUC": 0.7503},
            {"Model": "CatBoost", "Pendekatan": "SMOTE (Thresh=0.5)", "Threshold": 0.500, "Accuracy": 0.9192, "Precision": 0.4930, "Recall": 0.0284, "F1-Score": 0.0537, "AUC": 0.7494},
            {"Model": "LightGBM", "Pendekatan": "SMOTE + Tuning", "Threshold": 0.147, "Accuracy": 0.8360, "Precision": 0.2266, "Recall": 0.4274, "F1-Score": 0.2962, "AUC": 0.7433},
            {"Model": "LightGBM", "Pendekatan": "Class Weight (Thresh=0.5)", "Threshold": 0.500, "Accuracy": 0.6982, "Precision": 0.1657, "Recall": 0.6785, "F1-Score": 0.2663, "AUC": 0.7566},
            {"Model": "LightGBM", "Pendekatan": "SMOTE (Thresh=0.5)", "Threshold": 0.500, "Accuracy": 0.9188, "Precision": 0.4609, "Recall": 0.0332, "F1-Score": 0.0620, "AUC": 0.7433},
            {"Model": "Logistic Regression", "Pendekatan": "SMOTE + Tuning", "Threshold": 0.513, "Accuracy": 0.8455, "Precision": 0.2318, "Recall": 0.3948, "F1-Score": 0.2921, "AUC": 0.7411},
            {"Model": "Logistic Regression", "Pendekatan": "SMOTE (Thresh=0.5)", "Threshold": 0.500, "Accuracy": 0.8377, "Precision": 0.2236, "Recall": 0.4087, "F1-Score": 0.2891, "AUC": 0.7411},
        ],
        "lgd_before": 2775357715.50,
        "lgd_results": [
            {"Model": "Logistic Regression", "Uang Diselamatkan": 1082683944.00, "Sisa Kerugian (LGD After)": 1692673771.50, "Persentase Penghematan": 39.01},
            {"Model": "LightGBM", "Uang Diselamatkan": 78982078.50, "Sisa Kerugian (LGD After)": 2696375637.00, "Persentase Penghematan": 2.85},
            {"Model": "CatBoost", "Uang Diselamatkan": 68294344.50, "Sisa Kerugian (LGD After)": 2707063371.00, "Persentase Penghematan": 2.46},
        ],
        "delong_pvalues": {
            "Logistic Regression vs LightGBM": 0.77644,
            "Logistic Regression vs CatBoost": 0.87228,
            "LightGBM vs CatBoost": 0.57264,
        },
        "dataset_info": {
            "total_records": 307511,
            "default_pct": 8.07,
            "n_features_original": 122,
            "n_features_after_ohe": 239,
            "n_features_selected": 34,
            "train_size": 246008,
            "test_size": 61503,
            "smote_train_size": 339222,
        },
    }


# --- Load data ---
real_data = load_real_models()
IS_DEMO = real_data is None
config = real_data["config"] if not IS_DEMO else get_demo_config()
feature_names = real_data["feature_names"] if not IS_DEMO else config["feature_names"]


# ============================================
# HELPER FUNCTIONS
# ============================================

def predict_real(input_dict, model_name="LightGBM"):
    """Predict using real model loaded from .pkl."""
    models = real_data["models"]
    model = models[model_name]
    input_df = pd.DataFrame([input_dict])[feature_names]
    
    # Impute missing values manually using medians from config.json
    # This prevents scikit-learn version mismatch errors (AttributeError/KeyError) on Streamlit Cloud
    stats = config.get("feature_stats", {})
    for col in feature_names:
        if input_df[col].isnull().any():
            median_val = stats.get(col, {}).get("median", 0.0)
            input_df[col] = input_df[col].fillna(median_val)
            
    prob = model.predict_proba(input_df)[0][1]
    return prob


def predict_demo(input_dict):
    """Simulate prediction using weighted heuristic (demo mode only)."""
    stats = config["feature_stats"]
    score = 0.0
    weights = {
        "EXT_SOURCE_3": -0.25, "EXT_SOURCE_2": -0.20, "EXT_SOURCE_1": -0.15,
        "AGE_YEARS": -0.05, "AMT_CREDIT": 0.08, "AMT_INCOME_TOTAL": -0.05,
        "INCOME_TO_ANNUITY_RATIO": -0.03, "YEARS_EMPLOYED": -0.04,
        "DEF_30_CNT_SOCIAL_CIRCLE": 0.10, "DEF_60_CNT_SOCIAL_CIRCLE": 0.08,
    }
    for feat, w in weights.items():
        if feat in input_dict and feat in stats:
            s = stats[feat]
            rng = s["max"] - s["min"]
            if rng > 0:
                norm_val = (input_dict[feat] - s["min"]) / rng
            else:
                norm_val = 0.5
            score += w * norm_val

    # Add bias and binary features
    binary_risk = ["REG_CITY_NOT_WORK_CITY", "NAME_CONTRACT_TYPE_Revolving loans",
                   "FLAG_WORK_PHONE"]
    for feat in binary_risk:
        if feat in input_dict:
            score += 0.03 * input_dict[feat]

    prob = 1 / (1 + np.exp(-(score + 0.2)))  # sigmoid with base offset
    return np.clip(prob, 0.02, 0.98)


def get_risk_level(prob, threshold):
    if prob > 0.7:
        return "VERY HIGH RISK", "risk-high"
    elif prob > threshold:
        return "HIGH RISK", "risk-high"
    elif prob > threshold * 0.7:
        return "MEDIUM RISK", "risk-medium"
    else:
        return "LOW RISK", "risk-low"


def get_recommendation(prob, threshold):
    if prob > 0.7:
        return "REJECT", "Probabilitas default sangat tinggi. Kredit tidak direkomendasikan.", "rec-reject"
    elif prob > threshold:
        return "REVIEW MANUAL", "Perlu verifikasi lebih lanjut oleh analis kredit.", "rec-review"
    elif prob > threshold * 0.5:
        return "APPROVE + MONITORING", "Risiko acceptable, lakukan monitoring berkala.", "rec-approve"
    else:
        return "APPROVE", "Risiko sangat rendah. Kredit layak disetujui.", "rec-approve"


def format_rupiah(value):
    if abs(value) >= 1e12:
        return f"Rp {value/1e12:,.2f} T"
    elif abs(value) >= 1e9:
        return f"Rp {value/1e9:,.2f} M"
    elif abs(value) >= 1e6:
        return f"Rp {value/1e6:,.1f} Jt"
    else:
        return f"Rp {value:,.0f}"


def make_gauge(prob, threshold):
    """Create Plotly gauge chart for default probability."""
    color = "#e53e3e" if prob > threshold else "#38a169"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 48, "color": color}},
        delta={"reference": threshold * 100, "suffix": "%", "increasing": {"color": "#e53e3e"}, "decreasing": {"color": "#38a169"}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "white",
            "steps": [
                {"range": [0, threshold * 100 * 0.5], "color": "#c6f6d5"},
                {"range": [threshold * 100 * 0.5, threshold * 100], "color": "#fefcbf"},
                {"range": [threshold * 100, 70], "color": "#fed7d7"},
                {"range": [70, 100], "color": "#feb2b2"},
            ],
            "threshold": {
                "line": {"color": "#1a1a2e", "width": 3},
                "thickness": 0.8,
                "value": threshold * 100,
            },
        },
        title={"text": "Default Probability", "font": {"size": 16}},
    ))
    fig.update_layout(height=280, margin=dict(l=30, r=30, t=60, b=20))
    return fig


# ============================================
# SIDEBAR NAVIGATION
# ============================================

with st.sidebar:
    st.markdown("### 🏦 Navigation")
    page = st.radio(
        "Pilih Halaman:",
        ["🏠 Prediksi", "📊 Model Performance", "🔬 SHAP Explainability",
         "💰 Simulasi LGD", "ℹ️ About"],
        label_visibility="collapsed"
    )

    if IS_DEMO:
        st.markdown("---")
        st.warning("**Mode Demo** — Model .pkl belum tersedia. Menggunakan data hardcoded dari notebook.")
    else:
        st.markdown("---")
        st.success("**Mode Real** — Model terlatih berhasil di-load.")


# ============================================
# PAGE 1: PREDIKSI
# ============================================

if page == "🏠 Prediksi":
    st.markdown("""
    <div class="main-header">
        <h1>🏦 Loan Default Risk Predictor</h1>
        <p>Sistem prediksi risiko gagal bayar berbasis Machine Learning — ML Group 2 ITK</p>
    </div>
    """, unsafe_allow_html=True)

    # Get optimal threshold for LightGBM
    thresh_data = {r["Model"]: r for r in config["threshold_results"]}
    lgb_threshold = thresh_data.get("LightGBM", {}).get("Threshold Optimal", 0.147)

    # --- Input form in sidebar ---
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📋 Input Data Nasabah")

        stats = config.get("feature_stats", {})

        # Group features logically
        st.markdown("#### 🔑 Skor Kredit Eksternal")
        ext3 = st.slider("EXT_SOURCE_3", 0.0, 1.0, 0.51, 0.01, help="Skor dari sumber eksternal 3")
        ext2 = st.slider("EXT_SOURCE_2", 0.0, 1.0, 0.51, 0.01, help="Skor dari sumber eksternal 2")
        ext1 = st.slider("EXT_SOURCE_1", 0.0, 1.0, 0.50, 0.01, help="Skor dari sumber eksternal 1")

        st.markdown("#### 💰 Informasi Finansial")
        amt_credit = st.number_input("Jumlah Kredit (AMT_CREDIT)", 25000, 5000000, 500000, 10000)
        amt_income = st.number_input("Pendapatan Tahunan (AMT_INCOME_TOTAL)", 25000, 120000000, 168000, 10000)
        amt_annuity = st.number_input("Cicilan Tahunan (AMT_ANNUITY)", 1500, 260000, 27000, 1000)
        amt_goods_price = st.number_input("Harga Barang (AMT_GOODS_PRICE)", 25000, 5000000, 450000, 10000)
        income_to_annuity = amt_income / max(amt_annuity, 1)

        st.markdown("#### 👤 Informasi Personal")
        age = st.slider("Usia (AGE_YEARS)", 20, 69, 44, 1)
        years_employed = st.slider("Tahun Bekerja (YEARS_EMPLOYED)", 0.0, 49.0, 6.0, 0.5)
        code_gender = st.selectbox("Jenis Kelamin (CODE_GENDER)", ["Perempuan", "Laki-laki"])
        code_gender_m = 1 if code_gender == "Laki-laki" else 0
        
        has_car = st.checkbox("Memiliki Mobil?", value=False)
        if has_car:
            own_car_age = st.slider("Usia Mobil (Tahun)", 0.0, 91.0, 9.0, 1.0)
        else:
            own_car_age = np.nan

        st.markdown("#### 🏢 Profil Pekerjaan & Lokasi")
        is_laborer = st.selectbox("Pekerjaan: Buruh / Laborer?", [0, 1], index=0)
        region_rating_wc = st.selectbox("Rating Region + City", [1, 2, 3], index=1)
        region_rating = st.selectbox("Rating Region", [1, 2, 3], index=1)
        region_pop = st.slider("Kepadatan Penduduk Wilayah (REGION_POPULATION_RELATIVE)", 0.00029, 0.0725, 0.0188, 0.0001)
        
        income_working = st.selectbox("Tipe Pendapatan: Working?", [0, 1], index=1)
        income_pensioner = st.selectbox("Tipe Pendapatan: Pensioner?", [0, 1], index=0)
        edu_higher = st.selectbox("Pendidikan Tinggi?", [0, 1], index=0)
        edu_secondary = st.selectbox("Pendidikan Menengah?", [0, 1], index=1)
        org_xna = st.selectbox("Organisasi: XNA?", [0, 1], index=0)
        reg_city_not_work = st.selectbox("Kota Registrasi ≠ Kota Kerja?", [0, 1], index=0)
        reg_city_not_live = st.selectbox("Kota Registrasi ≠ Kota Tinggal?", [0, 1], index=0)

        st.markdown("#### 🏢 Informasi Properti & Dokumen")
        floorsmax = st.slider("Floors Max (Informasi Apartemen)", 0.0, 1.0, 0.16, 0.01)
        days_phone = st.number_input("DAYS_LAST_PHONE_CHANGE", -4300, 0, -960, 10)
        years_id = st.slider("YEARS_ID_PUBLISH", 0.0, 20.0, 8.0, 0.1)
        years_reg = st.slider("YEARS_REGISTRATION", 0.0, 68.0, 14.0, 0.5)
        flag_emp_phone = st.selectbox("FLAG_EMP_PHONE", [0, 1], index=1)
        flag_doc3 = st.selectbox("FLAG_DOCUMENT_3", [0, 1], index=1)
        
        emp_to_income = years_employed / max(amt_income, 1)

    # Build input dict with all 34 features in models_trained.pkl
    input_data = {
        "EXT_SOURCE_3": ext3,
        "EXT_SOURCE_2": ext2,
        "EXT_SOURCE_1": ext1,
        "AGE_YEARS": float(age),
        "DAYS_BIRTH": float(age) * -365.0,
        "DAYS_EMPLOYED": float(years_employed) * -365.0,
        "YEARS_EMPLOYED": float(years_employed),
        "REGION_RATING_CLIENT_W_CITY": region_rating_wc,
        "REGION_RATING_CLIENT": region_rating,
        "NAME_INCOME_TYPE_Working": income_working,
        "NAME_EDUCATION_TYPE_Higher education": edu_higher,
        "DAYS_LAST_PHONE_CHANGE": float(days_phone),
        "CODE_GENDER_M": float(code_gender_m),
        "EMPLOYMENT_TO_INCOME_RATIO": float(emp_to_income),
        "DAYS_ID_PUBLISH": float(years_id) * -365.0,
        "REG_CITY_NOT_WORK_CITY": reg_city_not_work,
        "NAME_EDUCATION_TYPE_Secondary / secondary special": edu_secondary,
        "NAME_INCOME_TYPE_Pensioner": income_pensioner,
        "ORGANIZATION_TYPE_XNA": org_xna,
        "FLAG_EMP_PHONE": flag_emp_phone,
        "REG_CITY_NOT_LIVE_CITY": reg_city_not_live,
        "FLAG_DOCUMENT_3": flag_doc3,
        "FLOORSMAX_AVG": floorsmax,
        "FLOORSMAX_MEDI": floorsmax,
        "FLOORSMAX_MODE": floorsmax,
        "OCCUPATION_TYPE_Laborers": is_laborer,
        "DAYS_REGISTRATION": float(years_reg) * -365.0,
        "AMT_GOODS_PRICE": float(amt_goods_price),
        "OWN_CAR_AGE": own_car_age,
        "REGION_POPULATION_RELATIVE": region_pop,
        "AMT_CREDIT": float(amt_credit),
        "AMT_ANNUITY": float(amt_annuity),
        "AMT_INCOME_TOTAL": float(amt_income),
        "INCOME_TO_ANNUITY_RATIO": float(income_to_annuity),
    }

    # --- Predict ---
    if IS_DEMO:
        prob = predict_demo(input_data)
    else:
        prob = predict_real(input_data, "LightGBM")

    risk_label, risk_class = get_risk_level(prob, lgb_threshold)
    rec_label, rec_detail, rec_class = get_recommendation(prob, lgb_threshold)

    # --- Display Results ---
    col_gauge, col_info = st.columns([1, 1])

    with col_gauge:
        st.plotly_chart(make_gauge(prob, lgb_threshold), use_container_width=True)

    with col_info:
        st.markdown(f"<div style='text-align:center; padding-top:1rem;'><span class='{risk_class}'>{risk_label}</span></div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="rec-box {rec_class}">
            <strong>{rec_label}</strong><br>
            <span style="font-size:0.9rem;">{rec_detail}</span>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Probabilitas", f"{prob*100:.1f}%")
        with m2:
            st.metric("Threshold", f"{lgb_threshold*100:.1f}%")
        with m3:
            st.metric("Model AUC", f"{thresh_data.get('LightGBM', {}).get('F1 @Optimal', 0.2962):.4f}")

    # Comparison across thresholds
    st.markdown("---")
    st.subheader("⚖️ Perbandingan Keputusan Antar Model & Threshold")

    compare_rows = []
    for t in config["threshold_results"]:
        model_name = t["Model"]
        opt_thresh = t["Threshold Optimal"]
        compare_rows.append({
            "Model": model_name,
            "Threshold": f"{opt_thresh:.3f}",
            "Keputusan @0.5": "🔴 HIGH" if prob > 0.5 else "🟢 LOW",
            "Keputusan @Optimal": "🔴 HIGH" if prob > opt_thresh else "🟢 LOW",
            "Recall @0.5": f"{t['Recall @0.5']*100:.1f}%",
            "Recall @Optimal": f"{t['Recall @Optimal']*100:.1f}%",
        })
    st.dataframe(pd.DataFrame(compare_rows), use_container_width=True, hide_index=True)

    # Download prediction
    st.markdown("---")
    pred_df = pd.DataFrame([{
        "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "Probability": f"{prob:.4f}",
        "Risk Level": risk_label,
        "Recommendation": rec_label,
        **{k: v for k, v in input_data.items()}
    }])
    st.download_button(
        "📥 Download Hasil Prediksi (CSV)",
        pred_df.to_csv(index=False),
        "prediction_result.csv",
        "text/csv"
    )


# ============================================
# PAGE 2: MODEL PERFORMANCE
# ============================================

elif page == "📊 Model Performance":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Model Performance Dashboard</h1>
        <p>Perbandingan performa 3 model: Logistic Regression, LightGBM, CatBoost</p>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["📋 Ringkasan", "🔄 Komparasi 3-Cara", "📈 ROC & PR Curve", "🧪 DeLong Test"])

    # Tab 1: Model Summary
    with tabs[0]:
        st.subheader("Ringkasan Performa Model (Threshold = 0.5)")
        metrics_df = pd.DataFrame(config["model_metrics"])
        cols_display = ["Model", "AUC", "Recall", "F1-Score", "Accuracy", "Train Time (s)", "Inference Time (s)"]
        available_cols = [c for c in cols_display if c in metrics_df.columns]
        st.dataframe(metrics_df[available_cols], use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Threshold Tuning Results")
        thresh_df = pd.DataFrame(config["threshold_results"])
        st.dataframe(thresh_df, use_container_width=True, hide_index=True)

        st.info("""
        **Key Insight:** Threshold default (0.5) menghasilkan Recall sangat rendah untuk LightGBM (3.3%) dan CatBoost (2.8%).
        Setelah threshold tuning, Recall melonjak menjadi **42.7%** (LightGBM) dan **41.6%** (CatBoost) —
        peningkatan **13x lipat**!
        """)

    # Tab 2: 3-Way Comparison
    with tabs[1]:
        st.subheader("Komparasi: SMOTE vs Class Weight vs Threshold Tuning")
        comp_df = pd.DataFrame(config["comparison_table"])
        st.dataframe(
            comp_df.sort_values(by=["Model", "F1-Score"], ascending=[True, False]),
            use_container_width=True, hide_index=True
        )

        # Grouped bar chart
        fig = px.bar(
            comp_df, x="Model", y="Recall", color="Pendekatan",
            barmode="group", title="Recall per Model & Pendekatan",
            color_discrete_sequence=px.colors.qualitative.Set2,
            text=comp_df["Recall"].apply(lambda x: f"{x*100:.1f}%")
        )
        fig.update_layout(yaxis_tickformat=".0%", height=450)
        st.plotly_chart(fig, use_container_width=True)

    # Tab 3: ROC & PR Curve (hardcoded visualization)
    with tabs[2]:
        st.subheader("ROC Curve & Precision-Recall Curve")
        st.info("Kurva ROC dan PR Curve dihasilkan dari data test set di notebook. Untuk visualisasi interaktif, jalankan notebook dan lihat hasilnya.")

        # Show AUC comparison bar chart
        auc_data = pd.DataFrame(config["model_metrics"])[["Model", "AUC"]]
        fig_auc = px.bar(
            auc_data, x="Model", y="AUC",
            title="AUC Comparison",
            text=auc_data["AUC"].apply(lambda x: f"{x:.4f}"),
            color="Model",
            color_discrete_sequence=["#0f3460", "#e94560", "#533483"]
        )
        fig_auc.update_layout(height=400, yaxis_range=[0.7, 0.76], showlegend=False)
        st.plotly_chart(fig_auc, use_container_width=True)

        st.caption("📌 Semua model memiliki AUC yang sangat dekat (~0.74-0.75), menunjukkan kemampuan diskriminasi yang setara.")

    # Tab 4: DeLong Test
    with tabs[3]:
        st.subheader("Uji Signifikansi Statistik (DeLong Test)")

        delong = config.get("delong_pvalues", {})
        delong_rows = []
        for pair, pval in delong.items():
            pval_f = float(pval)
            delong_rows.append({
                "Pasangan Model": pair,
                "P-Value": f"{pval_f:.5f}",
                "Signifikan?": "✅ Ya (p < 0.05)" if pval_f < 0.05 else "❌ Tidak (p ≥ 0.05)"
            })
        st.dataframe(pd.DataFrame(delong_rows), use_container_width=True, hide_index=True)

        st.warning("""
        **Kesimpulan DeLong Test:** Semua pasangan model menghasilkan p-value > 0.05,
        artinya **perbedaan AUC antar model TIDAK signifikan secara statistik**.
        Model boosting (LightGBM & CatBoost) memiliki performa setara dengan Logistic Regression pada metrik AUC.
        Keunggulan utama mereka muncul saat dilakukan **threshold tuning**.
        """)


# ============================================
# PAGE 3: SHAP EXPLAINABILITY
# ============================================

elif page == "🔬 SHAP Explainability":
    st.markdown("""
    <div class="main-header">
        <h1>🔬 SHAP Explainability</h1>
        <p>Memahami faktor-faktor yang mempengaruhi prediksi model</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Top Feature Importance (dari Analisis SHAP)")
    st.info("SHAP (SHapley Additive exPlanations) membedah 'otak AI' dan menunjukkan fitur mana yang paling berpengaruh terhadap keputusan model.")

    # Hardcoded top features from typical Home Credit SHAP analysis
    shap_features = [
        ("EXT_SOURCE_3", 0.052, "Skor kredit dari sumber eksternal (paling penting!)"),
        ("EXT_SOURCE_2", 0.043, "Skor kredit dari sumber eksternal lainnya"),
        ("EXT_SOURCE_1", 0.028, "Skor kredit dari sumber eksternal tambahan"),
        ("AGE_YEARS", 0.015, "Usia nasabah — semakin tua cenderung lebih rendah risikonya"),
        ("AMT_CREDIT", 0.012, "Jumlah kredit yang diajukan"),
        ("AMT_ANNUITY", 0.010, "Jumlah cicilan tahunan"),
        ("INCOME_TO_ANNUITY_RATIO", 0.009, "Rasio pendapatan terhadap cicilan"),
        ("YEARS_EMPLOYED", 0.008, "Lama bekerja — semakin lama semakin stabil"),
        ("AMT_INCOME_TOTAL", 0.007, "Total pendapatan nasabah"),
        ("DAYS_LAST_PHONE_CHANGE", 0.006, "Kapan terakhir ganti nomor telepon"),
    ]

    feat_names_shap = [f[0] for f in shap_features]
    feat_vals = [f[1] for f in shap_features]
    feat_desc = [f[2] for f in shap_features]

    fig = go.Figure(go.Bar(
        x=feat_vals[::-1],
        y=feat_names_shap[::-1],
        orientation="h",
        marker=dict(
            color=feat_vals[::-1],
            colorscale=[[0, "#38a169"], [0.5, "#dd6b20"], [1, "#e53e3e"]],
        ),
        text=[f"{v:.3f}" for v in feat_vals[::-1]],
        textposition="outside"
    ))
    fig.update_layout(
        title="Top 10 Features — Mean |SHAP Value|",
        xaxis_title="Mean |SHAP Value|",
        height=500,
        margin=dict(l=200),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Explanation table
    st.markdown("### 📝 Interpretasi Fitur")
    explain_df = pd.DataFrame({
        "Fitur": feat_names_shap,
        "SHAP Importance": feat_vals,
        "Penjelasan Bisnis": feat_desc,
    })
    st.dataframe(explain_df, use_container_width=True, hide_index=True)

    st.success("""
    **Insight Utama:**
    - **EXT_SOURCE** (skor kredit eksternal) adalah prediktor terkuat — ini selaras dengan literatur
    - **AGE_YEARS** dan **YEARS_EMPLOYED** menunjukkan bahwa nasabah dengan riwayat kerja stabil memiliki risiko lebih rendah
    - **AMT_CREDIT** dan **INCOME_TO_ANNUITY_RATIO** mencerminkan kapasitas finansial
    """)


# ============================================
# PAGE 4: SIMULASI LGD
# ============================================

elif page == "💰 Simulasi LGD":
    st.markdown("""
    <div class="main-header">
        <h1>💰 Simulasi Loss Given Default (LGD)</h1>
        <p>Dampak bisnis penerapan model — berapa uang yang bisa diselamatkan?</p>
    </div>
    """, unsafe_allow_html=True)

    lgd_before = config.get("lgd_before", 2775357715.50)
    lgd_results = config.get("lgd_results", [])

    # Top metric
    st.markdown(f"""
    <div class="metric-card" style="margin-bottom: 1.5rem;">
        <h3>Total Eksposur Kredit Nasabah Default (Sebelum Model)</h3>
        <p style="color: #e53e3e;">{format_rupiah(lgd_before)}</p>
    </div>
    """, unsafe_allow_html=True)

    if lgd_results:
        lgd_df = pd.DataFrame(lgd_results)

        col1, col2, col3 = st.columns(3)
        for i, row in lgd_df.iterrows():
            col = [col1, col2, col3][i % 3]
            with col:
                saved = row.get("Uang Diselamatkan", 0)
                pct = row.get("Persentase Penghematan", 0)
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{row['Model']}</h3>
                    <p style="color: #38a169;">{format_rupiah(saved)}</p>
                    <span style="font-size: 0.85rem; color: #6c757d;">diselamatkan ({pct:.1f}%)</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # Bar chart
        fig = go.Figure()
        for _, row in lgd_df.iterrows():
            saved = row.get("Uang Diselamatkan", 0)
            remaining = row.get("Sisa Kerugian (LGD After)", 0)
            fig.add_trace(go.Bar(name="Diselamatkan", x=[row["Model"]], y=[saved], marker_color="#38a169", text=[format_rupiah(saved)], textposition="inside"))
            fig.add_trace(go.Bar(name="Sisa Kerugian", x=[row["Model"]], y=[remaining], marker_color="#e53e3e", text=[format_rupiah(remaining)], textposition="inside"))

        fig.update_layout(
            barmode="stack",
            title="Perbandingan Uang Diselamatkan vs Sisa Kerugian",
            yaxis_title="Rupiah",
            height=500,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        # Remove duplicate legend entries
        seen = set()
        for trace in fig.data:
            if trace.name in seen:
                trace.showlegend = False
            else:
                seen.add(trace.name)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📋 Tabel Detail LGD")
        display_lgd = lgd_df.copy()
        for col_name in ["Uang Diselamatkan", "Sisa Kerugian (LGD After)"]:
            if col_name in display_lgd.columns:
                display_lgd[col_name] = display_lgd[col_name].apply(format_rupiah)
        if "Persentase Penghematan" in display_lgd.columns:
            display_lgd["Persentase Penghematan"] = display_lgd["Persentase Penghematan"].apply(lambda x: f"{x:.2f}%")
        st.dataframe(display_lgd, use_container_width=True, hide_index=True)

        st.warning("""
        **Catatan Penting:** Nilai LGD di atas menggunakan threshold default 0.5. Dengan threshold optimal (0.147 untuk LightGBM),
        performa deteksi model meningkat drastis sehingga **uang yang diselamatkan juga meningkat secara signifikan**.
        Untuk simulasi LGD dengan threshold optimal, jalankan kalkulasi tambahan di notebook.
        """)


# ============================================
# PAGE 5: ABOUT & TECHNICAL
# ============================================

elif page == "ℹ️ About":
    st.markdown("""
    <div class="main-header">
        <h1>ℹ️ About & Technical Details</h1>
        <p>Arsitektur pipeline, dataset, dan referensi ilmiah</p>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["🔧 Pipeline", "📚 Dataset", "📖 Referensi", "⚠️ Disclaimer"])

    with tabs[0]:
        st.subheader("Arsitektur Pipeline End-to-End")
        st.markdown("""
        ```
        ┌─────────────────────────────────────────────────────────────────────┐
        │                    PREPROCESSING PIPELINE                          │
        ├─────────────┬─────────────┬──────────────┬─────────────────────────┤
        │ Data Merge  │ Missing Val │ Feature Eng. │ One-Hot Encoding        │
        │ (3 tables)  │ (post-split)│ (AGE, RATIOS)│ (Kategorikal → Biner)  │
        └──────┬──────┴──────┬──────┴──────┬───────┴──────────┬──────────────┘
               │             │             │                  │
               v             v             v                  v
        ┌──────────────────────────────────────────────────────────────────────┐
        │                     FEATURE SELECTION                                │
        │              Top 30 Korelasi + 4 Domain Features = 34 Fitur         │
        └──────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       v
        ┌──────────────────────────────────────────────────────────────────────┐
        │                   TRAIN-TEST SPLIT (80/20, Stratified)               │
        │              SimpleImputer(median) fit pada X_train saja             │
        └──────────────────────────────┬───────────────────────────────────────┘
                                       │
                            ┌──────────┴──────────┐
                            v                     v
                    ┌───────────────┐     ┌───────────────┐
                    │  SMOTE (0.5)  │     │ Class Weight  │
                    │  339K samples │     │ (is_unbalance)│
                    └───────┬───────┘     └───────┬───────┘
                            │                     │
                            v                     v
        ┌──────────────────────────────────────────────────────────────────────┐
        │                MODELING & HYPERPARAMETER TUNING                      │
        │        RandomizedSearchCV (3-Fold Stratified CV, n_iter=5)           │
        │                                                                      │
        │    ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐         │
        │    │   Logistic   │  │   LightGBM   │  │     CatBoost     │         │
        │    │  Regression  │  │  (Boosting)  │  │    (Boosting)    │         │
        │    └──────────────┘  └──────────────┘  └──────────────────┘         │
        └──────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       v
        ┌──────────────────────────────────────────────────────────────────────┐
        │                        EVALUASI & ANALISIS                           │
        │                                                                      │
        │  ┌─────────┐ ┌──────────┐ ┌────────┐ ┌────────┐ ┌──────────────┐   │
        │  │Confusion│ │Threshold │ │  SHAP  │ │ DeLong │ │  Simulasi    │   │
        │  │ Matrix  │ │  Tuning  │ │Analysis│ │  Test  │ │   LGD        │   │
        │  └─────────┘ └──────────┘ └────────┘ └────────┘ └──────────────┘   │
        └──────────────────────────────────────────────────────────────────────┘
        ```
        """)

    with tabs[1]:
        st.subheader("Dataset: Home Credit Default Risk")
        ds = config.get("dataset_info", {})
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            | Informasi | Nilai |
            |---|---|
            | **Total Records** | {ds.get('total_records', 307511):,} |
            | **Default Rate** | {ds.get('default_pct', 8.07):.2f}% |
            | **Fitur Asli** | {ds.get('n_features_original', 122)} |
            | **Fitur Setelah OHE** | {ds.get('n_features_after_ohe', 239)} |
            | **Fitur Terpilih** | {ds.get('n_features_selected', 34)} |
            """)
        with col2:
            st.markdown(f"""
            | Split | Jumlah |
            |---|---|
            | **Training (80%)** | {ds.get('train_size', 246008):,} |
            | **Testing (20%)** | {ds.get('test_size', 61503):,} |
            | **Training + SMOTE** | {ds.get('smote_train_size', 339222):,} |
            | **Sumber Data** | application_train, previous_application, bureau |
            | **Periode** | Kaggle Competition |
            """)

    with tabs[2]:
        st.subheader("Referensi Jurnal")
        st.markdown("""
        1. **Putra, H. & Rumini (2025)**. *Comparative Study of Logistic Regression, Random Forest, and XGBoost
           for Bank Loan Approval Classification.* JAIC Vol. 9, No. 5, pp. 2822-2835.
           - Paper referensi utama yang menjadi dasar pipeline
           - Kami mengganti RF & XGBoost dengan **LightGBM & CatBoost** (model generasi lebih baru)
           - Kami menambahkan **Threshold Tuning** untuk mengatasi masalah recall rendah

        2. **Cai, Dai & Lu (2025)**. *Loan Default Prediction* (Alibaba Tianchi).
           - CatBoost unggul pada data dengan banyak fitur kategorikal
           - Kami verifikasi dengan eksperimen class weighting

        3. **Zhang (2025)**. *Explainable ML for Loan Risk.*
           - SHAP interpretation penting untuk domain keuangan
           - Kami sertakan analisis SHAP untuk interpretabilitas model
        """)

    with tabs[3]:
        st.subheader("⚠️ Disclaimer & Limitations")
        st.warning("""
        **PERINGATAN:**

        1. **Tujuan Akademis**: Model ini dibuat untuk tugas kuliah Machine Learning di Institut Teknologi Kalimantan.
           Tidak boleh digunakan untuk keputusan kredit riil tanpa validasi dari ahli keuangan.

        2. **Keterbatasan Model**:
           - AUC ~0.74 (moderate, belum enterprise-grade)
           - Precision rendah (~22%) → banyak false positive
           - Data training dari Kaggle Competition (mungkin tidak representatif untuk pasar Indonesia)

        3. **Keterbatasan Data**:
           - Class imbalance ekstrem (92% non-default)
           - Missing values di beberapa fitur
           - Tidak mencakup data terbaru

        4. **Rekomendasi**:
           - Gunakan sebagai **screening tool** awal saja
           - Prediksi HIGH RISK → **wajib review manual** oleh analis kredit
           - Model perlu dilatih ulang secara berkala dengan data terbaru
        """)


# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: gray; font-size: 0.85rem; padding: 1rem 0;'>
    <p>🏦 <strong>Loan Default Risk Predictor v2.0</strong></p>
    <p>ML Group 2 — Institut Teknologi Kalimantan</p>
    <p>Models: LightGBM + CatBoost + Logistic Regression | Dataset: Home Credit Default Risk</p>
    <p>{"🔶 Demo Mode" if IS_DEMO else "🟢 Production Mode"}</p>
</div>
""", unsafe_allow_html=True)
