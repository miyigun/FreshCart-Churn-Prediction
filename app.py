import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from src.config import PROCESSED_DATA_DIR, MODEL_DIR

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="FreshCart Churn Prediction",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- HIGH CONTRAST DARK THEME CSS ---
st.markdown("""
<style>
    /* Global Settings (Ana Uygulama) */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* --- SIDEBAR DÜZELTMESİ --- */
    [data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d;
    }
    
    /* Sidebar içindeki tüm metinleri beyaz yap */
    [data-testid="stSidebar"] * {
        color: #e6edf3 !important;
    }

    /* --- RADYO BUTONU VE CHECKBOX METİNLERİ (KESİN ÇÖZÜM) --- */
    
    /* Radyo butonu seçenek metinleri (span ve p etiketleri) */
    .stRadio label span, .stRadio label p {
        color: #ffffff !important;
        font-size: 1rem;
    }
    
    /* Radyo butonu başlığı (Label) */
    .stRadio > label {
        color: #ffffff !important;
        font-weight: bold;
        font-size: 1.1rem;
    }
    
    /* Radyo butonu seçeneklerinin kapsayıcısı */
    div[role="radiogroup"] {
        color: #ffffff !important;
    }

    /* --- DİĞER ELEMANLAR --- */
    
    /* Selectbox Etiketi */
    .stSelectbox label {
        color: #ffffff !important;
        font-weight: bold;
    }
    
    /* Selectbox Kutusu */
    .stSelectbox > div > div {
        background-color: #21262d !important;
        color: #ffffff !important;
        border: 1px solid #58a6ff;
    }
    
    /* Custom Info Box */
    .info-box {
        background-color: #1f2937;
        border: 1px solid #58a6ff;
        padding: 1.5rem;
        border-radius: 5px;
        margin-bottom: 2rem;
    }
    .info-box h4 {
        color: #58a6ff !important;
        margin-top: 0;
    }
    .info-box p {
        color: #e5e7eb !important;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #21262d;
        border: 1px solid #484f58;
        padding: 15px;
        border-radius: 10px;
    }
    div[data-testid="stMetric"] label {
        color: #8b949e !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #ffffff !important;
    }

    /* Header Fix */
    header[data-testid="stHeader"] {
        background-color: #0e1117 !important;
    }
    
    /* Başlıklar */
    h1, h2, h3 {
        color: #58a6ff !important;
    }
    
    /* Genel Paragraf Metinleri */
    p {
        color: #e6edf3;
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
@st.cache_resource
def load_artifacts():
    """Loads the trained model and necessary metadata."""
    try:
        model = joblib.load(MODEL_DIR / 'final_model_optimized.pkl')
    except FileNotFoundError:
        st.error("⚠️ Model file not found. Please check the 'models' directory.")
        st.stop()
    
    try:
        # First try loading from models directory, then processed
        features_path = MODEL_DIR / 'feature_names.json'
        if not features_path.exists():
             features_path = PROCESSED_DATA_DIR / 'model_features.json'
             
        with open(features_path, 'r') as f:
            feature_names = json.load(f)
            
    except FileNotFoundError:
        st.error("⚠️ Feature names file not found.")
        st.stop()
        
    try:
        data = pd.read_parquet(PROCESSED_DATA_DIR / 'final_features_advanced.parquet')
        cols_to_keep = ['user_id', 'is_churn'] + feature_names
        cols_to_keep = [c for c in cols_to_keep if c in data.columns]
        data = data[cols_to_keep]
    except:
        data = pd.DataFrame()

    return model, feature_names, data

# --- LOAD DATA ---
try:
    model, feature_names, df = load_artifacts()
except Exception as e:
    st.error(f"System Error: {e}")
    st.stop()

# --- SIDEBAR ---
st.sidebar.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=80)
st.sidebar.title("FreshCart AI")

page = st.sidebar.radio("NAVIGATION", ["🏠 Prediction Hub", "📊 Model Analytics", "📈 Deep Insights"])

st.sidebar.markdown("---")

# --- SIDEBAR FOOTER KISMI ---
st.sidebar.markdown("""
### 👨‍💻 Developed By
<div style="margin-top: -10px;">
    <h4 style="margin-bottom: 0px; color: #ffffff;">Murat IYIGUN</h4>
    <p style="margin-top: 0px; font-size: 0.9rem; color: #8b949e; font-style: italic;">
        Data Scientist & AI Engineer
    </p>
</div>
""", unsafe_allow_html=True)

# --- PAGE 1: PREDICTION HUB ---
if page == "🏠 Prediction Hub":
    # HEADER
    st.title("🛒 Customer Churn Prediction System")
    
    # MISSION STATEMENT (GERİ EKLENDİ)
    st.markdown("""
    <div class="info-box">
        <h4>🎯 Project Mission</h4>
        <p>
            This system leverages <strong>Advanced Machine Learning (LightGBM)</strong> to predict customer churn risk 
            <strong>14 days in advance</strong>. By analyzing purchasing behavior, velocity, and basket trends, 
            we empower the marketing team to take proactive actions and secure revenue.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if df.empty:
        st.warning("⚠️ Data stream offline.")
    else:
        # SELECTION AREA
        st.subheader("👤 Customer Profile Selection")
        
        sel_col1, sel_col2 = st.columns([1, 2])
        
        with sel_col1:
            input_method = st.radio("Source:", ["ID List", "Random Sample"], horizontal=True)
        
        with sel_col2:
            if input_method == "ID List":
                selected_user_id = st.selectbox("Search Customer ID:", df['user_id'].head(100).tolist())
            else:
                if st.button("🎲 Generate Random Profile", type="primary"):
                    selected_user_id = df['user_id'].sample(1).values[0]
                else:
                    selected_user_id = df['user_id'].iloc[0]

        # PREDICTION
        customer_data = df[df['user_id'] == selected_user_id].iloc[0]
        input_features = customer_data[feature_names].to_frame().T
        churn_prob = model.predict(input_features)[0]
        THRESHOLD = 0.38 
        is_churn = churn_prob >= THRESHOLD

        st.markdown("---")
        
        # RESULTS DASHBOARD
        r1, r2, r3 = st.columns([1.2, 1.5, 2.3])
        
        # 1. RISK STATUS
        with r1:
            st.markdown("### ⚡ Risk Status")
            if is_churn:
                st.metric("Prediction", "HIGH RISK", f"{churn_prob*100:.1f}% Prob", delta_color="inverse")
            else:
                st.metric("Prediction", "LOYAL", f"{churn_prob*100:.1f}% Prob", delta_color="normal")
        
        # 2. BEHAVIORAL DNA
        with r2:
            st.markdown("### 🧬 Behavioral DNA")
            st.info(f"""
            - **Recency:** {customer_data.get('days_since_last_order', 0):.0f} days ago
            - **Frequency:** {customer_data.get('total_orders', 0):.0f} total orders
            - **Basket Size:** {customer_data.get('avg_basket_size', 0):.1f} items
            - **Velocity:** {customer_data.get('purchase_velocity', 0):.2f} score
            """)

        # 3. SHAP EXPLANATION
        with r3:
            st.markdown("### 🧠 AI Reasoning (SHAP)")
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(input_features)
                
                # Dark mode uyumlu plot
                plt.style.use('dark_background')
                fig, ax = plt.subplots(figsize=(8, 4))
                fig.patch.set_alpha(0.0)
                ax.patch.set_alpha(0.0)
                
                shap.plots.waterfall(shap_values[0], max_display=5, show=False)
                
                # Eksen yazılarını beyaz yap
                for text in ax.get_yticklabels() + ax.get_xticklabels():
                    text.set_color('white')
                    text.set_fontsize(10)
                    
                st.pyplot(fig, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                st.warning("Explanation unavailable.")

# --- PAGE 2: MODEL ANALYTICS ---
elif page == "📊 Model Analytics":
    st.title("📊 System Performance Metrics")
    st.markdown("Evaluation results on test data (20% hold-out set).")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC Score", "0.92", "Excellent")
    m2.metric("F1-Score", "0.84", "Balanced")
    m3.metric("Recall Rate", "85%", "High Capture")
    m4.metric("Proj. Revenue Impact", "$1.8M", "Annual Saved")

    st.markdown("---")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📉 ROC & Precision-Recall Curves")
        try: st.image("plots/13_roc_pr_curves.png", use_column_width=True)
        except: st.info("Visualization not available.")
            
    with c2:
        st.markdown("#### 🔑 Feature Importance")
        try: st.image("plots/14_feature_importance.png", use_column_width=True)
        except: st.info("Visualization not available.")

    st.markdown("#### 💰 ROI Optimization Analysis")
    try:
        st.image("plots/20_threshold_optimization.png", use_column_width=True)
    except:
        st.info("ROI Chart not available.")

# --- PAGE 3: DATA INSIGHTS ---
elif page == "📈 Deep Insights":
    st.title("📈 Exploratory Intelligence")
    st.markdown("Discovering hidden patterns in customer behavior.")
    
    tab1, tab2 = st.tabs(["🌍 Market Overview", "🤖 AI Drivers"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### ⏰ Order Timing Habits")
            try: st.image("plots/02_orders_univariate.png", use_column_width=True)
            except: st.info("Data unavailable.")
        with col2:
            st.markdown("##### 📦 Product Affinity")
            try: st.image("plots/04_product_metrics.png", use_column_width=True)
            except: st.info("Data unavailable.")
                
    with tab2:
        st.markdown("##### 🧠 Global Explainability (SHAP)")
        try:
            st.image("plots/16_shap_summary.png", use_column_width=True)
            st.info("Feature Impact Direction: Red = High Value, Blue = Low Value.")
        except: st.info("SHAP summary unavailable.")

# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.caption("v1.0.2 | Production Build")