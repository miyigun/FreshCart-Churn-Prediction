import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import sqlite3
import plotly.express as px  # For interactive charts
import sys
import os

# --- PATH CONFIGURATION ---
# Get the absolute path of the current directory (where app.py is)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add it to sys.path if not already there to ensure imports work correctly
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.config import PROCESSED_DATA_DIR, MODEL_DIR

# --- MONITORING IMPORT ---
# DB dosyası 'src/monitoring/db.py' yolunda olduğu için
# Python'un src paketinden import ediyoruz.
try:
    from src.monitoring.db import init_db, log_prediction, get_connection
except ImportError as e:
    st.error(f"Monitoring modülü yüklenirken hata oluştu: {e}")
    st.stop()

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="FreshCart Churn Prediction",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- INIT DB ---
# Initialize monitoring database when app starts
init_db()

# --- HIGH CONTRAST DARK THEME CSS ---
st.markdown("""
<style>
    /* Global Settings (Main App) */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* --- SIDEBAR FIX --- */
    [data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d;
    }
    
    /* Make all text in Sidebar white */
    [data-testid="stSidebar"] * {
        color: #e6edf3 !important;
    }

    /* --- RADIO BUTTON AND CHECKBOX TEXTS --- */
    .stRadio label span, .stRadio label p {
        color: #ffffff !important;
        font-size: 1rem;
    }
    .stRadio > label {
        color: #ffffff !important;
        font-weight: bold;
        font-size: 1.1rem;
    }
    div[role="radiogroup"] {
        color: #ffffff !important;
    }

    /* --- OTHER ELEMENTS --- */
    .stSelectbox label {
        color: #ffffff !important;
        font-weight: bold;
    }
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
        margin-bottom: 0;
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
    
    /* Headings */
    h1, h2, h3 {
        color: #58a6ff !important;
    }
    /* General Paragraph Texts */
    p {
        color: #e6edf3;
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
@st.cache_resource
def load_artifacts():
    """Loads the trained model and necessary metadata."""
    # 1. Load Model
    try:
        model = joblib.load(MODEL_DIR / 'final_model_optimized.pkl')
    except FileNotFoundError:
        st.error("⚠️ Model file (final_model_optimized.pkl) not found in models directory.")
        st.stop()
    
    # 2. Load Feature Names
    # App looks at models/feature_names.json first, then processed/model_features.json
    feature_names = []
    feature_file_used = ""
    
    try:
        path_primary = MODEL_DIR / 'feature_names.json'
        path_secondary = PROCESSED_DATA_DIR / 'model_features.json'
        
        if path_primary.exists():
            with open(path_primary, 'r') as f:
                feature_names = json.load(f)
            feature_file_used = "models/feature_names.json"
        elif path_secondary.exists():
            with open(path_secondary, 'r') as f:
                feature_names = json.load(f)
            feature_file_used = "data/processed/model_features.json"
        else:
            st.error("⚠️ Feature list JSON file not found in models/ or data/processed/.")
            st.stop()
            
    except Exception as e:
        st.error(f"⚠️ Error loading feature names: {e}")
        st.stop()
        
    # 3. Load Data
    try:
        data_path = PROCESSED_DATA_DIR / 'final_features_advanced.parquet'
        data = pd.read_parquet(data_path)
        
        # Validate Columns immediately to prevent late KeyErrors
        missing_cols = [col for col in feature_names if col not in data.columns]
        if missing_cols:
            st.warning(f"⚠️ Data Mismatch detected! The feature list in '{feature_file_used}' expects columns not found in the parquet file: {missing_cols}")
            # Fail safe: Only keep columns that actually exist
            feature_names = [col for col in feature_names if col in data.columns]
        
        cols_to_keep = ['user_id', 'is_churn'] + feature_names
        # Ensure user_id and is_churn exist too
        cols_to_keep = [c for c in cols_to_keep if c in data.columns]
        
        data = data[cols_to_keep]
        
    except FileNotFoundError:
        st.warning("⚠️ Parquet data not found. App will run in Model-Only mode (no historical data).")
        data = pd.DataFrame()
    except Exception as e:
        st.error(f"⚠️ Error loading data: {e}")
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

# NEW OPTION ADDED TO NAVIGATION
page = st.sidebar.radio("NAVIGATION", [
    "🏠 Prediction Hub", 
    "📊 Model Analytics", 
    "📈 Deep Insights",
    "⚡ System Monitoring"
])

st.sidebar.markdown("---")

# --- SIDEBAR FOOTER ---
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
    
    st.markdown("""
    <div class="info-box">
        <h4>🎯 Project Mission</h4>
        <p>
            This system leverages <strong>Advanced Machine Learning (LightGBM)</strong> to predict customer churn risk 
            <strong>14 days in advance</strong>. Real-time predictions are logged for drift monitoring.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if df.empty:
        st.warning("⚠️ Data stream offline.")
    else:
        # SELECTION AREA
        st.subheader("👤 Customer Profile Selection")
        
        sel_col1, sel_col2, _ = st.columns([1, 1.5, 2])
        
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

        # --- LOGGING ---
        # Save to DB as soon as prediction is made
        log_prediction(
            user_id=int(selected_user_id),
            features=customer_data,
            prob=float(churn_prob),
            label=int(is_churn),
            model_version='v1.0.2'
        )
        # ---------------

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
                
                plt.style.use('dark_background')
                fig, ax = plt.subplots(figsize=(8, 4))
                fig.patch.set_alpha(0.0)
                ax.patch.set_alpha(0.0)
                
                shap.plots.waterfall(shap_values[0], max_display=5, show=False)
                
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
    m1.metric("ROC-AUC Score", "0.77", "Good Stability")
    m2.metric("F1-Score", "0.60", "Recall Focused")
    m3.metric("Recall Rate", "81%", "High Capture")
    m4.metric("Proj. Revenue Impact", "$1.7M", "Annual Saved")

    st.markdown("---")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📉 ROC & Precision-Recall Curves")
        try: st.image("plots/13_roc_pr_curves.png", use_container_width=True)
        except: st.info("Visualization not available.")
            
    with c2:
        st.markdown("#### 🔑 Feature Importance")
        try: st.image("plots/14_feature_importance.png", use_container_width=True)
        except: st.info("Visualization not available.")

    st.markdown("#### 💰 ROI Optimization Analysis")
    try:
        st.image("plots/20_threshold_optimization.png", use_container_width=True)
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
            try: st.image("plots/02_orders_univariate.png", use_container_width=True)
            except: st.info("Data unavailable.")
        with col2:
            st.markdown("##### 📦 Product Affinity")
            try: st.image("plots/04_product_metrics.png", use_container_width=True)
            except: st.info("Data unavailable.")
                
    with tab2:
        st.markdown("##### 🧠 Global Explainability (SHAP)")
        try:
            st.image("plots/16_shap_summary.png", use_container_width=True)
            st.info("Feature Impact Direction: Red = High Value, Blue = Low Value.")
        except: st.info("SHAP summary unavailable.")

# --- PAGE 4: SYSTEM MONITORING (NEW) ---
elif page == "⚡ System Monitoring":
    st.title("⚡ Live System Monitoring")
    st.markdown("Real-time tracking of model predictions and data drift.")

    # Fetch logs from database
    try:
        conn = get_connection()
        logs_df = pd.read_sql("SELECT * FROM predictions ORDER BY timestamp DESC", conn)
        conn.close()
    except Exception as e:
        st.error(f"Connection Error: {e}")
        logs_df = pd.DataFrame()

    if logs_df.empty:
        st.info("Waiting for incoming predictions to generate logs...")
    else:
        # KPI ROW
        st.subheader("📡 Live Statistics")
        k1, k2, k3, k4 = st.columns(4)
        
        total_preds = len(logs_df)
        churn_rate = logs_df['predicted_label'].mean() * 100
        avg_conf = logs_df['predicted_prob'].mean() * 100
        last_active = logs_df['timestamp'].iloc[0]

        k1.metric("Total Predictions", f"{total_preds}", "+1 (Live)")
        k2.metric("Avg Predicted Churn Rate", f"{churn_rate:.1f}%", "Target < 20%")
        k3.metric("Avg Confidence", f"{avg_conf:.1f}%")
        k4.metric("Last Activity", last_active.split('.')[0]) # Clean up seconds

        st.markdown("---")
        
        # VISUALIZATION ROW
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Prediction Distribution")
            fig = px.histogram(
                logs_df, 
                x="predicted_prob", 
                nbins=20, 
                title="Predicted Probability Distribution",
                color_discrete_sequence=['#58a6ff'],
                template="plotly_dark"
            )
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.markdown("#### ⚠️ Drift Monitor: Purchase Velocity")
            # Compare Baseline (Training Data) with Live Data
            # Get average velocity from training data (from global df)
            baseline_mean = df['purchase_velocity'].mean()
            current_mean = logs_df['purchase_velocity'].mean()
            
            fig = px.box(
                logs_df, 
                y="purchase_velocity", 
                title=f"Live Velocity Dist. (Baseline: {baseline_mean:.2f})",
                color_discrete_sequence=['#FF4B4B'],
                template="plotly_dark"
            )
            # Baseline reference line
            fig.add_hline(y=baseline_mean, line_dash="dash", line_color="green", annotation_text="Training Baseline")
            st.plotly_chart(fig, use_container_width=True)

        # RAW LOGS
        with st.expander("📝 View Raw Prediction Logs", expanded=False):
            st.dataframe(logs_df.style.highlight_max(axis=0))

# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.caption("v1.0.2 | Production Build")