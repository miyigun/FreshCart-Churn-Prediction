import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import sqlite3
import plotly.express as px  # İnteraktif grafikler için
import sys
import os

# --- YOL YAPILANDIRMASI ---
# Mevcut dizinin mutlak yolunu al (app.py'nin olduğu yer)
current_dir = os.path.dirname(os.path.abspath(__file__))
# İçe aktarmaların doğru çalışmasını sağlamak için sys.path'e ekle (eğer zaten ekli değilse)
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.config import PROCESSED_DATA_DIR, MODEL_DIR

# --- İZLEME MODÜLÜ İÇE AKTARMA ---
# DB dosyası 'src/monitoring/db.py' yolunda olduğu için
# Python'un src paketinden import ediyoruz.
try:
    from src.monitoring.db import init_db, log_prediction, get_connection
except ImportError as e:
    st.error(f"İzleme modülü yüklenirken hata oluştu: {e}")
    st.stop()

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(
    page_title="FreshCart Customer Churn Prediction",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DB'Yİ BAŞLAT ---
# Uygulama başladığında izleme veritabanını başlat
init_db()

# --- YÜKSEK KONTRASTLI KOYU TEMA CSS ---
st.markdown("""
<style>
    /* Genel Ayarlar (Ana Uygulama) */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* --- KENAR ÇUBUĞU DÜZELTMESİ --- */
    [data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d;
    }
    
    /* Kenar Çubuğundaki tüm metinleri beyaz yap */
    [data-testid="stSidebar"] * {
        color: #e6edf3 !important;
    }

    /* --- RADYO BUTONU VE ONAY KUTUSU METİNLERİ --- */
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

    /* --- DİĞER ELEMANLAR --- */
    .stSelectbox label {
        color: #ffffff !important;
        font-weight: bold;
    }
    .stSelectbox > div > div {
        background-color: #21262d !important;
        color: #ffffff !important;
        border: 1px solid #58a6ff;
    }
    
    /* Özel Bilgi Kutusu */
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

    /* Metrik Kartları */
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

    /* Başlık Düzeltmesi */
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

# --- YARDIMCI FONKSİYONLAR ---
@st.cache_resource
def load_artifacts():
    """Eğitilmiş modeli ve gerekli meta verileri yükler."""
    # 1. Modeli Yükle
    try:
        model = joblib.load(MODEL_DIR / 'final_model_optimized.pkl')
    except FileNotFoundError:
        st.error("Model dosyası (final_model_optimized.pkl) models dizininde bulunamadı.")
        st.stop()
    
    # 2. Özellik Adlarını Yükle
    # Uygulama önce models/feature_names.json'a, sonra processed/model_features.json'a bakar
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
            st.error("Özellik listesi JSON dosyası models/ veya data/processed/ dizininde bulunamadı.")
            st.stop()
            
    except Exception as e:
        st.error(f"Özellik adları yüklenirken hata: {e}")
        st.stop()
        
    # 3. Veriyi Yükle
    try:
        data_path = PROCESSED_DATA_DIR / 'final_features_advanced.parquet'
        data = pd.read_parquet(data_path)
        
        # Geç KeyErrors'ı önlemek için Sütunları hemen doğrula
        missing_cols = [col for col in feature_names if col not in data.columns]
        if missing_cols:
            st.warning(f"Veri Uyuşmazlığı tespit edildi! '{feature_file_used}' içindeki özellik listesi, parke dosyasında bulunmayan sütunlar bekliyor: {missing_cols}")
            # Güvenli mod: Sadece gerçekten var olan sütunları tut
            feature_names = [col for col in feature_names if col in data.columns]
        
        cols_to_keep = ['user_id', 'is_churn'] + feature_names
        # user_id ve is_churn'ün de var olduğundan emin ol
        cols_to_keep = [c for c in cols_to_keep if c in data.columns]
        
        data = data[cols_to_keep]
        
    except FileNotFoundError:
        st.warning("Parquet verisi bulunamadı. Uygulama sadece Model Modunda çalışacak (geçmiş veri yok).")
        data = pd.DataFrame()
    except Exception as e:
        st.error(f"Veri yüklenirken hata: {e}")
        data = pd.DataFrame()

    return model, feature_names, data

# --- VERİYİ YÜKLE ---
try:
    model, feature_names, df = load_artifacts()
except Exception as e:
    st.error(f"Sistem Hatası: {e}")
    st.stop()

# --- KENAR ÇUBUĞU ---
st.sidebar.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=80)
st.sidebar.title("FreshCart AI")

# NAVİGASYONA YENİ SEÇENEK EKLENDİ
page = st.sidebar.radio("NAVİGASYON", [
    "🏠 Tahmin Merkezi", 
    "📊 Model Analizi", 
    "📈 Derinlemesine Analiz",
    "⚡ Sistem İzleme"
])

st.sidebar.markdown("---")

# --- KENAR ÇUBUĞU ALT BİLGİSİ ---
st.sidebar.markdown("""
### Geliştiren
<div style="margin-top: -10px;">
    <h4 style="margin-bottom: 0px; color: #ffffff;">Murat IYIGUN</h4>
    <p style="margin-top: 0px; font-size: 0.9rem; color: #8b949e; font-style: italic;">
        Veri Bilimci & Yapay Zeka Mühendisi
    </p>
</div>
""", unsafe_allow_html=True)

# --- SAYFA 1: TAHMİN MERKEZİ ---
if page == "🏠 Tahmin Merkezi":
    # BAŞLIK
    st.title("🛒 Müşteri Kaybı Tahmin Sistemi")
    
    st.markdown("""
    <div class="info-box">
        <h4>🎯 Proje Misyonu</h4>
        <p>
            Bu sistem, müşteri kaybı riskini <strong>14 gün önceden</strong> tahmin etmek için 
            <strong>İleri Düzey Makine Öğrenmesi (LightGBM)</strong> kullanır. Gerçek zamanlı tahminler, veri kayması (drift) takibi için kaydedilir.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if df.empty:
        st.warning("⚠️ Veri akışı çevrimdışı.")
    else:
        # SEÇİM ALANI
        st.subheader("👤 Müşteri Profili Seçimi")
        
        sel_col1, sel_col2, _ = st.columns([1, 1.5, 2])
        
        with sel_col1:
            input_method = st.radio("Kaynak:", ["ID Listesi", "Rastgele Örnek"], horizontal=True)
        
        with sel_col2:
            if input_method == "ID Listesi":
                selected_user_id = st.selectbox("Müşteri ID'si Ara:", df['user_id'].head(100).tolist())
            else:
                if st.button("🎲 Rastgele Profil Oluştur", type="primary"):
                    selected_user_id = df['user_id'].sample(1).values[0]
                else:
                    selected_user_id = df['user_id'].iloc[0]

        # TAHMİN
        customer_data = df[df['user_id'] == selected_user_id].iloc[0]
        input_features = customer_data[feature_names].to_frame().T
        churn_prob = model.predict(input_features)[0]
        THRESHOLD = 0.38 
        is_churn = churn_prob >= THRESHOLD

        # --- GÜNLÜK KAYDI (LOGGING) ---
        # Tahmin yapılır yapılmaz veritabanına kaydet
        log_prediction(
            user_id=int(selected_user_id),
            features=customer_data,
            prob=float(churn_prob),
            label=int(is_churn),
            model_version='v1.0.2'
        )
        # ---------------

        st.markdown("---")
        
        # SONUÇLAR PANOSU
        r1, r2, r3 = st.columns([1.2, 1.5, 2.3])
        
        # 1. RİSK DURUMU
        with r1:
            st.markdown("### Risk Durumu")
            if is_churn:
                st.metric("Tahmin", "YÜKSEK RİSK", f"{churn_prob*100:.1f}% Olasılık", delta_color="inverse")
            else:
                st.metric("Tahmin", "SADIK", f"{churn_prob*100:.1f}% Olasılık", delta_color="normal")
        
        # 2. DAVRANIŞSAL DNA
        with r2:
            st.markdown("### Davranışsal DNA")
            st.info(f"""
            - **Yenilik:** {customer_data.get('days_since_last_order', 0):.0f} gün önce
            - **Sıklık:** {customer_data.get('total_orders', 0):.0f} toplam sipariş
            - **Sepet Büyüklüğü:** {customer_data.get('avg_basket_size', 0):.1f} ürün
            - **Hız:** {customer_data.get('purchase_velocity', 0):.2f} skor
            """)

        # 3. SHAP AÇIKLAMASI
        with r3:
            st.markdown("### Yapay Zeka Gerekçesi (SHAP)")
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
                st.warning("Açıklama mevcut değil.")

# --- SAYFA 2: MODEL ANALİZİ ---
elif page == "📊 Model Analizi":
    st.title("📊 Sistem Performans Metrikleri")
    st.markdown("Test verisi (ayrılmış %20'lik set) üzerindeki değerlendirme sonuçları.")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC Skoru", "0.77", "İyi Stabilite")
    m2.metric("F1-Skoru", "0.60", "Duyarlılık Odaklı")
    m3.metric("Duyarlılık Oranı", "81%", "Yüksek Yakalama")
    m4.metric("Tahmini Gelir Etkisi", "1.7M $", "Yıllık Tasarruf")

    st.markdown("---")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📉 ROC ve Precision-Recall Eğrileri")
        try: st.image("plots/13_roc_pr_curves.png", use_container_width=True)
        except: st.info("Görselleştirme mevcut değil.")
            
    with c2:
        st.markdown("#### 🔑 Özellik Önemi")
        try: st.image("plots/14_feature_importance.png", use_container_width=True)
        except: st.info("Görselleştirme mevcut değil.")

    st.markdown("#### 💰 ROI Optimizasyon Analizi")
    try:
        st.image("plots/20_threshold_optimization.png", use_container_width=True)
    except:
        st.info("ROI Grafiği mevcut değil.")

# --- SAYFA 3: VERİ ANALİZİ ---
elif page == "📈 Derinlemesine Analiz":
    st.title("📈 Keşifsel Zeka")
    st.markdown("Müşteri davranışlarındaki gizli kalıpları keşfetme.")
    
    tab1, tab2 = st.tabs(["🌍 Pazar Genel Bakışı", "🤖 Yapay Zeka Etkenleri"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### ⏰ Sipariş Zamanlama Alışkanlıkları")
            try: st.image("plots/02_orders_univariate.png", use_container_width=True)
            except: st.info("Veri mevcut değil.")
        with col2:
            st.markdown("##### 📦 Ürün Yakınlığı")
            try: st.image("plots/04_product_metrics.png", use_container_width=True)
            except: st.info("Veri mevcut değil.")
                
    with tab2:
        st.markdown("##### 🧠 Genel Açıklanabilirlik (SHAP)")
        try:
            st.image("plots/16_shap_summary.png", use_container_width=True)
            st.info("Özellik Etki Yönü: Kırmızı = Yüksek Değer, Mavi = Düşük Değer.")
        except: st.info("SHAP özeti mevcut değil.")

# --- SAYFA 4: SİSTEM İZLEME (YENİ) ---
elif page == "⚡ Sistem İzleme":
    st.title("⚡ Canlı Sistem İzleme")
    st.markdown("Model tahminlerinin ve veri kaymasının gerçek zamanlı takibi.")

    # Veritabanından günlük kayıtlarını al
    try:
        conn = get_connection()
        logs_df = pd.read_sql("SELECT * FROM predictions ORDER BY timestamp DESC", conn)
        conn.close()
    except Exception as e:
        st.error(f"Bağlantı Hatası: {e}")
        logs_df = pd.DataFrame()

    if logs_df.empty:
        st.info("Günlük oluşturmak için gelen tahminler bekleniyor...")
    else:
        # KPI SATIRI
        st.subheader("📡 Canlı İstatistikler")
        k1, k2, k3, k4 = st.columns(4)
        
        total_preds = len(logs_df)
        churn_rate = logs_df['predicted_label'].mean() * 100
        avg_conf = logs_df['predicted_prob'].mean() * 100
        last_active = logs_df['timestamp'].iloc[0]

        k1.metric("Toplam Tahmin", f"{total_preds}", "+1 (Canlı)")
        k2.metric("Ort. Tahmini Kayıp Oranı", f"{churn_rate:.1f}%", "Hedef < 20%")
        k3.metric("Ort. Güven", f"{avg_conf:.1f}%")
        k4.metric("Son Aktivite", last_active.split('.')[0]) # Saniyeleri temizle

        st.markdown("---")
        
        # GÖRSELLEŞTİRME SATIRI
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Tahmin Dağılımı")
            fig = px.histogram(
                logs_df, 
                x="predicted_prob", 
                nbins=20, 
                title="Tahmin Edilen Olasılık Dağılımı",
                color_discrete_sequence=['#58a6ff'],
                template="plotly_dark"
            )
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.markdown("#### Kayma Monitörü: Satın Alma Hızı")
            # Temel (Eğitim Verisi) ile Canlı Veriyi Karşılaştır
            # Eğitim verisinden ortalama hızı al (genel df'den)
            baseline_mean = df['purchase_velocity'].mean()
            current_mean = logs_df['purchase_velocity'].mean()
            
            fig = px.box(
                logs_df, 
                y="purchase_velocity", 
                title=f"Canlı Hız Dağ. (Temel: {baseline_mean:.2f})",
                color_discrete_sequence=['#FF4B4B'],
                template="plotly_dark"
            )
            # Temel referans çizgisi
            fig.add_hline(y=baseline_mean, line_dash="dash", line_color="green", annotation_text="Eğitim Temeli")
            st.plotly_chart(fig, use_container_width=True)

        # HAM GÜNLÜK KAYITLARI
        with st.expander("Ham Tahmin Günlüklerini Görüntüle", expanded=False):
            st.dataframe(logs_df.style.highlight_max(axis=0))

# --- ALT BİLGİ ---
st.sidebar.markdown("---")
st.sidebar.caption("v1.0.3 | Üretim Sürümü")