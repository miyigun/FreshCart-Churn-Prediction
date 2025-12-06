"""
FreshCart Churn Prediction - Yapılandırma Dosyası
================================================
Tüm proje ayarları, yolları ve iş kuralları bu dosyada tanımlanmıştır.
"""

from pathlib import Path
from typing import Dict, List
import os

# ============================================================================
# PROJE YOLLARI
# ============================================================================

# Kök dizin
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
LOG_DIR = ROOT_DIR / "logs"
NOTEBOOK_DIR = ROOT_DIR / "notebooks"

# Veri alt dizinleri
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Dizinler mevcut değilse oluştur
for directory in [DATA_DIR, MODEL_DIR, LOG_DIR, RAW_DATA_DIR, 
                  PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# VERİ DOSYALARI
# ============================================================================

# Ham veri dosyaları (Instacart)
RAW_DATA_FILES = {
    'orders': RAW_DATA_DIR / 'orders.csv',
    'order_products_prior': RAW_DATA_DIR / 'order_products__prior.csv',
    'order_products_train': RAW_DATA_DIR / 'order_products__train.csv',
    'products': RAW_DATA_DIR / 'products.csv',
    'aisles': RAW_DATA_DIR / 'aisles.csv',
    'departments': RAW_DATA_DIR / 'departments.csv'
}

# İşlenmiş veri dosyaları
PROCESSED_DATA_FILES = {
    'train_features': PROCESSED_DATA_DIR / 'train_features.parquet',
    'test_features': PROCESSED_DATA_DIR / 'test_features.parquet',
    'customer_features': PROCESSED_DATA_DIR / 'customer_features.parquet',
    'train_labels': PROCESSED_DATA_DIR / 'train_labels.parquet',
    'test_labels': PROCESSED_DATA_DIR / 'test_labels.parquet'
}

# ============================================================================
# İŞ KURALLARI VE TANIMLARI
# ============================================================================

# Müşteri Kaybı (Churn) Tanımı
CHURN_DEFINITION = {
    'days_threshold': 30,  # 30 günden fazla süredir sipariş vermeyen müşteriler kayıp olarak kabul edilir
    'min_orders': 3,       # Geçmişinde en az 3 siparişi olmalı
    'observation_window': 90,  # Son 90 günlük veri
    'prediction_horizon': 14   # Sonraki 14 gün için tahmin yap
}

# İş Metrikleri
BUSINESS_METRICS = {
    'avg_customer_value': 150,     # Ortalama müşteri değeri ($)
    'avg_order_value': 50,         # Ortalama sipariş değeri ($)
    'retention_cost': 10,          # Müşteriyi elde tutma kampanya maliyeti ($)
    'acquisition_cost': 45,        # Yeni müşteri kazanım maliyeti ($)
    'target_churn_rate': 0.18      # Hedef müşteri kaybı oranı
}

# Özellik Grupları
FEATURE_GROUPS = {
    'rfm_features': [
        'recency', 'frequency', 'monetary',
        'days_since_first_order', 'days_since_last_order',
        'total_orders', 'avg_days_between_orders'
    ],
    'behavioral_features': [
        'avg_basket_size', 'avg_order_hour', 'avg_order_dow',
        'weekend_order_ratio', 'night_order_ratio',
        'order_frequency_last_7d', 'order_frequency_last_30d'
    ],
    'product_features': [
        'unique_products', 'unique_aisles', 'unique_departments',
        'reorder_rate', 'avg_items_per_order',
        'product_diversity_score', 'favorite_aisle', 'favorite_department'
    ],
    'time_features': [
        'order_count_trend', 'basket_value_trend',
        'order_regularity_score', 'seasonality_score'
    ]
}

# ============================================================================
# MODEL YAPILANDIRMASI
# ============================================================================

# Tekrarlanabilirlik için rastgelelik tohumu (seed)
RANDOM_STATE = 42

# Eğitim-test ayrımı
TRAIN_TEST_SPLIT = {
    'test_size': 0.2,
    'validation_size': 0.1,
    'stratify': True,
    'method': 'time_based'  # zaman_tabanlı veya rastgele
}

# Model Parametreleri - Baseline
BASELINE_PARAMS = {
    'logistic_regression': {
        'max_iter': 1000,
        'random_state': RANDOM_STATE,
        'class_weight': 'balanced'
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': RANDOM_STATE,
        'class_weight': 'balanced',
        'n_jobs': -1
    }
}

# Model Parametreleri - Gelişmiş
ADVANCED_MODEL_PARAMS = {
    'lightgbm': {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': RANDOM_STATE
    },
    'xgboost': {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_STATE
    },
    'catboost': {
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'learning_rate': 0.1,
        'depth': 6,
        'random_state': RANDOM_STATE,
        'verbose': False
    }
}

# Hiperparametre Optimizasyonu (Optuna)
OPTUNA_CONFIG = {
    'n_trials': 100,
    'timeout': 3600,  # 1 saat
    'n_jobs': -1,
    'show_progress_bar': True
}

# Özellik Seçimi
FEATURE_SELECTION = {
    'method': 'shap',  # shap, önem (importance), özyinelemeli (recursive)
    'n_features': 50,
    'threshold': 0.01
}

# ============================================================================
# DEĞERLENDİRME METRİKLERİ
# ============================================================================

EVALUATION_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1',
    'roc_auc',
    'average_precision'
]

# Sınıflandırma için eşik değeri
CLASSIFICATION_THRESHOLD = 0.5

# İş metriği eşikleri
PERFORMANCE_THRESHOLDS = {
    'min_precision': 0.80,  # Minimum hassasiyet (precision)
    'min_recall': 0.75,     # Minimum duyarlılık (recall)
    'min_f1': 0.77,         # Minimum F1 skoru
    'min_auc': 0.85         # Minimum AUC
}

# ============================================================================
# ÖN İŞLEME
# ============================================================================

# Eksik değer yönetimi
MISSING_VALUE_STRATEGY = {
    'numeric': 'median',  # ortalama (mean), medyan (median), mod (mode)
    'categorical': 'mode'
}

# Aykırı değer tespiti
OUTLIER_CONFIG = {
    'method': 'iqr',  # iqr, zscore, isolation_forest
    'threshold': 3.0
}

# Ölçeklendirme
SCALING_CONFIG = {
    'method': 'standard',  # standard, minmax, robust
    'columns': 'numeric'   # sayısal (numeric), tümü (all), özel (custom)
}

# Kodlama (Encoding)
ENCODING_CONFIG = {
    'categorical_method': 'target',  # onehot, label, target, binary
    'high_cardinality_threshold': 10
}

# ============================================================================
# API VE DAĞITIM
# ============================================================================

# API Yapılandırması
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'reload': True,
    'log_level': 'info'
}

# Model sunma
MODEL_SERVING = {
    'model_path': MODEL_DIR / 'final_model.pkl',
    'preprocessor_path': MODEL_DIR / 'preprocessor.pkl',
    'batch_size': 1000,
    'timeout': 30
}

# ============================================================================
# İZLEME VE GÜNLÜK KAYDI
# ============================================================================

# Günlük kaydı (logging) yapılandırması
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': LOG_DIR / 'freshcart_churn.log'
}

# İzleme metrikleri
MONITORING_METRICS = {
    'model_performance': ['precision', 'recall', 'f1', 'auc'],
    'business_metrics': ['churn_rate', 'retention_rate', 'campaign_roi'],
    'system_metrics': ['response_time', 'error_rate', 'throughput']
}

# Veri kayması (data drift) tespiti
DATA_DRIFT_CONFIG = {
    'enabled': True,
    'check_interval': 'daily',
    'threshold': 0.05
}

# ============================================================================
# GÖRSELLEŞTİRME
# ============================================================================

VISUALIZATION_CONFIG = {
    'style': 'seaborn',
    'palette': 'Set2',
    'figsize': (12, 6),
    'dpi': 100,
    'save_format': 'png'
}

# Çizim dizinleri
PLOT_DIR = ROOT_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# ============================================================================
# YARDIMCI FONKSİYONLAR
# ============================================================================

def get_config() -> Dict:
    """Tüm yapılandırmayı bir sözlük olarak döndürür"""
    return {
        'paths': {
            'root': ROOT_DIR,
            'data': DATA_DIR,
            'models': MODEL_DIR,
            'logs': LOG_DIR
        },
        'business': BUSINESS_METRICS,
        'churn_definition': CHURN_DEFINITION,
        'model': ADVANCED_MODEL_PARAMS,
        'evaluation': EVALUATION_METRICS
    }

def print_config():
    """Yapılandırmanın bir özetini yazdırır"""
    print("=" * 80)
    print("FreshCart Müşteri Kaybı Tahmini - Yapılandırma Özeti")
    print("=" * 80)
    print(f"\n Proje Kök Dizini: {ROOT_DIR}")
    print(f" Veri Dizini: {DATA_DIR}")
    print(f" Modeller Dizini: {MODEL_DIR}")
    print(f"\n Müşteri Kaybı Tanımı: {CHURN_DEFINITION['days_threshold']} gün")
    print(f"Ort. Müşteri Değeri: ${BUSINESS_METRICS['avg_customer_value']}")
    print(f"Rastgelelik Durumu: {RANDOM_STATE}")
    print(f"\n Yapılandırma başarıyla yüklendi!")
    print("=" * 80)


if __name__ == "__main__":
    print_config()