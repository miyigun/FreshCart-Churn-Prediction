"""
FreshCart Churn Prediction - Configuration File
================================================
All project settings, paths, and business rules are defined in this file.
"""

from pathlib import Path
from typing import Dict, List
import os

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Root directory
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
LOG_DIR = ROOT_DIR / "logs"
NOTEBOOK_DIR = ROOT_DIR / "notebooks"

# Data subdirectories
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, LOG_DIR, RAW_DATA_DIR, 
                  PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA FILES
# ============================================================================

# Raw data files (Instacart)
RAW_DATA_FILES = {
    'orders': RAW_DATA_DIR / 'orders.csv',
    'order_products_prior': RAW_DATA_DIR / 'order_products__prior.csv',
    'order_products_train': RAW_DATA_DIR / 'order_products__train.csv',
    'products': RAW_DATA_DIR / 'products.csv',
    'aisles': RAW_DATA_DIR / 'aisles.csv',
    'departments': RAW_DATA_DIR / 'departments.csv'
}

# Processed data files
PROCESSED_DATA_FILES = {
    'train_features': PROCESSED_DATA_DIR / 'train_features.parquet',
    'test_features': PROCESSED_DATA_DIR / 'test_features.parquet',
    'customer_features': PROCESSED_DATA_DIR / 'customer_features.parquet',
    'train_labels': PROCESSED_DATA_DIR / 'train_labels.parquet',
    'test_labels': PROCESSED_DATA_DIR / 'test_labels.parquet'
}

# ============================================================================
# BUSINESS RULES & DEFINITIONS
# ============================================================================

# Churn Definition
CHURN_DEFINITION = {
    'days_threshold': 30,  # Customers who haven't ordered in more than 30 days are considered churned
    'min_orders': 3,       # Must have at least 3 orders in their history
    'observation_window': 90,  # Data from the last 90 days
    'prediction_horizon': 14   # Predict for the next 14 days
}

# Business Metrics
BUSINESS_METRICS = {
    'avg_customer_value': 150,     # Average customer value ($)
    'avg_order_value': 50,         # Average order value ($)
    'retention_cost': 10,          # Retention campaign cost ($)
    'acquisition_cost': 45,        # New customer acquisition cost ($)
    'target_churn_rate': 0.18      # Target churn rate
}

# Feature Groups
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
# MODEL CONFIGURATION
# ============================================================================

# Random seed for reproducibility
RANDOM_STATE = 42

# Train-test split
TRAIN_TEST_SPLIT = {
    'test_size': 0.2,
    'validation_size': 0.1,
    'stratify': True,
    'method': 'time_based'  # time_based or random
}

# Model Parameters - Baseline
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

# Model Parameters - Advanced
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

# Hyperparameter Optimization (Optuna)
OPTUNA_CONFIG = {
    'n_trials': 100,
    'timeout': 3600,  # 1 hour
    'n_jobs': -1,
    'show_progress_bar': True
}

# Feature Selection
FEATURE_SELECTION = {
    'method': 'shap',  # shap, importance, recursive
    'n_features': 50,
    'threshold': 0.01
}

# ============================================================================
# EVALUATION METRICS
# ============================================================================

EVALUATION_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1',
    'roc_auc',
    'average_precision'
]

# Threshold for classification
CLASSIFICATION_THRESHOLD = 0.5

# Business metric thresholds
PERFORMANCE_THRESHOLDS = {
    'min_precision': 0.80,  # Minimum precision
    'min_recall': 0.75,     # Minimum recall
    'min_f1': 0.77,         # Minimum F1
    'min_auc': 0.85         # Minimum AUC
}

# ============================================================================
# PREPROCESSING
# ============================================================================

# Missing value handling
MISSING_VALUE_STRATEGY = {
    'numeric': 'median',  # mean, median, mode
    'categorical': 'mode'
}

# Outlier detection
OUTLIER_CONFIG = {
    'method': 'iqr',  # iqr, zscore, isolation_forest
    'threshold': 3.0
}

# Scaling
SCALING_CONFIG = {
    'method': 'standard',  # standard, minmax, robust
    'columns': 'numeric'   # numeric, all, custom
}

# Encoding
ENCODING_CONFIG = {
    'categorical_method': 'target',  # onehot, label, target, binary
    'high_cardinality_threshold': 10
}

# ============================================================================
# API & DEPLOYMENT
# ============================================================================

# API Configuration
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'reload': True,
    'log_level': 'info'
}

# Model serving
MODEL_SERVING = {
    'model_path': MODEL_DIR / 'final_model.pkl',
    'preprocessor_path': MODEL_DIR / 'preprocessor.pkl',
    'batch_size': 1000,
    'timeout': 30
}

# ============================================================================
# MONITORING & LOGGING
# ============================================================================

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': LOG_DIR / 'freshcart_churn.log'
}

# Monitoring metrics
MONITORING_METRICS = {
    'model_performance': ['precision', 'recall', 'f1', 'auc'],
    'business_metrics': ['churn_rate', 'retention_rate', 'campaign_roi'],
    'system_metrics': ['response_time', 'error_rate', 'throughput']
}

# Data drift detection
DATA_DRIFT_CONFIG = {
    'enabled': True,
    'check_interval': 'daily',
    'threshold': 0.05
}

# ============================================================================
# VISUALIZATION
# ============================================================================

VISUALIZATION_CONFIG = {
    'style': 'seaborn',
    'palette': 'Set2',
    'figsize': (12, 6),
    'dpi': 100,
    'save_format': 'png'
}

# Plot directories
PLOT_DIR = ROOT_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_config() -> Dict:
    """Return all configuration as a dictionary"""
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
    """Print a summary of the configuration"""
    print("=" * 80)
    print("FreshCart Churn Prediction - Configuration Summary")
    print("=" * 80)
    print(f"\n📁 Project Root: {ROOT_DIR}")
    print(f"📊 Data Directory: {DATA_DIR}")
    print(f"🤖 Models Directory: {MODEL_DIR}")
    print(f"\n🎯 Churn Definition: {CHURN_DEFINITION['days_threshold']} days")
    print(f"💰 Avg Customer Value: ${BUSINESS_METRICS['avg_customer_value']}")
    print(f"🎲 Random State: {RANDOM_STATE}")
    print(f"\n✅ Configuration loaded successfully!")
    print("=" * 80)


if __name__ == "__main__":
    print_config()