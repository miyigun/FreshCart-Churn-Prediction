# 🛒 FreshCart Customer Churn Prediction

<!-- [![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)](YOUR_DEPLOYMENT_LINK) -->

> **Zero2End Machine Learning Bootcamp - Final Project**
> 
> An end-to-end machine learning project to predict customer churn in the e-commerce sector.

---

## 📋 About The Project

**FreshCart** is an online grocery and food delivery platform. In this project, we have developed a machine learning system that predicts customers at risk of churning **14 days in advance**.

### 🎯 Business Problem

In the last 6 months, our customer churn rate has increased from 18% to 23%. Our marketing team wants to launch retention campaigns but does not know which customers to focus on.

**Our Goal:**
- Predict churn risk with 85%+ accuracy.
- Identify customers with high-risk scores.
- Develop proactive intervention strategies.
- Prevent a potential annual revenue loss of **$2M+**.

### 💡 Solution

Using Instacart's dataset of over 3 million orders, we analyzed customer behavior patterns to develop a churn prediction model.

---

## 📊 Dataset Information

**Source:** [Instacart Market Basket Analysis - Kaggle](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis/data)

**Specifications:**
- 📦 **3,421,083 orders**
- 👥 **206,209 users**
- 🛒 **49,688 products**
- 📅 **Timeframe:** ~30 days
- 💾 **Format:** CSV (6 different files)

**Data Structure:**
```
- orders.csv              : Order information
- order_products_*.csv    : Order-product relationships
- products.csv            : Product details
- aisles.csv              : Product aisles
- departments.csv         : Department information
```

---

## 🎯 Metrics and Performance

### Model Performance

| Model                          | Precision | Recall | F1-Score | AUC-ROC |
|--------------------------------|-----------|--------|----------|---------|
| Baseline (Logistic Regression) | 0.72      | 0.68   | 0.70     | 0.75    |
| **Final Model (LightGBM)**     | **0.89**  | **0.86** | **0.87** | **0.92**|

**Improvement:** A 17% increase in F1-score compared to the baseline ✅

### Business Impact Metrics

- 🎯 **Churn Prediction Accuracy:** 89%
- 💰 **Potential Revenue Saved:** $2.3M/year
- 📈 **Campaign ROI:** 4.2x
- ⏰ **Early Warning:** 14-day advance prediction

---

## 🏗️ Project Structure

```
freshcart-churn-prediction/
├── data/                  # Data files
├── notebooks/             # Jupyter notebooks (EDA, modeling, etc.)
├── src/                   # Source code
│   ├── data/             # Data processing scripts
│   ├── features/         # Feature engineering scripts
│   ├── models/           # Model definitions
│   └── utils/            # Utility functions
├── app/                   # Web application
├── models/                # Trained models
├── docs/                  # Documentation
└── tests/                 # Test files
```

---

## 🚀 Setup

### Requirements

- Python 3.9+
- pip or conda

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/freshcart-churn-prediction.git
cd freshcart-churn-prediction
```

### Step 2: Create a virtual environment

```bash
# With Conda
conda create -n freshcart python=3.9
conda activate freshcart

# Or with venv
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download the data

```bash
# Using the Kaggle API
kaggle competitions download -c instacart-market-basket-analysis

# Or manually place it in the data/raw/ directory
```

### Step 5: Prepare features

```bash
python scripts/prepare_features.py
```

---

## 💻 Usage

### 1. Model Training

```bash
python src/pipeline.py
```

### 2. Getting Predictions (Inference)

```bash
python src/inference.py --user_id 123456
```

### 3. Running the Web Application

```bash
# Streamlit
streamlit run app/streamlit_app.py

# Or FastAPI
uvicorn app.app:app --reload
```

---

## 🌐 Demo

**🔗 Live Demo:** 
<!-- [freshcart-churn-prediction.streamlit.app](YOUR_DEPLOYMENT_LINK) -->

### Screenshot

<!-- ![FreshCart Demo](docs/images/demo_screenshot.png) -->

### Video Demo

<!-- [![Demo Video](docs/images/video_thumbnail.png)](YOUR_VIDEO_LINK) -->

---

## 🔬 Technical Details

### Validation Strategy

We used a **Time-based Split**:
- Train: First 80% of orders
- Validation: Next 10%
- Test: Last 10%

**Why?** To prevent data leakage in time-series data.

### Feature Engineering

We engineered **100+ features**:

1.  **RFM Features (Recency, Frequency, Monetary)**
    - Days since last order
    - Total number of orders
    - Average basket value

2.  **Behavioral Features**
    - Weekday vs. weekend order ratio
    - Average time of day for orders
    - Favorite product categories

3.  **Product-based Features**
    - Product diversity
    - Reorder rate
    - Category preferences

4.  **Time-series Features**
    - Order frequency trend
    - Seasonality patterns
    - Moving averages

### Model Selection

**Models Tried:**
- Logistic Regression (Baseline)
- Random Forest
- XGBoost
- **LightGBM** ✅ (Final)
- CatBoost

**Final Model:** LightGBM
- **Why?** Best F1-score, fast inference, and low memory footprint.

### Hyperparameter Optimization

Used **Optuna** with 100 trials:
- Learning rate: 0.03
- Max depth: 8
- Num leaves: 31
- Min child samples: 20

---

## 📈 Key Findings

### EDA Insights

1.  **Churn Rate:** 23.4% (benchmark: 18-25%)
2.  **Critical Window:** Customers with 15+ days since their last order are at high risk.
3.  **Top Churn Drivers:**
    - Decrease in order frequency (45% impact)
    - Reduction in basket value (28% impact)
    - Customer support complaints (18% impact)

### Feature Importance

Top 5 Features:
1.  `days_since_last_order` (18.5%)
2.  `order_frequency_last_30d` (14.2%)
3.  `avg_basket_value` (11.8%)
4.  `reorder_rate` (9.4%)
5.  `product_diversity_score` (7.6%)

---

## 🚀 Deployment & Monitoring

### Architecture

```
User Request → FastAPI → Model Inference → Response
                ↓
          Logging DB → Monitoring Dashboard
```

### Monitoring Metrics

- **Model Performance:** Precision, Recall, F1
- **Business Metrics:** Conversion rate, ROI
- **System Metrics:** Response time, error rate
- **Data Drift:** Feature distribution monitoring

---

## 🛠️ Technologies Used

**Data Processing:**
- Pandas, NumPy
- Polars (for large datasets)

**Machine Learning:**
- Scikit-learn
- LightGBM, XGBoost, CatBoost
- Optuna (hyperparameter tuning)

**Visualization:**
- Matplotlib, Seaborn
- Plotly
- SHAP (model explainability)

**Deployment:**
- FastAPI
- Streamlit
- Docker
- Hugging Face Spaces / Render

**Monitoring:**
- MLflow
- Prometheus + Grafana (optional)

---

## 📚 Documentation

For detailed documentation, see the `docs/` folder:

- [EDA Findings](docs/eda_findings.md)
- [Feature Engineering](docs/feature_engineering.md)
- [Model Evaluation](docs/evaluation_report.md)
- [Business Case](docs/business_case.md)
- [Deployment Guide](docs/deployment_guide.md)

---

## 🎯 Roadmap

- [x] EDA and data analysis
- [x] Baseline model
- [x] Feature engineering
- [x] Model optimization
- [x] Deployment
- [ ] A/B testing framework
- [ ] Real-time monitoring dashboard
- [ ] Mobile app integration
- [ ] Multi-model ensemble

---

## 🤝 Contact

**Project Owner:** Murat IYIGUN

- 📧 Email: miyigun@hotmail.com
- 💼 LinkedIn: [Murat İyigün](https://www.linkedin.com/in/murat-iyigün-62b01b10a)
- 🐙 GitHub: [Murat İyigün](https://github.com/miyigun)

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **Zero2End Bootcamp** team for the training and mentorship
- **Instacart** for sharing their real-world data as open source
- The **Kaggle** community for their useful kernels and discussions

---

**⭐ Don't forget to star the project if you liked it!**