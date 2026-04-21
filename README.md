# ❤️ Heart Disease Risk Prediction System

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5-orange)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)](https://streamlit.io)
[![Pandas](https://img.shields.io/badge/Pandas-2.0-green)](https://pandas.pydata.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## 📌 Overview

This project predicts a patient's **10-year risk of developing coronary heart disease** using machine learning. Built with **Random Forest** and deployed as an interactive **Streamlit web app**.

**Live Demo:** [Click here to try the app](https://your-app-url.streamlit.app)

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Source** | Framingham Heart Study |
| **Samples** | 4,240 patient records |
| **Features** | 15 health indicators |
| **Target** | TenYearCHD (0 = No Disease, 1 = Disease) |
| **Class Distribution** | 84.8% No Disease, 15.2% Disease (Imbalanced) |

**Features include:** age, blood pressure, cholesterol, glucose, smoking status, BMI, diabetes, hypertension, and more.

---

## 🔧 Tech Stack

| Category | Technologies |
|----------|--------------|
| **Language** | Python 3.10 |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn |
| **Balancing** | SMOTE (Imbalanced-learn) |
| **Deployment** | Streamlit |

---

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | ROC-AUC |
|-------|----------|-----------|--------|---------|
| Logistic Regression | 66.02% | 64.22% | 66.71% | 71.11% |
| Decision Tree | 73.11% | 71.47% | 73.63% | 73.18% |
| **Random Forest** | **78.46%** | **77.12%** | **78.67%** | **87.69%** |
| Gradient Boosting | 74.50% | 72.55% | 75.79% | 83.72% |
| KNN | 71.09% | 66.91% | 79.25% | 78.16% |

### 🏆 Best Model: Random Forest

- **Recall:** 78.67% (Catches ~79 out of 100 sick patients)
- **ROC-AUC:** 87.69% (Excellent at distinguishing healthy vs sick)
- **Why Random Forest?** Highest ROC-AUC and balanced performance

---

## 🔄 Complete Project Pipeline
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Load Data │
│ - 4,240 rows, 16 columns │
└─────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Data Cleaning │
│ - Handle null values (mean for numbers, mode for text) │
│ - Remove duplicates │
└─────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Exploratory Data Analysis (EDA) │
│ - Correlation heatmap │
│ - Distribution plots │
└─────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Outlier Detection │
│ - IQR (Interquartile Range) method │
│ - Identify extreme values │
└─────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────┐
│ Step 5: Class Imbalance Check │
│ - Before: 84% No Disease, 16% Disease │
│ - Problem: Model will be biased toward "No Disease" │
└─────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────┐
│ Step 6: SMOTE Balancing │
│ - Created synthetic disease cases │
│ - After: 50% No Disease, 50% Disease (3,596 each) │
└─────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────┐
│ Step 7: Feature Selection │
│ - Selected top 6 features by correlation │
│ - age, sysBP, prevalentHyp, diaBP, glucose, diabetes │
└─────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────┐
│ Step 8: Train-Test Split │
│ - 80% Training, 20% Testing │
└─────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────┐
│ Step 9: Feature Scaling │
│ - StandardScaler (fit on train, transform both) │
└─────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────┐
│ Step 10: Model Training │
│ - 5 models compared: LR, DT, RF, GB, KNN │
│ - Random Forest selected as best │
└─────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────┐
│ Step 11: Model Saving │
│ - Saved as heart_disease_model.pkl │
│ - Saved scaler.pkl for preprocessing │
└─────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────┐
│ Step 12: Streamlit Deployment │
│ - Interactive web app for real-time predictions │
└─────────────────────────────────────────────────────────


---

## 🎯 Top 6 Features Used

| Rank | Feature | Correlation with Target | Description |
|------|---------|------------------------|-------------|
| 1 | **age** | 0.225 | Patient's age in years |
| 2 | **sysBP** | 0.216 | Systolic blood pressure |
| 3 | **prevalentHyp** | 0.177 | Prevalent hypertension (0/1) |
| 4 | **diaBP** | 0.145 | Diastolic blood pressure |
| 5 | **glucose** | 0.120 | Blood glucose level |
| 6 | **diabetes** | 0.097 | Diabetes status (0/1) |

---
deployed app link:https://heartdiseaseprediction-phqsrwfzttfghakvjy3e3q.streamlit.app/
## 🚀 Local Setup

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Installation Steps

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/heart-disease-prediction.git
cd heart-disease-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run heart_disease.py
