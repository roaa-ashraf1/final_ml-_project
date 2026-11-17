# 📊 End-to-End Telco Customer Churn Prediction (MLOps Pipeline)

![CI/CD Pipeline](https://shields.io)
![Python](https://shields.io)
![FastAPI](https://shields.io)
![Docker](https://shields.io)
![MLflow](https://shields.io)

An end-to-end machine learning and MLOps solution to predict customer churn in the telecommunications industry. This project covers exploratory data analysis, automated preprocessing pipelines, hyperparameter tuning with MLflow tracking, model explainability (SHAP & LIME), REST API deployment with FastAPI, Docker containerization, CI/CD automation via GitHub Actions, and post-deployment drift monitoring.

---

## 🏗️ Project Architecture & Directory Structure

```text
telco-churn-mlops/
├── .github/
│   └── workflows/
│       └── ci-cd.yml                          # GitHub Actions CI/CD (Pytest + Docker Smoke Test)
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw Telco dataset
│   └── cleaned_telco_data.joblib              # Cleaned dataset artifact
├── models/
│   ├── xgboost_pipeline.joblib                # Production XGBoost Pipeline (Winning Model)
│   ├── random_forest_pipeline.joblib          # Tuned Random Forest Pipeline
│   └── logistic_regression_pipeline.joblib    # Tuned Logistic Regression Pipeline
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb              # EDA, missing value handling & feature importance
│   ├── 02_modeling_and_tuning.ipynb           # GridSearchCV, nested MLflow logging & model export
│   ├── 03_interpretability_shap_lime.ipynb    # Global (SHAP) & Local (LIME) explanations
│   ├── 04_api_testing.ipynb                   # API contract, validation & consistency tests
│   └── 05_ml_monitoring.ipynb                 # Data & prediction drift monitoring (PSI & KS Test)
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                       # Reusable Scikit-Learn ColumnTransformer pipeline
│   └── app.py                                 # FastAPI REST service (/predict, /model-info)
├── tests/
│   └── test_api.py                            # Automated Pytest suite for FastAPI endpoints
├── .dockerignore
├── .gitignore
├── Dockerfile                                 # Production container configuration
├── requirements.txt                           # Production runtime dependencies
├── requirements-dev.txt                       # Testing & development dependencies
└── README.md
```

---

## 🚀 Key MLOps Capabilities

1. **Data Preprocessing & Feature Engineering (`src/preprocessing.py`):**
   - Handled missing values in `TotalCharges` using median imputation.
   - Built a leakage-free Scikit-Learn `ColumnTransformer` applying `StandardScaler` to numerical features and `OneHotEncoder` to categorical features.
   - Conducted tree-based feature importance analysis identifying `Contract`, `tenure`, `InternetService`, and `MonthlyCharges` as key churn drivers.

2. **Experiment Tracking & Tuning (`notebooks/02_modeling_and_tuning.ipynb`):**
   - Evaluated **Logistic Regression**, **Random Forest**, and **XGBoost** using 5-fold cross-validation (`GridSearchCV`).
   - Logged all hyperparameter trials as **nested runs in MLflow** across `ROC-AUC`, `Accuracy`, `Precision`, `Recall`, and `F1-Score`.
   - Selected the tuned **XGBoost Pipeline** as the production model and serialized it via `joblib`.

3. **Model Explainability (`notebooks/03_interpretability_shap_lime.ipynb`):**
   - **Global Explanations (SHAP):** Generated summary beeswarm and feature ranking plots showing how month-to-month contracts and fiber optic service increase churn probability.
   - **Local Explanations (LIME):** Provided instance-level visual breakdowns explaining individual high-risk customer predictions.

4. **REST API Serving (`src/app.py`):**
   - Built with **FastAPI** and strict **Pydantic** schema validation (`extra="forbid"`).
   - Exposes `GET /` (health check), `GET /model-info` (model metadata & feature contract), and `POST /predict` (real-time JSON inference).

5. **Containerization & CI/CD (`Dockerfile` & `.github/workflows/ci-cd.yml`):**
   - Packaged the API and serialized pipeline into a lightweight `python:3.11-slim` Docker container.
   - Automated `pytest` execution, Docker image builds, and container health smoke tests on every push via **GitHub Actions**.

6. **Post-Deployment Monitoring (`notebooks/05_ml_monitoring.ipynb`):**
   - Implemented **Population Stability Index (PSI)** and **Kolmogorov-Smirnov (KS)** tests to detect feature drift, prediction probability shift, and performance degradation.

---

## 💻 Local Setup & Execution

### 1. Clone the Repository & Create Virtual Environment
```bash
git clone https://github.com
cd final_ml_-project/telco-churn-mlops

python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

pip install -r requirements-dev.txt
```

### 2. Launch MLflow Tracking UI
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open **`http://127.0.0.1:5000`** in your browser to inspect model metrics and parameter trials.

### 3. Run FastAPI Locally
```bash
uvicorn src.app:app --reload --port 8000
```
Access interactive Swagger documentation at **`http://127.0.0`**.

### 4. Run Automated Tests
```bash
python -m pytest -v
```

---

## 🐳 Docker Deployment

Build and run the containerized prediction service:

```bash
# Build the Docker image
docker build -t telco-churn-api:1.0.0 .

# Run the container on port 8002
docker run -d --name telco-churn-container -p 8002:8000 telco-churn-api:1.0.0

# Verify container status
docker ps
```
Visit **`http://127.0.0`** to test the live containerized API.

---

## 📡 Sample API Request & Response

### Request (`POST /predict`)
```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 2,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 89.5,
  "TotalCharges": 179.0
}
```

### Response (`200 OK`)
```json
{
  "prediction": 1,
  "class_name": "Yes",
  "probability": 0.7842
}
```
