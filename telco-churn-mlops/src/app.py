import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="REST API for predicting customer churn using the tuned XGBoost Pipeline.",
    version="1.0.0")

model_pipeline = joblib.load(BASE_DIR / "models" / "xgboost_pipeline.joblib")
data = joblib.load(BASE_DIR / "data" / "cleaned_telco_data.joblib")


class_names = {
    0: "No",
    1: "Yes"}



class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float
    model_config = {"extra": "forbid"}

def prepare_input(input_data: CustomerData) -> pd.DataFrame:
    return pd.DataFrame([input_data.model_dump()])

@app.get("/")
def home():
    return {
        "status": "API is running",
        "model": "XGBoost Churn Pipeline",
        "docs_url": "/docs"
    }

@app.get("/model-info")
def model_info():
    feature_cols = [col for col in data.columns if col != "Churn"]
    return {
        "model_name": "telco_churn_classifier",
        "model_version": "1.0.0",
        "model_type": "XGBClassifier (Pipeline)",
        "number_of_features": len(CustomerData.model_fields),
        "classes":
            [
                "no",
                "yes"
            ]
    }



@app.post("/predict")
def predict(input_data: CustomerData):
    input_df = prepare_input(input_data)

    pred_array = model_pipeline.predict(input_df)
    prob_array = model_pipeline.predict_proba(input_df)

    prediction = int(pred_array.item())
    predicted_probability = float(prob_array.flatten()[prediction])

    return {
        "prediction": prediction,
        "class_name": class_names[prediction],
        "probability": round(predicted_probability, 4)
    }