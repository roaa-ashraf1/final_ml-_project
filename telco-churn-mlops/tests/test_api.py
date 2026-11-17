import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

valid_payload = {
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

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "API is running"

def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json()["number_of_features"] == 19

def test_valid_prediction():
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    result = response.json()
    assert result["prediction"] in [0, 1]
    assert 0.0 <= result["probability"] <= 1.0

def test_missing_field():
    payload = valid_payload.copy()
    payload.pop("tenure")
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_extra_field():
    payload = valid_payload.copy()
    payload["unexpected_feature"] = 123
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
