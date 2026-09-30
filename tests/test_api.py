from fastapi.testclient import TestClient

from src.api import app

def valid_payload():
    return {
        "BorrState": "PA",
        "BankState": "DE",
        "GrossApproval": 25000.0,
        "SBAGuaranteedApproval": 12500.0,
        "ApprovalFY": 2020,
        "ProcessingMethod": "SBA Express Program",
        "InitialInterestRate": 11.04,
        "FixedorVariableInterestInd": "V",
        "TermInMonths": 120.0,
        "NaicsCode": 236220,
        "BankFDICNumber": "18409.0",
        "BankNCUANumber": None,
        "FranchiseCode": None,
        "ProjectState": "PA",
        "SBADistrictOffice": "PHILADELPHIA DISTRICT OFFICE",
        "BusinessType": "PARTNERSHIP",
        "BusinessAge": "Unanswered",
        "RevolverStatus": "Y",
        "JobsSupported": 5.0,
        "CollateralInd": "N",
    }

def test_health_reports_model_loaded():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["model_loaded"] is True

def test_predict_returns_probability_and_flag():
    with TestClient(app) as client:
        response = client.post("/predict", json=valid_payload())
        assert response.status_code == 200
        body = response.json()
        assert 0.0 <= body["probability_of_chargeoff"] <= 1.0
        assert isinstance(body["flagged_high_risk"], bool)
        assert body["threshold_used"] > 0

def test_predict_rejects_missing_required_field():
    with TestClient(app) as client:
        payload = valid_payload()
        del payload["GrossApproval"]
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

def test_predict_accepts_null_optional_fields():
    with TestClient(app) as client:
        payload = valid_payload()
        payload["InitialInterestRate"] = None
        payload["BusinessAge"] = None
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
