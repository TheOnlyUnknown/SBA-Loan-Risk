import json
from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.data_loader import (
    derive_naics_sector,
    derive_lender_type,
    derive_is_franchise,
    clean_business_age,
)
from src.features import select_model_features
from src.model_io import load_model

MODEL_PATH = "models/xgboost_pipeline.joblib"
METADATA_PATH = "models/metadata.json"

model_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_state["model"] = load_model(MODEL_PATH)
    with open(METADATA_PATH) as f:
        model_state["metadata"] = json.load(f)
    yield
    model_state.clear()

app = FastAPI(title="SBA Loan Default Risk API", lifespan=lifespan)

class LoanApplication(BaseModel):
    BorrState: str
    BankState: str
    GrossApproval: float
    SBAGuaranteedApproval: float
    ApprovalFY: int
    ProcessingMethod: str
    InitialInterestRate: Optional[float] = None
    FixedorVariableInterestInd: Optional[str] = None
    TermInMonths: float
    NaicsCode: int
    BankFDICNumber: Optional[str] = None
    BankNCUANumber: Optional[str] = None
    FranchiseCode: Optional[str] = None
    ProjectState: str
    SBADistrictOffice: str
    BusinessType: str
    BusinessAge: Optional[str] = None
    RevolverStatus: str
    JobsSupported: float
    CollateralInd: str

class PredictionResponse(BaseModel):
    probability_of_chargeoff: float
    flagged_high_risk: bool
    threshold_used: float

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": "model" in model_state,
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication):
    df = pd.DataFrame([application.model_dump()])

    df = derive_naics_sector(df)
    df = derive_lender_type(df)
    df = derive_is_franchise(df)
    df = clean_business_age(df)

    model = model_state["model"]
    metadata = model_state["metadata"]

    X = select_model_features(df.drop(columns=["NaicsCode", "BankFDICNumber", "BankNCUANumber", "FranchiseCode"]))
    probability = float(model.predict_proba(X)[:, 1][0])
    threshold = metadata["threshold"]

    return PredictionResponse(
        probability_of_chargeoff=probability,
        flagged_high_risk=probability >= threshold,
        threshold_used=threshold,
    )
