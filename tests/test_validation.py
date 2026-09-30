import pandas as pd
import pytest

from src.validation import validate_raw

def valid_row():
    return {
        "LoanStatus": "P I F",
        "ApprovalDate": "2020-01-01",
        "GrossApproval": 100000.0,
        "SBAGuaranteedApproval": 75000.0,
        "TermInMonths": 120.0,
        "InitialInterestRate": 6.5,
        "JobsSupported": 5.0,
        "FixedorVariableInterestInd": "F",
    }

def test_validate_raw_accepts_valid_data():
    df = pd.DataFrame([valid_row()])
    validated = validate_raw(df)
    assert len(validated) == 1

def test_validate_raw_rejects_unknown_loan_status():
    row = valid_row()
    row["LoanStatus"] = "WITHDRN"
    df = pd.DataFrame([row])
    with pytest.raises(Exception):
        validate_raw(df)

def test_validate_raw_rejects_negative_gross_approval():
    row = valid_row()
    row["GrossApproval"] = -500.0
    df = pd.DataFrame([row])
    with pytest.raises(Exception):
        validate_raw(df)

def test_validate_raw_rejects_guaranteed_exceeding_gross():
    row = valid_row()
    row["SBAGuaranteedApproval"] = row["GrossApproval"] + 1000.0
    df = pd.DataFrame([row])
    with pytest.raises(Exception):
        validate_raw(df)

def test_validate_raw_rejects_null_approval_date():
    row = valid_row()
    row["ApprovalDate"] = None
    df = pd.DataFrame([row])
    with pytest.raises(Exception):
        validate_raw(df)

def test_validate_raw_rejects_negative_term_in_months():
    row = valid_row()
    row["TermInMonths"] = -12.0
    df = pd.DataFrame([row])
    with pytest.raises(Exception):
        validate_raw(df)

def test_validate_raw_allows_null_interest_rate():
    row = valid_row()
    row["InitialInterestRate"] = None
    df = pd.DataFrame([row])
    df["InitialInterestRate"] = df["InitialInterestRate"].astype(float)
    validated = validate_raw(df)
    assert len(validated) == 1
