import pandas as pd
import numpy as np

RESOLVED_STATUSES = ["P I F","CHGOFF"]

DATE_COLS = ["ApprovalDate", "PaidInFullDate", "ChargeOffDate"]


ALLOWED_FEATURE_COLS = [
    "Program", "BorrState", "BorrZip",
    "BankName", "BankFDICNumber", "BankNCUANumber", "BankCity", "BankState", "BankZip",
    "GrossApproval", "SBAGuaranteedApproval",
    "ApprovalDate", "ApprovalFY",
    "ProcessingMethod", "InitialInterestRate", "FixedorVariableInterestInd", "TermInMonths",
    "NaicsSector", "FranchiseCode", "FranchiseName",
    "ProjectCounty", "ProjectState", "SBADistrictOffice", "CongressionalDistrict",
    "BusinessType", "BusinessAge",
    "RevolverStatus", "JobsSupported", "CollateralInd",
]

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    for col in DATE_COLS:
        df[col] = pd.to_datetime(df[col], errors = "coerce")
    df["AsOfDate"] = pd.to_datetime(df["AsOfDate"], errors = "coerce")
    return df

def filter_resolved(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["LoanStatus"].isin(RESOLVED_STATUSES)].copy()

def build_target(df: pd.DataFrame) -> pd.Series:
    y = (df["LoanStatus"] == "CHGOFF").astype(int)
    y.name = "target_chargeoff"
    return y

def compute_seasoning_cutoff(df: pd.DataFrame, seasoning_percentile: float = 0.90):
    chgoff = df[df["LoanStatus"] == "CHGOFF"].copy()
    chgoff = chgoff.dropna(subset=["ApprovalDate", "ChargeOffDate"])

    months_to_chargeoff = (chgoff["ChargeOffDate"] - chgoff["ApprovalDate"]).dt.days / 30.44
    months_to_chargeoff = months_to_chargeoff[months_to_chargeoff >= 0]

    seasoning_months = float(np.percentile(months_to_chargeoff, seasoning_percentile * 100))

    as_of_date = df["AsOfDate"].max()
    cutoff_date = as_of_date - pd.DateOffset(months=int(round(seasoning_months)))

    return as_of_date, seasoning_months, cutoff_date

def derive_naics_sector(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
    df["NaicsSector"] = df["NaicsCode"].astype(str).str.zfill(6).str[:2]
    return df

def apply_seasoning_cutoff(df: pd.DataFrame, cutoff_date) -> pd.DataFrame:

    return df[df["ApprovalDate"] <= cutoff_date].copy()

def load_and_label(csv_path: str, seasoning_percentile: float = 0.90):
    df = pd.read_csv(csv_path, low_memory=False)
    df = parse_dates(df)

    resolved = filter_resolved(df)
    as_of_date, seasoning_months, cutoff_date = compute_seasoning_cutoff(resolved, seasoning_percentile)
    seasoned = apply_seasoning_cutoff(resolved, cutoff_date)
    seasoned = derive_naics_sector(seasoned)

    y = build_target(seasoned)
    X = seasoned[ALLOWED_FEATURE_COLS].copy()

    return X, y
