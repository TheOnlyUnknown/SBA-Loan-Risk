import pandas as pd
import numpy as np

RESOLVED_STATUSES = ["P I F","CHGOFF"]

DATE_COLS = ["ApprovalDate", "PaidInFullDate", "ChargeOffDate"]

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