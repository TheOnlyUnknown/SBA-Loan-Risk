"""
Phase 3 (coding) — loading and labeling the SBA FOIA data.

See README.md "Leakage Rule" and "Time-Aware Seasoning Cutoff" for the
reasoning behind the constants below. This file is the implementation
of those two decisions, not the place to re-litigate them.
"""

import pandas as pd
import numpy as np

# LoanStatus values that represent a real, settled outcome. Everything
# else (EXEMPT, CANCLD, COMMIT) is either withheld, never-funded, or
# still outstanding -- not usable as a label.
RESOLVED_STATUSES = ["P I F", "CHGOFF"]

# Columns that need to go from string -> real datetime before we can do
# arithmetic on them (e.g. "how many months between approval and
# charge-off").
DATE_COLS = ["ApprovalDate", "ChargeOffDate", "PaidInFullDate"]


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the raw string date columns to real pandas datetimes, plus
    AsOfDate (the FOIA snapshot date -- not a loan attribute, but we need
    it as a date to measure loan age against later).
    """
    for col in DATE_COLS:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["AsOfDate"] = pd.to_datetime(df["AsOfDate"], errors="coerce")
    return df


def filter_resolved(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only loans with a real, settled outcome (P I F or CHGOFF).
    Drops EXEMPT, CANCLD, COMMIT -- none of those are a usable label.
    """
    return df[df["LoanStatus"].isin(RESOLVED_STATUSES)].copy()


def build_target(df: pd.DataFrame) -> pd.Series:
    """
    LoanStatus gets read here, once, to build the label -- then it's done.
    It does not go into the feature set; it's the answer, not a predictor.
    """
    y = (df["LoanStatus"] == "CHGOFF").astype(int)
    y.name = "target_chargeoff"
    return y


def compute_seasoning_cutoff(df: pd.DataFrame, seasoning_percentile: float = 0.90):
    """
    The problem: a loan approved recently hasn't had time to default yet,
    even if it eventually would. If we don't account for that, our
    "resolved" population skews toward older loans that had time to
    reveal an outcome, while the model gets used on brand-new loans in
    production. That mismatch makes offline metrics look better than
    real performance.

    The fix: look ONLY at loans that already charged off, and measure how
    many months it took them, from approval to charge-off. Take a high
    percentile of that distribution (default: the 90th) as the
    "seasoning window" -- the amount of time a loan needs to have been
    alive before we trust that it's not going to default.

    Returns:
        as_of_date:      the FOIA snapshot date (today, for this data)
        seasoning_months: the computed window, in months
        cutoff_date:      loans approved on/before this date are "old enough"
    """
    chgoff = df[df["LoanStatus"] == "CHGOFF"].copy()
    chgoff = chgoff.dropna(subset=["ApprovalDate", "ChargeOffDate"])

    months_to_chargeoff = (chgoff["ChargeOffDate"] - chgoff["ApprovalDate"]).dt.days / 30.44
    months_to_chargeoff = months_to_chargeoff[months_to_chargeoff >= 0]  # drop bad rows

    seasoning_months = float(np.percentile(months_to_chargeoff, seasoning_percentile * 100))

    as_of_date = df["AsOfDate"].max()
    cutoff_date = as_of_date - pd.DateOffset(months=int(round(seasoning_months)))

    return as_of_date, seasoning_months, cutoff_date
