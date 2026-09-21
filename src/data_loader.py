import pandas as pd
import numpy as np

RESOLVED_STATUSES = ["P I F","CHGOFF"]

DATE_COLS = ["ApprovalDate", "PaidInFullDate", "ChargeOffDate"]

# Origination-time columns only -- see docs/leakage_audit.md for the
# full column-by-column reasoning. Everything here is known at the
# moment a loan is approved. Leakage columns (PaidInFullDate,
# ChargeOffDate, GrossChargeOffAmount, LoanStatus) and the gray-area
# ones (FirstDisbursementDate, SoldSecMrktInd) are deliberately absent.
ALLOWED_FEATURE_COLS = [
    "BorrState", "BorrZip",
    "BankName", "LenderType", "BankCity", "BankState", "BankZip",
    "GrossApproval", "SBAGuaranteedApproval",
    "ApprovalDate", "ApprovalFY",
    "ProcessingMethod", "InitialInterestRate", "FixedorVariableInterestInd", "TermInMonths",
    "NaicsSector", "IsFranchise",
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
    # Collapse the 6-digit NaicsCode down to its first 2 digits -- the
    # broad industry sector. 877 exact codes is too sparse to model
    # reliably (median 10 loans each); 24 sectors is not (median 1,054
    # each), and sector alone still separates risk (3.6%-8.7% default
    # rate range across sectors). zfill(6) guards against codes that
    # lost a leading zero when read as a number.
    df = df.copy()
    df["NaicsSector"] = df["NaicsCode"].astype(str).str.zfill(6).str[:2]
    return df

def derive_lender_type(df: pd.DataFrame) -> pd.DataFrame:
    # BankFDICNumber/BankNCUANumber are blank because of WHICH KIND of
    # lender it is, not because data is missing. 36,084 loans have only
    # an FDIC number (banks), 1,263 have only an NCUA number (credit
    # unions), 1,891 have neither -- real non-depository SBA lenders
    # like Newtek and ReadyCap, not broken rows. 41 rows have both
    # (a data quirk); those get folded into "Bank" since FDIC coverage
    # is the dominant signal there.
    df = df.copy()
    has_fdic = df["BankFDICNumber"].notna()
    has_ncua = df["BankNCUANumber"].notna()

    df["LenderType"] = "NonDepository"
    df.loc[has_fdic, "LenderType"] = "Bank"
    df.loc[~has_fdic & has_ncua, "LenderType"] = "CreditUnion"
    return df

def derive_is_franchise(df: pd.DataFrame) -> pd.DataFrame:
    # FranchiseCode is blank for 86.7% of loans because those businesses
    # genuinely aren't franchises -- not because the value is unknown.
    # Individual franchise codes are far too sparse to use directly
    # (1,304 of 1,329 codes have fewer than 30 loans each), so this
    # collapses to a single yes/no signal instead.
    df = df.copy()
    df["IsFranchise"] = df["FranchiseCode"].notna().astype(int)
    return df

def clean_zip_codes(df: pd.DataFrame) -> pd.DataFrame:
    # BorrZip/BankZip are stored as numbers, which silently drops the
    # leading zero on any zip starting with 0 (New England, NJ, PR) --
    # 02360 becomes 2360. Affects 9.3% of rows for BorrZip alone. Pad
    # back to 5 digits and keep as text: a zip is an identifier, not a
    # quantity, same reasoning as NaicsCode.
    df = df.copy()
    df["BorrZip"] = df["BorrZip"].astype(str).str.zfill(5)
    df["BankZip"] = df["BankZip"].astype(str).str.zfill(5)
    return df

def clean_business_age(df: pd.DataFrame) -> pd.DataFrame:
    # The raw data already has an explicit "Unanswered" category for
    # this -- SBA itself treats "don't know" as an answer. A small
    # number of rows (0.29%) are a true blank instead of that string;
    # fold them into the same category so there's one "unknown" value,
    # not two different-looking ones.
    df = df.copy()
    df["BusinessAge"] = df["BusinessAge"].fillna("Unanswered")
    return df

def clean_congressional_district(df: pd.DataFrame) -> pd.DataFrame:
    # Stored as a float (e.g. 12.0), which invites a model to treat it as
    # a quantity -- district 12 isn't "twice" district 6. It's an
    # identifier, same reasoning as zip codes and NAICS codes. 0 means
    # "at-large district" (a state with only one district has no number
    # to assign) -- not a true zero, but it's already its own distinct
    # category once treated as text, so no special-casing needed. No
    # nulls in the current seasoned population, but fillna defensively
    # (same "Unanswered" pattern as BusinessAge) before the int cast so
    # this doesn't crash if a future data pull has any.
    df = df.copy()
    district = df["CongressionalDistrict"]
    district_str = district.fillna(-1).astype(int).astype(str)
    district_str = district_str.where(district.notna(), "Unanswered")
    df["CongressionalDistrict"] = district_str
    return df

def apply_seasoning_cutoff(df: pd.DataFrame, cutoff_date) -> pd.DataFrame:
    # Keep only loans approved on/before the cutoff -- old enough that a
    # non-default outcome is trustworthy, not just "hasn't happened yet".
    return df[df["ApprovalDate"] <= cutoff_date].copy()

def load_and_label(csv_path: str, seasoning_percentile: float = 0.90):
    df = pd.read_csv(csv_path, low_memory=False)
    df = parse_dates(df)

    resolved = filter_resolved(df)
    as_of_date, seasoning_months, cutoff_date = compute_seasoning_cutoff(resolved, seasoning_percentile)
    seasoned = apply_seasoning_cutoff(resolved, cutoff_date)
    seasoned = derive_naics_sector(seasoned)
    seasoned = derive_lender_type(seasoned)
    seasoned = derive_is_franchise(seasoned)
    seasoned = clean_zip_codes(seasoned)
    seasoned = clean_business_age(seasoned)
    seasoned = clean_congressional_district(seasoned)

    y = build_target(seasoned)
    X = seasoned[ALLOWED_FEATURE_COLS].copy()

    return X, y
