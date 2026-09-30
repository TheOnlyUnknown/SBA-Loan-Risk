import pandas as pd

from src.data_loader import (
    build_target,
    filter_resolved,
    derive_naics_sector,
    derive_lender_type,
    derive_is_franchise,
    clean_zip_codes,
    clean_business_age,
    clean_congressional_district,
    compute_seasoning_cutoff,
    apply_seasoning_cutoff,
)

def test_build_target_labels_chargeoff_as_one():
    df = pd.DataFrame({"LoanStatus": ["CHGOFF", "P I F", "CHGOFF"]})
    y = build_target(df)
    assert y.tolist() == [1, 0, 1]

def test_filter_resolved_keeps_only_resolved_statuses():
    df = pd.DataFrame({"LoanStatus": ["CHGOFF", "P I F", "EXEMPT", "CANCLD", "COMMIT"]})
    resolved = filter_resolved(df)
    assert set(resolved["LoanStatus"]) == {"CHGOFF", "P I F"}
    assert len(resolved) == 2

def test_derive_naics_sector_preserves_leading_zero():
    df = pd.DataFrame({"NaicsCode": [722513, 81234, 111110]})
    result = derive_naics_sector(df)
    assert result["NaicsSector"].tolist() == ["72", "08", "11"]

def test_derive_lender_type_classifies_by_fdic_ncua_presence():
    df = pd.DataFrame({
        "BankFDICNumber": [123, None, None, 456],
        "BankNCUANumber": [None, 789, None, 999],
    })
    result = derive_lender_type(df)
    assert result["LenderType"].tolist() == ["Bank", "CreditUnion", "NonDepository", "Bank"]

def test_derive_is_franchise_flags_nonnull_franchise_code():
    df = pd.DataFrame({"FranchiseCode": [1234, None, 5678]})
    result = derive_is_franchise(df)
    assert result["IsFranchise"].tolist() == [1, 0, 1]

def test_clean_zip_codes_zero_pads_to_five_digits():
    df = pd.DataFrame({"BorrZip": [1234, 90210], "BankZip": [501, 33101]})
    result = clean_zip_codes(df)
    assert result["BorrZip"].tolist() == ["01234", "90210"]
    assert result["BankZip"].tolist() == ["00501", "33101"]

def test_clean_business_age_fills_nulls_as_unanswered():
    df = pd.DataFrame({"BusinessAge": ["Existing or more than 2 years old", None]})
    result = clean_business_age(df)
    assert result["BusinessAge"].tolist() == ["Existing or more than 2 years old", "Unanswered"]

def test_clean_congressional_district_fills_nulls_and_converts_to_string():
    df = pd.DataFrame({"CongressionalDistrict": [5.0, None, 12.0]})
    result = clean_congressional_district(df)
    assert result["CongressionalDistrict"].tolist() == ["5", "Unanswered", "12"]

def test_compute_seasoning_cutoff_uses_percentile_of_time_to_chargeoff():
    df = pd.DataFrame({
        "LoanStatus": ["CHGOFF", "CHGOFF", "CHGOFF"],
        "ApprovalDate": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-01"]),
        "ChargeOffDate": pd.to_datetime(["2020-02-01", "2020-04-01", "2020-06-01"]),
        "AsOfDate": pd.to_datetime(["2026-01-01"] * 3),
    })
    as_of_date, seasoning_months, cutoff_date = compute_seasoning_cutoff(df, seasoning_percentile=0.5)
    assert seasoning_months > 0
    assert cutoff_date < as_of_date

def test_apply_seasoning_cutoff_filters_to_loans_before_cutoff():
    df = pd.DataFrame({"ApprovalDate": pd.to_datetime(["2020-01-01", "2021-06-01", "2022-01-01"])})
    result = apply_seasoning_cutoff(df, pd.Timestamp("2021-01-01"))
    assert len(result) == 1
