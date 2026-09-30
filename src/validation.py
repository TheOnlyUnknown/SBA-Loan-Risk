import pandera.pandas as pa
from pandera import Column, Check

KNOWN_LOAN_STATUSES = ["EXEMPT", "P I F", "CANCLD", "COMMIT", "CHGOFF"]

raw_loan_schema = pa.DataFrameSchema(
    columns={
        "LoanStatus": Column(str, Check.isin(KNOWN_LOAN_STATUSES)),
        "ApprovalDate": Column(str, nullable=False),
        "GrossApproval": Column(float, Check.gt(0)),
        "SBAGuaranteedApproval": Column(float, Check.gt(0)),
        "TermInMonths": Column(float, Check.ge(0)),
        "InitialInterestRate": Column(float, Check.ge(0), nullable=True),
        "JobsSupported": Column(float, Check.ge(0)),
        "FixedorVariableInterestInd": Column(str, Check.isin(["F", "V"]), nullable=True),
    },
    checks=Check(
        lambda df: df["SBAGuaranteedApproval"] <= df["GrossApproval"],
        error="SBAGuaranteedApproval must never exceed GrossApproval",
    ),
    strict=False,
    coerce=False,
)

def validate_raw(df):
    return raw_loan_schema.validate(df, lazy=True)
