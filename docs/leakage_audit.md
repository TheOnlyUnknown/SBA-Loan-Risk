# Schema & Leakage Audit — FOIA 7(a) FY2020-Present

Source: `data/raw/foia_7a_fy2020_present.csv` (as of 2026-06-30 snapshot,
downloaded 2026-09-20)

Shape: 388,338 rows x 42 columns

## Target

`LoanStatus` value counts:

| Status | Count | Meaning |
|---|---|---|
| EXEMPT | 242,061 | Individual status withheld from disclosure (verify against data dictionary — likely active/current loans below a reporting threshold or not yet resolved) |
| P I F | 68,201 | Paid in full — non-default, resolved |
| CANCLD | 50,104 | Approved but cancelled/never fully disbursed |
| COMMIT | 21,079 | Committed/disbursed, still outstanding — not yet resolved |
| CHGOFF | 6,893 | Charged off — default, resolved |

Only `P I F` and `CHGOFF` are resolved, unambiguous outcomes. That's
75,094 rows (~19.3% of the file) as the actual modeling population for
the MVP binary classifier.

**Important, not just a filtering footnote:** restricting to resolved
loans introduces a maturation/selection bias. This file only goes back to
FY2020, and charge-offs take time to happen — a loan originated in 2025
hasn't had time to default yet, so it's sitting in `COMMIT`, not
`CHGOFF`. If you train only on `P I F`/`CHGOFF`, you are implicitly
training on a population skewed toward earlier vintages in the window
(2020-2022) that have had time to season, and your model's real-world
performance on brand-new loans will look better in offline eval than it
will in production, where every loan is unseasoned at the point you'd
actually use the score. Two ways to handle this, to decide in Phase 5:
(1) explicitly document the model as "predicts eventual outcome
conditional on the loan having resolved within the observation window,"
which is a legitimate but narrower claim, or (2) restrict training to
loans old enough to have plausibly resolved (e.g. origination FY2020-2022
only) so the resolved subset isn't systematically younger than the
population you'd deploy on. Don't skip this decision — it's exactly the
kind of thing that produces a model that backtests well and quietly
underperforms live.

Also verify what `EXEMPT` actually means in the SBA data dictionary
before assuming it's discardable — if it turns out to include resolved-but-suppressed
outcomes, the modeling population changes.

## Column classification

**Allowed — known at origination:**
Program, BorrState, BorrZip, BankName, BankFDICNumber, BankNCUANumber,
BankCity, BankState, BankZip, GrossApproval, SBAGuaranteedApproval,
ApprovalDate, ApprovalFY, ProcessingMethod, InitialInterestRate,
FixedorVariableInterestInd, TermInMonths, NaicsCode, NaicsDescription,
FranchiseCode, FranchiseName, ProjectCounty, ProjectState,
SBADistrictOffice, CongressionalDistrict, BusinessType, BusinessAge,
RevolverStatus, JobsSupported, CollateralInd

**Excluded — leakage (only known/populated once the loan is performing):**
- `PaidInFullDate` — only populated if the loan resolved as paid-in-full; directly encodes the label.
- `ChargeOffDate` — only populated if the loan defaulted; this literally *is* the label.
- `GrossChargeOffAmount` — non-zero/non-null only on defaults; same problem.
- `LoanStatus` — this is the target itself, not a feature. Drop from X after deriving y.

**Gray area — needs a decision, not an assumption:**
- `FirstDisbursementDate` — happens after the approval/underwriting decision that this model is meant to support. If "origination" means "the point where SBA/lender decides to approve," this should be excluded, since it isn't known yet at that decision point. Recommend excluding for the MVP; revisit only if the framing changes to "known at disbursement" instead of "known at approval."
- `SoldSecMrktInd` — secondary-market sale can happen well after origination. Needs the data dictionary to confirm timing before it's allowed in.

**Excluded — identifiers / low value, not a leakage issue:**
`AsOfDate` (dataset snapshot metadata, not a loan attribute), `LocationID`,
`BorrName`, `BorrStreet`, `BorrCity`, `BankStreet` (all PII/free-text
identifiers with no standalone predictive value at this stage; geographic
signal is already captured via `ProjectState`/`ProjectCounty`/`BorrState`).

## Next step (Phase 5)

Build the loading + labeling function: filter to `LoanStatus` in
{`P I F`, `CHGOFF`}, decide the vintage-cutoff question above, derive
`y = 1 if CHGOFF else 0`, then restrict `X` to the allowed column list.
Check class balance and missingness on the allowed columns before any
encoding decisions.
