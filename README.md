# SBA Loan Default Risk Classifier

Project 1 of an 8-project, 12-month ML engineering portfolio. Predicts loan
charge-off vs. paid-in-full for SBA 7(a) and 504 loans, using only
information available at the moment a loan is originated. Built for a
lender's credit/risk team persona.

## Problem Framing

Binary classification: will this loan be charged off (default) or paid in
full? Success is measured by recall, precision, and F1 on the default
class, plus calibration (predicted probabilities should match observed
default rates) — not raw accuracy, which is misleading on an imbalanced
target where most loans are paid in full.

MVP scope is the outcome classifier only. A secondary regression target
(e.g. loss severity, time-to-default) is deferred to a later iteration.

## Data

US Small Business Administration 7(a) and 504 FOIA loan dataset (public,
no API key required): https://data.sba.gov/en/dataset/7-a-504-foia

### Leakage Rule

Only information known at the moment a loan is originated is allowed as a
feature — loan amount, term, industry (NAICS code), business type, SBA
guaranteed percentage, and similar origination-time fields. Anything only
known or recorded after the loan starts performing — a charge-off date,
for instance — is excluded, because the model would effectively be
reading the answer before making its prediction. The tell if this rule
ever gets broken by accident: suspiciously high accuracy on a problem
that's genuinely hard to predict well.

### Time-Aware Seasoning Cutoff

Filtering to resolved loans (paid-in-full or charged-off) isn't enough on
its own. This file starts at FY2020, and charge-offs take time to happen
— a loan approved last year hasn't had time to default yet, so it sits
in an unresolved status, not charged-off. Training only on resolved
loans without accounting for this means the training population skews
toward older, already-seasoned loans, while the model will actually be
scored on brand-new, unseasoned ones — offline performance would look
better than real performance.

The fix: derive a seasoning window from the data itself — the time it
actually took charged-off loans to charge off (e.g. the 90th percentile
of that distribution) — and only include loans old enough, as of the
snapshot date, to have had a fair chance to show that outcome. Not an
arbitrary "just use FY2020-2022" cutoff; a cutoff computed from observed
time-to-default.

## Project Structure

```
sba-loan-risk/
├── data/
│   ├── raw/            # untouched downloaded FOIA CSVs
│   └── processed/      # cleaned / feature-engineered datasets
├── src/                # pipeline code (loading, features, model, eval)
├── notebooks/          # exploratory analysis
├── tests/              # unit tests
└── docs/               # schema notes, decisions, writeups
```

## Status

- Phase 1-3: project scaffolding, framing, and repo structure — done.
- Phase 4 (current): pull the FY2020-Present FOIA CSV, inspect schema and
  target distribution, sort columns into origination-time (allowed) vs.
  post-origination (leakage) before any feature work starts.
