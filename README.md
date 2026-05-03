# HMDA Mortgage Approval Prediction & Fair Lending Audit

Mortgage approval model built on 2023 CFPB HMDA data for California. The project predicts loan approval outcomes and audits the model for racial disparate impact using SHAP and Adverse Impact Ratio analysis.

---

## Data

**Source:** CFPB HMDA Data Browser - California, 2023

HMDA (Home Mortgage Disclosure Act) is mandatory regulatory disclosure. Every U.S. lender above an asset threshold submits it to the CFPB annually. It covers all loan applications with applicant demographics, loan terms, and outcomes.

**Scope filter applied:**
- Home purchase loans only (not refinance)
- Conventional loans only (not FHA/VA)
- First lien only
- Originated or denied outcomes only (withdrawn/incomplete excluded)

| | Count |
|---|---|
| Raw file | 962,073 rows, 99 columns |
| After scope filter | 210,315 rows |
| After cleaning | 196,113 rows |
| Approval rate | 87.8% |
| Denial rate | 12.2% |

---

## Data Quality Issues Found

| Problem | Finding | Fix |
|---|---|---|
| Mixed units | income in $K, loan_amount in $ | Converted loan_amount and property_value to $K |
| Negative income | 83 rows, 95% denied | Dropped |
| Income outliers | Max $118M/yr | Capped at 99.5th percentile ($2,265K) |
| LTV outliers | Max LTV 29,167% | Capped at 200 |
| DTI format | Mixed string buckets + integers + "Exempt" | Mapped to numeric midpoints |
| DTI MNAR | 7.95% missing for denied vs 5.88% for approved | Created `dti_missing_flag` binary feature |
| Data leakage | `rate_spread`, `interest_rate` 0% populated for denied loans | Dropped before any modeling |

The leakage issue is worth noting: an early model including `rate_spread` achieved AUC = 0.9996. After removing it, honest AUC is 0.8490.

---

## EDA - Key Findings

### Denial Rate by Race

| Race | n | Denial Rate | AIR vs White |
|---|---|---|---|
| Black or African American | 4,802 | 20.2% | 0.905 |
| American Indian / Alaska Native | 1,094 | 19.5% | 0.913 |
| Native Hawaiian / Pacific Islander | 524 | 20.8% | 0.898 |
| White | 93,419 | 11.8% | 1.000 |
| Asian | 44,204 | 9.6% | 1.025 |

Adverse Impact Ratio (AIR) = approval rate of group / approval rate of White applicants. AIR < 0.80 is the legal threshold for disparate impact. No group falls below 0.80 in the raw data, but the gaps are statistically significant (chi-square p = 1.6x10^-253).

Denied Black applicants had higher median income than denied White applicants, meaning income alone does not explain the disparity.

### Denial Rate by Age

Youngest applicants (<25) face a 20.2% denial rate vs 10.5% for the 25-34 group, consistent with shorter credit histories. Oldest applicants (>74) also face elevated rates at 15.5%.

### Top Stated Denial Reasons

1. Debt-to-income ratio - 38%
2. Credit history - 14%
3. Collateral - 12%

---

## Feature Engineering

11 features engineered from pre-decision data only:

| Feature | Logic |
|---|---|
| `loan_to_income_ratio` | loan_amount / income |
| `dti_missing_flag` | 1 if DTI is null (MNAR signal) |
| `high_dti_flag` | DTI > 43% (QM threshold) |
| `high_ltv_flag` | LTV > 80% (PMI threshold) |
| `low_income_tract_flag` | Tract income < 80% of MSA median |
| `minority_tract_flag` | Tract minority population > 50% |
| `non_conforming_flag` | Jumbo loan indicator |
| `co_applicant_present` | Joint application flag |
| `county_approval_rate` | County mean approval rate (target encoding) |
| `non_primary_flag` | Investment or second home |
| `applicant_age_enc` | Ordinal encoding of HMDA age bands |

---

## Models

80/20 stratified train/test split. LightGBM cross-validated with 5-fold CV.

| Model | Test AUC | F1 | Balanced Accuracy |
|---|---|---|---|
| Decision Tree (depth=5) | 0.7978 | 0.9528 | 0.6836 |
| Random Forest (n=100) | 0.8300 | 0.9534 | 0.7142 |
| LightGBM | 0.8490 | 0.9581 | 0.7255 |

LightGBM CV AUC: 0.8401 +/- 0.0032. Optimal threshold: 0.85.

### SHAP - Top Features

`dti_numeric` is the strongest predictor by a wide margin (mean |SHAP| = 0.582), followed by `property_value_k` (0.325) and `loan_amount_k` (0.207). `loan_term`, `county_approval_rate`, and `loan_to_income_ratio` round out the top 6.

`derived_ethnicity_enc` ranks 8th (0.094) and `derived_race_enc` ranks 11th (0.082) — both inside the top 12. The model is picking up on demographic patterns in the training data. This is what motivates the fair model comparison below.

---

## Results

### Error Analysis (at threshold 0.85)

| Error Type | Count | Rate |
|---|---|---|
| False Negative (wrongly denied) | 412 | 1.05% |
| False Positive (wrongly approved) | 2,565 | 6.54% |

False Negative rate by race:

| Race | FN Rate |
|---|---|
| Black or African American | 3.52% |
| American Indian / Alaska Native | 1.94% |
| White | 1.24% |
| Asian | 0.89% |

### Fair Model - No Demographics

A second LightGBM was trained with race, ethnicity, and sex removed to simulate a legally compliant model.

| Metric | Full Model | Fair Model |
|---|---|---|
| Test AUC | 0.8490 | 0.8458 |
| Balanced Accuracy | 0.7255 | 0.7296 |

AIR and False Negative rate by race:

| Race | Full AIR | Fair AIR | Full FN% | Fair FN% |
|---|---|---|---|---|
| Black or African American | 0.934 | 0.962 | 3.52% | 2.17% |
| American Indian / Alaska Native | 0.929 | 0.938 | 1.94% | 1.29% |
| White | 1.000 | 1.000 | 1.24% | 1.90% |
| Asian | 1.024 | 1.040 | 0.89% | 0.64% |

Removing demographic features costs 0.003 AUC and improves AIR for every minority group. The wrongful denial rate for Black applicants drops from 3.52% to 2.17%.

---

## Stack

Python, pandas, scikit-learn, LightGBM, SHAP, seaborn, scipy
