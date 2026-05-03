# Mortgage Approval Prediction & Fair Lending Audit

**Built a fair lending audit model on 196K+ CFPB mortgage applications - predicted approval decisions and quantified demographic approval gaps using SHAP and disparate impact analysis, translating regulatory compliance requirements into actionable business insights.**

---

## Background

HMDA (Home Mortgage Disclosure Act) data is mandatory regulatory disclosure. Every U.S. lender above an asset threshold must submit it to the CFPB annually. Banks use it internally for CRA compliance, fair lending audits, and underwriting strategy reviews.

This project treats it as a real compliance problem: build a model that predicts approval outcomes, audit it for racial disparate impact, and quantify the fairness tradeoff at different decision thresholds.

---

## Dataset

- **Source:** [CFPB HMDA Data Browser](https://ffiec.cfpb.gov/data-browser/) - California, 2023
- **Raw file:** 962,073 rows x 99 columns
- **Scope filter:** Home purchase, conventional, first lien, originated or denied only
- **After filtering + cleaning:** 196,113 applications
- **Class split:** 87.8% approved / 12.2% denied

---

## Project Structure

```
notebooks/
  01_eda.ipynb         # Data exploration - problem discovery - cleaning - validation - EDA
  02_modeling.ipynb    # Feature engineering - model comparison - SHAP - error analysis - fair model

data/
  state_CA.csv         # Raw HMDA file (not committed - see .gitignore)
  hmda_filtered.parquet  # Processed output of notebook 1

outputs/               # All charts saved here by the notebooks
```

---

## Methodology

### Iterative Approach
The project follows a proper iterative cycle - not a single straight pass:

1. **Raw exploration** - understand structure, all 99 columns, target distribution
2. **Problem discovery** - 10 documented data quality issues (mixed units, MNAR, leakage)
3. **Preprocessing** - explicit decision for every problem found
4. **Validation** - before/after distributions confirming each step worked
5. **EDA** - fairness analysis on clean data
6. **Modeling** - three models compared, cross-validated, error analysis
7. **Fair model audit** - two-model comparison with and without demographic features

### Key Data Quality Findings

| Problem | Finding | Decision |
|---|---|---|
| Mixed units | income in $K, loan_amount in $ | Converted to $K for ratios |
| Negative income | 83 rows (95% denied - data error) | Dropped |
| Income outliers | Max $118M/yr; P99.5 = $2,265K | Capped at P99.5 |
| LTV outliers | Max LTV 29,167% - data error | Capped at 200 |
| DTI format | Mixed string buckets + integers + "Exempt" | Mapped to numeric midpoints |
| DTI MNAR | 7.95% missing for denied vs 5.88% for approved | Created `dti_missing_flag` |
| Data leakage | `rate_spread` + `interest_rate`: 0% populated for denied loans | Dropped before any modeling |

The leakage finding is critical: an early model *including* `rate_spread` achieved AUC = 0.9996. After removing it, the honest AUC is **0.8490**.

---

## EDA - Fair Lending Findings

### Denial Rate by Race

| Race | n | Denial Rate | AIR vs White |
|---|---|---|---|
| Native Hawaiian / Pacific Islander | 524 | 20.8% | 0.898 |
| **Black or African American** | 4,802 | **20.2%** | 0.905 |
| American Indian / Alaska Native | 1,094 | 19.5% | 0.913 |
| 2+ minority races | 474 | 18.4% | 0.926 |
| White | 93,419 | 11.8% | 1.000 |
| Asian | 44,204 | 9.6% | 1.025 |

**Adverse Impact Ratio (AIR):** approval rate of group / approval rate of White applicants. AIR < 0.80 triggers legal scrutiny under the 80% rule.

No group falls below 0.80 in the raw data - but Black applicants face 1.7x the denial rate of White applicants.

### Denial Rate by Age Group

| Age Group | n | Denial Rate |
|---|---|---|
| <25 | 4,522 | 20.2% |
| >74 | 3,587 | 15.5% |
| 55-64 | 24,118 | 14.6% |
| 35-44 | 61,592 | 11.0% |
| 25-34 | 53,860 | 10.5% |

Youngest applicants face denial rates nearly double the 25-34 group, consistent with shorter credit histories. Oldest applicants (>74) also face elevated rates, likely reflecting fixed retirement income scoring worse on DTI.

### Income Does Not Explain the Racial Gap
Denied Black applicants had **higher median income** than denied White applicants. Within the denied population, Black applicants are on average better-off financially than their White counterparts - suggesting income alone does not explain the disparity.

### Statistical Significance
- Chi-square (race vs approval): chi2 = 1,199, **p = 1.6x10^-253**
- Mann-Whitney U (income): p approx 0 - approved applicants earn more
- Median DTI: Approved = 41.0% | Denied = 49.0%

### Top Denial Reasons (Lender-Stated)
1. Debt-to-income ratio - 38% of denials
2. Credit history - 14%
3. Collateral - 12%

---

## Feature Engineering (11 Features)

| Feature | Logic |
|---|---|
| `loan_to_income_ratio` | loan_amount / income - affordability normalized for loan size |
| `dti_missing_flag` | Binary: DTI null = MNAR signal of denial risk |
| `high_dti_flag` | DTI > 43% - regulatory QM threshold |
| `high_ltv_flag` | LTV > 80% - PMI threshold |
| `low_income_tract_flag` | Tract income < 80% of MSA median |
| `minority_tract_flag` | Tract minority population > 50% |
| `non_conforming_flag` | Jumbo loan indicator |
| `co_applicant_present` | Joint application flag |
| `county_approval_rate` | County-level mean approval rate (target encoding) |
| `non_primary_flag` | Investment / second home indicator |
| `applicant_age_enc` | Ordinal encoding of HMDA age bands (<25 to >74) |

---

## Model Comparison

All models use 80/20 stratified train/test split. LightGBM additionally cross-validated with 5-fold CV.

| Model | Test AUC | F1 | Balanced Accuracy |
|---|---|---|---|
| Decision Tree (depth=5) | 0.7978 | 0.9528 | 0.6836 |
| Random Forest (n=100) | 0.8300 | 0.9534 | 0.7142 |
| **LightGBM** | **0.8490** | **0.9581** | **0.7255** |

**LightGBM 5-fold CV AUC: 0.8401 +/- 0.0032** - consistent with test score, confirming no overfitting.

Decision tree depth sweep showed severe overfitting at unlimited depth (CV AUC = 0.69 vs depth=7 CV AUC = 0.80), confirming the signal is real but not overwhelming.

---

## SHAP Feature Importance (LightGBM)

| Rank | Feature | Mean |SHAP| |
|---|---|---|
| 1 | `dti_numeric` | 0.582 |
| 2 | `property_value_k` | 0.325 |
| 3 | `loan_amount_k` | 0.207 |
| 4 | `loan_term` | 0.185 |
| 5 | `county_approval_rate` | 0.127 |
| 6 | `loan_to_income_ratio` | 0.108 |
| 7 | `loan_to_value_ratio` | 0.107 |
| 8 | `derived_ethnicity_enc` | **0.094** |
| 9 | `income` | 0.092 |
| 11 | `derived_race_enc` | 0.082 |

**Finding:** Both ethnicity and race appear in the top SHAP features. The model is using demographic variables as predictors - reflecting structural patterns in the training data. This is why removing them in the fair model is meaningful.

---

## Error Analysis

At the optimal threshold (0.85):

| Error Type | Count | Rate |
|---|---|---|
| False Negative (wrongly denied) | 412 | 1.05% |
| False Positive (wrongly approved) | 2,565 | 6.54% |

### False Negative Rate by Race

| Race | FN Rate | Relative to White |
|---|---|---|
| **Black or African American** | **3.52%** | **2.8x** |
| American Indian / Alaska Native | 1.94% | 1.6x |
| White | 1.24% | 1.0x (reference) |
| Asian | 0.89% | 0.7x |

Among applicants who should genuinely be approved, Black applicants are 2.8x more likely to be wrongly denied by the full model.

---

## Fair Lending Audit - Model With vs Without Demographics

A second LightGBM was trained using only financial and geographic features (race, ethnicity, sex removed). This simulates what a legally compliant production model would look like.

| Metric | Full Model | Fair Model | Difference |
|---|---|---|---|
| Test AUC | 0.8490 | 0.8458 | -0.003 |
| CV AUC | 0.8401 | 0.8384 | -0.002 |
| Balanced Accuracy | 0.7255 | 0.7296 | +0.004 |

**AUC cost of removing demographics: 0.003 - essentially nothing.**

### AIR Comparison: Full vs Fair Model

| Race | Full AIR | Fair AIR | Change |
|---|---|---|---|
| Black or African American | 0.934 | **0.962** | +0.027 |
| American Indian / Alaska Native | 0.929 | **0.938** | +0.009 |
| Native Hawaiian / Pacific Islander | 0.858 | **0.867** | +0.009 |
| 2+ minority races | 0.964 | **0.974** | +0.010 |

Every minority group has a higher AIR when demographics are removed. Removing demographic inputs makes the model more equitable across all groups at almost zero performance cost.

### False Negative Rate: Full vs Fair Model

| Race | Full FN Rate | Fair FN Rate | Change |
|---|---|---|---|
| **Black or African American** | **3.52%** | **2.17%** | **-1.35pp** |
| American Indian / Alaska Native | 1.94% | 1.29% | -0.65pp |
| White | 1.24% | 1.90% | +0.66pp |
| Asian | 0.89% | 0.64% | -0.25pp |

The wrongful denial rate for Black applicants drops from 3.52% to 2.17% (38% reduction) when demographics are excluded. The fair model shifts errors slightly toward White applicants but substantially reduces harm to minority applicants.

---

## Limitations

1. **No credit score** - HMDA does not include credit scores, the most important underwriting variable. Some of the observed racial gap may be a proxy for credit score differences.
2. **Race in full model** - Using race as a model input would be illegal in production. The fair model (Section above) is the deployment-appropriate version.
3. **County encoding** - Mean encoding computed on the full dataset; production use would require cross-validated encoding.

---

## Resume Bullets

> **Built a LightGBM mortgage approval model on 196K CFPB HMDA applications with 12.2% denial rate class imbalance, achieving AUC = 0.8490 (CV: 0.8401 +/- 0.0032) after engineering 11 domain features including DTI missing-data flags, age group bands, and neighborhood income indicators - outperforming Random Forest by 1.9 AUC points and Decision Tree by 5.1 points.**

> **Identified and removed data leakage from 8 post-origination HMDA fields (rate_spread, interest_rate, etc.) that were 0% populated for denied loans - correcting inflated AUC from 0.9996 to an honest 0.8490.**

> **Conducted a full fair lending audit: built a demographically-blind model (race/ethnicity/sex removed) that achieves AIR = 0.962 for Black applicants vs 0.934 in the full model, reducing wrongful denials for Black applicants by 38% - at a cost of only 0.003 AUC points, demonstrating fair and accurate are not in conflict for this dataset.**

---

## Tech Stack

Python - pandas - scikit-learn - LightGBM - SHAP - seaborn - matplotlib - scipy
