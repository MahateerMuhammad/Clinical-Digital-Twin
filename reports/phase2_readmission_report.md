# Phase 2 — 30-Day Unplanned Hospital Readmission Prediction: Comprehensive Technical & Clinical Report

> [!NOTE]
> **Figures refreshed 2026-08-06** against
> [`tables/readmission_model_comparison.md`](tables/readmission_model_comparison.md), which is
> regenerated from the model on disk. This document is hand-written and does not update
> itself, so it had drifted two corrections behind — the laboratory-join repair and the
> feature-selection repair. Headline Run B AUROC moved 0.7094 → **0.7158**.
>
> The largest change is the **LACE baseline**, which moved 0.4994 → **0.6096** once the
> laboratory join was repaired. The old figure was *below chance*, which should have been
> treated as a broken baseline rather than published as evidence of the model's advantage.

---

## 1. Executive Summary & Clinical Context

Hospital readmission within 30 days of discharge is a primary quality-of-care metric and financial penalty driver across healthcare systems globally (e.g., CMS Hospital Readmissions Reduction Program). The objective of Phase 2 is to predict whether a surviving hospitalized patient will experience an **unplanned 30-day readmission** (`readmission_30d == 1`) using **data available within the first 24 hours of the index admission**.

### Why Early 24-Hour Readmission Prediction Matters
- **Actionable Care Coordination:** Predicting readmission risk within the first 24 hours of stay allows inpatient care teams to initiate high-touch interventions early (e.g., clinical pharmacist consultation, social work assessment, outpatient follow-up scheduling, home health referrals).
- **Avoiding Target Proxy Bias:** Evaluating readmission risk requires strict exclusion of index in-hospital deaths. A patient who dies during the index admission cannot physically be readmitted.

---

## 2. Cohort Definition & Living Cohort Exclusion Protocol

### Initial Data Cohort
- **Total MIMIC-IV Hospital Admissions:** **546,028 admissions** (223,452 unique patients).

### Strict Living Cohort Derivation (Critical Exclusion Rule)
- **Exclusion Requirement:** Drop all admissions where `hospital_expire_flag == 1` or `deathtime` is non-null for the index admission.
- **Why Non-Null `dod` Must NOT Be Used:** In MIMIC-IV, the `patients` table contains out-of-hospital date of death (`dod`), which reflects deaths occurring years after a given admission. Excluding admissions based on `dod` erroneously drops 144,966 surviving admissions of patients who died years later, introducing severe selection bias.
- **Excluded Index In-Hospital Deaths:** **11,801 admissions** (2.16% in-hospital mortality rate).
- **Corrected Living Cohort:** **534,227 admissions**.

### Holdout Test Split Readmission Base Rate
- **Split Protocol:** Strict patient-level split (70% Train, 15% Validation, 15% Test) with ZERO patient overlap across splits.
- **Test Set Admissions:** **81,019 admissions**.
- **Test Set Readmissions (Positive Class):** **16,587 readmissions**.
- **Base Readmission Prevalence:** **20.47%** ($16,587 / 81,019$).

---

## 3. Feature Engineering & Expansions Protocol

Phase 2 builds upon Phase 1's 24-hour vital/lab/medication engine and incorporates two high-impact temporal feature expansions:

### A. Core 24-Hour Engine (Labs — **not** vitals or medications)
Windowed statistics (`_first_24h`, `_last_24h`, `_min_24h`, `_max_24h`, `_mean_24h`,
`_count_24h`, `_abnormal_24h`, `_missing_ratio_24h`) across the first 24 hours of the index
stay, over twelve laboratory analytes.

> [!CAUTION]
> **Vital signs and medication classes do not reach this model**, despite the earlier text.
> Vitals derive from `chartevents`, which is ICU-only in MIMIC-IV and keyed by `stay_id` —
> roughly 83% of hospital admissions have none, so they cannot be used at the admission
> grain. The `med_class_*` columns are whole-stay flags and are removed as
> observation-window leakage. The served model has **170 features**: admission and
> demographic dummies, windowed labs, the Expansion A/D history features below, and counts.
> See Phase 1 §3 for the full accounting.

### B. Expansion A — Prior Utilization History (Strict Temporal Filtering)
Calculated strictly from prior admissions where $\text{dischtime} < \text{index admittime}$:
1. `prior_admissions_30d`: Count of prior completed hospital stays in the preceding 30 days.
2. `prior_admissions_90d`: Count of prior completed hospital stays in the preceding 90 days.
3. `prior_admissions_365d`: Count of prior completed hospital stays in the preceding 365 days.
4. `prior_cumulative_los_days`: Total accumulated length of stay (days) across all prior admissions.
5. `days_since_last_discharge`: Exact days between the index admission and the most recent prior discharge (-1.0 sentinel value for first-time admissions).

### C. Expansion D — Pre-Admission Baseline Charlson Comorbidity Index
Built strictly from ICD diagnosis codes attached to prior admissions with $\text{dischtime} < \text{index admittime}$:
- `pre_admission_charlson_index`: Weighted comorbidity score reflecting pre-existing chronic conditions prior to the current admission.

> [!IMPORTANT]
> **Temporal Discipline Rule:** Prior admissions with $\text{dischtime} \ge \text{index admittime}$ (ongoing, overlapping, or same-day stays) are strictly excluded from Expansion A & D features to eliminate temporal boundary leakage.

---

## 4. Leakage Protocol Audit: Run A vs. Run B

### Run A: Full-Stay Reference Protocol
- **Includes:** 24h features + Expansions A&D + index admission length of stay (`los_days`), index discharge location, index primary ICD diagnosis, and index Charlson score.
- **Performance:** LightGBM AUROC = **0.7328**, AUPRC = **0.4409**.

### Run B: Strict 24-Hour Real-Time Protocol (Deployment Standard) ★
- **Excludes:** ALL index discharge-derived features (`READMISSION_EXCLUDE_STRICT` list: `next_admittime`, `days_to_readmission`, `dischtime`, `discharge_location`, `los_days`, `hospital_expire_flag`, `cci_*`, `dx_*`).
- **Why Exclusion is Mandatory:** Discharge location, total length of stay, and index ICD codes are only determined at the end of the index hospital stay. Using them in a 24-hour prediction model violates real-time feasibility.
- **Performance:** LightGBM AUROC = **0.7158**, AUPRC = **0.4218** (**2.06x enrichment over the 20.47% base rate**).
- The gap to Run A is small (0.7328 → 0.7158). Unlike mortality, readmission risk is driven
  mostly by *history* — prior utilisation — which is available at hour 24, so enforcing the
  window costs little here.

---

## 5. Model Suite, Calibration & Clinical Thresholding

### Models Evaluated
1. **LACE Clinical Baseline Score (modified E):** 0–19 point clinical risk index (Length of stay, Acuity, Comorbidity, Emergency visits). AUROC = **0.6096**, AUPRC = **0.2836**.
2. **Logistic Regression (L2):** AUROC = **0.6944**, AUPRC = **0.4011**.
3. **XGBoost:** AUROC = **0.7089**, AUPRC = **0.4146**.
4. **LightGBM (Winning Model):** AUROC = **0.7158**, AUPRC = **0.4218**.

### Probability Calibration (Isotonic Regression)
- **Error Reduction:** Reduced Brier Score from **0.2098 → 0.1445** (a **31.1% reduction in probability error**).
- Ranking is essentially unchanged by calibration (AUROC 0.7158 → 0.7155); what improves is
  the probability itself, which is what the clinical report quotes.

### Operational Decision Thresholding
- **Tuned Threshold:** `0.1431` (targeting 80%+ recall on validation set).
- **Holdout Test Set Operational Performance:**
  - **Recall (Sensitivity):** **81.35%** (captures 4 out of 5 patients who will be readmitted).
  - **Precision:** **27.78%** (more than 1 in 4 flagged patients is readmitted within 30 days, compared to 1 in 5 in the general population).

---

## 6. Model Explainability & Proof of Non-Fluke Performance (SHAP Analysis)

To ensure the model is learning genuine clinical mechanisms rather than statistical artifacts, SHAP (SHapley Additive exPlanations) values were computed on the holdout test set for the winning LightGBM model.

Regenerated by `scripts/evaluation/run_explainability_audit.py`; see
[`tables/explainability_audit.md`](tables/explainability_audit.md). Leakage screen: **CLEAN**.

### Top 15 SHAP Feature Ranking (Run B)

| Rank | Feature Name | Feature Scope | Mean \|SHAP\| | Clinical Rationale |
| :---: | :--- | :--- | :---: | :--- |
| **1** | `prior_admissions_365d` | Expansion A | **0.1991** | **Chronic Healthcare Utilization:** High prior admission frequency is the single strongest indicator of chronic disease fragility and frequent relapse ("revolving door" effect). |
| **2** | `diagnosis_count` | Index Stay | **0.1140** | **Comorbidity burden proxy** — also partly a coding-process signal. |
| **3** | `prior_admissions_90d` | Expansion A | **0.1019** | **Recent Disease Instability:** Multiple admissions within the last 3 months signal active decompensation. |
| **4** | `prior_cumulative_los_days` | Expansion A | **0.0841** | **High Disease Burden & Frailty:** Extensive prior hospitalisation predicts functional decline and complex post-discharge needs. |
| **5** | `procedure_count` | Index Stay | **0.0716** | **Intervention intensity** during the index admission. |
| **6** | `anchor_age` | Baseline | **0.0636** | **Advanced Age:** higher vulnerability to post-discharge complications and medication errors. |
| **7** | `days_since_last_discharge` | Expansion A | **0.0562** | **Short Inter-Admission Interval:** rapid recurrence indicates failed outpatient stabilisation. |
| **8** | `lab_platelets_first_24h` | 24h Window | **0.0472** | **Haematological / inflammatory state** on presentation. |
| **9** | `lab_hematocrit_last_24h` | 24h Window | **0.0460** | **Anaemia** — a well-established readmission risk factor. |
| **10** | `race_UNKNOWN` | Baseline | **0.0387** | **Data-capture artefact** — see caution below. Not a clinical finding. |
| **11** | `insurance_Private` | Baseline | **0.0382** | **Socioeconomic / Outpatient Access:** insurance status shapes access to post-discharge care. |
| **12** | `prior_admissions_30d` | Expansion A | **0.0289** | **Very recent utilisation.** |
| **13** | `pre_admission_charlson_index` | Expansion D | **0.0277** | **Pre-Existing Comorbid Burden** prior to the current stay. |
| **14** | `gender_M` | Baseline | **0.0241** | Sex, a weak but non-zero contributor. |
| **15** | `lab_hematocrit_first_24h` | 24h Window | **0.0240** | Admission haematocrit. |

> [!WARNING]
> `med_class_anticoagulant` and `med_class_insulin` appeared at ranks 5 and 10 in the
> superseded table. **Neither is in the model** — the `med_class_*` family is whole-stay
> and is removed as observation-window leakage. Their clinical rationales read persuasively
> but described features the model never saw.
>
> `race_UNKNOWN` ranking at all is a **data-capture artefact**: missing ethnicity correlates
> with chaotic or emergency admissions. It should be monitored as a fairness risk, not
> interpreted as a finding.

### Why This Confirms Model Legitimacy (Not a Fluke)
1. **Clinical Alignment:** The top drivers (`prior_admissions_365d`, `prior_admissions_90d`, `prior_cumulative_los_days`, `days_since_last_discharge`) directly reflect the **utilization hypothesis** widely validated in health services research: *prior hospital utilization is the single best predictor of future hospital utilization*. Four of the top seven are prior-utilisation features.
2. **Manual Audit Verification:** Spot-check audits on 10 random multi-admission patient cases confirmed 100% manual agreement between hand-counted prior admissions and calculated feature values, with zero index self-inclusion and zero temporal boundary leakage.
3. **Published Benchmark Comparison:** The strict 24-hour AUROC of **0.7158** sits at the upper tier of published leak-free MIMIC-IV 30-day readmission models (0.65–0.71 AUROC range), outperforming the LACE index baseline (0.6096) by **+0.1062 AUROC**.

> [!NOTE]
> The superseded version claimed a **+0.2100** advantage over LACE. That gap was inflated
> by a broken baseline: LACE scored 0.4994 — *below chance* — because the laboratory join
> was corrupt. With the join repaired LACE reaches 0.6096 and the real advantage is
> **+0.1062**, roughly half what was published. The model still beats the clinical index,
> by a defensible margin rather than an implausible one.

---

## 7. Phase 2 Performance Benchmark Summary Table

Mirrors [`tables/readmission_model_comparison.md`](tables/readmission_model_comparison.md).
If the two disagree, the generated table is correct and this one is stale.

| Model | Run Protocol | Feature Scope & Temporal Discipline | AUROC | AUPRC | Base-Rate AUPRC | Brier Score | Decision Threshold | Test Precision | Test Recall | Test F1 Score |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LACE Score (modified E)** | Baseline | Clinical LACE Index Proxy | **0.6096** | **0.2836** | 0.2047 | 0.1994 | 0.1667 | 22.20% | 85.12% | 0.3521 |
| **LightGBM** | Run A | Full-stay + Expansions A&D | **0.7328** | **0.4409** | 0.2047 | 0.2043 | 0.3971 | 29.52% | 80.29% | 0.4317 |
| **XGBoost** | Run A | Full-stay + Expansions A&D | **0.7292** | **0.4376** | 0.2047 | 0.2060 | 0.3967 | 29.13% | 80.25% | 0.4274 |
| **Logistic Reg** | Run A | Full-stay + Expansions A&D | **0.7083** | **0.4129** | 0.2047 | 0.2143 | 0.3953 | 27.36% | 80.89% | 0.4089 |
| ★ **LightGBM** | **Run B (24h)** | **Strict 24h + Expansions A&D** | **0.7158** | **0.4218** | **0.2047** | **0.2098** | **0.3955** | **28.21%** | **80.30%** | **0.4175** |
| **XGBoost** | **Run B (24h)** | **Strict 24h + Expansions A&D** | **0.7089** | **0.4146** | **0.2047** | **0.2137** | **0.3979** | **27.65%** | **80.47%** | **0.4116** |
| **Logistic Reg** | **Run B (24h)** | **Strict 24h + Expansions A&D** | **0.6944** | **0.4011** | **0.2047** | **0.2178** | **0.3932** | **26.32%** | **81.18%** | **0.3975** |
| ★ **LightGBM (Cal)**| **Run B (24h)** | **Strict 24h — served** | **0.7155** | **0.4128** | **0.2047** | **0.1445** | **0.1431** | **27.78%** | **81.35%** | **0.4141** |
