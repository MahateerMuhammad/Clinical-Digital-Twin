# Phase 4 — Two-Stage Length of Stay (LOS) Prediction: Technical & Clinical Report

> [!NOTE]
> **Figures refreshed 2026-08-06** against
> [`tables/los_two_stage_results.md`](tables/los_two_stage_results.md), which is regenerated
> from the models on disk. This document is hand-written and does not update itself, so it
> had drifted two corrections behind — the laboratory-join repair and the feature-selection
> repair.
>
> Stage A hospital LOS moved 0.8114 → **0.9001** AUROC and 0.5434 → **0.7576** AUPRC, and
> Stage B regression improved materially as well (deployment MAE 1.36 → **1.05** days,
> $R^2$ 0.1162 → **0.2563**). ICU LOS Stage A moved 0.6381 → **0.8527**. Feature count
> moved 110 → **170**.

---

## 1. Executive Summary & Literature Rationale

Predicting Hospital Length of Stay (`los_days`) and Intensive Care Unit Length of Stay (`icu_los_days`) at hospital admission ($t = 0$) is essential for hospital resource management, bed capacity planning, and triage optimization.

### Why a Two-Stage Framework is Mandatory
Multiple MIMIC-IV LOS studies have consistently demonstrated that direct regression across the full LOS spectrum using early admission-time features performs poorly ($R^2 \approx 0$). This occurs because hospital length of stay exhibits a **heavy right-tailed distribution**—a small fraction of patients stay for multiple weeks or months due to acute complications, secondary infections, or placement delays. Early presentation features available at admission cannot reliably predict these extreme long-stay outliers.

To resolve this, we implement the literature-recommended **Two-Stage Framework**:
1. **Stage A (Short vs. Long Stay Classification):** Classifies whether a patient will have a normal/short stay vs. an extra-long stay using empirical data-driven 75th percentile thresholds.
2. **Stage B (Short-Bucket Duration Regression):** Trains specialized regressors strictly on admissions within the "short" stay bucket ($ \le 75\text{th percentile threshold} $), providing exact duration estimates for short stays while explicitly flagging long stays for clinical review without generating misleading exact-day estimates for extreme outliers.

---

## 2. Cohort Definition, Thresholds & Data Splitting

### Cohort Statistics & Empirical 75th Percentile Thresholds
All thresholds are derived strictly from the **training split** ($70\%$) to prevent threshold data leakage:

*   **Hospital LOS Cohort (`los_days`):**
    *   **Total Admissions:** **546,028 admissions** across **223,452 unique patients**.
    *   **Distribution:** Median = $2.82$ days, 75th Percentile Threshold = **$5.63$ days** ($5.6306$ days), Max = $515.56$ days.
    *   **Prevalence of Long Stay ($> 5.63$ days):** $25.00\%$ in Train ($95,347 / 381,403$), $24.72\%$ in Val ($20,224 / 81,819$), **$24.86\%$ in Test ($20,585 / 82,806$)**.

*   **ICU LOS Cohort (`icu_los_days`):**
    *   **Evaluated Cohort:** Filtered strictly to positive ICU admissions (**85,229 admissions** across **65,355 unique patients**).
    *   **Distribution:** Median = $2.05$ days, 75th Percentile Threshold = **$4.18$ days** ($4.1795$ days), Max = $226.54$ days.
    *   **Prevalence of Long ICU Stay ($> 4.18$ days):** $25.00\%$ in Train ($14,939 / 59,756$), $24.23\%$ in Val ($3,052 / 12,597$), **$24.98\%$ in Test ($3,216 / 12,876$)**.

> [!NOTE]
> **Fixed Training Split Threshold Protocol:** The 75th percentile empirical cutoffs ($5.63$ days for hospital LOS, $4.18$ days for ICU LOS) were computed **once on the training split (`df_train`)** and applied as a fixed numerical scalar cutoff across validation and test splits (`(target > p75_threshold)`). Thresholds were **never recomputed separately per split**, eliminating threshold leakage.

> [!IMPORTANT]
> **Model Scope & Task Distinction:** The ICU LOS model (`icu_los_days`) is evaluated **strictly on admissions that already had an ICU stay** ($has\_icu\_stay == 1$, $N = 85,229$). This model predicts the duration of ICU stay *conditional on ICU admission*. It is structurally distinct from the **Phase 3 ICU Admission Risk Model**, which predicts the *binary likelihood of needing ICU admission* across all general hospital admissions ($N = 546,028$). The two models serve complementary clinical roles and must not be conflated.

### Patient-Level Splitting Protocol (Zero Data Overlap)
The cohort is strictly split by `subject_id` using the shared `patient_split.parquet` protocol (Train: $70\%$, Val: $15\%$, Test: $15\%$). Automated assertions confirmed **zero patient overlap** across splits.

---

## 3. Strict Leakage Audit & Protocol Evolution

> [!CAUTION]
> **Vital signs do not reach this model.** Stage A has **170 features** and **zero**
> `vital_*` or `news2_*` columns, verified against `feature_names_in_`.
>
> This is an absence, not one of the exclusions listed below. Vitals derive from
> `chartevents`, which is ICU-only in MIMIC-IV and keyed by `stay_id`; roughly 83% of
> hospital admissions have none, and none merges to the admission grain. The
> composition is admission and demographic dummies, 24-hour windowed labs,
> prior-utilisation history and counts. See Phase 1 §3 for the full accounting.
>
> The distinction matters for reading the exclusion list: 154 columns were dropped
> *deliberately* as leakage, and it would be easy to assume vitals were among them and
> could be recovered by relaxing the protocol. They could not — they were never there.

To maintain 100% real-time admission discipline, we applied `LOS_EXCLUDE_STRICT` (dropping 154 leakage columns):
1. **Outcome & Resolution Proxies Dropped:** `dischtime`, `discharge_location`, `deathtime`, `los_days`, `los_hours`, `dod`, `hospital_expire_flag`, `next_admittime`, `days_to_readmission`, `readmission_30d`, `icu_los_days`, `has_icu_stay`, `n_icu_stays`.
2. **Post-Hoc Diagnosis Codes Dropped:** `charlson_comorbidity_index`, `cci_*`, `dx_*`, `primary_icd_code`. Baseline comorbidities are dynamically computed from prior historical stays (`pre_admission_charlson_index`).
3. **Full-Stay Aggregates Dropped:** Excluded all medication/procedure counts, lab trajectory stats (`median`, `max`, `min`, `slope`, `last`, counts), care unit transfers (`first_careunit`, `last_careunit`), and clinical notes readability metrics. Retained 110 leak-free presentation predictor features.

---

## 4. Stage A — Short vs. Long Stay Classification Results

Stage A models (Logistic Regression, XGBoost, LightGBM) were trained with 3-fold `GroupKFold` cross-validation on `subject_id`. Isotonic probability calibration was applied to the winning LightGBM model.

### Test Set Performance ($N = 82,806$ Hospital LOS / $N = 12,876$ ICU LOS)

| Target | Model Name | Run Protocol | AUROC | AUPRC | Base Rate AUPRC | Brier Score | Decision Threshold | F1 Score | Precision | Recall |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hospital LOS** | Logistic Regression | Stage A (Admission) | 0.8873 | 0.7356 | 0.2486 | 0.1403 | 0.5089 | 0.6628 | 0.5641 | 0.8034 |
| **Hospital LOS** | XGBoost | Stage A (Admission) | 0.8973 | 0.7527 | 0.2486 | 0.1343 | 0.5349 | 0.6768 | 0.5877 | 0.7978 |
| ★ **Hospital LOS** | **LightGBM** | Stage A (Admission) | **0.9001** | **0.7576** | 0.2486 | 0.1323 | 0.5524 | **0.6845** | 0.5989 | 0.7985 |
| ★ **Hospital LOS** | **LightGBM (Calibrated)** | Stage A — served | **0.9001** | **0.7503** | 0.2486 | **0.1072** | 0.2852 | 0.6791 | 0.5775 | 0.8241 |
| **ICU LOS** | Logistic Regression | Stage A (Admission) | 0.8374 | 0.6689 | 0.2498 | 0.1618 | 0.3912 | 0.5895 | 0.4589 | 0.8237 |
| **ICU LOS** | XGBoost | Stage A (Admission) | 0.8527 | 0.6970 | 0.2498 | 0.1521 | 0.3999 | 0.6058 | 0.4781 | 0.8265 |
| **ICU LOS** | **LightGBM** | Stage A (Admission) | **0.8527** | **0.6962** | 0.2498 | 0.1489 | 0.3886 | **0.6095** | 0.4835 | 0.8243 |
| **ICU LOS** | **LightGBM (Calibrated)** | Stage A (Admission) | **0.8523** | **0.6830** | 0.2498 | **0.1239** | 0.1919 | 0.6037 | 0.4714 | 0.8392 |

### Classification Takeaways
*   **Hospital LOS Classification:** LightGBM achieves an AUROC of **0.9001** and AUPRC of **0.7576** (a **3.05x enrichment** over the $0.2486$ base rate).
*   **ICU LOS Classification:** LightGBM achieves an AUROC of **0.8527** and AUPRC of **0.6962** (a **2.79x enrichment** over the $0.2498$ base rate).
*   **Probability Calibration:** Isotonic regression reduces Hospital LOS Brier Score from **0.1323 → 0.1072** ($19.0\%$ error reduction) and ICU LOS Brier Score from **0.1489 → 0.1239** ($16.8\%$ error reduction), with no measurable ranking cost on hospital LOS.
*   **ICU LOS was the weakest model in the project and no longer is.** It moved from AUROC
    0.6381 — barely better than guessing — to **0.8527** once the laboratory join was
    repaired. The earlier figure reflected a cohort where roughly half of admissions had no
    labs, which hurt the ICU sub-cohort hardest because it is the most lab-dense.

---

## 5. Stage B — Duration Regression Performance (Short Bucket)

Stage B regressors (XGBoost `reg:squarederror` and LightGBM `regression`) were trained exclusively on short-stay admissions ($ \le 75\text{th percentile threshold} $). Performance is evaluated under two distinct protocols:
1.  **Primary Deployment Scenario:** Evaluated on admissions predicted as short stay by Stage A (`Stage A output == 0`), capturing realistic classification error propagation.
2.  **Optimistic Upper Bound:** Evaluated on actual short stay admissions (`actual_los <= threshold`).

| Target | Model Name | Evaluation Scope | Sample Size ($N$) | MAE (days) | RMSE (days) | $R^2$ Score |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Hospital LOS** | **LightGBM Regressor** | Predicted Short Bucket (Deployment Primary) | 53,429 | **1.0548** | **2.0101** | **0.2563** |
| **Hospital LOS** | **XGBoost Regressor** | Predicted Short Bucket (Deployment Primary) | 53,429 | **1.0607** | **2.0140** | **0.2534** |
| **Hospital LOS** | **LightGBM Regressor** | Actual Short Bucket (Optimistic Bound) | 62,221 | **0.7963** | **1.0365** | **0.5201** |
| **Hospital LOS** | **XGBoost Regressor** | Actual Short Bucket (Optimistic Bound) | 62,221 | **0.8017** | **1.0411** | **0.5158** |
| **ICU LOS** | **LightGBM Regressor** | Predicted Short Bucket (Deployment Primary) | 7,150 | **0.9669** | **1.7026** | **0.0170** |
| **ICU LOS** | **XGBoost Regressor** | Predicted Short Bucket (Deployment Primary) | 7,150 | **0.9676** | **1.7037** | **0.0157** |
| **ICU LOS** | **LightGBM Regressor** | Actual Short Bucket (Optimistic Bound) | 9,660 | **0.7298** | **0.8990** | **0.1460** |
| **ICU LOS** | **XGBoost Regressor** | Actual Short Bucket (Optimistic Bound) | 9,660 | **0.7299** | **0.8989** | **0.1461** |

### Regression Takeaways & Low $R^2$ Framing
*   **Hospital LOS Duration Accuracy:** Within the short stay bucket ($ \le 5.63$ days), LightGBM/XGBoost regressors predict hospital stay duration with a **Mean Absolute Error (MAE) of ~1.05 days** under the realistic deployment pipeline — improved from ~1.36 days.
*   **ICU LOS Duration Accuracy:** Within the short ICU stay bucket ($ \le 4.18$ days), actual short-stay duration is predicted with an **MAE of 0.73 days** (about 17 hours error).
*   **The deployment bucket is larger now** (44,580 → 53,429 admissions) because the stronger
    Stage A classifier routes more patients correctly, which is why deployment MAE improved
    even though the optimistic bound improved less.

> [!WARNING]
> **Explicit Stage B Framing Statement:** Even within the restricted short-stay bucket ($ \le 5.63$ days for hospital, $ \le 4.18$ days for ICU), Stage B regressors yield low coefficient of determination scores ($R^2 = 0.2563$ for hospital LOS, $R^2 = 0.0170$ for ICU LOS under primary deployment). Exact length of stay prediction at the second of hospital admission using early-presentation features remains inherently noisy and low-variance. **Stage B outputs must be read as a rough directional estimate rather than a precise forecast.** The primary clinical utility of this pipeline resides in Stage A binary risk stratification (identifying patients at high risk of prolonged hospitalization), while Stage B provides approximate duration bounds for short-stay planning.
>
> ICU LOS Stage B is the weakest component in the project. Its deployment $R^2$ has moved
> from **negative** (−0.0645, worse than predicting the mean) to **0.0170** — no longer
> harmful, but still close to no explanatory power. Duration of an ICU stay is not
> predictable from admission-time features on this cohort, and should be presented as a
> planning heuristic only.

> [!CAUTION]
> **This model is still not served from a presentation payload** — the only one of the
> five that is not. Measured on the held-out split, a payload supports AUROC **0.731**
> for Stage A hospital LOS against **0.900** from the full admission record: **57.9%**
> of the validated discrimination, below the 66.7% floor.
>
> This is a large improvement on the earlier figure of 0.489 — *below chance*, the
> model ranking patients backwards — which came from a payload schema that omitted
> the admission and prior-utilisation features. It is no longer backwards, and it is
> no longer far from the floor. It remains withheld because length of stay depends on
> discharge planning, placement and social circumstances that a presentation payload
> does not describe, and lowering the floor to admit it would defeat the floor's
> purpose. See
> [`tables/payload_fidelity_evaluation.md`](tables/payload_fidelity_evaluation.md).

---

## 6. Explainability & SHAP Feature Attribution

The top-10 SHAP features driving Stage A classification (Long vs. Short Stay) for LightGBM are:

Regenerated by `scripts/evaluation/run_explainability_audit.py`; see
[`tables/explainability_audit.md`](tables/explainability_audit.md). Leakage screen: **CLEAN**.

### Hospital LOS Stage A Top SHAP Features:
1.  `admission_type_EU OBSERVATION` (1.1676) — Emergency observation intake.
2.  `diagnosis_count` (0.9656) — Comorbidity burden and coding-process proxy.
3.  `procedure_count` (0.5230) — Intervention intensity.
4.  `admission_type_DIRECT OBSERVATION` (0.2600) — Direct observation route.
5.  `prior_cumulative_los_days` (0.1199) — Historical hospital utilization.
6.  `admit_hour` (0.1071) — Time of arrival.
7.  `days_since_last_discharge` (0.0948) — Recency of prior utilisation.
8.  `pre_admission_charlson_index` (0.0926) — Baseline chronic comorbidity.
9.  `lab_platelets_first_24h` (0.0524) — Presenting platelet count.
10. `admission_type_SURGICAL SAME DAY ADMISSION` (0.0507) — Planned surgical intake.
11. `lab_creatinine_first_24h` (0.0494) — Presenting renal function.
12. `admission_location_PROCEDURE SITE` (0.0493) — Procedural admission route.
13. `admission_location_PHYSICIAN REFERRAL` (0.0468) — Referral route.
14. `marital_status_MARRIED` (0.0420) — See caution below.
15. `marital_status_SINGLE` (0.0418) — See caution below.

Long stays are driven predominantly by **admission pathway** — the top four features are
all routing or workload counts — followed by prior utilisation history and, more weakly,
presenting labs.

> [!CAUTION]
> **This is an administrative model more than a physiological one, and it should be
> described as such.** The earlier version of this section said long stays are driven by
> "initial metabolic/renal lab derangements"; in the current model the strongest lab feature
> ranks 9th with roughly 1/22nd the influence of the top feature.
>
> **`marital_status_MARRIED` and `marital_status_SINGLE` appearing in the top 15 is a
> social-determinant signal, not a clinical one.** Discharge planning is materially harder
> for patients without a partner at home, so marital status genuinely predicts length of
> stay — but a model that lengthens its own prediction because a patient is single is
> encoding a placement constraint, not an illness. This should be monitored as an equity
> risk if Stage A output is ever used to prioritise beds or discharge resources.
