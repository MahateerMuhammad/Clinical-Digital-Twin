# Phase 1 — In-Hospital Mortality Prediction: Comprehensive Technical & Clinical Report

> [!NOTE]
> **Figures refreshed 2026-08-06** against
> [`tables/mortality_model_comparison.md`](tables/mortality_model_comparison.md), which is
> regenerated from the model on disk. This document is written by hand and does not
> update itself, so it had drifted two corrections behind: the laboratory-join repair
> (odd `hadm_id` admissions carried no labs at all) and the feature-selection repair
> (which restored creatinine, BUN and haematocrit, and closed a `lab_*_mean` full-stay
> leak). Headline Run C AUROC moved 0.9484 → **0.9442** and AUPRC 0.4554 → **0.3800**.
>
> The earlier text also named the strict 24-hour protocol "Run B". There are now three
> protocols and the strict window is **Run C**; the naming below is corrected throughout.

---

## 1. Executive Summary & Clinical Context

In-hospital mortality prediction is a core foundational task in critical care informatics and digital twin modeling. The objective of Phase 1 is to predict whether a hospitalized/ICU patient will die during their index hospital stay (`hospital_expire_flag == 1`) using **only data available within the first 24 hours of admission**.

### Why Real-Time 24-Hour Risk Stratification Matters
- **Early Triage & Resource Allocation:** Identifying high-risk deteriorating patients early allows clinical teams to escalate care, transfer patients to intensive care units (ICUs), or initiate continuous hemodynamic monitoring.
- **Actionable Clinical Windows:** Predictions made after discharge or late in a multi-week stay are clinically useless for acute intervention. A strict 24-hour observation window ensures predictions occur when clinical interventions can still change patient outcomes.
- **Benchmark Alignment:** Standard severity-of-illness scores (e.g., SAPS II, APACHE II, OASIS) use 24-hour physiological windows as their gold-standard evaluation protocol.

---

## 2. Cohort Definition, Filtering & Data Splitting

### Raw Data Cohort
- **Source Dataset:** MIMIC-IV v2.2 (Clinical Database of ICU and ED stays from Beth Israel Deaconess Medical Center).
- **Initial Hospital Admissions:** **546,028 admissions** representing **223,452 unique patients**.
- **In-Hospital Deaths:** **11,801 admissions** resulted in in-hospital death.
- **Base Mortality Rate:** **2.16%** ($11,801 / 546,028$).

### Strict Patient-Level Splitting Protocol (Zero Data Leakage)
A major vulnerability in clinical machine learning is **patient overlap across splits**. If a patient with 5 hospital admissions has 3 stays in the training set and 2 in the test set, models can memorize patient-specific baseline quirks (e.g., chronic baseline creatinine) rather than learning generalizable disease dynamics.

To prevent patient-level leakage:
- **Split Unit:** Unique Patient ID (`subject_id`).
- **Split Ratios:** 70% Training (`156,416` patients / `381,403` admissions), 15% Validation (`33,517` patients / `81,819` admissions), 15% Holdout Test (`33,519` patients / `82,806` admissions).
- **Verification:** Automated assertion verified **ZERO patient overlap** across Train, Validation, and Test sets.
- Admission counts are uneven because the split is on patients, and patients differ in how
  many admissions they have. That is the point: splitting on admissions would put the same
  patient on both sides.

---

## 3. Comprehensive Feature Engineering Protocol (First 24 Hours)

Features were engineered strictly from data recorded between `admittime` and `admittime + 24 hours`.

> [!CAUTION]
> **This section described the feature set as designed, not as built.** The served Run C
> model has **164 features**, composed as follows — verified against
> `booster_.feature_name()`:
>
> | Group | In the served model |
> | :--- | ---: |
> | Admission & demographic one-hot dummies | 81 |
> | Laboratory, 24-hour windowed | 69 |
> | Counts, timing and other | 14 |
> | **Vital signs** | **0** |
> | **Medication classes** | **0** |
>
> The subsections below are retained because they document the engineering *intent*, but
> two of the four groups never reach the model, for different and important reasons:
>
> - **Vital signs are absent from the dataset entirely.** They derive from `chartevents`,
>   which in MIMIC-IV is **ICU-only** and keyed by `stay_id`, not `hadm_id`.
>   `vitals_features.parquet` covers 94,442 ICU stays against 546,028 hospital admissions,
>   so roughly **83% of the cohort has no charted vitals at all** and none merges to the
>   admission grain. An admission-level model over the full cohort therefore cannot use
>   them. This is the single largest limitation of Phase 1 and was previously unstated.
> - **Medication classes exist but are excluded by Run C.** The seven `med_class_*` columns
>   are whole-stay flags — they record what the patient received across the entire
>   admission, including after hour 24 — so they are observation-window leakage and are
>   removed by `MORTALITY_EXCLUDE_RUN_C`.
>
> Neither absence is a bug. Both change what the model is, and the earlier text implied a
> physiologically rich model that does not exist.

### A. Demographics & Admission Baseline — **in the model** (as 81 one-hot dummies)
- `anchor_age`: Patient age at admission (clipped at 91 per MIMIC de-identification standards).
- `gender`: Binary indicator (Male / Female).
- `admission_type`: Emergency, Urgent, Elective, EU Observation, Direct Admit.
- `admission_location`: Emergency Room, Transfer from Hospital, Physician Referral, Clinic.
- `insurance`: Medicare, Medicaid, Other/Private.
- `race`: Categorical mapping of self-reported ethnicity.

### B. Physiological Vital Signs — **NOT in the model** (ICU-only source; see caution above)
Six key vital signs were tracked across the first 24 hours: **Heart Rate, Systolic Blood Pressure (SBP), Diastolic Blood Pressure (DBP), Oxygen Saturation (SpO2), Respiratory Rate, and Temperature**.
These are built in `data/interim/features/vitals_features.parquet` and are used by
**Phase 5 (deterioration)**, whose cohort is ICU-adjacent. They are unavailable to Phase 1.
For each vital sign, 6 summary statistics were computed:
1. `min`: Lowest value recorded in 24 hours (captures severe hypotension, hypoxia, hypothermia).
2. `max`: Highest value recorded in 24 hours (captures severe tachycardia, hypertensive crisis, fever).
3. `mean`: Overall 24-hour baseline average.
4. `std`: Volatility / instability measure over 24 hours.
5. `slope`: Rate of change ($\Delta / \Delta t$) over the 24-hour window (captures acute trajectory/deterioration).
6. `first` / `last`: Entry vitals vs end-of-window vitals.

### C. Laboratory Test Panels — **in the model** (69 windowed features)
Twelve laboratory analytes survive selection: **Creatinine, BUN, White Blood Cell (WBC)
Count, Haemoglobin, Haematocrit, Platelets, Potassium, Sodium, Chloride, Glucose,
Bicarbonate, Anion Gap** — plus point-of-care and whole-blood assay variants
(`_wb`, `_poc`) where they are a distinct measurement.

**Lactate is not among them.** It was the single strongest driver in the superseded SHAP
table, but it does not survive the 24-hour window at usable coverage on this cohort — it is
drawn selectively, mostly in ICU, so it is missing for most ward admissions.

Aggregations are windowed: `_first_24h`, `_last_24h`, and where available `_min_24h`,
`_max_24h`, `_mean_24h`, plus `_count_24h`, `_abnormal_24h` and `_missing_ratio_24h`.
The full-stay forms (`lab_*_max`, `lab_*_mean` and the rest) are excluded — a
`lab_*_mean` computed over the whole admission was found leaking and closed in the
feature-selection repair.

### D. Early Medications & Interventions — **NOT in the model** (whole-stay leakage)
Prescriptions and administrations initiated within the first 24 hours were categorized into 8 high-acuity drug classes:
1. `med_class_vasopressor`: Norepinephrine, Epinephrine, Dopamine, Vasopressin (indicates septic/cardiogenic shock).
2. `med_class_inotrope`: Dobutamine, Milrinone (indicates acute heart failure).
3. `med_class_anticoagulant`: Heparin, Enoxaparin, Warfarin.
4. `med_class_opioid`: Morphine, Fentanyl, Hydromorphone.
5. `med_class_sedative`: Propofol, Midazolam, Lorazepam.
6. `med_class_antibiotic`: Vancomycin, Cefepime, Piperacillin/Tazobactam.
7. `med_class_insulin`: Regular, Glargine, Humalog.
8. `med_class_diuretic`: Furosemide, Bumetanide.

Seven of these classes are built into the dataset (`med_class_antibiotic`,
`med_class_anticoagulant`, `med_class_insulin`, `med_class_opioid`, `med_class_statin`,
`med_class_beta_blocker`, `med_class_ace_inhibitor`), but as **whole-stay** flags rather
than 24-hour ones — they record what was given at any point in the admission. Run C
therefore excludes them all. They remain available to the unseen-patient agent
(Phase 11), where the clinician states the current medication list explicitly and there is
no future to leak from.

---

## 4. Strict Leakage Protocol Audit: Run A vs. Run B vs. Run C

Three feature-inclusion protocols were implemented and compared. All figures are LightGBM
on the held-out test set.

### Run A: Full-Stay Reference Protocol (Post-Hoc / Retrospective Baseline)
- **Includes:** First 24h vitals/labs + post-hoc ICD-9/10 diagnosis codes, procedure codes, and full-stay Charlson Comorbidity Index (CCI).
- **Result:** AUROC = **0.9966**, AUPRC = **0.9048**.

### Run B: Leak-Free Full-Stay Protocol
- **Excludes:** ALL ICD diagnosis codes, procedure codes, primary ICD categories, and full-stay CCI scores (`MORTALITY_EXCLUDE` list).
- **Why Exclusion is Mandatory:** In hospital electronic health record systems (including MIMIC-IV), ICD diagnosis codes are generated by professional medical coders **after patient discharge or death**. Including ICD codes in a 24-hour prediction model introduces **severe retrospective data leakage**—the model is essentially "cheating" by reading diagnostic summaries produced at the end of the stay.
- **Result:** AUROC = **0.9915**, AUPRC = **0.8550**.
- **Still not deployable:** Run B removes the diagnosis codes but keeps aggregates computed
  over the *whole* stay, which are equally unavailable at hour 24.

### Run C: Strict 24-Hour Real-Time Protocol (Deployment Standard) ★
- **Excludes:** everything Run B excludes, **plus** every full-stay aggregate — the model
  sees only what existed by `admittime + 24h` (`MORTALITY_EXCLUDE_RUN_C`).
- **Result:** AUROC = **0.9442**, AUPRC = **0.3800** (17.6x enrichment over the 2.16% base
  mortality rate).

> [!IMPORTANT]
> **Leakage Audit Finding:** Run C is the sole valid, leak-free deployment model for
> Phase 1 and the figure quoted throughout the rest of the project. Run A and Run B are
> upper-bound references demonstrating the magnitude of diagnostic and full-stay leakage
> respectively. The gap between them is the honest cost of predicting in real time:
> AUPRC falls from 0.9048 to 0.3800 once the model may only use the first 24 hours.

---

## 4b. Laboratory Coverage — and What Missingness Itself Predicts

Only **73.4%** of held-out test admissions have at least one laboratory result inside the
24-hour window. The remaining 26.6% reach the model with every lab feature missing (NaN —
see Phase 1 §3 and `src/llm/feature_space.py`). Performance is therefore reported
stratified, because a single headline figure averages two quite different populations:

| Subgroup | Admissions | Deaths | Base rate | AUROC |
| :--- | ---: | ---: | ---: | ---: |
| **All test admissions** | 82,806 | 1,787 | 2.16% | **0.9442** |
| Has ≥1 lab in 24h | 60,783 | 1,581 | 2.60% | 0.9350 |
| **No labs in 24h** | 22,023 | 206 | **0.94%** | **0.9673** |

Two things follow, and the second matters more.

**The model is not degraded by missing labs.** AUROC is *higher* in the no-lab subgroup, so
NaN-filling is working as intended — the boosters route missing values down their learned
default branch rather than treating them as zero readings.

**But "no bloods were drawn" is itself a strong predictor.** The no-lab subgroup dies at
**0.94%** against 2.60% for the rest — barely a third the rate. That is not a physiological
finding; it is a care-process one. A clinician who orders no bloods in the first 24 hours
has already judged the patient to be low-acuity, and the model reads that judgement.

> [!CAUTION]
> This is a real limitation of the model and is easy to mistake for skill. Part of the
> headline 0.9442 comes from inferring acuity from **whether tests were ordered**, not from
> what they showed. Two consequences:
>
> - **It will not transfer cleanly.** A hospital that draws routine bloods on every
>   admission removes this signal entirely, and performance there will be lower than 0.9442
>   suggests.
> - **It is circular for decision support.** Using the model to decide who needs attention,
>   when part of its confidence comes from clinicians having already decided who needs
>   attention, risks confirming existing triage rather than improving it.
>
> The same mechanism appears in Phase 3 (`lab_total_count_24h`, `lab_unique_items_24h` rank
> 3rd and 11th) and was severe enough in Phase 5 to justify excluding those features from
> the promoted model entirely. Phase 1 does not use explicit count features, but
> missingness carries the same information implicitly.

---

## 5. Model Training, Calibration & Threshold Optimization

### Model Architectures Evaluated
1. **Logistic Regression (L2 Regularized):** Linear baseline with balanced class weights.
2. **XGBoost (Extreme Gradient Boosting):** Tree-based ensemble with `scale_pos_weight = 45.3` (ratio of negative to positive cases).
3. **LightGBM (Light Gradient Boosting Machine):** Fast histogram-based tree ensemble with `class_weight='balanced'`.

### Hyperparameter Tuning Strategy
Models were tuned using 3-Fold Patient `GroupKFold` cross-validation on the training set:
- **LightGBM Best Hyperparameters:** `num_leaves: 63`, `learning_rate: 0.03`, `n_estimators: 350`, `min_child_samples: 50`, `subsample: 0.8`, `colsample_bytree: 0.8`.

### Probability Calibration (Isotonic Regression)
Uncalibrated gradient boosted trees often output uncalibrated probability estimates due to class reweighting.
- **Method:** Isotonic Regression fitted on out-of-sample validation predictions.
- **Calibration Performance:** Reduced Brier Score from **0.0768 → 0.0164** (a **78.6% reduction in probability error**) on Run C.
- **Cost:** calibration trades a little ranking quality for far better probabilities —
  AUROC 0.9442 → 0.9438, AUPRC 0.3800 → 0.3608. The calibrated model is the one served,
  because the reports quote probabilities rather than ranks.

### Clinical Decision Thresholding
Standard 0.5 decision thresholds fail severely on highly imbalanced clinical tasks (2.16% base rate).
- **Target Strategy:** Decision threshold tuned on validation predictions to achieve **~80% sensitivity (Recall)**, ensuring 4 out of 5 deteriorating patients are flagged early for clinical intervention.
- **Calibrated Operational Threshold:** `0.0437` (84.44% Recall, 14.24% Precision, F1 = 0.2437).
- Precision is low by construction at this operating point: at a 2.16% base rate, flagging
  84% of deaths necessarily flags many survivors too. The threshold encodes a deliberate
  choice to miss few deaths rather than to raise few alerts.

---

## 6. Model Explainability Analysis (SHAP)

SHAP (SHapley Additive exPlanations) values were computed on the holdout test set for the
served **Run C** LightGBM model. Regenerated by
`scripts/evaluation/run_explainability_audit.py`; see
[`tables/explainability_audit.md`](tables/explainability_audit.md).

> [!WARNING]
> The table previously printed here listed `lab_lactate_max`, `vital_sp_o2_min`,
> `med_class_vasopressor` and seven more — **none of which the served model contains**.
> Those are full-stay aggregates and whole-stay drug flags, exactly the families Run C
> removes as observation-window leakage. It was a hand-written ranking for a model that
> no longer existed, and it read as confirmation that the model was clinically sound
> while describing a different model entirely.

### Top 15 SHAP Feature Drivers for In-Hospital Mortality (Run C)

| Rank | Feature Name | Mean \|SHAP\| | Clinical & Process Rationale |
| :---: | :--- | :---: | :--- |
| **1** | `diagnosis_count` | **0.8789** | **Comorbidity burden proxy.** The number of coded diagnoses; largely a process signal for how complex the admission is. |
| **2** | `anchor_age` | **0.6073** | **Baseline Physiological Reserve:** advanced age reduces resilience to acute stress. |
| **3** | `admission_type_EW EMER.` | **0.3612** | **Acuity of presentation:** emergency-ward arrival versus elective booking. |
| **4** | `procedure_count` | **0.3289** | **Intervention intensity** during the stay. |
| **5** | `admission_type_EU OBSERVATION` | **0.2749** | Observation pathway — a low-acuity admission route. |
| **6** | `lab_wbc_last_24h` | **0.2318** | **Severe infection / leukocytosis** at the end of the observation window. |
| **7** | `lab_creatinine_first_24h` | **0.2052** | **Acute Kidney Injury:** renal dysfunction on presentation. |
| **8** | `lab_platelets_first_24h` | **0.1505** | **Coagulopathy / DIC:** thrombocytopenia reflects systemic inflammation. |
| **9** | `admission_location_PHYSICIAN REFERRAL` | **0.1291** | Referral route — again a presentation-acuity signal. |
| **10** | `lab_bun_first_24h` | **0.1180** | **Renal function and volume status.** |
| **11** | `lab_anion_gap_last_24h` | **0.0963** | **Metabolic acidosis** — the lactate proxy available inside the window. |
| **12** | `admission_type_SURGICAL SAME DAY ADMISSION` | **0.0944** | Elective surgical pathway. |
| **13** | `admit_hour` | **0.0875** | Time-of-day of arrival; out-of-hours presentation correlates with acuity. |
| **14** | `lab_hematocrit_first_24h` | **0.0842** | **Anaemia / haemorrhage.** |
| **15** | `race_UNKNOWN` | **0.0687** | Recorded ethnicity missing — a data-capture artefact, flagged below. |

### How to read this ranking

The corrected ranking is **more honest and less flattering** than the one it replaces.
The strongest drivers are not physiology but *administrative* — diagnosis count,
procedure count, admission route. That is what a 24-hour model looks like once the
full-stay physiology it used to lean on is taken away: it falls back on how sick the
admission pathway implies the patient is.

Genuine physiology still appears and is clinically coherent — WBC, creatinine,
platelets, BUN, anion gap and haematocrit occupy six of the top fourteen, and all are
windowed (`_first_24h` / `_last_24h`) rather than full-stay. The automated leakage screen
reports **CLEAN**: no feature in the top 15 belongs to a removed leak family.

Two cautions worth stating plainly:
- `diagnosis_count` and `procedure_count` are counts available at admission, but they
  encode clinical process as much as patient state. They are not a mechanism.
- `race_UNKNOWN` ranking at all is a **data-capture artefact**, not a clinical finding.
  Missing ethnicity correlates with chaotic or emergency admissions. It should be
  watched as a fairness risk rather than interpreted.

---

## 7. Phase 1 Performance Benchmark Summary Table

Mirrors [`tables/mortality_model_comparison.md`](tables/mortality_model_comparison.md).
If the two disagree, the generated table is correct and this one is stale.

| Model | Run Protocol | Feature Scope | AUROC | AUPRC | Base-Rate AUPRC | Brier Score | Threshold | Test Precision | Test Recall | Test F1 Score |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | Run A | Full-Stay (ICD Included) | **0.9966** | **0.9048** | 0.0216 | 0.0108 | 0.9357 | 85.84% | 81.42% | 0.8357 |
| **XGBoost** | Run A | Full-Stay (ICD Included) | **0.9961** | **0.8940** | 0.0216 | 0.0167 | 0.9322 | 84.27% | 81.25% | 0.8274 |
| **Logistic Reg** | Run A | Full-Stay (ICD Included) | **0.9919** | **0.7876** | 0.0216 | 0.0292 | 0.9394 | 67.16% | 80.47% | 0.7322 |
| **LightGBM** | Run B | Leak-Free Full-Stay | **0.9915** | **0.8550** | 0.0216 | 0.0164 | 0.8773 | 77.65% | 81.09% | 0.7933 |
| **XGBoost** | Run B | Leak-Free Full-Stay | **0.9907** | **0.8428** | 0.0216 | 0.0221 | 0.8611 | 74.76% | 81.20% | 0.7784 |
| **Logistic Reg** | Run B | Leak-Free Full-Stay | **0.9827** | **0.7116** | 0.0216 | 0.0414 | 0.8855 | 54.45% | 81.53% | 0.6529 |
| ★ **LightGBM** | **Run C (24h)** | **Strict 24h** | **0.9442** | **0.3800** | **0.0216** | **0.0768** | **0.5743** | **16.02%** | **80.30%** | **0.2671** |
| **XGBoost** | **Run C (24h)** | **Strict 24h** | **0.9421** | **0.3654** | **0.0216** | **0.0853** | **0.6024** | **15.86%** | **80.19%** | **0.2648** |
| **Logistic Reg** | **Run C (24h)** | **Strict 24h** | **0.9296** | **0.2568** | **0.0216** | **0.1064** | **0.5852** | **13.56%** | **81.42%** | **0.2325** |
| ★ **LightGBM (Cal)** | **Run C (24h)** | **Strict 24h — served** | **0.9438** | **0.3608** | **0.0216** | **0.0164** | **0.0437** | **14.24%** | **84.44%** | **0.2437** |
