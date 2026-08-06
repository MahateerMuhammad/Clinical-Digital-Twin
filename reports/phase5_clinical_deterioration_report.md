# Phase 5 — Clinical Deterioration Prediction (Ward-to-ICU Escalation): Technical & Clinical Report

> [!CAUTION]
> **SUPERSEDED — Phase 5 was rebuilt on 2026-08-06 as a landmark analysis.**
>
> The design documented below has a windowing leak: features were windowed to
> `admittime + 24h` while the prediction cutoff was nominally `t_event − 6h`. On this
> cohort **12,236 of 31,282 positive cases (39%) transferred to ICU before hour 24**, so
> their feature window reached past the event and absorbed post-transfer ICU laboratory
> draws. Every headline figure in this document is inflated by an unquantified margin.
>
> **The current model is documented in
> [`tables/deterioration_landmark_results.md`](tables/deterioration_landmark_results.md).**
> It fixes the leak structurally rather than by filtering columns: a landmark T = 24h is
> fixed, only admissions still at risk at T enter the cohort, and the outcome is ICU
> transfer within 48h of T. Every patient is event-free when the feature window closes, so
> no feature can contain post-event data — **0 of the positives are affected**, against 39%
> before.
>
> | | Superseded (below) | Landmark (current) |
> | :--- | ---: | ---: |
> | AUROC | 0.8231 | **0.7679** |
> | AUPRC | 0.3739 | **0.1016** |
> | Base rate | 5.95% | 2.19% |
> | Enrichment (AUPRC ÷ base) | 6.28x | **4.64x** |
> | Positives leaking post-event data | 12,236 (39%) | **0** |
>
> The raw AUPRC drop overstates the change — base rate fell too, and AUPRC scales with it.
> Compared as enrichment the model is still weaker, which is the expected direction once
> post-event information is removed. Part of the loss is the leak going away and part is the
> landmark task being genuinely harder; the two cannot be separated from these numbers.
>
> This document is retained because §3's leakage audit is a good record of how the problem
> was found. Read §5–§7 as history, not as current performance.

---

## 1. Executive Summary & Clinical Context

Early identification of physiological deterioration in general ward patients is a cornerstone of modern patient safety and hospital early warning systems (EWS). The goal of **Phase 5** is to predict acute clinical deterioration requiring ward-to-ICU transfer with a **6-hour pre-transfer lead time window** ($t = t_{\text{event}} - 6\text{h}$) using only data available prior to the cutoff.

### The Clinical Tradeoff of Prediction Lead Windows
Published early-warning-system literature demonstrates a fundamental clinical tradeoff regarding prediction window lengths:
* **Shorter Windows (< 2–4 hours)**: Yield artificially high technical performance scores (AUROC > 0.95) because severe physiological instability (e.g., severe hypotension, tachypnea) is already manifest. However, they provide insufficient lead time for clinical teams to intervene, reverse deterioration, or arrange ICU bed transfers.
* **Longer Windows (> 12–24 hours)**: Provide ample lead time but suffer from meaningful physiological signal degradation and high false-alarm rates due to the unpredictable nature of acute clinical decompensation.
* **The 6-Hour Window Choice**: A **6-hour pre-transfer window** is chosen as a clinically actionable starting point. It offers clinicians a realistic lead time to mobilize Rapid Response Teams (RRT), run diagnostic panels, and initiate early resuscitation while maintaining strong predictive signal.

---

## 2. Derivable Proxy Event Definition & Limitations

### Primary Proxy Event
In observational EHR databases like MIMIC-IV, physiological deterioration is operationalized via the derivable proxy of **unplanned transfer from a general medical/surgical ward to an Intensive Care Unit (ICU)** where the ward stay prior to transfer exceeds 6 hours (`time_to_icu > 6.0 hours`).

### Explicit Clinical Limitations
1. **Uncaptured Ward Deterioration**: Patients who experience severe physiological deterioration on the ward but are not transferred to an ICU due to Comfort Measures Only (CMO) status, Do-Not-Resuscitate (DNR) directives, or ceiling-of-care limitations are misclassified as $0$.
2. **Direct Ward Mortality**: Sudden catastrophic events (e.g., fatal pulmonary embolism or cardiac arrest) resulting in immediate death on the general floor before ICU transfer can be arranged are uncaptured by the transfer proxy.
3. **ICU Bed Availability Noise**: Transfers to the ICU are confounded by hospital bed capacity; a severely deteriorating patient may be held on the ward longer during periods of ICU bed saturation.

---

## 3. Data Leakage Audit, Discovery & Distributional Audit

An exhaustive multi-stage audit was conducted to investigate initial baseline models that achieved an artificially inflated **0.9999 AUROC / 0.9999 AUPRC** and subsequent intermediate models sitting at **0.916 AUROC**.

### A. Audit 1: Feature Availability Leakage (`vital_*` Features)
* **Root Cause**: In MIMIC-IV, vital signs in `vitals_features.parquet` are derived from the ICU `chartevents` table, which exists **only for stays with an assigned `stay_id` (ICU stay)**.
* **Empirical Diagnostic**:
  - For **Target = 0** (Ward non-transfers): `vital_*` features were **100.00% missing (0 / 460,786 rows non-null)**.
  - For **Target = 1** (Ward-to-ICU transfers): `vital_*` features were **99.99% non-missing (28,759 / 28,761 rows non-null)**.
* **Resolution**: Completely excluded all ICU `chartevents`-derived vitals (`vital_*`) and calculated scores (`news2_*`).

### B. Audit 2: Distributional Leakage in Lab Counts & Order Frequency (`lab_anion_gap_count`, `lab_unique_items`)
While the missingness differential check caught table presence leaks, a second subtle leakage vector was discovered in **order frequency and full-stay count features**. Features like `lab_anion_gap_count` and `lab_unique_items` were present (non-null) for most patients in both classes, but their **numerical distribution** leaked stay duration and downstream clinician concern.

#### Empirical Distributional Check (Target = 0 vs Target = 1):

| Feature Name | Target = 0 Mean (Median) | Target = 1 Mean (Median) | Distributional Ratio | Leakage Mechanism |
| :--- | :---: | :---: | :---: | :--- |
| `lab_anion_gap_count` | 4.35 (3.0) | 17.30 (11.0) | **3.98x** | Full-Stay Lab Draw Accumulation |
| `lab_glucose_abnormal_count` | 2.75 (1.0) | 13.04 (7.0) | **4.74x** | Post-Hoc Abnormal Accumulation |
| `lab_wbc_abnormal_count` | 1.55 (0.0) | 8.51 (4.0) | **5.49x** | Post-Hoc Abnormal Accumulation |
| `lab_glucose_poc_missing_ratio` | 0.96 (1.0) | 0.45 (0.0) | **0.47x** | Full-Stay Missingness Ratio |
| `lab_unique_items` | 11.53 (12.0) | 15.23 (16.0) | **1.32x** | Workup Intensity Aggregate |
| `lab_potassium_wb_count` | 0.14 (0.0) | 3.79 (1.0) | **26.24x** | Post-Transfer ICU Draw Leakage |

* **Mechanistic Impact**: Patients who deteriorate have longer ward stays or receive post-transfer ICU draws, resulting in 4x–26x higher lab counts accumulated over the full stay. Ordering frequency acted as a downstream consequence of clinical suspicion rather than an antecedent predictor.
* **Resolution**: Excluded all full-stay `*_count`, `*_abnormal_count`, `*_missing_ratio`, `lab_unique_items`, and `unique_*` features from the candidate feature set.
* **Performance Impact**: AUROC dropped from **0.9162 $\rightarrow$ 0.8968** at the time, confirming the elimination of full-stay order-frequency leakage. Closing a further `lab_*_mean` full-stay leak subsequently took it to **0.8231**.

### C. Open issue — the 24-hour window can extend past the event

The count and missingness families excluded above were later **reintroduced in windowed
form** (`lab_total_count_24h`, `lab_unique_items_24h`, `lab_glucose_abnormal_count_24h`,
`lab_glucose_poc_missing_ratio_24h`, …). The automated screen passes them because
`_24h`-suffixed features are the corrected, windowed form of a removed full-stay family.

> [!CAUTION]
> **For this phase specifically, that reasoning is incomplete, and the model's top two
> laboratory drivers are affected.**
>
> The `_24h` suffix means *the first 24 hours of the admission*. Phase 5's prediction
> cutoff is not hour 24 — it is **6 hours before the ICU transfer**, and the transfer time
> varies per patient. Measured on the cohort: **12,236 of 31,282 positive cases (39.1%)
> transfer to ICU before hour 24** (median time to transfer 39.1 h). For those patients the
> 24-hour feature window extends *past the event*, and therefore includes post-transfer ICU
> laboratory draws — which is precisely the "Post-Transfer ICU Draw Leakage" mechanism
> identified for `lab_potassium_wb_count` in the table above.
>
> This is consistent with what the SHAP ranking showed: `lab_unique_items_24h` and
> `lab_total_count_24h` ranked **1st and 3rd**, and three more count/missingness
> features appeared in the top 15 (§7). Those measure *how much testing was ordered*, which
> is exactly the quantity that inflates once a patient reaches the ICU.

> [!NOTE]
> **This issue is now fixed.** The rebuild
> ([`tables/deterioration_landmark_results.md`](tables/deterioration_landmark_results.md))
> uses a landmark design in which every patient is event-free when the feature window
> closes, so the exposure is **0%** rather than 39%.
>
> Two findings from the rebuild are worth recording here:
>
> 1. **Why the exclusion list missed these features.** `DETERIORATION_EXCLUDE_STRICT`
>    contains `*_count`, `*_abnormal_count`, `*_missing_ratio` and `lab_total_count` — but
>    those globs require the name to *end* there. `lab_total_count_24h` does not match
>    `*_count`, so every windowed variant of a deliberately-excluded family passed straight
>    through the screen. A name-based filter silently stopped filtering when the naming
>    convention changed.
>
> 2. **How much of the model was clinical concern.** The rebuild trains both with and
>    without those features. Retaining them lifts AUPRC from 0.1016 to 0.1219 — **+20%
>    relative**. So testing volume was a real contributor but not the whole story: the
>    model is mostly reading physiology. They are excluded from the promoted model anyway,
>    because a signal that reflects local testing habits will not transfer to another site.

---

## 4. Cohort Partitioning & Automated Leakage Surety Assertions

### Zero-Overlap Patient Partitioning Protocol
Cohort of **489,547 general ward admissions** partitioned at the patient level (`subject_id`):
* **Training Set**: 341,802 admissions (20,086 positive events).
* **Validation Set**: 73,391 admissions (4,284 positive events).
* **Holdout Test Set**: 74,354 admissions (4,391 positive events; **Test Base Rate: 5.95%**).

### Automated Assertion Suite Results
1. **Patient Split Integrity Assertions**: `Train/Val Overlap: 0` | `Train/Test Overlap: 0` | `Val/Test Overlap: 0` $\rightarrow$ **PASSED**.
2. **Forbidden Column Assertions**: Evaluated against forbidden patterns (`vital_*`, `news2_*`, `first_careunit*`, `last_careunit*`, `cci_*`, `dx_*`, and the **full-stay** `*_count`, `*_abnormal_count`, `*_missing_ratio`, `lab_unique_items` families) $\rightarrow$ **0 Forbidden Columns Found** $\rightarrow$ **PASSED**.
3. **Distributional & Availability Differential Assertion** $\rightarrow$ **PASSED**.

### ~~FINAL SURETY RATING: 100% LEAK-FREE VERIFIED~~ — **WITHDRAWN**

> [!CAUTION]
> **This rating is withdrawn and should not be cited.** The assertion suite is sound for
> what it tests, and all three assertions genuinely pass. But it tests for *forbidden
> column names*, and the residual issue in §3C is not a column-name problem — the
> offending features carry the permitted `_24h` suffix and pass every check while still
> reaching past the event for 39% of positive cases.
>
> "100% leak-free verified" states a guarantee that no name-based screen can provide. What
> the suite establishes is narrower and still worth having: **no feature from a known
> forbidden family is present, and no patient appears in two splits.** That is what should
> be claimed.

---

## 5. Final Leak-Free Performance Metrics

Models were trained using class-imbalance weighting on **85 features** and evaluated on
**74,354 holdout test admissions**:

| Model Name | Test AUROC | Test AUPRC | Test Base Rate | Brier Pre-Calib | Brier Post-Calib | AUPRC Enrichment vs Base Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.7865** | **0.3229** | 0.0595 | 0.1751 | N/A | **5.43x** |
| ★ **XGBoost** *(promoted / served)* | **0.8196** | **0.3771** | 0.0595 | 0.1679 | **0.0454** | **6.34x** |
| **LightGBM** | **0.8231** | **0.3739** | 0.0595 | 0.1636 | N/A | **6.28x** |

> [!IMPORTANT]
> **The served model is XGBoost, not LightGBM**, despite the artifact being named
> `phase5_deterioration_lightgbm_winning.pkl`. LightGBM has the higher AUROC (0.8231 vs
> 0.8196) but XGBoost has the higher **AUPRC** (0.3771 vs 0.3739), which is the selection
> criterion for a 5.95% base-rate task — ranking the positives well matters more than
> ranking the whole cohort. Only XGBoost has a fitted isotonic calibrator.
>
> The filename is misleading and is retained only because the serving layer addresses it by
> that path. Verify with `type(joblib.load(...)).__name__`, not by reading the filename.

### Probability Calibration Analysis
Isotonic calibration on validation predictions reduced the Brier score of the served
XGBoost model from **0.1679 $\rightarrow$ 0.0454** (**73.0% error reduction**).

> [!WARNING]
> **Calibration is not optional for this model.** Trained with class-imbalance weighting,
> its raw output reports roughly **79%** deterioration risk against a **5.95%** base rate.
> Serving the uncalibrated booster puts that figure directly into a clinical report. This
> is not hypothetical — it occurred: the runner had no calibrator entry for
> `deterioration`, so the raw output was served until the omission was found.

---

## 6. Plausibility Justification & Cross-Phase Performance Alignment

Why is a leak-free AUROC of **0.8231** plausible for Phase 5 relative to Phase 1
(Mortality Run C: **0.9442**) and Phase 3 (ICU Triage: **0.9219**)?

1. **Comparison to Phase 1 (Mortality Run C AUROC 0.9442)**:
   - Phase 1 evaluates unconstrained in-hospital mortality across *all* hospital admissions at registration ($t=0$). Unconstrained mortality includes massive, unambiguous acute physiological failures present immediately at admission (e.g., out-of-hospital cardiac arrest, catastrophic trauma, end-stage septic shock).
   - In Phase 5, all patients admitted directly to the ICU at baseline ($t \le 6\text{h}$) are **removed**, filtering out the most obvious acute presentations. Predicting which general ward patient breaks down >6 hours into their stay involves a mandatory 6-hour pre-transfer lead window, causing natural physiological signal decay. An AUROC of **0.8231** sits well below unconstrained immediate mortality, reflecting the added noise and difficulty of predicting ward deterioration ahead of time.

2. **Comparison to Phase 3 (Registration ICU Triage AUROC 0.9219)**:
   - Phase 3 predicts ICU admission at the single instant of emergency registration ($t=0$).
   - Phase 5 is now the **weakest** of the two, not the stronger one. The superseded text
     argued the reverse — that Phase 5's richer inpatient context explained it beating
     Phase 3 (0.8968 vs 0.8469). Both figures have since moved and the ordering reversed,
     so that argument no longer describes anything. It was also built on features the
     model does not contain: `med_class_*` and `lab_*_median` are cited as Phase 5's
     advantage, and neither family is in the served feature set.
   - The current ordering is the more intuitive one. Phase 5 has the harder task: it must
     predict a *future* transition with a mandatory lead time, on a cohort from which the
     obviously-sick patients have already been removed.

> [!NOTE]
> **A caution about plausibility arguments generally.** This section originally justified
> 0.8968 as sitting "appropriately" between the other phases. It read as confirmation, but
> the figure it was defending was inflated by a leak, and the reasoning accommodated it
> without difficulty. A number that can be justified after the fact is not thereby
> validated — the leakage audit in §3, not the narrative here, is what carries the weight.

---

## 7. Leak-Free Feature Attribution (Top 10 SHAP Features)

Regenerated by `scripts/evaluation/run_explainability_audit.py`; see
[`tables/explainability_audit.md`](tables/explainability_audit.md).

```
Top 15 SHAP Features for Clinical Deterioration (served XGBoost, 85 features):
 1. lab_unique_items_24h              0.3690  (workup breadth — see caution)
 2. anchor_age                        0.3477  (physiological reserve)
 3. lab_total_count_24h               0.2182  (testing volume — see caution)
 4. admit_hour                        0.1566  (out-of-hours presentation)
 5. lab_wbc_last_24h                  0.1149  (systemic inflammatory response)
 6. lab_glucose_abnormal_count_24h    0.0954  (abnormal-result accumulation — see caution)
 7. lab_bun_first_24h                 0.0725  (renal function / volume status)
 8. lab_glucose_poc_missing_ratio_24h 0.0652  (POC testing pattern — see caution)
 9. lab_hematocrit_wb_count_24h       0.0645  (whole-blood draw frequency — see caution)
10. lab_platelets_first_24h           0.0632  (coagulopathy / sepsis)
11. lab_bicarbonate_last_24h          0.0621  (metabolic acidosis)
12. lab_glucose_max_24h               0.0449  (glycaemic instability)
13. lab_chloride_last_24h             0.0443  (electrolyte / acid-base)
14. lab_potassium_count_24h           0.0433  (draw frequency — see caution)
15. lab_hematocrit_last_24h           0.0363  (anaemia / haemorrhage)
```

> [!CAUTION]
> **The superseded list above was for a model that no longer exists.** It named
> `med_class_*` at ranks 1–4 and `lab_*_median` at 5–7. Neither family is in the served
> feature set — `med_class_*` are whole-stay flags and the full-stay `_median` aggregates
> were removed by the same audit this report documents. The clinical rationales attached to
> them read convincingly and described nothing the model uses.
>
> **Six of the current top 15 are testing-frequency or missingness features**
> (ranks 1, 3, 6, 8, 9, 14), and they occupy the two strongest laboratory positions. These
> measure *how much testing was ordered*, not what the results were, and they are the
> features most exposed to the windowing issue in §3C. The claim that predictions are
> "driven purely by baseline lab medians, medication categories, and admission timing" was
> not true of this model and is withdrawn.
>
> Genuine physiology is present — WBC, BUN, platelets, bicarbonate, glucose, chloride and
> haematocrit occupy seven of the top fifteen — but it is not what the model leans on hardest.

---

## 8. Retroactive Recommendation for Phases 1–4

The two-stage leakage assertion suite developed in Phase 5 (checking both **availability missingness differentials** and **value distribution accumulation ratios**) represents the strongest leakage audit standard in this codebase. 

### Recommended Retroactive Spot-Check:
It is recommended to apply this distributional count check retroactively to spot-check top features in Phases 1–4:
- Confirm that any `count` or aggregate metrics in Phase 1 (Mortality) and Phase 3 (ICU Triage) are strictly bounded to $t = 0$ data and do not accumulate post-admission values.
