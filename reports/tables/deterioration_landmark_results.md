# Phase 5 (rebuilt) — Clinical Deterioration as a Landmark Analysis

_Generated 2026-08-06 10:13 UTC by `scripts/pipelines/run_deterioration_landmark.py`._

## 1. What changed and why

The original Phase 5 windowed features to `admittime + 24h` while the prediction
cutoff was nominally `t_event − 6h`. Those are different instants, and on this
cohort **12,236 of 31,282 positive cases (39%) transferred to ICU before hour 24** —
so their feature window reached past the event and absorbed post-transfer ICU
laboratory draws. The model's two strongest laboratory drivers were
`lab_unique_items_24h` and `lab_total_count_24h`: testing volume, which is exactly
what inflates once a patient reaches intensive care.

This rebuild fixes it **structurally**. A landmark T = 24h is fixed, and only
admissions still at risk at T — still in hospital, not yet in ICU — enter the cohort.
The outcome is ICU transfer within 48h of T. Every patient is therefore
event-free at the moment the feature window closes, so no feature can contain
post-event information. Cases and controls also get an identical observation window,
which the original case-control design did not provide.

## 2. Cohort

| | |
| :--- | ---: |
| At-risk admissions at T | 361,672 |
| Deterioration events within 48h | 7,937 |
| Base rate | 2.19% |
| Train / val / test | 252,820 / 54,342 / 54,510 |

> [!NOTE]
> The cohort is smaller than the original 492,068, and patients who deteriorate in
> the first 24 hours are no longer represented. That is a genuine narrowing of scope,
> not a modelling trick: this model answers *"will a patient still stable at 24 hours
> deteriorate over the next two days?"* — it does not cover early crashes. Those need
> a separate model at a shorter landmark.

## 3. Results

Two feature sets are trained. **Primary** excludes windowed testing-volume and
missingness features (`*_count_24h`, `*_abnormal_count_24h`, `*_missing_ratio_24h`,
`lab_total_count_24h`, `lab_unique_items_24h`); **sensitivity** retains them. Those
features are knowable at the landmark, but they describe how much testing a clinician
ordered rather than the patient's state, so the difference measures how much of the
model reads clinical concern.

### Primary — physiology only (129 features)

| Model | AUROC | AUPRC | Brier |
| :--- | ---: | ---: | ---: |
| LogisticRegression | 0.7552 | 0.0657 | 0.2076 |
| ★ XGBoost | 0.7679 | 0.1016 | 0.1439 |
| LightGBM | 0.7584 | 0.0951 | 0.1145 |
| ★ XGBoost (Calibrated) | 0.7665 | 0.0953 | 0.0208 |

### Sensitivity — testing volume retained (170 features)

| Model | AUROC | AUPRC | Brier |
| :--- | ---: | ---: | ---: |
| LogisticRegression | 0.7760 | 0.0810 | 0.1984 |
| ★ XGBoost | 0.7776 | 0.1219 | 0.1385 |
| LightGBM | 0.7654 | 0.1078 | 0.1097 |
| ★ XGBoost (Calibrated) | 0.7768 | 0.1156 | 0.0204 |

## 4. How much of the model is clinical concern?

Retaining testing-volume features moves AUPRC from **0.1016** to **0.1219** (+0.0203, +20% relative) and AUROC from 0.7679 to 0.7776.

So roughly a 20% share of the model's precision comes from how much testing was ordered rather than from what the results were. That is meaningful but not dominant — the model is mostly reading physiology, and the original audit was right to be suspicious of these features without being right that they were the whole story. They stay out of the promoted model: a signal that reflects clinician concern will not transfer to a site with different testing habits.

The **primary** model is the one promoted.

## 5. Comparison with the superseded design

| | Superseded (fixed 24h window) | Landmark (this) |
| :--- | ---: | ---: |
| Cohort | 492,068 | 361,672 |
| Events | 31,282 (6.36%) | 7,937 (2.19%) |
| Positives with feature window past the event | **12,236 (39%)** | **0** |
| AUROC | 0.8231 | 0.7679 |
| AUPRC | 0.3739 | 0.1016 |
| AUPRC ÷ base rate (enrichment) | 6.28x | 4.63x |

> [!IMPORTANT]
> **These two columns are not like-for-like and the raw AUPRC drop overstates the
> change.** The base rate fell from 5.95% to 2.19%, and AUPRC scales with base rate, so the
> figures must be compared as enrichment: **6.28x → 4.63x**.
>
> Even on that fairer footing the model is weaker, which is the expected direction: the
> superseded figure was measured on a task where 39% of positives carried post-event
> information. Some of the loss is the leak being removed, and some is the landmark task
> being genuinely harder — predicting deterioration in a patient who has *already been
> stable for 24 hours* is a harder question than predicting it across all comers. The
> two causes cannot be separated from these numbers alone, and no attempt is made to.

## 6. Is this good enough to be useful?

AUROC **0.7679** sits within the range published for ward
early-warning scores predicting ICU transfer — NEWS2 and its variants typically report
0.65–0.78 on comparable tasks. This model is at the upper end of that band while using
**no vital signs at all** (they are unavailable outside the ICU in MIMIC-IV; see the
Phase 1 report §3), which is the more notable result here.

The honest summary: this is a credible ward-deterioration model, materially weaker than
the number it replaces, and the number it replaces was not real.
