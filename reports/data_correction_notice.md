
# Data Correction Notice — Identifier Precision Loss in Laboratory Joins

**Status:** code corrected; data rebuilt; **all phases retrained — 1–5 on 2026-07-29, 6–7 on 2026-07-30**
**Affects:** all model results published before 2026-07-29
**Baseline (pre-correction) reports retained at:** `reports/baseline_pre_id_fix/`

> Figures in `reports/tables/` are current and citable. Figures in
> `reports/baseline_pre_id_fix/` are the superseded pre-correction values, retained
> only for the before/after comparison in
> [`tables/model_comparison_before_after.md`](tables/model_comparison_before_after.md).

> [!CAUTION]
> **A second correction followed this one. The post-correction figures in this document
> are themselves superseded.**
>
> This notice records the identifier/laboratory-join repair of 2026-07-29 and its
> 0.9490 → 0.9062 effect on Phase 1 Run C. A later **feature-selection repair** restored
> creatinine, BUN and haematocrit to the selected set and closed a `lab_*_mean` full-stay
> leak, moving Phase 1 Run C again to **0.9442 / 0.3800**. Phases 2–5 moved likewise; see
> [`tables/model_comparison_before_after.md`](tables/model_comparison_before_after.md) for
> the full then-and-now.
>
> Every "after" figure below should be read as *"after the join repair, before the
> feature-selection repair"*. The document is deliberately left otherwise unedited: it is a
> record of what one correction did, and revising its numbers would destroy that.

> [!IMPORTANT]
> Several distinct defects were corrected, not one. The identifier fix **raised**
> performance by restoring laboratory data to half the cohort; the leakage fixes
> **lowered** it. The headline mortality figure moved 0.9490 → 0.9062 AUROC as the
> net of these opposing effects (§6). Phases 6 and 7 carried their own drifted copies
> of the exclusion list and were corrected separately (§8) — Phase 6's conclusion
> reversed, and Phase 7's positive result became a null one.

---

## 1. What happened

The preprocessing pipeline reduced memory by downcasting numeric columns to the
smallest safe dtype (`src/utils/io_utils.optimise_dtypes`). The float branch was
unconditional:

```python
for col in df.select_dtypes(include=["float64"]).columns:
    df[col] = df[col].astype(np.float32)
```

`float32` has a 24-bit significand, so it represents consecutive integers exactly
only up to 2²⁴ = 16,777,216. MIMIC-IV `hadm_id` values are approximately 2.2 × 10⁷,
above that limit, where the representable spacing is 2. **Every odd `hadm_id`
was therefore rounded to an even neighbour.**

The defect only reached columns that arrived as `float64`. An identifier becomes
`float64` only when it contains nulls, which in MIMIC-IV is true for exactly two
tables: `labevents` (labs drawn outside an inpatient admission) and `emar`. All
other tables carried non-null integer identifiers and were unaffected.

## 2. Measured effect

Verified against the source CSV and the derived artifacts:

| Quantity | Expected | Observed before correction |
| :--- | ---: | ---: |
| Odd `hadm_id` in `admissions` | 273,333 / 546,028 (50.1%) | 273,333 (50.1%) — source intact |
| Odd `hadm_id` surviving in `labevents_clean` | ~50% | **0 of 621,011 (0.0%)** |
| Distinct lab `hadm_id` matching a real admission | ~100% | 4,578 / 8,588 (53%) |
| Odd `hadm_id` in `laboratory_features` | ~50% | **0 of 422,977 (0.0%)** |

Downstream, in `admission_level_selected.parquet`, non-null rate of
`lab_anion_gap_median` by admission-ID parity:

| `hadm_id` parity | Admissions | With laboratory features |
| :--- | ---: | ---: |
| Even | 272,695 | 75.6% |
| Odd | 273,333 | **0.0%** |

**50.1% of admissions carried no laboratory features at all**, and an unquantified
subset of even-ID admissions received laboratory values belonging to the adjacent
odd-ID admission that had been rounded onto them.

Since `lab_*` features dominate the Phase 8 SHAP rankings for mortality, ICU
admission, length of stay and deterioration, every model trained before this
correction learned from a feature matrix in which half the cohort appeared never
to have had bloodwork.

## 3. Correction

**Code** (`src/utils/io_utils.py`): identifier columns are excluded from float
downcasting. Nullable integer identifiers are converted to pandas `Int32`, which
is exact and still halves the memory of `float64`. Non-identifier floats continue
to be downcast, preserving the memory benefit. A magnitude guard additionally
protects any unnamed column holding integral values above 2²⁴.

**Data** (`scripts/dev/run_id_corruption_rebuild.py`): `labevents` was re-read from source CSV
— `data/interim/raw_cache/` could not be used, as it was written by the same
defective code — followed by rebuilds of laboratory features and all processed
datasets. Three interim tables (`admissions`, `chartevents`, `radiology_detail`)
had separately been overwritten by synthetic test fixtures and were also restored;
see §5.

**Verification** (`scripts/dev/run_id_corruption_rebuild.py --verify`) asserts that odd
identifiers are present at ~50% in the repaired artifacts, and that laboratory
coverage is within 5 percentage points between even- and odd-ID admissions.

## 4. Effect on published results

`patient_split.parquet` was deliberately **not** regenerated. It is keyed on
`subject_id`, which was never affected, so holding the split fixed makes
pre- and post-correction metrics directly comparable on an identical test cohort.

See `reports/tables/model_comparison_before_after.md` for the paired comparison.

Any figure in the baseline reports that derives from laboratory features should be
treated as superseded. This includes the Phase 1–5 performance tables, the Phase 8
SHAP rankings, and the Phase 9 risk-tier cutoffs — the last of which are
additionally hardcoded in `src/llm/model_runner.py` and quoted as system constants
in `src/llm/report_composer.py`. Both have since been updated; see §7.

## 5. Related defect: test suite overwrote production data

`DataCleaner.clean_table` defaulted to `save=True` and wrote to
`data/interim/{table}_clean.parquet`. The regression tests called it with synthetic
fixtures and did not override that default, so **every execution of
`scripts/dev/run_tests.py` overwrote real interim tables with a handful of synthetic rows.**
`admissions_clean.parquet` (23 MB → 7 KB), `chartevents_clean.parquet`
(542 MB → 5 KB) and `radiology_detail_clean.parquet` were destroyed this way.

`clean_table` now defaults to `save=False`; only `src/data/pipeline.py` passes
`save=True`. The three tables were restored during the rebuild.

---

## 6. Observation-window leakage (corrected 2026-07-29)

Retraining on the repaired data exposed three features carrying outcome information
into protocols published as *strict 24-hour observation window*. All three were
already excluded from some strict lists and missing from others — copy-paste drift
across five hand-maintained exclusion lists. They are now shared constants in
`src/features/leakage_filters.py`.

| Feature family | Why it leaks | Evidence |
| :--- | :--- | :--- |
| Discharge-note statistics (`sentence_count`, `medical_keyword_count`, `negation_count`, …) | The discharge summary is authored *at discharge*; its length describes the whole admission | ranked 4th and 9th of 66 features by SHAP |
| `med_class_*` | `groupby(hadm_id).max()` over all prescriptions, no time filter — "prescribed at any point" | `med_class_opioid` present for **94.8%** of deaths vs **56.6%** of survivors (1.68×); `med_class_statin` shows no gradient (0.93×), so the effect is end-of-life comfort-care prescribing, not general acuity |
| `lab_*_min` / `_max` / `_median` | `charttime` was used only to *order* records, never to filter them, so these were whole-admission extremes | 30 of the 40 lab features surviving Run C; `lab_bicarbonate_min` ranked 2nd by SHAP |

### The laboratory window

Rather than delete the lab aggregates, `src/features/laboratory.py` gained
`build_lab_features_windowed()`, producing a parallel `_24h`-suffixed feature set
restricted to `charttime <= admittime + 24h`. Full-stay protocols (mortality Run A/B,
readmission Run A) continue to use the whole-admission columns; the strict protocols
consume only the windowed ones.

Two deliberate choices:

* **Pre-admission draws are retained.** 15% of first draws carry a negative offset
  because they were taken in the emergency department before inpatient registration.
  They are on the chart at hour zero and are legitimately available to an
  admission-time model.
* **Feature selection must not collapse the twins.** `lab_creatinine_first` and
  `lab_creatinine_first_24h` are identical for the 92.2% of admissions whose first
  draw falls inside the window, so the duplicate filter would have discarded whichever
  merged second — silently stripping labs from either Run B or Run C depending on
  merge order. `_twin_pairs()` in `src/features/feature_selection.py` exempts them.

### Net effect

Both corrections applied at once, so each phase moved in the direction of whichever
dominated. Phases 3 and 4 lost no features and therefore show the identifier repair
in isolation; mortality Run C and Phase 5 lost the most leaked features. All figures
are LightGBM on the same held-out test patients (`patient_split.parquet` was held
fixed across the correction).

| Phase | Protocol | AUROC before | AUROC after | AUPRC before | AUPRC after |
| :--- | :--- | ---: | ---: | ---: | ---: |
| 1 Mortality | **Run C (24h) ★** | 0.9490 | **0.9062** | 0.4706 | **0.3281** |
| 1 Mortality | Run B (full-stay) | 0.9835 | 0.9917 | 0.7331 | 0.8556 |
| 1 Mortality | Run A (leaky bound) | 0.9940 | 0.9966 | 0.8303 | 0.9035 |
| 2 Readmission | **Run B (24h) ★** | 0.7094 | **0.7072** | 0.4195 | **0.4173** |
| 2 Readmission | Run A (full-stay) | 0.7299 | 0.7324 | 0.4371 | 0.4407 |
| 3 ICU admission | ★ | 0.8469 | **0.8969** | 0.5369 | **0.6875** |
| 4 LOS Stage A | ★ | 0.8114 | **0.8350** | 0.5434 | **0.5958** |
| 5 Deterioration | ★ | 0.8878 | **0.8221** | 0.4302 | **0.3735** |

A demographics-only model (age, admission type, admission location; no laboratory
data) scores **0.8260** AUROC / 0.1484 AUPRC on the mortality task. Run C's 0.9062
should be read against that floor rather than against zero.

### Report AUPRC, not AUROC

At 2.16% prevalence, AUROC flatters every model here. Run C's AUPRC of 0.3281 is a
**15× lift** over the base rate and is the honest headline. At the deployed operating
point (80% recall) precision is **10%** — roughly nine false alarms per true positive.
The superseded 0.9490 figure implied a far better trade-off than exists.

### Known remaining limitations

* `lab_total_count_24h` (number of assays ordered in the first 24 hours) ranks 5th by
  SHAP in Run C and 2nd in Phase 3. It is a **care-intensity proxy**, not physiology —
  it reflects how concerned the treating team was. It is genuinely observable within
  the window, so it is retained, but a model intended to stand on physiology alone
  should exclude it.
* The LACE clinical baseline was scored on the *post-exclusion* frame, collapsing its
  L component to a constant and falling C back to an age proxy (AUROC 0.4994, chance).
  Corrected 2026-07-29 by snapshotting LACE's inputs before exclusions. MIMIC-IV's
  hosp module carries no emergency-department visit history, so the E component is
  scored from prior admissions as a documented proxy and the baseline is reported as
  **LACE (modified E)** — AUROC 0.6096.

## 7. Serving-layer staleness

`models/` (training output) and `models/best_models/` (what `src/llm/model_runner.py`
loads) use different naming schemes, and nothing in the codebase bridged them —
promotion was a manual copy-and-rename. After the 2026-07-29 retrain, `best_models/`
still held 2026-07-21 artifacts, so Phase 9 and the entire LLM layer continued
serving the pre-correction models with no error raised. `scripts/maintenance/promote_models.py` now
performs the mapping explicitly and archives what it replaces.

Phase 9 tier cutoffs (`0.0094 / 0.1119 / 0.2171`, hardcoded in
`src/llm/model_runner.py` and quoted in `src/llm/report_composer.py`) are the
50th/80th/95th percentiles of the calibrated model's test predictions and are
invalidated by any retrain. They must be recomputed after promotion.

---

## 8. Phases 6 & 7 (corrected 2026-07-30)

Both Kaggle notebooks carried their own hand-copied `MORTALITY_EXCLUDE_RUN_C`, and
both had drifted from `src/features/leakage_filters.py`.

### Phase 6 — the sequence-vs-tabular comparison was not like-for-like

The notebook's copy was missing 15 patterns, so the sequence models received static
features (`lab_*_min/_max/_median/_first`, `med_class_*`, the discharge-note
statistics) that the tabular baseline was denied. Correcting it changes the result:

| Model | AUROC before | AUROC after | AUPRC before | AUPRC after |
| :--- | ---: | ---: | ---: | ---: |
| LSTM / GRU | 0.9738 | **0.8957** | 0.6933 | **0.2988** |
| Transformer Encoder | 0.9740 | **0.8970** | 0.6996 | **0.3009** |
| LightGBM tabular (Run C) | 0.9062 | 0.9062 | 0.3281 | 0.3281 |

The leak was worth ~0.077 AUROC and ~0.39 AUPRC to the sequence models; the tabular
baseline, which never had those features, is unchanged. The corrected conclusion is
that engineered 24-hour summaries capture what raw event ordering offers, at a
fraction of the compute — the opposite of what the previous report claimed, and the
reason the tabular model is the one deployed.

### Phase 7 — embeddings were graded on their own inputs

`cci_*`, `dx_*` and `med_class_*` drive triplet mining and the retrieval metrics, but
were also encoder inputs, so the embedding was scored on what it had memorised. They
are now held as supervision only. Separately, six of the seven rows in the published
benchmark table were hardcoded literals from an earlier run — `eval_7_techniques_full()`
was called once. The table now reports only spaces measured in the run that produced it.

| Space | Disease % | Lab MAE | Med Jaccard % | Mortality enrichment |
| :--- | ---: | ---: | ---: | ---: |
| Naive Raw Features | 36.3 | **0.123** | **50.2** | 0.73x |
| Multi-Task Triplet AE | **37.3** | 0.182 | 49.8 | 0.79x |
| LightGBM Tree-Leaf AE | 35.0 | 0.219 | 48.8 | 1.06x |
| Dual-Head Hybrid AE | 36.1 | 0.200 | 49.7 | 1.04x |

**No representation achieves a mortality-retrieval CI separated from the naive
baseline.** Triplet disease match fell from a reported 71.7% to 37.3% once the labels
were withheld from the encoder. On this cohort, learned patient embeddings do not
improve outcome-aligned twin retrieval over raw scaled features; they remain useful as
dimensionality reduction for Level 5 evidence lookup. This is a negative result from a
correctly specified experiment, and supersedes the positive result from a broken one.

### Consequence for the serving layer

`src/llm/report_composer.py` held the Phase 9 tier rates in two places
(`SYSTEM_CONSTANTS` and `TIER_CONTEXT`); updating one left the other stale and the
grounding verifier correctly refused to emit a report quoting an ungroundable number.
`TIER_CONTEXT` is now derived from `SYSTEM_CONSTANTS`. Tier 4 observed mortality is
**21.52%**, not the 15.05% previously quoted.

---

## 9. Emergency-department features (added 2026-08-07, **not yet consumed**)

> [!IMPORTANT]
> **The ED feature set is staged, not in production. No published figure in this
> repository uses it.**

MIMIC-IV-ED was integrated to address the finding recorded in §6 and repeated in the
Phase 1 and Phase 5 reports: `chartevents` is ICU-only, so `admission_level_selected`
carries **no `vital_*` columns at all** and every model predicts physiology through
proxies. `src/features/emergency.py` produces 66 features — triage vitals, serial ED
vitals, ED length of stay, arrival mode, and home-medication reconciliation — for the
202,415 admissions (37.1%) with a linked ED stay.

**What was verified before staging it:**

| Check | Result |
| :--- | :--- |
| New patients introduced | **0** — ED joins onto admissions, never creates them |
| Cohort fingerprint | unchanged (`7aed4ec6a8d4ab7d`) |
| ED `intime` precedes `admittime` | 99.7% (median −4.8h) |
| Observations after `admittime` | dropped (41% of ED vital rows) |
| Availability leakage (`ed_available` alone) | AUROC **0.5097** mortality, 0.5018 ICU, 0.4985 readmission |
| Missing ED data | NaN, never 0.0 |

The availability check is the one that mattered: partial coverage meant "has ED data"
could have carried the outcome by itself, which is the mechanism that forced the Phase
5 rebuild. Measured, it does not — 0.5097 is noise. The guard is retained in every
exclusion list regardless, and `tests/test_ed_features.py` pins it so a future change
to the linkage rule cannot quietly turn presence into a severity marker.

**Why the figure of 37.1% and not ~69%.** An earlier estimate used
`admission_location == 'EMERGENCY ROOM'`, which covers 244,179 admissions. The ED
*module* is a separate partial capture: only 56.5% of those admissions have an ED
record. The correct coverage is 37.1% of the cohort, and the gain is **0% → 37%
vitals coverage**, not the 16% → 75% first projected.

**Status and cost of adoption.** Consuming these features requires rebuilding
`admission_level_selected.parquet` and retraining Phases 1–5. Because the patient split
is untouched, before/after models remain directly comparable on identical test
patients — so this is a controlled experiment rather than a rebuild, and the current
results stay valid and citable until it is run.
`reports/tables/ed_feature_coverage.md` derives its status line from the selected
matrix, so it will report adoption automatically rather than on trust.

## 10. Payload serving schema (corrected 2026-08-10)

Four of the five models were withheld from payload-based reports because a
presentation payload retained too little of their validated discrimination. That was
recorded as an intrinsic limit of predicting for an unseen patient. It was mostly a
defect in the serving path.

**What was wrong.** `LiveModelRunner._convert_payload_to_series` built the feature
vector by writing booster column names directly — `gender_M = 1.0` — instead of
emitting the source categorical and letting `encode_admission_frame` expand it, which
is how the models were fitted. One-hot families therefore arrived with a single
member set and every sibling missing. A female patient reached Phase 5 as
`gender_M = 0.0` with `gender_F` absent: not male, sex unknown.

Separately, `payload_validation.py` never asked for race, language, insurance,
marital status, admission type/location, or prior utilisation. Those expansions are
86 of the mortality model's 164 features, and `prior_*` is the Expansion A&D block
Phase 2 was built on — the readmission model was being asked its question with the
patient's own readmission history withheld.

**Why it was invisible.** The stored-row path was never affected, and it is the
reference the payload path is measured against. `admission_level_selected.parquet`
holds these columns as `CategoricalDtype` carrying every level, so `get_dummies`
emitted the full family with its zeros. Only a payload, which carries plain strings,
lost the siblings. The two paths diverged in exactly the place the comparison could
not see, and the shortfall was attributed to the payload concept rather than its
implementation.

**Effect.** Coverage of the mortality feature space went from 18.3% to 67.7%
(deterioration 24.0% → 91.5%). Retention against the unchanged 66.7% floor:

| Task | Before | After | Served |
| :--- | ---: | ---: | :--- |
| mortality | 78.0% | **85.6%** | was, still is |
| readmission | 32.6% | **81.7%** | now served |
| icu_admission | 24.0% | **75.6%** | now served |
| deterioration | −0.8% | **91.6%** | now served |
| hospital_los | −2.7% | 57.9% | still withheld |

Reference AUROCs are unchanged — 0.9448 / 0.7062 / 0.9209 / 0.8997 / 0.7858 — which
is what makes the before/after comparison sound: only the payload arm moved.

**What was not done.** The retention floor was not lowered, and `hospital_los` stays
withheld at 57.9%. It is near the boundary, and that is the point of having fixed the
boundary in advance: length of stay depends on discharge planning and social
circumstances a presentation payload does not describe. `diagnosis_count` and
`procedure_count` were also deliberately left unmapped — they are counts of what the
hospital has coded by hour 24, and claiming them would have raised the coverage
figure without a clinician being able to supply them.

**Pinned by.** `tests/test_feature_space_onehot.py` (the known-zero vs unknown
distinction, per row) and `tests/test_payload_categoricals.py`, which includes a test
that every field the converter reads is one the schema actually asks for — the
specific gap that let this persist.

---

## 11. Reproducing the audit

```bash
python scripts/dev/run_id_corruption_rebuild.py --audit             # damage report, read-only
python scripts/dev/run_id_corruption_rebuild.py --verify            # acceptance test
python scripts/evaluation/run_explainability_audit.py                      # Phase 8 SHAP + leakage screen
python scripts/maintenance/recompute_risk_tiers.py --write-report --patch   # Phase 9 tiers + report
python scripts/maintenance/promote_models.py                                # dry run: models/ -> best_models/
```

`scripts/evaluation/run_explainability_audit.py` is the standing regression test for this correction: it
screens every model's SHAP top-15 against the removed families and fails loudly if one
reappears. Both `reports/tables/explainability_audit.md` and
`reports/tables/risk_stratification.md` previously had no generator and went stale
silently after the retrain; both are now regenerated from the models on disk.

