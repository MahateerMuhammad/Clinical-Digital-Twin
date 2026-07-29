
# Data Correction Notice — Identifier Precision Loss in Laboratory Joins

**Status:** code corrected; data rebuilt; **models retrained 2026-07-29 (Phases 1–5)**
**Affects:** all model results published before 2026-07-29
**Baseline (pre-correction) reports retained at:** `reports/baseline_pre_id_fix/`

> Figures in `reports/tables/` are current and citable. Figures in
> `reports/baseline_pre_id_fix/` are the superseded pre-correction values, retained
> only for the before/after comparison in
> [`tables/model_comparison_before_after.md`](tables/model_comparison_before_after.md).

> [!IMPORTANT]
> Three defects were corrected in this round, not one. The identifier fix **raised**
> performance by restoring laboratory data to half the cohort; two separate leakage
> fixes **lowered** it. The headline mortality figure moved 0.9490 → 0.9062 AUROC as
> the net of these opposing effects. See §6.

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

**Data** (`run_id_corruption_rebuild.py`): `labevents` was re-read from source CSV
— `data/interim/raw_cache/` could not be used, as it was written by the same
defective code — followed by rebuilds of laboratory features and all processed
datasets. Three interim tables (`admissions`, `chartevents`, `radiology_detail`)
had separately been overwritten by synthetic test fixtures and were also restored;
see §5.

**Verification** (`run_id_corruption_rebuild.py --verify`) asserts that odd
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
in `src/llm/report_composer.py`; both require updating to the new values.

## 5. Related defect: test suite overwrote production data

`DataCleaner.clean_table` defaulted to `save=True` and wrote to
`data/interim/{table}_clean.parquet`. The regression tests called it with synthetic
fixtures and did not override that default, so **every execution of
`run_tests.py` overwrote real interim tables with a handful of synthetic rows.**
`admissions_clean.parquet` (23 MB → 7 KB), `chartevents_clean.parquet`
(542 MB → 5 KB) and `radiology_detail_clean.parquet` were destroyed this way.

`clean_table` now defaults to `save=False`; only `src/data/pipeline.py` passes
`save=True`. The three tables were restored during the rebuild.

## 6. Reproducing the audit

```bash
python run_id_corruption_rebuild.py --audit    # damage report, read-only
python run_id_corruption_rebuild.py --verify   # acceptance test
```

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
serving the pre-correction models with no error raised. `promote_models.py` now
performs the mapping explicitly and archives what it replaces.

Phase 9 tier cutoffs (`0.0094 / 0.1119 / 0.2171`, hardcoded in
`src/llm/model_runner.py` and quoted in `src/llm/report_composer.py`) are the
50th/80th/95th percentiles of the calibrated model's test predictions and are
invalidated by any retrain. They must be recomputed after promotion.
