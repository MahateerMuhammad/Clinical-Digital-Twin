
# Data Correction Notice — Identifier Precision Loss in Laboratory Joins

**Status:** code corrected; data rebuild in progress; models **not yet retrained**
**Affects:** all model results published before this notice
**Baseline (pre-correction) reports retained at:** `reports/baseline_pre_id_fix/`

> Until the status line above reads "models retrained", every performance figure in
> `reports/` is still the pre-correction value and should not be cited.

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
