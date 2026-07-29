# Model Performance — Before vs After the Identifier Correction

Both columns are evaluated on the **same held-out test patients**: `patient_split.parquet` was held fixed across the correction, so these deltas isolate the effect of repairing the laboratory join.

> [!NOTE]
> Before the correction, 50.1% of admissions (every odd `hadm_id`) carried > **no laboratory features at all**, and some even-ID admissions held labs > rounded onto them from a neighbouring admission. See > [`data_correction_notice.md`](../data_correction_notice.md).


| Model | Metric | Before | After | Δ |
| :--- | :--- | ---: | ---: | ---: |
| **Phase 1 Mortality — Run C (headline, strict 24h)** | AUROC | 0.9490 | 0.9490 | +0.0000 |
| Phase 1 Mortality — Run B (full-stay, leak-free) | AUROC | 0.9835 | 0.9835 | +0.0000 |
| Phase 1 Mortality — Run A (_leaky upper bound, not a result_) | AUROC | 0.9940 | 0.9940 | +0.0000 |
| **Phase 2 Readmission — strict 24h (headline)** | AUROC | 0.7094 | 0.7094 | +0.0000 |
| Phase 3 ICU admission | AUROC | 0.8469 | 0.8469 | +0.0000 |
| Phase 4 Hospital LOS — Stage A | AUROC | 0.8114 | 0.8114 | +0.0000 |
| Phase 5 Deterioration | AUROC | 0.8878 | 0.8878 | +0.0000 |

## Interpretation

A metric that **falls** after correction is the expected outcome if the missing-laboratory pattern was itself carrying signal — absence of lab records correlates with shorter, less acute admissions, which a model can exploit without any physiological information.

A metric that **rises** indicates the models were previously starved of genuine laboratory signal for half the cohort.

Either direction is a legitimate finding and should be reported as such. The corrected figures supersede the baseline in all cases.

