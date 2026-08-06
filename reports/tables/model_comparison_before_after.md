# Model Performance — Before vs After the Identifier Correction

> [!CAUTION]
> **Historical record. The "After" column is superseded and is not current performance.**
>
> This table documents one specific event — the laboratory-join repair. A **second**
> correction followed it (the feature-selection repair, which restored creatinine, BUN and
> haematocrit and closed a `lab_*_mean` full-stay leak), and every figure moved again:
>
> | Model | "After" below | Current |
> | :--- | ---: | ---: |
> | Phase 1 Mortality — Run C | 0.9062 | **0.9442** |
> | Phase 2 Readmission — strict 24h | 0.7072 | **0.7158** |
> | Phase 3 ICU admission | 0.8969 | **0.9219** |
> | Phase 4 Hospital LOS — Stage A | 0.8350 | **0.9001** |
> | Phase 5 Deterioration | 0.8221 | **0.8231** |
>
> The table is left unedited because rewriting it would misrepresent what the identifier
> correction actually did. For current figures see the generated tables in this directory.

Both columns are evaluated on the **same held-out test patients**: `patient_split.parquet` was held fixed across the correction, so these deltas isolate the effect of repairing the laboratory join.

> [!NOTE]
> Before the correction, 50.1% of admissions (every odd `hadm_id`) carried > **no laboratory features at all**, and some even-ID admissions held labs > rounded onto them from a neighbouring admission. See > [`data_correction_notice.md`](../data_correction_notice.md).


| Model | Metric | Before | After | Δ |
| :--- | :--- | ---: | ---: | ---: |
| **Phase 1 Mortality — Run C (headline, strict 24h)** | AUROC | 0.9490 | 0.9062 | -0.0428 ⚠ |
| Phase 1 Mortality — Run B (full-stay, leak-free) | AUROC | 0.9835 | 0.9917 | +0.0082 |
| Phase 1 Mortality — Run A (_leaky upper bound, not a result_) | AUROC | 0.9940 | 0.9966 | +0.0026 |
| **Phase 2 Readmission — strict 24h (headline)** | AUROC | 0.7094 | 0.7072 | -0.0022 |
| Phase 3 ICU admission | AUROC | 0.8469 | 0.8969 | +0.0500 ↑ |
| Phase 4 Hospital LOS — Stage A | AUROC | 0.8114 | 0.8350 | +0.0236 ↑ |
| Phase 5 Deterioration | AUROC | 0.8878 | 0.8221 | -0.0657 ⚠ |

## Interpretation

A metric that **falls** after correction is the expected outcome if the missing-laboratory pattern was itself carrying signal — absence of lab records correlates with shorter, less acute admissions, which a model can exploit without any physiological information.

A metric that **rises** indicates the models were previously starved of genuine laboratory signal for half the cohort.

Either direction is a legitimate finding and should be reported as such. The corrected figures supersede the baseline in all cases.

