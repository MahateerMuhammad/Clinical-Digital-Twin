# Feature Engineering Report

Generated: 2026-07-20T12:51:31.137031Z

## Feature Selection Summary

- **n_features_in**: 71
- **n_features_out**: 57
- **n_constant**: 1
- **n_duplicate**: 0
- **n_near_zero_variance**: 3
- **n_high_missing**: 4
- **n_highly_correlated_pairs**: 8
- **n_dropped**: 14

## Dropped Features (Categorized by Reason)

### Genuine Clinical Missingness (> Threshold)
*(Note: In prior runs, all vitals/labs were dropped here due to the 1.5M row-cap bug. These are now true clinical drops.)*
- icu_duration_days
- icu_duration_hours
- los
- n_icu_stays_per_admission

### Constant / Zero Variance
- _invalid_time_order
- _is_duplicate
- _is_duplicate
- cci_aids

### Duplicates / Highly Correlated
- admit_year
- cci_renal_disease
- charlson_comorbidity_index
- diagnosis_count
- dx_diabetes
- icu_duration_hours
- n_icu_stays
- procedure_count
