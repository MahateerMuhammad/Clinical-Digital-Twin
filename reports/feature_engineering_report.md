# Feature Engineering Report

Generated: 2026-07-19T15:43:21.289967Z

## Feature Selection Summary

- **n_features_in**: 328
- **n_features_out**: 166
- **n_constant**: 3
- **n_duplicate**: 9
- **n_near_zero_variance**: 9
- **n_high_missing**: 91
- **n_highly_correlated_pairs**: 196
- **n_dropped**: 162

## Dropped Features (Categorized by Reason)

### Genuine Clinical Missingness (> Threshold)
*(Note: In prior runs, all vitals/labs were dropped here due to the 1.5M row-cap bug. These are now true clinical drops.)*
- icu_duration_days
- icu_duration_hours
- lab_anion_gap_change
- lab_anion_gap_slope
- lab_anion_gap_std
- lab_bicarbonate_change
- lab_bicarbonate_slope
- lab_bicarbonate_std
- lab_bun_change
- lab_bun_slope
- lab_bun_std
- lab_chloride_change
- lab_chloride_slope
- lab_chloride_std
- lab_chloride_wb_change
- lab_chloride_wb_first
- lab_chloride_wb_last
- lab_chloride_wb_max
- lab_chloride_wb_mean
- lab_chloride_wb_median
- lab_chloride_wb_min
- lab_chloride_wb_slope
- lab_chloride_wb_std
- lab_creatinine_change
- lab_creatinine_slope
- lab_creatinine_std
- lab_creatinine_wb_change
- lab_creatinine_wb_first
- lab_creatinine_wb_last
- lab_creatinine_wb_max
- lab_creatinine_wb_mean
- lab_creatinine_wb_median
- lab_creatinine_wb_min
- lab_creatinine_wb_slope
- lab_creatinine_wb_std
- lab_glucose_change
- lab_glucose_poc_change
- lab_glucose_poc_first
- lab_glucose_poc_last
- lab_glucose_poc_max
- lab_glucose_poc_mean
- lab_glucose_poc_median
- lab_glucose_poc_min
- lab_glucose_poc_slope
- lab_glucose_poc_std
- lab_glucose_slope
- lab_glucose_std
- lab_hematocrit_wb_change
- lab_hematocrit_wb_first
- lab_hematocrit_wb_last
- _... and 41 more_

### Constant / Zero Variance
- _invalid_time_order
- _is_duplicate
- _is_duplicate
- cci_aids
- lab_creatinine_wb_change
- lab_creatinine_wb_slope
- lab_creatinine_wb_std
- lab_hematocrit_wb_abnormal_count
- lab_hematocrit_wb_abnormal_count
- lab_potassium_wb_slope
- lab_potassium_wb_std
- note_count

### Duplicates / Highly Correlated
- admit_year
- cci_renal_disease
- char_count
- char_count
- char_count
- char_count
- days_to_readmission
- days_to_readmission
- days_to_readmission
- diagnosis_count
- dx_stroke
- icu_duration_hours
- lab_anion_gap_mean
- lab_bicarbonate_count
- lab_bicarbonate_count
- lab_bicarbonate_mean
- lab_bicarbonate_missing_ratio
- lab_bicarbonate_missing_ratio
- lab_bun_count
- lab_bun_count
- lab_bun_count
- lab_bun_count
- lab_bun_count
- lab_bun_count
- lab_bun_mean
- lab_chloride_count
- lab_chloride_mean
- lab_chloride_missing_ratio
- lab_chloride_wb_mean
- lab_chloride_wb_mean
- lab_chloride_wb_mean
- lab_chloride_wb_mean
- lab_chloride_wb_mean
- lab_chloride_wb_median
- lab_chloride_wb_median
- lab_chloride_wb_median
- lab_creatinine_count
- lab_creatinine_count
- lab_creatinine_count
- lab_creatinine_count
- lab_creatinine_count
- lab_creatinine_count
- lab_creatinine_count
- lab_creatinine_count
- lab_creatinine_count
- lab_creatinine_max
- lab_creatinine_mean
- lab_creatinine_mean
- lab_creatinine_mean
- lab_creatinine_mean
- _... and more_
