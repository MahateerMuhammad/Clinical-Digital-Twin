# Feature Engineering Report

Generated: 2026-07-29T08:17:06.437373Z

## Feature Selection Summary

- **n_features_in**: 557
- **n_features_out**: 267
- **n_constant**: 5
- **n_duplicate**: 1
- **n_near_zero_variance**: 14
- **n_high_missing**: 167
- **n_highly_correlated_pairs**: 411
- **n_dropped**: 290

## Dropped Features (Categorized by Reason)

### Genuine Clinical Missingness (> Threshold)
*(Note: In prior runs, all vitals/labs were dropped here due to the 1.5M row-cap bug. These are now true clinical drops.)*
- icu_duration_days
- icu_duration_hours
- lab_anion_gap_change_24h
- lab_anion_gap_slope_24h
- lab_anion_gap_std_24h
- lab_bicarbonate_change_24h
- lab_bicarbonate_slope_24h
- lab_bicarbonate_std_24h
- lab_bun_change_24h
- lab_bun_slope_24h
- lab_bun_std_24h
- lab_chloride_change_24h
- lab_chloride_slope_24h
- lab_chloride_std_24h
- lab_chloride_wb_change
- lab_chloride_wb_change_24h
- lab_chloride_wb_first
- lab_chloride_wb_first_24h
- lab_chloride_wb_last
- lab_chloride_wb_last_24h
- lab_chloride_wb_max
- lab_chloride_wb_max_24h
- lab_chloride_wb_mean
- lab_chloride_wb_mean_24h
- lab_chloride_wb_median
- lab_chloride_wb_median_24h
- lab_chloride_wb_min
- lab_chloride_wb_min_24h
- lab_chloride_wb_slope
- lab_chloride_wb_slope_24h
- lab_chloride_wb_std
- lab_chloride_wb_std_24h
- lab_creatinine_change_24h
- lab_creatinine_slope_24h
- lab_creatinine_std_24h
- lab_creatinine_wb_change
- lab_creatinine_wb_change_24h
- lab_creatinine_wb_first
- lab_creatinine_wb_first_24h
- lab_creatinine_wb_last
- lab_creatinine_wb_last_24h
- lab_creatinine_wb_max
- lab_creatinine_wb_max_24h
- lab_creatinine_wb_mean
- lab_creatinine_wb_mean_24h
- lab_creatinine_wb_median
- lab_creatinine_wb_median_24h
- lab_creatinine_wb_min
- lab_creatinine_wb_min_24h
- lab_creatinine_wb_slope
- _... and 117 more_

### Constant / Zero Variance
- _invalid_time_order
- _is_duplicate
- _is_duplicate
- cci_aids
- lab_creatinine_wb_abnormal_count_24h
- lab_creatinine_wb_change
- lab_creatinine_wb_change_24h
- lab_creatinine_wb_count_24h
- lab_creatinine_wb_slope
- lab_creatinine_wb_slope_24h
- lab_creatinine_wb_std
- lab_creatinine_wb_std_24h
- lab_hematocrit_wb_abnormal_count
- lab_hematocrit_wb_abnormal_count
- lab_hematocrit_wb_abnormal_count_24h
- lab_hematocrit_wb_abnormal_count_24h
- note_count
- readability_flesch
- readability_flesch

### Duplicates / Highly Correlated
- admit_year
- cci_renal_disease
- char_count
- diagnosis_count
- icu_duration_hours
- lab_anion_gap_count
- lab_anion_gap_count
- lab_anion_gap_count
- lab_anion_gap_count
- lab_anion_gap_count
- lab_anion_gap_count
- lab_anion_gap_count
- lab_anion_gap_count
- lab_anion_gap_count
- lab_anion_gap_count
- lab_anion_gap_count
- lab_anion_gap_count
- lab_anion_gap_count_24h
- lab_anion_gap_count_24h
- lab_anion_gap_count_24h
- lab_anion_gap_count_24h
- lab_anion_gap_count_24h
- lab_anion_gap_count_24h
- lab_anion_gap_mean
- lab_anion_gap_mean_24h
- lab_anion_gap_mean_24h
- lab_anion_gap_missing_ratio
- lab_anion_gap_missing_ratio
- lab_anion_gap_missing_ratio
- lab_anion_gap_missing_ratio_24h
- lab_bicarbonate_count
- lab_bicarbonate_count
- lab_bicarbonate_count
- lab_bicarbonate_count
- lab_bicarbonate_count
- lab_bicarbonate_count
- lab_bicarbonate_count
- lab_bicarbonate_count
- lab_bicarbonate_count
- lab_bicarbonate_count
- lab_bicarbonate_count
- lab_bicarbonate_count_24h
- lab_bicarbonate_count_24h
- lab_bicarbonate_count_24h
- lab_bicarbonate_count_24h
- lab_bicarbonate_count_24h
- lab_bicarbonate_max_24h
- lab_bicarbonate_mean
- lab_bicarbonate_mean_24h
- lab_bicarbonate_mean_24h
- _... and more_
