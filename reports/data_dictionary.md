# Data Dictionary

Generated: 2026-08-01T11:57:01.954824Z

## Dataset Summary

| dataset                  |   n_rows |   n_cols |
|:-------------------------|---------:|---------:|
| patient_level            |   223452 |       14 |
| admission_level          |   546028 |      565 |
| icu_level                |    94458 |      136 |
| time_series              | 71323299 |       10 |
| clinical_notes           |   331793 |       13 |
| similarity               |   546028 |       35 |
| admission_level_selected |   546028 |      275 |

## Feature Dictionary (sample)

| dataset         | feature                              | dtype          |   n_missing |   pct_missing |   n_unique | example                |
|:----------------|:-------------------------------------|:---------------|------------:|--------------:|-----------:|:-----------------------|
| patient_level   | subject_id                           | int32          |           0 |          0    |     223452 | 10000032               |
| patient_level   | n_admissions                         | int64          |           0 |          0    |         97 | 4                      |
| patient_level   | los_days_mean                        | float64        |           0 |          0    |      79406 | 1.4444444444444444     |
| patient_level   | los_days_max                         | float64        |           0 |          0    |      36730 | 2.2222222222222223     |
| patient_level   | los_days_sum                         | float64        |           0 |          0    |      62960 | 5.777777777777778      |
| patient_level   | ever_inhosp_mortality                | int8           |           0 |          0    |          2 | 0                      |
| patient_level   | ever_readmission_30d                 | int8           |           0 |          0    |          2 | 1                      |
| patient_level   | ever_icu_stay                        | int8           |           0 |          0    |          2 | 1                      |
| patient_level   | insurance                            | category       |        5723 |          2.56 |          5 | Medicaid               |
| patient_level   | race                                 | category       |           0 |          0    |         33 | WHITE                  |
| patient_level   | gender                               | category       |           0 |          0    |          2 | F                      |
| patient_level   | anchor_age                           | int8           |           0 |          0    |         73 | 52                     |
| patient_level   | anchor_year                          | int16          |           0 |          0    |         99 | 2180                   |
| patient_level   | anchor_year_group                    | category       |           0 |          0    |          5 | 2014 - 2016            |
| admission_level | subject_id                           | int32          |           0 |          0    |     223452 | 10000032               |
| admission_level | hadm_id                              | int32          |           0 |          0    |     546028 | 22595853               |
| admission_level | admittime                            | datetime64[us] |           0 |          0    |     534919 | 2180-05-06 22:23:00    |
| admission_level | dischtime                            | datetime64[us] |           0 |          0    |     528871 | 2180-05-07 17:15:00    |
| admission_level | deathtime                            | datetime64[us] |      534238 |         97.84 |      11788 | 2134-12-06 12:54:00    |
| admission_level | admission_type                       | category       |           0 |          0    |          9 | URGENT                 |
| admission_level | admit_provider_id                    | category       |           4 |          0    |       2045 | P49AFC                 |
| admission_level | admission_location                   | category       |           1 |          0    |         11 | TRANSFER FROM HOSPITAL |
| admission_level | discharge_location                   | category       |      149818 |         27.44 |         13 | HOME                   |
| admission_level | insurance                            | category       |        9355 |          1.71 |          5 | Medicaid               |
| admission_level | language                             | category       |         775 |          0.14 |         25 | English                |
| admission_level | marital_status                       | category       |       13619 |          2.49 |          4 | WIDOWED                |
| admission_level | race                                 | category       |           0 |          0    |         33 | WHITE                  |
| admission_level | edregtime                            | datetime64[us] |      166788 |         30.55 |     372692 | 2180-05-06 19:17:00    |
| admission_level | edouttime                            | datetime64[us] |      166788 |         30.55 |     372755 | 2180-05-06 23:30:00    |
| admission_level | hospital_expire_flag                 | int8           |           0 |          0    |          2 | 0                      |
| admission_level | _is_duplicate                        | int8           |           0 |          0    |          1 | 0                      |
| admission_level | _invalid_time_order                  | int8           |           0 |          0    |          2 | 0                      |
| admission_level | n_icu_stays                          | float64        |           0 |          0    |         10 | 0.0                    |
| admission_level | icu_los_days                         | float32        |           0 |          0    |      77872 | 0.0                    |
| admission_level | has_icu_stay                         | int8           |           0 |          0    |          2 | 0                      |
| admission_level | los_days                             | float64        |           0 |          0    |      39931 | 0.7861111111111111     |
| admission_level | los_hours                            | float64        |           0 |          0    |      39931 | 18.866666666666667     |
| admission_level | admit_hour                           | int32          |           0 |          0    |         24 | 22                     |
| admission_level | admit_dow                            | int32          |           0 |          0    |          7 | 5                      |
| admission_level | admit_month                          | int32          |           0 |          0    |         12 | 5                      |
| admission_level | admit_year                           | int32          |           0 |          0    |        108 | 2180                   |
| admission_level | weekend_admission                    | int8           |           0 |          0    |          2 | 1                      |
| admission_level | night_admission                      | int8           |           0 |          0    |          2 | 1                      |
| admission_level | next_admittime                       | datetime64[us] |      223452 |         40.92 |     318726 | 2180-06-26 18:27:00    |
| admission_level | days_to_readmission                  | float64        |      223452 |         40.92 |     222636 | 50.05                  |
| admission_level | readmission_30d                      | int8           |           0 |          0    |          2 | 0                      |
| admission_level | gender                               | category       |           0 |          0    |          2 | F                      |
| admission_level | anchor_age                           | int8           |           0 |          0    |         73 | 52                     |
| admission_level | anchor_year                          | int16          |           0 |          0    |         99 | 2180                   |
| admission_level | anchor_year_group                    | category       |           0 |          0    |          5 | 2014 - 2016            |
| admission_level | dod                                  | datetime64[us] |      401062 |         73.45 |      21981 | 2180-09-09 00:00:00    |
| admission_level | lab_anion_gap_mean_24h               | float32        |      175614 |         32.16 |        548 | 9.0                    |
| admission_level | lab_bicarbonate_mean_24h             | float32        |      175419 |         32.13 |        590 | 28.0                   |
| admission_level | lab_bun_mean_24h                     | float32        |      170390 |         31.21 |       1739 | 25.0                   |
| admission_level | lab_chloride_mean_24h                | float32        |      172258 |         31.55 |        892 | 105.0                  |
| admission_level | lab_chloride_wb_mean_24h             | float32        |      523733 |         95.92 |        511 | 104.6                  |
| admission_level | lab_creatinine_mean_24h              | float32        |      167239 |         30.63 |       1961 | 0.3                    |
| admission_level | lab_creatinine_wb_mean_24h           | float32        |      543513 |         99.54 |        255 | 0.8                    |
| admission_level | lab_glucose_mean_24h                 | float32        |      178254 |         32.65 |       3739 | 99.0                   |
| admission_level | lab_glucose_poc_mean_24h             | float32        |      518834 |         95.02 |       3787 | 173.5                  |
| admission_level | lab_hematocrit_mean_24h              | float32        |      154457 |         28.29 |       4692 | 37.6                   |
| admission_level | lab_hematocrit_wb_mean_24h           | float32        |      523533 |         95.88 |       1551 | 25.0                   |
| admission_level | lab_hemoglobin_mean_24h              | float32        |      162680 |         29.79 |       1978 | 12.7                   |
| admission_level | lab_hemoglobin_wb_mean_24h           | float32        |      523533 |         95.88 |       1678 | 8.2                    |
| admission_level | lab_platelets_mean_24h               | float32        |      160372 |         29.37 |       5462 | 71.0                   |
| admission_level | lab_potassium_mean_24h               | float32        |      169631 |         31.07 |       1076 | 4.5                    |
| admission_level | lab_potassium_wb_mean_24h            | float32        |      510401 |         93.48 |       1413 | 3.7                    |
| admission_level | lab_sodium_mean_24h                  | float32        |      172887 |         31.66 |        889 | 137.0                  |
| admission_level | lab_sodium_wb_mean_24h               | float32        |      520770 |         95.37 |        839 | 137.0                  |
| admission_level | lab_wbc_mean_24h                     | float32        |      162724 |         29.8  |       5160 | 4.2                    |
| admission_level | lab_anion_gap_median_24h             | float32        |      175614 |         32.16 |         94 | 9.0                    |
| admission_level | lab_bicarbonate_median_24h           | float32        |      175419 |         32.13 |        104 | 28.0                   |
| admission_level | lab_bun_median_24h                   | float32        |      170390 |         31.21 |        405 | 25.0                   |
| admission_level | lab_chloride_median_24h              | float32        |      172258 |         31.55 |        146 | 105.0                  |
| admission_level | lab_chloride_wb_median_24h           | float32        |      523733 |         95.92 |        130 | 104.0                  |
| admission_level | lab_creatinine_median_24h            | float32        |      167239 |         30.63 |        590 | 0.3                    |
| admission_level | lab_creatinine_wb_median_24h         | float32        |      543513 |         99.54 |        215 | 0.8                    |
| admission_level | lab_glucose_median_24h               | float32        |      178254 |         32.65 |       1170 | 99.0                   |
| admission_level | lab_glucose_poc_median_24h           | float32        |      518834 |         95.02 |        854 | 154.5                  |
| admission_level | lab_hematocrit_median_24h            | float32        |      154457 |         28.29 |       1221 | 37.6                   |
| admission_level | lab_hematocrit_wb_median_24h         | float32        |      523533 |         95.88 |        677 | 25.0                   |
| admission_level | lab_hemoglobin_median_24h            | float32        |      162680 |         29.79 |        465 | 12.7                   |
| admission_level | lab_hemoglobin_wb_median_24h         | float32        |      523533 |         95.88 |        374 | 8.2                    |
| admission_level | lab_platelets_median_24h             | float32        |      160372 |         29.37 |       1975 | 71.0                   |
| admission_level | lab_potassium_median_24h             | float32        |      169631 |         31.07 |        204 | 4.5                    |
| admission_level | lab_potassium_wb_median_24h          | float32        |      510401 |         93.48 |        212 | 3.7                    |
| admission_level | lab_sodium_median_24h                | float32        |      172887 |         31.66 |        143 | 137.0                  |
| admission_level | lab_sodium_wb_median_24h             | float32        |      520770 |         95.37 |        136 | 137.0                  |
| admission_level | lab_wbc_median_24h                   | float32        |      162724 |         29.8  |       2040 | 4.2                    |
| admission_level | lab_anion_gap_min_24h                | float32        |      175614 |         32.16 |         56 | 9.0                    |
| admission_level | lab_bicarbonate_min_24h              | float32        |      175419 |         32.13 |         60 | 28.0                   |
| admission_level | lab_bun_min_24h                      | float32        |      170390 |         31.21 |        215 | 25.0                   |
| admission_level | lab_chloride_min_24h                 | float32        |      172258 |         31.55 |         89 | 105.0                  |
| admission_level | lab_chloride_wb_min_24h              | float32        |      523733 |         95.92 |         82 | 101.0                  |
| admission_level | lab_creatinine_min_24h               | float32        |      167239 |         30.63 |        243 | 0.3                    |
| admission_level | lab_creatinine_wb_min_24h            | float32        |      543513 |         99.54 |        168 | 0.8                    |
| admission_level | lab_glucose_min_24h                  | float32        |      178254 |         32.65 |        644 | 99.0                   |
| admission_level | lab_glucose_poc_min_24h              | float32        |      518834 |         95.02 |        554 | 99.0                   |
| admission_level | lab_hematocrit_min_24h               | float32        |      154457 |         28.29 |        535 | 37.6                   |
| admission_level | lab_hematocrit_wb_min_24h            | float32        |      523533 |         95.88 |        382 | 25.0                   |
| admission_level | lab_hemoglobin_min_24h               | float32        |      162680 |         29.79 |        189 | 12.7                   |
| admission_level | lab_hemoglobin_wb_min_24h            | float32        |      523533 |         95.88 |        174 | 8.2                    |
| admission_level | lab_platelets_min_24h                | float32        |      160372 |         29.37 |       1128 | 71.0                   |
| admission_level | lab_potassium_min_24h                | float32        |      169631 |         31.07 |         88 | 4.5                    |
| admission_level | lab_potassium_wb_min_24h             | float32        |      510401 |         93.48 |        104 | 3.7                    |
| admission_level | lab_sodium_min_24h                   | float32        |      172887 |         31.66 |         90 | 137.0                  |
| admission_level | lab_sodium_wb_min_24h                | float32        |      520770 |         95.37 |         83 | 136.0                  |
| admission_level | lab_wbc_min_24h                      | float32        |      162724 |         29.8  |       1066 | 4.2                    |
| admission_level | lab_anion_gap_max_24h                | float32        |      175614 |         32.16 |         60 | 9.0                    |
| admission_level | lab_bicarbonate_max_24h              | float32        |      175419 |         32.13 |         56 | 28.0                   |
| admission_level | lab_bun_max_24h                      | float32        |      170390 |         31.21 |        229 | 25.0                   |
| admission_level | lab_chloride_max_24h                 | float32        |      172258 |         31.55 |         83 | 105.0                  |
| admission_level | lab_chloride_wb_max_24h              | float32        |      523733 |         95.92 |         79 | 109.0                  |
| admission_level | lab_creatinine_max_24h               | float32        |      167239 |         30.63 |        267 | 0.3                    |
| admission_level | lab_creatinine_wb_max_24h            | float32        |      543513 |         99.54 |        172 | 0.8                    |
| admission_level | lab_glucose_max_24h                  | float32        |      178254 |         32.65 |       1006 | 99.0                   |
| admission_level | lab_glucose_poc_max_24h              | float32        |      518834 |         95.02 |        597 | 332.0                  |
| admission_level | lab_hematocrit_max_24h               | float32        |      154457 |         28.29 |        497 | 37.6                   |
| admission_level | lab_hematocrit_wb_max_24h            | float32        |      523533 |         95.88 |        375 | 25.0                   |
| admission_level | lab_hemoglobin_max_24h               | float32        |      162680 |         29.79 |        180 | 12.7                   |
| admission_level | lab_hemoglobin_wb_max_24h            | float32        |      523533 |         95.88 |        173 | 8.2                    |
| admission_level | lab_platelets_max_24h                | float32        |      160372 |         29.37 |       1179 | 71.0                   |
| admission_level | lab_potassium_max_24h                | float32        |      169631 |         31.07 |         86 | 4.5                    |
| admission_level | lab_potassium_wb_max_24h             | float32        |      510401 |         93.48 |        125 | 3.7                    |
| admission_level | lab_sodium_max_24h                   | float32        |      172887 |         31.66 |         78 | 137.0                  |
| admission_level | lab_sodium_wb_max_24h                | float32        |      520770 |         95.37 |         81 | 138.0                  |
| admission_level | lab_wbc_max_24h                      | float32        |      162724 |         29.8  |       1162 | 4.2                    |
| admission_level | lab_anion_gap_std_24h                | float32        |      412575 |         75.56 |       1210 | 3.535534               |
| admission_level | lab_bicarbonate_std_24h              | float32        |      411682 |         75.4  |        934 | 2.1213202              |
| admission_level | lab_bun_std_24h                      | float32        |      409354 |         74.97 |       2096 | 3.535534               |
| admission_level | lab_chloride_std_24h                 | float32        |      406011 |         74.36 |       1318 | 0.0                    |
| admission_level | lab_chloride_wb_std_24h              | float32        |      535041 |         97.99 |        632 | 3.04959                |
| admission_level | lab_creatinine_std_24h               | float32        |      408007 |         74.72 |       3456 | 0.070710674            |
| admission_level | lab_creatinine_wb_std_24h            | float32        |      545778 |         99.95 |         99 | 0.1414213              |
| admission_level | lab_glucose_std_24h                  | float32        |      416547 |         76.29 |      12125 | 5.656854               |
| admission_level | lab_glucose_poc_std_24h              | float32        |      531986 |         97.43 |       8634 | 63.564926              |
| admission_level | lab_hematocrit_std_24h               | float32        |      404205 |         74.03 |      21072 | 1.1313698              |
| admission_level | lab_hematocrit_wb_std_24h            | float32        |      533767 |         97.75 |       2612 | 2.1679482              |
| admission_level | lab_hemoglobin_std_24h               | float32        |      416491 |         76.28 |       8114 | 0.35355338             |
| admission_level | lab_hemoglobin_wb_std_24h            | float32        |      533768 |         97.75 |       6389 | 0.719027               |
| admission_level | lab_platelets_std_24h                | float32        |      414135 |         75.85 |       8987 | 16.263456              |
| admission_level | lab_potassium_std_24h                | float32        |      402366 |         73.69 |       5466 | 0.21213217             |
| admission_level | lab_potassium_wb_std_24h             | float32        |      530047 |         97.07 |       6336 | 0.4652188              |
| admission_level | lab_sodium_std_24h                   | float32        |      406308 |         74.41 |       1170 | 1.4142135              |
| admission_level | lab_sodium_wb_std_24h                | float32        |      533124 |         97.64 |        817 | 0.8944272              |
| admission_level | lab_wbc_std_24h                      | float32        |      417437 |         76.45 |      14237 | 0.35355338             |
| admission_level | lab_anion_gap_count_24h              | float32        |      143008 |         26.19 |         16 | 1.0                    |
| admission_level | lab_bicarbonate_count_24h            | float32        |      143008 |         26.19 |         16 | 1.0                    |
| admission_level | lab_bun_count_24h                    | float32        |      143008 |         26.19 |         16 | 1.0                    |
| admission_level | lab_chloride_count_24h               | float32        |      143008 |         26.19 |         16 | 1.0                    |
| admission_level | lab_chloride_wb_count_24h            | float32        |      143008 |         26.19 |         16 | 0.0                    |
| admission_level | lab_creatinine_count_24h             | float32        |      143008 |         26.19 |         16 | 1.0                    |
| admission_level | lab_creatinine_wb_count_24h          | float32        |      143008 |         26.19 |         11 | 0.0                    |
| admission_level | lab_glucose_count_24h                | float32        |      143008 |         26.19 |         16 | 1.0                    |
| admission_level | lab_glucose_poc_count_24h            | float32        |      143008 |         26.19 |         24 | 0.0                    |
| admission_level | lab_hematocrit_count_24h             | float32        |      143008 |         26.19 |         15 | 1.0                    |
| admission_level | lab_hematocrit_wb_count_24h          | float32        |      143008 |         26.19 |         18 | 0.0                    |
| admission_level | lab_hemoglobin_count_24h             | float32        |      143008 |         26.19 |         15 | 1.0                    |
| admission_level | lab_hemoglobin_wb_count_24h          | float32        |      143008 |         26.19 |         18 | 0.0                    |
| admission_level | lab_platelets_count_24h              | float32        |      143008 |         26.19 |         15 | 1.0                    |
| admission_level | lab_potassium_count_24h              | float32        |      143008 |         26.19 |         16 | 1.0                    |
| admission_level | lab_potassium_wb_count_24h           | float32        |      143008 |         26.19 |         23 | 0.0                    |
| admission_level | lab_sodium_count_24h                 | float32        |      143008 |         26.19 |         16 | 1.0                    |
| admission_level | lab_sodium_wb_count_24h              | float32        |      143008 |         26.19 |         21 | 0.0                    |
| admission_level | lab_wbc_count_24h                    | float32        |      143008 |         26.19 |         15 | 1.0                    |
| admission_level | lab_anion_gap_missing_ratio_24h      | float32        |      143008 |         26.19 |          6 | 0.0                    |
| admission_level | lab_bicarbonate_missing_ratio_24h    | float32        |      143008 |         26.19 |         17 | 0.0                    |
| admission_level | lab_bun_missing_ratio_24h            | float32        |      143008 |         26.19 |         12 | 0.0                    |
| admission_level | lab_chloride_missing_ratio_24h       | float32        |      143008 |         26.19 |         12 | 0.0                    |
| admission_level | lab_chloride_wb_missing_ratio_24h    | float32        |      143008 |         26.19 |          8 | 1.0                    |
| admission_level | lab_creatinine_missing_ratio_24h     | float32        |      143008 |         26.19 |         10 | 0.0                    |
| admission_level | lab_creatinine_wb_missing_ratio_24h  | float32        |      143008 |         26.19 |          5 | 1.0                    |
| admission_level | lab_glucose_missing_ratio_24h        | float32        |      143008 |         26.19 |          8 | 0.0                    |
| admission_level | lab_glucose_poc_missing_ratio_24h    | float32        |      143008 |         26.19 |         28 | 1.0                    |
| admission_level | lab_hematocrit_missing_ratio_24h     | float32        |      143008 |         26.19 |         19 | 0.0                    |
| admission_level | lab_hematocrit_wb_missing_ratio_24h  | float32        |      143008 |         26.19 |          5 | 1.0                    |
| admission_level | lab_hemoglobin_missing_ratio_24h     | float32        |      143008 |         26.19 |         17 | 0.0                    |
| admission_level | lab_hemoglobin_wb_missing_ratio_24h  | float32        |      143008 |         26.19 |          3 | 1.0                    |
| admission_level | lab_platelets_missing_ratio_24h      | float32        |      143008 |         26.19 |         21 | 0.0                    |
| admission_level | lab_potassium_missing_ratio_24h      | float32        |      143008 |         26.19 |         12 | 0.0                    |
| admission_level | lab_potassium_wb_missing_ratio_24h   | float32        |      143008 |         26.19 |         11 | 1.0                    |
| admission_level | lab_sodium_missing_ratio_24h         | float32        |      143008 |         26.19 |          9 | 0.0                    |
| admission_level | lab_sodium_wb_missing_ratio_24h      | float32        |      143008 |         26.19 |          8 | 1.0                    |
| admission_level | lab_wbc_missing_ratio_24h            | float32        |      143008 |         26.19 |         16 | 0.0                    |
| admission_level | lab_anion_gap_abnormal_count_24h     | float32        |      143008 |         26.19 |         13 | 0.0                    |
| admission_level | lab_bicarbonate_abnormal_count_24h   | float32        |      143008 |         26.19 |         14 | 0.0                    |
| admission_level | lab_bun_abnormal_count_24h           | float32        |      143008 |         26.19 |         14 | 1.0                    |
| admission_level | lab_chloride_abnormal_count_24h      | float32        |      143008 |         26.19 |         12 | 0.0                    |
| admission_level | lab_chloride_wb_abnormal_count_24h   | float32        |      143008 |         26.19 |         14 | 0.0                    |
| admission_level | lab_creatinine_abnormal_count_24h    | float32        |      143008 |         26.19 |         14 | 1.0                    |
| admission_level | lab_creatinine_wb_abnormal_count_24h | float32        |      143008 |         26.19 |         10 | 0.0                    |
| admission_level | lab_glucose_abnormal_count_24h       | float32        |      143008 |         26.19 |         14 | 0.0                    |
| admission_level | lab_glucose_poc_abnormal_count_24h   | float32        |      143008 |         26.19 |         23 | 0.0                    |
| admission_level | lab_hematocrit_abnormal_count_24h    | float32        |      143008 |         26.19 |         14 | 0.0                    |
| admission_level | lab_hematocrit_wb_abnormal_count_24h | float32        |      143008 |         26.19 |          1 | 0.0                    |
| admission_level | lab_hemoglobin_abnormal_count_24h    | float32        |      143008 |         26.19 |         14 | 0.0                    |
| admission_level | lab_hemoglobin_wb_abnormal_count_24h | float32        |      143008 |         26.19 |         17 | 0.0                    |
| admission_level | lab_platelets_abnormal_count_24h     | float32        |      143008 |         26.19 |         15 | 1.0                    |
| admission_level | lab_potassium_abnormal_count_24h     | float32        |      143008 |         26.19 |         12 | 0.0                    |
| admission_level | lab_potassium_wb_abnormal_count_24h  | float32        |      143008 |         26.19 |         16 | 0.0                    |