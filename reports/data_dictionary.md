# Data Dictionary

Generated: 2026-07-28T11:30:48.545771Z

## Dataset Summary

| dataset                  |   n_rows |   n_cols |
|:-------------------------|---------:|---------:|
| patient_level            |   223452 |       14 |
| admission_level          |   546028 |      335 |
| icu_level                |    94458 |      136 |
| time_series              | 71323299 |       10 |
| clinical_notes           |   331793 |       13 |
| similarity               |   546028 |       35 |
| admission_level_selected |   546028 |      204 |

## Feature Dictionary (sample)

| dataset         | feature                          | dtype          |   n_missing |   pct_missing |   n_unique | example                |
|:----------------|:---------------------------------|:---------------|------------:|--------------:|-----------:|:-----------------------|
| patient_level   | subject_id                       | int32          |           0 |          0    |     223452 | 10000032               |
| patient_level   | n_admissions                     | int64          |           0 |          0    |         97 | 4                      |
| patient_level   | los_days_mean                    | float64        |           0 |          0    |      79406 | 1.4444444444444444     |
| patient_level   | los_days_max                     | float64        |           0 |          0    |      36730 | 2.2222222222222223     |
| patient_level   | los_days_sum                     | float64        |           0 |          0    |      62960 | 5.777777777777778      |
| patient_level   | ever_inhosp_mortality            | int8           |           0 |          0    |          2 | 0                      |
| patient_level   | ever_readmission_30d             | int8           |           0 |          0    |          2 | 1                      |
| patient_level   | ever_icu_stay                    | int8           |           0 |          0    |          2 | 1                      |
| patient_level   | insurance                        | category       |        5723 |          2.56 |          5 | Medicaid               |
| patient_level   | race                             | category       |           0 |          0    |         33 | WHITE                  |
| patient_level   | gender                           | category       |           0 |          0    |          2 | F                      |
| patient_level   | anchor_age                       | int8           |           0 |          0    |         73 | 52                     |
| patient_level   | anchor_year                      | int16          |           0 |          0    |         99 | 2180                   |
| patient_level   | anchor_year_group                | category       |           0 |          0    |          5 | 2014 - 2016            |
| admission_level | subject_id                       | int32          |           0 |          0    |     223452 | 10000032               |
| admission_level | hadm_id                          | int32          |           0 |          0    |     546028 | 22595853               |
| admission_level | admittime                        | datetime64[us] |           0 |          0    |     534919 | 2180-05-06 22:23:00    |
| admission_level | dischtime                        | datetime64[us] |           0 |          0    |     528871 | 2180-05-07 17:15:00    |
| admission_level | deathtime                        | datetime64[us] |      534238 |         97.84 |      11788 | 2134-12-06 12:54:00    |
| admission_level | admission_type                   | category       |           0 |          0    |          9 | URGENT                 |
| admission_level | admit_provider_id                | category       |           4 |          0    |       2045 | P49AFC                 |
| admission_level | admission_location               | category       |           1 |          0    |         11 | TRANSFER FROM HOSPITAL |
| admission_level | discharge_location               | category       |      149818 |         27.44 |         13 | HOME                   |
| admission_level | insurance                        | category       |        9355 |          1.71 |          5 | Medicaid               |
| admission_level | language                         | category       |         775 |          0.14 |         25 | English                |
| admission_level | marital_status                   | category       |       13619 |          2.49 |          4 | WIDOWED                |
| admission_level | race                             | category       |           0 |          0    |         33 | WHITE                  |
| admission_level | edregtime                        | datetime64[us] |      166788 |         30.55 |     372692 | 2180-05-06 19:17:00    |
| admission_level | edouttime                        | datetime64[us] |      166788 |         30.55 |     372755 | 2180-05-06 23:30:00    |
| admission_level | hospital_expire_flag             | int8           |           0 |          0    |          2 | 0                      |
| admission_level | _is_duplicate                    | int8           |           0 |          0    |          1 | 0                      |
| admission_level | _invalid_time_order              | int8           |           0 |          0    |          2 | 0                      |
| admission_level | n_icu_stays                      | float64        |           0 |          0    |         10 | 0.0                    |
| admission_level | icu_los_days                     | float32        |           0 |          0    |      77872 | 0.0                    |
| admission_level | has_icu_stay                     | int8           |           0 |          0    |          2 | 0                      |
| admission_level | los_days                         | float64        |           0 |          0    |      39931 | 0.7861111111111111     |
| admission_level | los_hours                        | float64        |           0 |          0    |      39931 | 18.866666666666667     |
| admission_level | admit_hour                       | int32          |           0 |          0    |         24 | 22                     |
| admission_level | admit_dow                        | int32          |           0 |          0    |          7 | 5                      |
| admission_level | admit_month                      | int32          |           0 |          0    |         12 | 5                      |
| admission_level | admit_year                       | int32          |           0 |          0    |        108 | 2180                   |
| admission_level | weekend_admission                | int8           |           0 |          0    |          2 | 1                      |
| admission_level | night_admission                  | int8           |           0 |          0    |          2 | 1                      |
| admission_level | next_admittime                   | datetime64[us] |      223452 |         40.92 |     318726 | 2180-06-26 18:27:00    |
| admission_level | days_to_readmission              | float64        |      223452 |         40.92 |     222636 | 50.05                  |
| admission_level | readmission_30d                  | int8           |           0 |          0    |          2 | 0                      |
| admission_level | gender                           | category       |           0 |          0    |          2 | F                      |
| admission_level | anchor_age                       | int8           |           0 |          0    |         73 | 52                     |
| admission_level | anchor_year                      | int16          |           0 |          0    |         99 | 2180                   |
| admission_level | anchor_year_group                | category       |           0 |          0    |          5 | 2014 - 2016            |
| admission_level | dod                              | datetime64[us] |      401062 |         73.45 |      21981 | 2180-09-09 00:00:00    |
| admission_level | procedure_count                  | float64        |      258524 |         47.35 |         41 | 1.0                    |
| admission_level | unique_procedure_count           | float64        |      258524 |         47.35 |         36 | 1.0                    |
| admission_level | major_procedure_count            | float64        |      258524 |         47.35 |         41 | 1.0                    |
| admission_level | has_major_procedure              | float64        |      258524 |         47.35 |          2 | 1.0                    |
| admission_level | lab_anion_gap_mean               | float32        |      139168 |         25.49 |       9045 | 9.0                    |
| admission_level | lab_bicarbonate_mean             | float32        |      139133 |         25.48 |      10531 | 28.0                   |
| admission_level | lab_bun_mean                     | float32        |      134160 |         24.57 |      20805 | 25.0                   |
| admission_level | lab_chloride_mean                | float32        |      136702 |         25.04 |      12571 | 105.0                  |
| admission_level | lab_chloride_wb_mean             | float32        |      510090 |         93.42 |        886 | 90.0                   |
| admission_level | lab_creatinine_mean              | float32        |      130195 |         23.84 |      19035 | 0.3                    |
| admission_level | lab_creatinine_wb_mean           | float32        |      542035 |         99.27 |        480 | 0.8                    |
| admission_level | lab_glucose_mean                 | float32        |      139776 |         25.6  |      29036 | 99.0                   |
| admission_level | lab_glucose_poc_mean             | float32        |      502359 |         92    |       7095 | 140.0                  |
| admission_level | lab_hematocrit_mean              | float32        |      115472 |         21.15 |      35813 | 37.6                   |
| admission_level | lab_hematocrit_wb_mean           | float32        |      508597 |         93.14 |       2875 | 25.0                   |
| admission_level | lab_hemoglobin_mean              | float32        |      123752 |         22.66 |      19901 | 12.7                   |
| admission_level | lab_hemoglobin_wb_mean           | float32        |      508596 |         93.14 |       2964 | 8.2                    |
| admission_level | lab_platelets_mean               | float32        |      122042 |         22.35 |      45525 | 71.0                   |
| admission_level | lab_potassium_mean               | float32        |      134495 |         24.63 |      11852 | 4.5                    |
| admission_level | lab_potassium_wb_mean            | float32        |      490754 |         89.88 |       3380 | 3.7                    |
| admission_level | lab_sodium_mean                  | float32        |      136233 |         24.95 |      11577 | 137.0                  |
| admission_level | lab_sodium_wb_mean               | float32        |      504784 |         92.45 |       1716 | 134.5                  |
| admission_level | lab_wbc_mean                     | float32        |      124478 |         22.8  |      33739 | 4.2                    |
| admission_level | lab_anion_gap_median             | float32        |      139168 |         25.49 |         89 | 9.0                    |
| admission_level | lab_bicarbonate_median           | float32        |      139133 |         25.48 |        100 | 28.0                   |
| admission_level | lab_bun_median                   | float32        |      134160 |         24.57 |        360 | 25.0                   |
| admission_level | lab_chloride_median              | float32        |      136702 |         25.04 |        122 | 105.0                  |
| admission_level | lab_chloride_wb_median           | float32        |      510090 |         93.42 |        137 | 90.0                   |
| admission_level | lab_creatinine_median            | float32        |      130195 |         23.84 |        552 | 0.3                    |
| admission_level | lab_creatinine_wb_median         | float32        |      542035 |         99.27 |        278 | 0.8                    |
| admission_level | lab_glucose_median               | float32        |      139776 |         25.6  |        979 | 99.0                   |
| admission_level | lab_glucose_poc_median           | float32        |      502359 |         92    |        878 | 140.0                  |
| admission_level | lab_hematocrit_median            | float32        |      115472 |         21.15 |       1140 | 37.6                   |
| admission_level | lab_hematocrit_wb_median         | float32        |      508597 |         93.14 |        784 | 25.0                   |
| admission_level | lab_hemoglobin_median            | float32        |      123752 |         22.66 |        427 | 12.7                   |
| admission_level | lab_hemoglobin_wb_median         | float32        |      508596 |         93.14 |        399 | 8.2                    |
| admission_level | lab_platelets_median             | float32        |      122042 |         22.35 |       1988 | 71.0                   |
| admission_level | lab_potassium_median             | float32        |      134495 |         24.63 |        174 | 4.5                    |
| admission_level | lab_potassium_wb_median          | float32        |      490754 |         89.88 |        226 | 3.7                    |
| admission_level | lab_sodium_median                | float32        |      136233 |         24.95 |        120 | 137.0                  |
| admission_level | lab_sodium_wb_median             | float32        |      504784 |         92.45 |        126 | 134.5                  |
| admission_level | lab_wbc_median                   | float32        |      124478 |         22.8  |       1861 | 4.2                    |
| admission_level | lab_anion_gap_min                | float32        |      139168 |         25.49 |         66 | 9.0                    |
| admission_level | lab_bicarbonate_min              | float32        |      139133 |         25.48 |         68 | 28.0                   |
| admission_level | lab_bun_min                      | float32        |      134160 |         24.57 |        184 | 25.0                   |
| admission_level | lab_chloride_min                 | float32        |      136702 |         25.04 |         84 | 105.0                  |
| admission_level | lab_chloride_wb_min              | float32        |      510090 |         93.42 |         84 | 90.0                   |
| admission_level | lab_creatinine_min               | float32        |      130195 |         23.84 |        211 | 0.3                    |
| admission_level | lab_creatinine_wb_min            | float32        |      542035 |         99.27 |        188 | 0.8                    |
| admission_level | lab_glucose_min                  | float32        |      139776 |         25.6  |        596 | 99.0                   |
| admission_level | lab_glucose_poc_min              | float32        |      502359 |         92    |        564 | 140.0                  |
| admission_level | lab_hematocrit_min               | float32        |      115472 |         21.15 |        523 | 37.6                   |
| admission_level | lab_hematocrit_wb_min            | float32        |      508597 |         93.14 |        419 | 25.0                   |
| admission_level | lab_hemoglobin_min               | float32        |      123752 |         22.66 |        183 | 12.7                   |
| admission_level | lab_hemoglobin_wb_min            | float32        |      508596 |         93.14 |        182 | 8.2                    |
| admission_level | lab_platelets_min                | float32        |      122042 |         22.35 |       1036 | 71.0                   |
| admission_level | lab_potassium_min                | float32        |      134495 |         24.63 |         90 | 4.5                    |
| admission_level | lab_potassium_wb_min             | float32        |      490754 |         89.88 |        108 | 3.7                    |
| admission_level | lab_sodium_min                   | float32        |      136233 |         24.95 |         93 | 137.0                  |
| admission_level | lab_sodium_wb_min                | float32        |      504784 |         92.45 |         87 | 134.0                  |
| admission_level | lab_wbc_min                      | float32        |      124478 |         22.8  |        885 | 4.2                    |
| admission_level | lab_anion_gap_max                | float32        |      139168 |         25.49 |         67 | 9.0                    |
| admission_level | lab_bicarbonate_max              | float32        |      139133 |         25.48 |         54 | 28.0                   |
| admission_level | lab_bun_max                      | float32        |      134160 |         24.57 |        253 | 25.0                   |
| admission_level | lab_chloride_max                 | float32        |      136702 |         25.04 |         76 | 105.0                  |
| admission_level | lab_chloride_wb_max              | float32        |      510090 |         93.42 |         81 | 90.0                   |
| admission_level | lab_creatinine_max               | float32        |      130195 |         23.84 |        271 | 0.3                    |
| admission_level | lab_creatinine_wb_max            | float32        |      542035 |         99.27 |        201 | 0.8                    |
| admission_level | lab_glucose_max                  | float32        |      139776 |         25.6  |       1594 | 99.0                   |
| admission_level | lab_glucose_poc_max              | float32        |      502359 |         92    |        624 | 140.0                  |
| admission_level | lab_hematocrit_max               | float32        |      115472 |         21.15 |        499 | 37.6                   |
| admission_level | lab_hematocrit_wb_max            | float32        |      508597 |         93.14 |        404 | 25.0                   |
| admission_level | lab_hemoglobin_max               | float32        |      123752 |         22.66 |        180 | 12.7                   |
| admission_level | lab_hemoglobin_wb_max            | float32        |      508596 |         93.14 |        184 | 8.2                    |
| admission_level | lab_platelets_max                | float32        |      122042 |         22.35 |       1363 | 71.0                   |
| admission_level | lab_potassium_max                | float32        |      134495 |         24.63 |        102 | 4.5                    |
| admission_level | lab_potassium_wb_max             | float32        |      490754 |         89.88 |        135 | 3.7                    |
| admission_level | lab_sodium_max                   | float32        |      136233 |         24.95 |         71 | 137.0                  |
| admission_level | lab_sodium_wb_max                | float32        |      504784 |         92.45 |         86 | 135.0                  |
| admission_level | lab_wbc_max                      | float32        |      124478 |         22.8  |       1309 | 4.2                    |
| admission_level | lab_anion_gap_std                | float32        |      229976 |         42.12 |      24864 | 2.5166116              |
| admission_level | lab_bicarbonate_std              | float32        |      229902 |         42.1  |      28239 | 3.0                    |
| admission_level | lab_bun_std                      | float32        |      227548 |         41.67 |      55062 | 3.6055512              |
| admission_level | lab_chloride_std                 | float32        |      226801 |         41.54 |      33017 | 2.8867514              |
| admission_level | lab_chloride_wb_std              | float32        |      525530 |         96.25 |       1542 | 3.04959                |
| admission_level | lab_creatinine_std               | float32        |      225349 |         41.27 |      49687 | 0.057735022            |
| admission_level | lab_creatinine_wb_std            | float32        |      545198 |         99.85 |        379 | 1.2919623              |
| admission_level | lab_glucose_std                  | float32        |      230971 |         42.3  |     122284 | 7.0237694              |
| admission_level | lab_glucose_poc_std              | float32        |      519391 |         95.12 |      18044 | 63.564926              |
| admission_level | lab_hematocrit_std               | float32        |      213724 |         39.14 |     174188 | 1.9091889              |
| admission_level | lab_hematocrit_wb_std            | float32        |      523236 |         95.83 |       5554 | 7.071068               |
| admission_level | lab_hemoglobin_std               | float32        |      223019 |         40.84 |      98876 | 0.4949746              |
| admission_level | lab_hemoglobin_wb_std            | float32        |      523236 |         95.83 |      12093 | 2.4748738              |
| admission_level | lab_platelets_std                | float32        |      221501 |         40.57 |     122563 | 0.70710677             |
| admission_level | lab_potassium_std                | float32        |      223909 |         41.01 |      67339 | 0.25166115             |
| admission_level | lab_potassium_wb_std             | float32        |      515848 |         94.47 |      13219 | 0.4652188              |
| admission_level | lab_sodium_std                   | float32        |      226225 |         41.43 |      29984 | 1.1547005              |
| admission_level | lab_sodium_wb_std                | float32        |      521879 |         95.58 |       2516 | 0.70710677             |
| admission_level | lab_wbc_std                      | float32        |      223767 |         40.98 |     138594 | 0.49497494             |
| admission_level | lab_anion_gap_count              | float32        |      108995 |         19.96 |        226 | 1.0                    |
| admission_level | lab_bicarbonate_count            | float32        |      108995 |         19.96 |        225 | 1.0                    |
| admission_level | lab_bun_count                    | float32        |      108995 |         19.96 |        222 | 1.0                    |
| admission_level | lab_chloride_count               | float32        |      108995 |         19.96 |        229 | 1.0                    |
| admission_level | lab_chloride_wb_count            | float32        |      108995 |         19.96 |         38 | 0.0                    |
| admission_level | lab_creatinine_count             | float32        |      108995 |         19.96 |        224 | 1.0                    |
| admission_level | lab_creatinine_wb_count          | float32        |      108995 |         19.96 |         22 | 0.0                    |
| admission_level | lab_glucose_count                | float32        |      108995 |         19.96 |        222 | 1.0                    |
| admission_level | lab_glucose_poc_count            | float32        |      108995 |         19.96 |        149 | 0.0                    |
| admission_level | lab_hematocrit_count             | float32        |      108995 |         19.96 |        223 | 1.0                    |
| admission_level | lab_hematocrit_wb_count          | float32        |      108995 |         19.96 |         59 | 0.0                    |
| admission_level | lab_hemoglobin_count             | float32        |      108995 |         19.96 |        216 | 1.0                    |
| admission_level | lab_hemoglobin_wb_count          | float32        |      108995 |         19.96 |         59 | 0.0                    |
| admission_level | lab_platelets_count              | float32        |      108995 |         19.96 |        234 | 1.0                    |
| admission_level | lab_potassium_count              | float32        |      108995 |         19.96 |        234 | 1.0                    |
| admission_level | lab_potassium_wb_count           | float32        |      108995 |         19.96 |        145 | 0.0                    |
| admission_level | lab_sodium_count                 | float32        |      108995 |         19.96 |        231 | 1.0                    |
| admission_level | lab_sodium_wb_count              | float32        |      108995 |         19.96 |         74 | 0.0                    |
| admission_level | lab_wbc_count                    | float32        |      108995 |         19.96 |        216 | 1.0                    |
| admission_level | lab_anion_gap_missing_ratio      | float32        |      108995 |         19.96 |         58 | 0.0                    |
| admission_level | lab_bicarbonate_missing_ratio    | float32        |      108995 |         19.96 |        117 | 0.0                    |
| admission_level | lab_bun_missing_ratio            | float32        |      108995 |         19.96 |        174 | 0.0                    |
| admission_level | lab_chloride_missing_ratio       | float32        |      108995 |         19.96 |         91 | 0.0                    |
| admission_level | lab_chloride_wb_missing_ratio    | float32        |      108995 |         19.96 |         13 | 1.0                    |
| admission_level | lab_creatinine_missing_ratio     | float32        |      108995 |         19.96 |        168 | 0.0                    |
| admission_level | lab_creatinine_wb_missing_ratio  | float32        |      108995 |         19.96 |          7 | 1.0                    |
| admission_level | lab_glucose_missing_ratio        | float32        |      108995 |         19.96 |         81 | 0.0                    |
| admission_level | lab_glucose_poc_missing_ratio    | float32        |      108995 |         19.96 |         61 | 1.0                    |
| admission_level | lab_hematocrit_missing_ratio     | float32        |      108995 |         19.96 |        192 | 0.0                    |
| admission_level | lab_hematocrit_wb_missing_ratio  | float32        |      108995 |         19.96 |          8 | 1.0                    |
| admission_level | lab_hemoglobin_missing_ratio     | float32        |      108995 |         19.96 |        198 | 0.0                    |
| admission_level | lab_hemoglobin_wb_missing_ratio  | float32        |      108995 |         19.96 |          5 | 1.0                    |
| admission_level | lab_platelets_missing_ratio      | float32        |      108995 |         19.96 |        395 | 0.0                    |
| admission_level | lab_potassium_missing_ratio      | float32        |      108995 |         19.96 |        164 | 0.0                    |
| admission_level | lab_potassium_wb_missing_ratio   | float32        |      108995 |         19.96 |         48 | 1.0                    |
| admission_level | lab_sodium_missing_ratio         | float32        |      108995 |         19.96 |         84 | 0.0                    |
| admission_level | lab_sodium_wb_missing_ratio      | float32        |      108995 |         19.96 |         17 | 1.0                    |
| admission_level | lab_wbc_missing_ratio            | float32        |      108995 |         19.96 |        297 | 0.0                    |
| admission_level | lab_anion_gap_abnormal_count     | float32        |      108995 |         19.96 |         99 | 0.0                    |
| admission_level | lab_bicarbonate_abnormal_count   | float32        |      108995 |         19.96 |        121 | 0.0                    |
| admission_level | lab_bun_abnormal_count           | float32        |      108995 |         19.96 |        182 | 1.0                    |
| admission_level | lab_chloride_abnormal_count      | float32        |      108995 |         19.96 |        125 | 0.0                    |
| admission_level | lab_chloride_wb_abnormal_count   | float32        |      108995 |         19.96 |         20 | 0.0                    |
| admission_level | lab_creatinine_abnormal_count    | float32        |      108995 |         19.96 |        170 | 1.0                    |
| admission_level | lab_creatinine_wb_abnormal_count | float32        |      108995 |         19.96 |         18 | 0.0                    |
| admission_level | lab_glucose_abnormal_count       | float32        |      108995 |         19.96 |        197 | 0.0                    |
| admission_level | lab_glucose_poc_abnormal_count   | float32        |      108995 |         19.96 |        133 | 0.0                    |
| admission_level | lab_hematocrit_abnormal_count    | float32        |      108995 |         19.96 |        220 | 0.0                    |
| admission_level | lab_hematocrit_wb_abnormal_count | float32        |      108995 |         19.96 |          1 | 0.0                    |
| admission_level | lab_hemoglobin_abnormal_count    | float32        |      108995 |         19.96 |        220 | 0.0                    |