# Data Dictionary

Generated: 2026-07-20T12:51:38.586050Z

## Dataset Summary

| dataset                  |   n_rows |   n_cols |
|:-------------------------|---------:|---------:|
| patient_level            |   223452 |       14 |
| admission_level          |   546028 |       78 |
| icu_level                |    94458 |       14 |
| time_series              |   546028 |        6 |
| similarity               |   546028 |       35 |
| admission_level_selected |   546028 |       64 |

## Feature Dictionary

| dataset                  | feature                         | dtype          |   n_missing |   pct_missing |   n_unique | example                            |
|:-------------------------|:--------------------------------|:---------------|------------:|--------------:|-----------:|:-----------------------------------|
| patient_level            | subject_id                      | int32          |           0 |          0    |     223452 | 10000032                           |
| patient_level            | n_admissions                    | int64          |           0 |          0    |         97 | 4                                  |
| patient_level            | los_days_mean                   | float64        |           0 |          0    |      79406 | 1.4444444444444444                 |
| patient_level            | los_days_max                    | float64        |           0 |          0    |      36730 | 2.2222222222222223                 |
| patient_level            | los_days_sum                    | float64        |           0 |          0    |      62960 | 5.777777777777778                  |
| patient_level            | ever_inhosp_mortality           | int8           |           0 |          0    |          2 | 0                                  |
| patient_level            | ever_readmission_30d            | int8           |           0 |          0    |          2 | 1                                  |
| patient_level            | ever_icu_stay                   | int8           |           0 |          0    |          2 | 1                                  |
| patient_level            | insurance                       | category       |        5723 |          2.56 |          5 | Medicaid                           |
| patient_level            | race                            | category       |           0 |          0    |         33 | WHITE                              |
| patient_level            | gender                          | category       |           0 |          0    |          2 | F                                  |
| patient_level            | anchor_age                      | int8           |           0 |          0    |         73 | 52                                 |
| patient_level            | anchor_year                     | int16          |           0 |          0    |         99 | 2180                               |
| patient_level            | anchor_year_group               | category       |           0 |          0    |          5 | 2014 - 2016                        |
| admission_level          | subject_id                      | int32          |           0 |          0    |     223452 | 10000032                           |
| admission_level          | hadm_id                         | int32          |           0 |          0    |     546028 | 22595853                           |
| admission_level          | admittime                       | datetime64[us] |           0 |          0    |     534919 | 2180-05-06 22:23:00                |
| admission_level          | dischtime                       | datetime64[us] |           0 |          0    |     528871 | 2180-05-07 17:15:00                |
| admission_level          | deathtime                       | datetime64[us] |      534238 |         97.84 |      11788 | 2134-12-06 12:54:00                |
| admission_level          | admission_type                  | category       |           0 |          0    |          9 | URGENT                             |
| admission_level          | admit_provider_id               | category       |           4 |          0    |       2045 | P49AFC                             |
| admission_level          | admission_location              | category       |           1 |          0    |         11 | TRANSFER FROM HOSPITAL             |
| admission_level          | discharge_location              | category       |      149818 |         27.44 |         13 | HOME                               |
| admission_level          | insurance                       | category       |        9355 |          1.71 |          5 | Medicaid                           |
| admission_level          | language                        | category       |         775 |          0.14 |         25 | English                            |
| admission_level          | marital_status                  | category       |       13619 |          2.49 |          4 | WIDOWED                            |
| admission_level          | race                            | category       |           0 |          0    |         33 | WHITE                              |
| admission_level          | edregtime                       | datetime64[us] |      166788 |         30.55 |     372692 | 2180-05-06 19:17:00                |
| admission_level          | edouttime                       | datetime64[us] |      166788 |         30.55 |     372755 | 2180-05-06 23:30:00                |
| admission_level          | hospital_expire_flag            | int8           |           0 |          0    |          2 | 0                                  |
| admission_level          | _is_duplicate                   | int8           |           0 |          0    |          1 | 0                                  |
| admission_level          | _invalid_time_order             | int8           |           0 |          0    |          2 | 0                                  |
| admission_level          | n_icu_stays                     | float64        |           0 |          0    |         10 | 0.0                                |
| admission_level          | icu_los_days                    | float32        |           0 |          0    |      77872 | 0.0                                |
| admission_level          | has_icu_stay                    | int8           |           0 |          0    |          2 | 0                                  |
| admission_level          | los_days                        | float64        |           0 |          0    |      39931 | 0.7861111111111111                 |
| admission_level          | los_hours                       | float64        |           0 |          0    |      39931 | 18.866666666666667                 |
| admission_level          | admit_hour                      | int32          |           0 |          0    |         24 | 22                                 |
| admission_level          | admit_dow                       | int32          |           0 |          0    |          7 | 5                                  |
| admission_level          | admit_month                     | int32          |           0 |          0    |         12 | 5                                  |
| admission_level          | admit_year                      | int32          |           0 |          0    |        108 | 2180                               |
| admission_level          | weekend_admission               | int8           |           0 |          0    |          2 | 1                                  |
| admission_level          | night_admission                 | int8           |           0 |          0    |          2 | 1                                  |
| admission_level          | next_admittime                  | datetime64[us] |      223452 |         40.92 |     318726 | 2180-06-26 18:27:00                |
| admission_level          | days_to_readmission             | float64        |      223452 |         40.92 |     222636 | 50.05                              |
| admission_level          | readmission_30d                 | int8           |           0 |          0    |          2 | 0                                  |
| admission_level          | gender                          | category       |           0 |          0    |          2 | F                                  |
| admission_level          | anchor_age                      | int8           |           0 |          0    |         73 | 52                                 |
| admission_level          | anchor_year                     | int16          |           0 |          0    |         99 | 2180                               |
| admission_level          | anchor_year_group               | category       |           0 |          0    |          5 | 2014 - 2016                        |
| admission_level          | dod                             | datetime64[us] |      401062 |         73.45 |      21981 | 2180-09-09 00:00:00                |
| admission_level          | diagnosis_count                 | int32          |           0 |          0    |         43 | 8                                  |
| admission_level          | unique_diagnosis_count          | int32          |           0 |          0    |         43 | 8                                  |
| admission_level          | primary_icd_code                | str            |         531 |          0.1  |      14222 | 5723                               |
| admission_level          | cci_myocardial_infarction       | int8           |           0 |          0    |          2 | 0                                  |
| admission_level          | cci_congestive_heart_failure    | int8           |           0 |          0    |          2 | 0                                  |
| admission_level          | cci_peripheral_vascular_disease | int8           |           0 |          0    |          2 | 0                                  |
| admission_level          | cci_cerebrovascular_disease     | int8           |           0 |          0    |          2 | 0                                  |
| admission_level          | cci_dementia                    | int8           |           0 |          0    |          2 | 0                                  |
| admission_level          | cci_copd                        | int8           |           0 |          0    |          2 | 1                                  |
| admission_level          | cci_connective_tissue_disease   | int8           |           0 |          0    |          2 | 0                                  |
| admission_level          | cci_peptic_ulcer                | int8           |           0 |          0    |          2 | 0                                  |
| admission_level          | cci_mild_liver_disease          | int8           |           0 |          0    |          2 | 0                                  |
| admission_level          | cci_diabetes_uncomplicated      | int8           |           0 |          0    |          2 | 0                                  |
| admission_level          | cci_diabetes_complicated        | int8           |           0 |          0    |          2 | 0                                  |
| admission_level          | cci_hemiplegia_paraplegia       | int8           |           0 |          0    |          2 | 0                                  |
| admission_level          | cci_renal_disease               | int8           |           0 |          0    |          2 | 0                                  |
| admission_level          | cci_malignancy                  | int8           |           0 |          0    |          2 | 0                                  |
| admission_level          | cci_severe_liver_disease        | int8           |           0 |          0    |          2 | 1                                  |
| admission_level          | cci_metastatic_tumor            | int8           |           0 |          0    |          2 | 0                                  |
| admission_level          | cci_aids                        | int8           |           0 |          0    |          2 | 0                                  |
| admission_level          | charlson_comorbidity_index      | int32          |           0 |          0    |         19 | 4                                  |
| admission_level          | dx_diabetes                     | int8           |           0 |          0    |          2 | 0                                  |
| admission_level          | dx_hypertension                 | int8           |           0 |          0    |          2 | 0                                  |
| admission_level          | dx_ckd                          | int8           |           0 |          0    |          2 | 0                                  |
| admission_level          | dx_cad                          | int8           |           0 |          0    |          2 | 0                                  |
| admission_level          | dx_copd_flag                    | int8           |           0 |          0    |          2 | 1                                  |
| admission_level          | dx_stroke                       | int8           |           0 |          0    |          2 | 0                                  |
| admission_level          | procedure_count                 | int32          |           0 |          0    |         42 | 1                                  |
| admission_level          | unique_procedure_count          | int32          |           0 |          0    |         37 | 1                                  |
| admission_level          | major_procedure_count           | int32          |           0 |          0    |         41 | 1                                  |
| admission_level          | has_major_procedure             | int8           |           0 |          0    |          2 | 1                                  |
| admission_level          | first_careunit                  | category       |      460786 |         84.39 |         16 | Medical Intensive Care Unit (MICU) |
| admission_level          | last_careunit                   | category       |      460786 |         84.39 |         16 | Medical Intensive Care Unit (MICU) |
| admission_level          | intime                          | datetime64[us] |      460786 |         84.39 |      85234 | 2180-07-23 14:00:00                |
| admission_level          | outtime                         | datetime64[us] |      460799 |         84.39 |      85228 | 2180-07-23 23:50:47                |
| admission_level          | los                             | float32        |      460799 |         84.39 |      77831 | 0.4102662                          |
| admission_level          | icu_duration_hours              | float64        |      460786 |         84.39 |      77867 | 9.846388888888889                  |
| admission_level          | icu_duration_days               | float32        |      460786 |         84.39 |      77872 | 0.4102662                          |
| admission_level          | n_icu_stays_per_admission       | float64        |      460786 |         84.39 |          9 | 1.0                                |
| admission_level          | age_x_diabetes                  | int8           |           0 |          0    |         74 | 0                                  |
| admission_level          | cci_x_age                       | int32          |           0 |          0    |        560 | 208                                |
| icu_level                | subject_id                      | int32          |           0 |          0    |      65366 | 10000032                           |
| icu_level                | hadm_id                         | int32          |           0 |          0    |      85242 | 29079034                           |
| icu_level                | stay_id                         | int32          |           0 |          0    |      94458 | 39553978                           |
| icu_level                | first_careunit                  | category       |           0 |          0    |         17 | Medical Intensive Care Unit (MICU) |
| icu_level                | last_careunit                   | category       |           0 |          0    |         17 | Medical Intensive Care Unit (MICU) |
| icu_level                | intime                          | datetime64[us] |           0 |          0    |      94449 | 2180-07-23 14:00:00                |
| icu_level                | outtime                         | datetime64[us] |          14 |          0.01 |      94442 | 2180-07-23 23:50:47                |
| icu_level                | los                             | float32        |          14 |          0.01 |      84932 | 0.4102662                          |
| icu_level                | _is_duplicate                   | int8           |           0 |          0    |          1 | 0                                  |
| icu_level                | icu_los_days                    | float32        |          14 |          0.01 |      84932 | 0.4102662                          |
| icu_level                | icu_los_hours                   | float32        |          14 |          0.01 |      84932 | 9.846389                           |
| icu_level                | hospital_expire_flag            | int8           |           0 |          0    |          2 | 0                                  |
| icu_level                | admittime                       | datetime64[us] |           0 |          0    |      84785 | 2180-07-23 12:35:00                |
| icu_level                | dischtime                       | datetime64[us] |           0 |          0    |      84710 | 2180-07-25 17:55:00                |
| time_series              | subject_id                      | int32          |           0 |          0    |     223452 | 10000032                           |
| time_series              | hadm_id                         | int32          |           0 |          0    |     546028 | 22595853                           |
| time_series              | admittime                       | datetime64[us] |           0 |          0    |     534919 | 2180-05-06 22:23:00                |
| time_series              | event_time                      | datetime64[us] |           0 |          0    |     534919 | 2180-05-06 22:23:00                |
| time_series              | event_type                      | str            |           0 |          0    |          1 | admission                          |
| time_series              | hours_since_admission           | float64        |           0 |          0    |          1 | 0.0                                |
| similarity               | subject_id                      | int32          |           0 |          0    |     223452 | 10000032                           |
| similarity               | hadm_id                         | int32          |           0 |          0    |     546028 | 22595853                           |
| similarity               | insurance                       | category       |        9355 |          1.71 |          5 | Medicaid                           |
| similarity               | marital_status                  | category       |       13619 |          2.49 |          4 | WIDOWED                            |
| similarity               | race                            | category       |           0 |          0    |         33 | WHITE                              |
| similarity               | gender                          | category       |           0 |          0    |          2 | F                                  |
| similarity               | anchor_age                      | int8           |           0 |          0    |         73 | 52                                 |
| similarity               | hospital_expire_flag            | int8           |           0 |          0    |          2 | 0                                  |
| similarity               | has_icu_stay                    | int8           |           0 |          0    |          2 | 0                                  |
| similarity               | los_days                        | float64        |           0 |          0    |      39931 | 0.7861111111111111                 |
| similarity               | readmission_30d                 | int8           |           0 |          0    |          2 | 0                                  |
| similarity               | cci_myocardial_infarction       | float64        |         531 |          0.1  |          2 | 0.0                                |
| similarity               | cci_congestive_heart_failure    | float64        |         531 |          0.1  |          2 | 0.0                                |
| similarity               | cci_peripheral_vascular_disease | float64        |         531 |          0.1  |          2 | 0.0                                |
| similarity               | cci_cerebrovascular_disease     | float64        |         531 |          0.1  |          2 | 0.0                                |
| similarity               | cci_dementia                    | float64        |         531 |          0.1  |          2 | 0.0                                |
| similarity               | cci_copd                        | float64        |         531 |          0.1  |          2 | 1.0                                |
| similarity               | cci_connective_tissue_disease   | float64        |         531 |          0.1  |          2 | 0.0                                |
| similarity               | cci_peptic_ulcer                | float64        |         531 |          0.1  |          2 | 0.0                                |
| similarity               | cci_mild_liver_disease          | float64        |         531 |          0.1  |          2 | 0.0                                |
| similarity               | cci_diabetes_uncomplicated      | float64        |         531 |          0.1  |          2 | 0.0                                |
| similarity               | cci_diabetes_complicated        | float64        |         531 |          0.1  |          2 | 0.0                                |
| similarity               | cci_hemiplegia_paraplegia       | float64        |         531 |          0.1  |          2 | 0.0                                |
| similarity               | cci_renal_disease               | float64        |         531 |          0.1  |          2 | 0.0                                |
| similarity               | cci_malignancy                  | float64        |         531 |          0.1  |          2 | 0.0                                |
| similarity               | cci_severe_liver_disease        | float64        |         531 |          0.1  |          2 | 1.0                                |
| similarity               | cci_metastatic_tumor            | float64        |         531 |          0.1  |          2 | 0.0                                |
| similarity               | cci_aids                        | float64        |         531 |          0.1  |          2 | 0.0                                |
| similarity               | dx_diabetes                     | float64        |         531 |          0.1  |          2 | 0.0                                |
| similarity               | dx_hypertension                 | float64        |         531 |          0.1  |          2 | 0.0                                |
| similarity               | dx_ckd                          | float64        |         531 |          0.1  |          2 | 0.0                                |
| similarity               | dx_cad                          | float64        |         531 |          0.1  |          2 | 0.0                                |
| similarity               | dx_copd_flag                    | float64        |         531 |          0.1  |          2 | 1.0                                |
| similarity               | dx_stroke                       | float64        |         531 |          0.1  |          2 | 0.0                                |
| similarity               | embedding_placeholder           | str            |           0 |          0    |          1 |                                    |
| admission_level_selected | subject_id                      | int32          |           0 |          0    |     223452 | 10000032                           |
| admission_level_selected | hadm_id                         | int32          |           0 |          0    |     546028 | 22595853                           |
| admission_level_selected | hospital_expire_flag            | int8           |           0 |          0    |          2 | 0                                  |
| admission_level_selected | los_days                        | float64        |           0 |          0    |      39931 | 0.7861111111111111                 |
| admission_level_selected | has_icu_stay                    | int8           |           0 |          0    |          2 | 0                                  |
| admission_level_selected | icu_los_days                    | float32        |           0 |          0    |      77872 | 0.0                                |
| admission_level_selected | readmission_30d                 | int8           |           0 |          0    |          2 | 0                                  |
| admission_level_selected | admittime                       | datetime64[us] |           0 |          0    |     534919 | 2180-05-06 22:23:00                |
| admission_level_selected | dischtime                       | datetime64[us] |           0 |          0    |     528871 | 2180-05-07 17:15:00                |
| admission_level_selected | deathtime                       | datetime64[us] |      534238 |         97.84 |      11788 | 2134-12-06 12:54:00                |
| admission_level_selected | admission_type                  | category       |           0 |          0    |          9 | URGENT                             |
| admission_level_selected | admit_provider_id               | category       |           4 |          0    |       2045 | P49AFC                             |
| admission_level_selected | admission_location              | category       |           1 |          0    |         11 | TRANSFER FROM HOSPITAL             |
| admission_level_selected | discharge_location              | category       |      149818 |         27.44 |         13 | HOME                               |
| admission_level_selected | insurance                       | category       |        9355 |          1.71 |          5 | Medicaid                           |
| admission_level_selected | language                        | category       |         775 |          0.14 |         25 | English                            |
| admission_level_selected | marital_status                  | category       |       13619 |          2.49 |          4 | WIDOWED                            |
| admission_level_selected | race                            | category       |           0 |          0    |         33 | WHITE                              |
| admission_level_selected | edregtime                       | datetime64[us] |      166788 |         30.55 |     372692 | 2180-05-06 19:17:00                |
| admission_level_selected | edouttime                       | datetime64[us] |      166788 |         30.55 |     372755 | 2180-05-06 23:30:00                |
| admission_level_selected | los_hours                       | float64        |           0 |          0    |      39931 | 18.866666666666667                 |
| admission_level_selected | admit_hour                      | int32          |           0 |          0    |         24 | 22                                 |
| admission_level_selected | admit_dow                       | int32          |           0 |          0    |          7 | 5                                  |
| admission_level_selected | admit_month                     | int32          |           0 |          0    |         12 | 5                                  |
| admission_level_selected | weekend_admission               | int8           |           0 |          0    |          2 | 1                                  |
| admission_level_selected | night_admission                 | int8           |           0 |          0    |          2 | 1                                  |
| admission_level_selected | next_admittime                  | datetime64[us] |      223452 |         40.92 |     318726 | 2180-06-26 18:27:00                |
| admission_level_selected | days_to_readmission             | float64        |      223452 |         40.92 |     222636 | 50.05                              |
| admission_level_selected | gender                          | category       |           0 |          0    |          2 | F                                  |
| admission_level_selected | anchor_age                      | int8           |           0 |          0    |         73 | 52                                 |
| admission_level_selected | anchor_year                     | int16          |           0 |          0    |         99 | 2180                               |
| admission_level_selected | anchor_year_group               | category       |           0 |          0    |          5 | 2014 - 2016                        |
| admission_level_selected | dod                             | datetime64[us] |      401062 |         73.45 |      21981 | 2180-09-09 00:00:00                |
| admission_level_selected | unique_diagnosis_count          | int32          |           0 |          0    |         43 | 8                                  |
| admission_level_selected | primary_icd_code                | str            |         531 |          0.1  |      14222 | 5723                               |
| admission_level_selected | cci_myocardial_infarction       | int8           |           0 |          0    |          2 | 0                                  |
| admission_level_selected | cci_congestive_heart_failure    | int8           |           0 |          0    |          2 | 0                                  |
| admission_level_selected | cci_peripheral_vascular_disease | int8           |           0 |          0    |          2 | 0                                  |
| admission_level_selected | cci_cerebrovascular_disease     | int8           |           0 |          0    |          2 | 0                                  |
| admission_level_selected | cci_dementia                    | int8           |           0 |          0    |          2 | 0                                  |
| admission_level_selected | cci_copd                        | int8           |           0 |          0    |          2 | 1                                  |
| admission_level_selected | cci_connective_tissue_disease   | int8           |           0 |          0    |          2 | 0                                  |
| admission_level_selected | cci_peptic_ulcer                | int8           |           0 |          0    |          2 | 0                                  |
| admission_level_selected | cci_mild_liver_disease          | int8           |           0 |          0    |          2 | 0                                  |
| admission_level_selected | cci_diabetes_uncomplicated      | int8           |           0 |          0    |          2 | 0                                  |
| admission_level_selected | cci_diabetes_complicated        | int8           |           0 |          0    |          2 | 0                                  |
| admission_level_selected | cci_hemiplegia_paraplegia       | int8           |           0 |          0    |          2 | 0                                  |
| admission_level_selected | cci_malignancy                  | int8           |           0 |          0    |          2 | 0                                  |
| admission_level_selected | cci_severe_liver_disease        | int8           |           0 |          0    |          2 | 1                                  |
| admission_level_selected | cci_metastatic_tumor            | int8           |           0 |          0    |          2 | 0                                  |
| admission_level_selected | dx_hypertension                 | int8           |           0 |          0    |          2 | 0                                  |
| admission_level_selected | dx_ckd                          | int8           |           0 |          0    |          2 | 0                                  |
| admission_level_selected | dx_cad                          | int8           |           0 |          0    |          2 | 0                                  |
| admission_level_selected | dx_copd_flag                    | int8           |           0 |          0    |          2 | 1                                  |
| admission_level_selected | dx_stroke                       | int8           |           0 |          0    |          2 | 0                                  |
| admission_level_selected | unique_procedure_count          | int32          |           0 |          0    |         37 | 1                                  |
| admission_level_selected | major_procedure_count           | int32          |           0 |          0    |         41 | 1                                  |
| admission_level_selected | has_major_procedure             | int8           |           0 |          0    |          2 | 1                                  |
| admission_level_selected | first_careunit                  | category       |      460786 |         84.39 |         16 | Medical Intensive Care Unit (MICU) |
| admission_level_selected | last_careunit                   | category       |      460786 |         84.39 |         16 | Medical Intensive Care Unit (MICU) |
| admission_level_selected | intime                          | datetime64[us] |      460786 |         84.39 |      85234 | 2180-07-23 14:00:00                |
| admission_level_selected | outtime                         | datetime64[us] |      460799 |         84.39 |      85228 | 2180-07-23 23:50:47                |
| admission_level_selected | age_x_diabetes                  | int8           |           0 |          0    |         74 | 0                                  |
| admission_level_selected | cci_x_age                       | int32          |           0 |          0    |        560 | 208                                |