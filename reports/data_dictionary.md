# Data Dictionary

Generated: 2026-07-19T15:43:25.540972Z

## Dataset Summary

| dataset                  |   n_rows |   n_cols |
|:-------------------------|---------:|---------:|
| patient_level            |     3006 |       14 |
| admission_level          |     7066 |      336 |
| icu_level                |     1217 |      136 |
| time_series              |  2326019 |       10 |
| clinical_notes           |     4286 |       13 |
| similarity               |     7066 |       35 |
| admission_level_selected |     7066 |      174 |

## Feature Dictionary (sample)

| dataset         | feature                         | dtype          |   n_missing |   pct_missing |   n_unique | example                |
|:----------------|:--------------------------------|:---------------|------------:|--------------:|-----------:|:-----------------------|
| patient_level   | subject_id                      | int32          |           0 |          0    |       3006 | 10000032               |
| patient_level   | n_admissions                    | int64          |           0 |          0    |         28 | 4                      |
| patient_level   | los_days_mean                   | float64        |           0 |          0    |       2812 | 1.4444444444444444     |
| patient_level   | los_days_max                    | float64        |           0 |          0    |       2718 | 2.2222222222222223     |
| patient_level   | los_days_sum                    | float64        |           0 |          0    |       2817 | 5.777777777777778      |
| patient_level   | ever_inhosp_mortality           | int8           |           0 |          0    |          2 | 0                      |
| patient_level   | ever_readmission_30d            | int8           |           0 |          0    |          2 | 1                      |
| patient_level   | ever_icu_stay                   | int8           |           0 |          0    |          2 | 1                      |
| patient_level   | insurance                       | category       |          69 |          2.3  |          5 | Medicaid               |
| patient_level   | race                            | category       |           0 |          0    |         33 | WHITE                  |
| patient_level   | gender                          | category       |           0 |          0    |          2 | F                      |
| patient_level   | anchor_age                      | int8           |           0 |          0    |         73 | 52                     |
| patient_level   | anchor_year                     | int16          |           0 |          0    |         94 | 2180                   |
| patient_level   | anchor_year_group               | category       |           0 |          0    |          5 | 2014 - 2016            |
| admission_level | subject_id                      | int32          |           0 |          0    |       3006 | 10000032               |
| admission_level | hadm_id                         | int32          |           0 |          0    |       7066 | 22595853               |
| admission_level | admittime                       | datetime64[ns] |           0 |          0    |       7065 | 2180-05-06 22:23:00    |
| admission_level | dischtime                       | datetime64[ns] |           0 |          0    |       7062 | 2180-05-07 17:15:00    |
| admission_level | deathtime                       | datetime64[ns] |        6908 |         97.76 |        158 | 2134-12-06 12:54:00    |
| admission_level | admission_type                  | category       |           0 |          0    |          9 | URGENT                 |
| admission_level | admit_provider_id               | category       |           1 |          0.01 |       1013 | P49AFC                 |
| admission_level | admission_location              | category       |           0 |          0    |         11 | TRANSFER FROM HOSPITAL |
| admission_level | discharge_location              | category       |        1914 |         27.09 |         13 | HOME                   |
| admission_level | insurance                       | category       |         114 |          1.61 |          5 | Medicaid               |
| admission_level | language                        | category       |           7 |          0.1  |         21 | English                |
| admission_level | marital_status                  | category       |         192 |          2.72 |          4 | WIDOWED                |
| admission_level | race                            | category       |           0 |          0    |         33 | WHITE                  |
| admission_level | edregtime                       | datetime64[ns] |        2170 |         30.71 |       4832 | 2180-05-06 19:17:00    |
| admission_level | edouttime                       | datetime64[ns] |        2170 |         30.71 |       4833 | 2180-05-06 23:30:00    |
| admission_level | hospital_expire_flag            | int8           |           0 |          0    |          2 | 0                      |
| admission_level | _is_duplicate                   | int8           |           0 |          0    |          1 | 0                      |
| admission_level | _invalid_time_order             | int8           |           0 |          0    |          2 | 0                      |
| admission_level | n_icu_stays                     | float64        |           0 |          0    |          5 | 0.0                    |
| admission_level | icu_los_days                    | float32        |           0 |          0    |       1118 | 0.0                    |
| admission_level | has_icu_stay                    | int8           |           0 |          0    |          2 | 0                      |
| admission_level | los_days                        | float64        |           0 |          0    |       5301 | 0.7861111111111111     |
| admission_level | los_hours                       | float64        |           0 |          0    |       5301 | 18.866666666666667     |
| admission_level | admit_hour                      | int32          |           0 |          0    |         24 | 22                     |
| admission_level | admit_dow                       | int32          |           0 |          0    |          7 | 5                      |
| admission_level | admit_month                     | int32          |           0 |          0    |         12 | 5                      |
| admission_level | admit_year                      | int32          |           0 |          0    |        100 | 2180                   |
| admission_level | weekend_admission               | int8           |           0 |          0    |          2 | 1                      |
| admission_level | night_admission                 | int8           |           0 |          0    |          2 | 1                      |
| admission_level | next_admittime                  | datetime64[ns] |        3006 |         42.54 |       4060 | 2180-06-26 18:27:00    |
| admission_level | days_to_readmission             | float64        |        3006 |         42.54 |       3903 | 50.05                  |
| admission_level | readmission_30d                 | int8           |           0 |          0    |          2 | 0                      |
| admission_level | gender                          | category       |           0 |          0    |          2 | F                      |
| admission_level | anchor_age                      | int8           |           0 |          0    |         73 | 52                     |
| admission_level | anchor_year                     | int16          |           0 |          0    |         94 | 2180                   |
| admission_level | anchor_year_group               | category       |           0 |          0    |          5 | 2014 - 2016            |
| admission_level | dod                             | datetime64[ns] |        5176 |         73.25 |        505 | 2180-09-09 00:00:00    |
| admission_level | diagnosis_count                 | float64        |           3 |          0.04 |         39 | 8.0                    |
| admission_level | unique_diagnosis_count          | float64        |           3 |          0.04 |         39 | 8.0                    |
| admission_level | primary_icd_code                | object         |           3 |          0.04 |       2335 | 5723                   |
| admission_level | cci_myocardial_infarction       | float64        |           3 |          0.04 |          2 | 0.0                    |
| admission_level | cci_congestive_heart_failure    | float64        |           3 |          0.04 |          2 | 0.0                    |
| admission_level | cci_peripheral_vascular_disease | float64        |           3 |          0.04 |          2 | 0.0                    |
| admission_level | cci_cerebrovascular_disease     | float64        |           3 |          0.04 |          2 | 0.0                    |
| admission_level | cci_dementia                    | float64        |           3 |          0.04 |          2 | 0.0                    |
| admission_level | cci_copd                        | float64        |           3 |          0.04 |          2 | 1.0                    |
| admission_level | cci_connective_tissue_disease   | float64        |           3 |          0.04 |          2 | 0.0                    |
| admission_level | cci_peptic_ulcer                | float64        |           3 |          0.04 |          2 | 0.0                    |
| admission_level | cci_mild_liver_disease          | float64        |           3 |          0.04 |          2 | 1.0                    |
| admission_level | cci_diabetes_uncomplicated      | float64        |           3 |          0.04 |          2 | 0.0                    |
| admission_level | cci_diabetes_complicated        | float64        |           3 |          0.04 |          2 | 0.0                    |
| admission_level | cci_hemiplegia_paraplegia       | float64        |           3 |          0.04 |          2 | 0.0                    |
| admission_level | cci_renal_disease               | float64        |           3 |          0.04 |          2 | 0.0                    |
| admission_level | cci_malignancy                  | float64        |           3 |          0.04 |          2 | 0.0                    |
| admission_level | cci_severe_liver_disease        | float64        |           3 |          0.04 |          2 | 1.0                    |
| admission_level | cci_metastatic_tumor            | float64        |           3 |          0.04 |          2 | 0.0                    |
| admission_level | cci_aids                        | float64        |           3 |          0.04 |          2 | 0.0                    |
| admission_level | charlson_comorbidity_index      | float64        |           3 |          0.04 |         16 | 5.0                    |
| admission_level | dx_diabetes                     | float64        |           3 |          0.04 |          2 | 0.0                    |
| admission_level | dx_hypertension                 | float64        |           3 |          0.04 |          2 | 0.0                    |
| admission_level | dx_ckd                          | float64        |           3 |          0.04 |          2 | 0.0                    |
| admission_level | dx_cad                          | float64        |           3 |          0.04 |          2 | 0.0                    |
| admission_level | dx_copd_flag                    | float64        |           3 |          0.04 |          2 | 1.0                    |
| admission_level | dx_stroke                       | float64        |           3 |          0.04 |          2 | 0.0                    |
| admission_level | icd_embedding_placeholder       | object         |           3 |          0.04 |          1 |                        |
| admission_level | procedure_count                 | float64        |        3217 |         45.53 |         23 | 1.0                    |
| admission_level | unique_procedure_count          | float64        |        3217 |         45.53 |         20 | 1.0                    |
| admission_level | major_procedure_count           | float64        |        3217 |         45.53 |         23 | 1.0                    |
| admission_level | has_major_procedure             | float64        |        3217 |         45.53 |          2 | 1.0                    |
| admission_level | lab_total_count                 | float64        |        4148 |         58.7  |        339 | 32.0                   |
| admission_level | lab_unique_items                | float64        |        4148 |         58.7  |         20 | 12.0                   |
| admission_level | lab_creatinine_mean             | float64        |        4362 |         61.73 |        876 | 0.43333330750465393    |
| admission_level | lab_creatinine_median           | float64        |        4362 |         61.73 |        151 | 0.4000000059604645     |
| admission_level | lab_creatinine_min              | float64        |        4362 |         61.73 |         85 | 0.4000000059604645     |
| admission_level | lab_creatinine_max              | float64        |        4362 |         61.73 |        108 | 0.5                    |
| admission_level | lab_creatinine_std              | float64        |        4960 |         70.2  |        921 | 0.05773502215743065    |
| admission_level | lab_creatinine_first            | float64        |        4362 |         61.73 |        103 | 0.5                    |
| admission_level | lab_creatinine_last             | float64        |        4362 |         61.73 |         95 | 0.4000000059604645     |
| admission_level | lab_creatinine_count            | float64        |        4148 |         58.7  |         67 | 3.0                    |
| admission_level | lab_creatinine_abnormal_count   | float64        |        4148 |         58.7  |         48 | 0.0                    |
| admission_level | lab_creatinine_missing_ratio    | float64        |        4148 |         58.7  |          5 | 0.0                    |
| admission_level | lab_sodium_mean                 | float64        |        4397 |         62.23 |        588 | 130.6666717529297      |
| admission_level | lab_sodium_median               | float64        |        4397 |         62.23 |         55 | 130.0                  |
| admission_level | lab_sodium_min                  | float64        |        4397 |         62.23 |         34 | 130.0                  |
| admission_level | lab_sodium_max                  | float64        |        4397 |         62.23 |         40 | 132.0                  |
| admission_level | lab_sodium_std                  | float64        |        4965 |         70.27 |        778 | 1.154700517654419      |
| admission_level | lab_sodium_first                | float64        |        4397 |         62.23 |         42 | 132.0                  |
| admission_level | lab_sodium_last                 | float64        |        4397 |         62.23 |         30 | 130.0                  |
| admission_level | lab_sodium_count                | float64        |        4148 |         58.7  |         63 | 3.0                    |
| admission_level | lab_sodium_abnormal_count       | float64        |        4148 |         58.7  |         32 | 3.0                    |
| admission_level | lab_sodium_missing_ratio        | float64        |        4148 |         58.7  |          5 | 0.0                    |
| admission_level | lab_potassium_mean              | float64        |        4392 |         62.16 |        823 | 4.9666666984558105     |
| admission_level | lab_potassium_median            | float64        |        4392 |         62.16 |         84 | 5.0                    |
| admission_level | lab_potassium_min               | float64        |        4392 |         62.16 |         39 | 4.699999809265137      |
| admission_level | lab_potassium_max               | float64        |        4392 |         62.16 |         61 | 5.199999809265137      |
| admission_level | lab_potassium_std               | float64        |        4950 |         70.05 |       1335 | 0.25166115164756775    |
| admission_level | lab_potassium_first             | float64        |        4392 |         62.16 |         55 | 4.699999809265137      |
| admission_level | lab_potassium_last              | float64        |        4392 |         62.16 |         40 | 5.199999809265137      |
| admission_level | lab_potassium_count             | float64        |        4148 |         58.7  |         67 | 3.0                    |
| admission_level | lab_potassium_abnormal_count    | float64        |        4148 |         58.7  |         22 | 1.0                    |
| admission_level | lab_potassium_missing_ratio     | float64        |        4148 |         58.7  |         12 | 0.0                    |
| admission_level | lab_bun_mean                    | float64        |        4387 |         62.09 |        863 | 32.0                   |
| admission_level | lab_bun_median                  | float64        |        4387 |         62.09 |        160 | 33.0                   |
| admission_level | lab_bun_min                     | float64        |        4387 |         62.09 |         84 | 28.0                   |
| admission_level | lab_bun_max                     | float64        |        4387 |         62.09 |        126 | 35.0                   |
| admission_level | lab_bun_std                     | float64        |        4975 |         70.41 |       1043 | 3.605551242828369      |
| admission_level | lab_bun_first                   | float64        |        4387 |         62.09 |        110 | 33.0                   |
| admission_level | lab_bun_last                    | float64        |        4387 |         62.09 |         98 | 35.0                   |
| admission_level | lab_bun_count                   | float64        |        4148 |         58.7  |         64 | 3.0                    |
| admission_level | lab_bun_abnormal_count          | float64        |        4148 |         58.7  |         55 | 3.0                    |
| admission_level | lab_bun_missing_ratio           | float64        |        4148 |         58.7  |          4 | 0.0                    |
| admission_level | lab_wbc_mean                    | float64        |        4311 |         61.01 |       1469 | 4.449999809265137      |
| admission_level | lab_wbc_median                  | float64        |        4311 |         61.01 |        459 | 4.449999809265137      |
| admission_level | lab_wbc_min                     | float64        |        4311 |         61.01 |        207 | 4.099999904632568      |
| admission_level | lab_wbc_max                     | float64        |        4311 |         61.01 |        312 | 4.800000190734863      |
| admission_level | lab_wbc_std                     | float64        |        4924 |         69.69 |       1716 | 0.49497494101524353    |
| admission_level | lab_wbc_first                   | float64        |        4311 |         61.01 |        267 | 4.099999904632568      |
| admission_level | lab_wbc_last                    | float64        |        4311 |         61.01 |        242 | 4.800000190734863      |
| admission_level | lab_wbc_count                   | float64        |        4148 |         58.7  |         55 | 2.0                    |
| admission_level | lab_wbc_abnormal_count          | float64        |        4148 |         58.7  |         45 | 0.0                    |
| admission_level | lab_wbc_missing_ratio           | float64        |        4148 |         58.7  |         23 | 0.0                    |
| admission_level | lab_hemoglobin_mean             | float64        |        4303 |         60.9  |       1221 | 11.549999237060547     |
| admission_level | lab_hemoglobin_median           | float64        |        4303 |         60.9  |        250 | 11.549999237060547     |
| admission_level | lab_hemoglobin_min              | float64        |        4303 |         60.9  |        121 | 11.199999809265137     |
| admission_level | lab_hemoglobin_max              | float64        |        4303 |         60.9  |        108 | 11.899999618530273     |
| admission_level | lab_hemoglobin_std              | float64        |        4921 |         69.64 |       1491 | 0.49497461318969727    |
| admission_level | lab_hemoglobin_first            | float64        |        4303 |         60.9  |        117 | 11.899999618530273     |
| admission_level | lab_hemoglobin_last             | float64        |        4303 |         60.9  |        111 | 11.199999809265137     |
| admission_level | lab_hemoglobin_count            | float64        |        4148 |         58.7  |         53 | 2.0                    |
| admission_level | lab_hemoglobin_abnormal_count   | float64        |        4148 |         58.7  |         52 | 2.0                    |
| admission_level | lab_hemoglobin_missing_ratio    | float64        |        4148 |         58.7  |         22 | 0.0                    |
| admission_level | lab_platelets_mean              | float64        |        4297 |         60.81 |       1710 | 94.5                   |
| admission_level | lab_platelets_median            | float64        |        4297 |         60.81 |        782 | 94.5                   |
| admission_level | lab_platelets_min               | float64        |        4297 |         60.81 |        461 | 94.0                   |
| admission_level | lab_platelets_max               | float64        |        4297 |         60.81 |        546 | 95.0                   |
| admission_level | lab_platelets_std               | float64        |        4913 |         69.53 |       1621 | 0.7071067690849304     |
| admission_level | lab_platelets_first             | float64        |        4297 |         60.81 |        488 | 94.0                   |
| admission_level | lab_platelets_last              | float64        |        4297 |         60.81 |        522 | 95.0                   |
| admission_level | lab_platelets_count             | float64        |        4148 |         58.7  |         56 | 2.0                    |
| admission_level | lab_platelets_abnormal_count    | float64        |        4148 |         58.7  |         43 | 2.0                    |
| admission_level | lab_platelets_missing_ratio     | float64        |        4148 |         58.7  |         27 | 0.0                    |
| admission_level | lab_glucose_mean                | float64        |        4431 |         62.71 |       1209 | 114.33333587646484     |
| admission_level | lab_glucose_median              | float64        |        4431 |         62.71 |        341 | 115.0                  |
| admission_level | lab_glucose_min                 | float64        |        4431 |         62.71 |        203 | 107.0                  |
| admission_level | lab_glucose_max                 | float64        |        4431 |         62.71 |        351 | 121.0                  |
| admission_level | lab_glucose_std                 | float64        |        5001 |         70.78 |       1562 | 7.023768901824951      |
| admission_level | lab_glucose_first               | float64        |        4431 |         62.71 |        278 | 115.0                  |
| admission_level | lab_glucose_last                | float64        |        4431 |         62.71 |        239 | 121.0                  |
| admission_level | lab_glucose_count               | float64        |        4148 |         58.7  |         66 | 3.0                    |
| admission_level | lab_glucose_abnormal_count      | float64        |        4148 |         58.7  |         54 | 3.0                    |
| admission_level | lab_glucose_missing_ratio       | float64        |        4148 |         58.7  |          6 | 0.0                    |
| admission_level | lab_hematocrit_mean             | float64        |        4253 |         60.19 |       1740 | 33.44999694824219      |
| admission_level | lab_hematocrit_median           | float64        |        4253 |         60.19 |        588 | 33.44999694824219      |
| admission_level | lab_hematocrit_min              | float64        |        4253 |         60.19 |        311 | 32.099998474121094     |
| admission_level | lab_hematocrit_max              | float64        |        4253 |         60.19 |        274 | 34.79999923706055      |
| admission_level | lab_hematocrit_std              | float64        |        4852 |         68.67 |       1807 | 1.9091888666152954     |
| admission_level | lab_hematocrit_first            | float64        |        4253 |         60.19 |        303 | 34.79999923706055      |
| admission_level | lab_hematocrit_last             | float64        |        4253 |         60.19 |        277 | 32.099998474121094     |
| admission_level | lab_hematocrit_count            | float64        |        4148 |         58.7  |         58 | 2.0                    |
| admission_level | lab_hematocrit_abnormal_count   | float64        |        4148 |         58.7  |         58 | 2.0                    |
| admission_level | lab_hematocrit_missing_ratio    | float64        |        4148 |         58.7  |         23 | 0.0                    |
| admission_level | lab_bicarbonate_mean            | float64        |        4421 |         62.57 |        582 | 24.0                   |
| admission_level | lab_bicarbonate_median          | float64        |        4421 |         62.57 |         52 | 24.0                   |
| admission_level | lab_bicarbonate_min             | float64        |        4421 |         62.57 |         34 | 21.0                   |
| admission_level | lab_bicarbonate_max             | float64        |        4421 |         62.57 |         35 | 27.0                   |
| admission_level | lab_bicarbonate_std             | float64        |        4989 |         70.61 |        765 | 3.0                    |
| admission_level | lab_bicarbonate_first           | float64        |        4421 |         62.57 |         34 | 21.0                   |
| admission_level | lab_bicarbonate_last            | float64        |        4421 |         62.57 |         35 | 27.0                   |
| admission_level | lab_bicarbonate_count           | float64        |        4148 |         58.7  |         64 | 3.0                    |
| admission_level | lab_bicarbonate_abnormal_count  | float64        |        4148 |         58.7  |         37 | 1.0                    |
| admission_level | lab_bicarbonate_missing_ratio   | float64        |        4148 |         58.7  |          4 | 0.0                    |
| admission_level | lab_chloride_mean               | float64        |        4400 |         62.27 |        653 | 100.33333587646484     |
| admission_level | lab_chloride_median             | float64        |        4400 |         62.27 |         67 | 102.0                  |
| admission_level | lab_chloride_min                | float64        |        4400 |         62.27 |         44 | 97.0                   |
| admission_level | lab_chloride_max                | float64        |        4400 |         62.27 |         42 | 102.0                  |
| admission_level | lab_chloride_std                | float64        |        4967 |         70.29 |        845 | 2.886751174926758      |
| admission_level | lab_chloride_first              | float64        |        4400 |         62.27 |         47 | 102.0                  |
| admission_level | lab_chloride_last               | float64        |        4400 |         62.27 |         39 | 97.0                   |
| admission_level | lab_chloride_count              | float64        |        4148 |         58.7  |         66 | 3.0                    |
| admission_level | lab_chloride_abnormal_count     | float64        |        4148 |         58.7  |         42 | 0.0                    |
| admission_level | lab_chloride_missing_ratio      | float64        |        4148 |         58.7  |          5 | 0.0                    |
| admission_level | lab_anion_gap_mean              | float64        |        4421 |         62.57 |        512 | 11.333333015441895     |
| admission_level | lab_anion_gap_median            | float64        |        4421 |         62.57 |         41 | 11.0                   |
| admission_level | lab_anion_gap_min               | float64        |        4421 |         62.57 |         26 | 9.0                    |
| admission_level | lab_anion_gap_max               | float64        |        4421 |         62.57 |         33 | 14.0                   |
| admission_level | lab_anion_gap_std               | float64        |        4989 |         70.61 |        749 | 2.5166115760803223     |