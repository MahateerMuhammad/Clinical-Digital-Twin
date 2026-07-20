# Data Dictionary

Generated: 2026-07-20T11:47:16.806927Z

## Dataset Summary

| dataset                  |   n_rows |   n_cols |
|:-------------------------|---------:|---------:|
| patient_level            |   223452 |       14 |
| admission_level          |   546028 |      336 |
| icu_level                |    94458 |      136 |
| time_series              | 71323299 |       10 |
| clinical_notes           |   331793 |       13 |
| similarity               |   546028 |       35 |
| admission_level_selected |   546028 |      187 |

## Feature Dictionary (sample)

| dataset         | feature                         | dtype          |   n_missing |   pct_missing |   n_unique | example                |
|:----------------|:--------------------------------|:---------------|------------:|--------------:|-----------:|:-----------------------|
| patient_level   | subject_id                      | int32          |           0 |          0    |     223452 | 10000032               |
| patient_level   | n_admissions                    | int64          |           0 |          0    |         97 | 4                      |
| patient_level   | los_days_mean                   | float64        |           0 |          0    |      79406 | 1.4444444444444444     |
| patient_level   | los_days_max                    | float64        |           0 |          0    |      36730 | 2.2222222222222223     |
| patient_level   | los_days_sum                    | float64        |           0 |          0    |      62960 | 5.777777777777778      |
| patient_level   | ever_inhosp_mortality           | int8           |           0 |          0    |          2 | 0                      |
| patient_level   | ever_readmission_30d            | int8           |           0 |          0    |          2 | 1                      |
| patient_level   | ever_icu_stay                   | int8           |           0 |          0    |          2 | 1                      |
| patient_level   | insurance                       | category       |        5723 |          2.56 |          5 | Medicaid               |
| patient_level   | race                            | category       |           0 |          0    |         33 | WHITE                  |
| patient_level   | gender                          | category       |           0 |          0    |          2 | F                      |
| patient_level   | anchor_age                      | int8           |           0 |          0    |         73 | 52                     |
| patient_level   | anchor_year                     | int16          |           0 |          0    |         99 | 2180                   |
| patient_level   | anchor_year_group               | category       |           0 |          0    |          5 | 2014 - 2016            |
| admission_level | subject_id                      | int32          |           0 |          0    |     223452 | 10000032               |
| admission_level | hadm_id                         | int32          |           0 |          0    |     546028 | 22595853               |
| admission_level | admittime                       | datetime64[ns] |           0 |          0    |     534919 | 2180-05-06 22:23:00    |
| admission_level | dischtime                       | datetime64[ns] |           0 |          0    |     528871 | 2180-05-07 17:15:00    |
| admission_level | deathtime                       | datetime64[ns] |      534238 |         97.84 |      11788 | 2134-12-06 12:54:00    |
| admission_level | admission_type                  | category       |           0 |          0    |          9 | URGENT                 |
| admission_level | admit_provider_id               | category       |           4 |          0    |       2045 | P49AFC                 |
| admission_level | admission_location              | category       |           1 |          0    |         11 | TRANSFER FROM HOSPITAL |
| admission_level | discharge_location              | category       |      149818 |         27.44 |         13 | HOME                   |
| admission_level | insurance                       | category       |        9355 |          1.71 |          5 | Medicaid               |
| admission_level | language                        | category       |         775 |          0.14 |         25 | English                |
| admission_level | marital_status                  | category       |       13619 |          2.49 |          4 | WIDOWED                |
| admission_level | race                            | category       |           0 |          0    |         33 | WHITE                  |
| admission_level | edregtime                       | datetime64[ns] |      166788 |         30.55 |     372692 | 2180-05-06 19:17:00    |
| admission_level | edouttime                       | datetime64[ns] |      166788 |         30.55 |     372755 | 2180-05-06 23:30:00    |
| admission_level | hospital_expire_flag            | int8           |           0 |          0    |          2 | 0                      |
| admission_level | _is_duplicate                   | int8           |           0 |          0    |          1 | 0                      |
| admission_level | _invalid_time_order             | int8           |           0 |          0    |          2 | 0                      |
| admission_level | n_icu_stays                     | float64        |           0 |          0    |         10 | 0.0                    |
| admission_level | icu_los_days                    | float32        |           0 |          0    |      77872 | 0.0                    |
| admission_level | has_icu_stay                    | int8           |           0 |          0    |          2 | 0                      |
| admission_level | los_days                        | float64        |           0 |          0    |      39931 | 0.7861111111111111     |
| admission_level | los_hours                       | float64        |           0 |          0    |      39931 | 18.866666666666667     |
| admission_level | admit_hour                      | int32          |           0 |          0    |         24 | 22                     |
| admission_level | admit_dow                       | int32          |           0 |          0    |          7 | 5                      |
| admission_level | admit_month                     | int32          |           0 |          0    |         12 | 5                      |
| admission_level | admit_year                      | int32          |           0 |          0    |        108 | 2180                   |
| admission_level | weekend_admission               | int8           |           0 |          0    |          2 | 1                      |
| admission_level | night_admission                 | int8           |           0 |          0    |          2 | 1                      |
| admission_level | next_admittime                  | datetime64[ns] |      223452 |         40.92 |     318726 | 2180-06-26 18:27:00    |
| admission_level | days_to_readmission             | float64        |      223452 |         40.92 |     222636 | 50.05                  |
| admission_level | readmission_30d                 | int8           |           0 |          0    |          2 | 0                      |
| admission_level | gender                          | category       |           0 |          0    |          2 | F                      |
| admission_level | anchor_age                      | int8           |           0 |          0    |         73 | 52                     |
| admission_level | anchor_year                     | int16          |           0 |          0    |         99 | 2180                   |
| admission_level | anchor_year_group               | category       |           0 |          0    |          5 | 2014 - 2016            |
| admission_level | dod                             | datetime64[ns] |      401062 |         73.45 |      21981 | 2180-09-09 00:00:00    |
| admission_level | diagnosis_count                 | float64        |         531 |          0.1  |         42 | 8.0                    |
| admission_level | unique_diagnosis_count          | float64        |         531 |          0.1  |         42 | 8.0                    |
| admission_level | primary_icd_code                | object         |         531 |          0.1  |      14222 | 5723                   |
| admission_level | cci_myocardial_infarction       | float64        |         531 |          0.1  |          2 | 0.0                    |
| admission_level | cci_congestive_heart_failure    | float64        |         531 |          0.1  |          2 | 0.0                    |
| admission_level | cci_peripheral_vascular_disease | float64        |         531 |          0.1  |          2 | 0.0                    |
| admission_level | cci_cerebrovascular_disease     | float64        |         531 |          0.1  |          2 | 0.0                    |
| admission_level | cci_dementia                    | float64        |         531 |          0.1  |          2 | 0.0                    |
| admission_level | cci_copd                        | float64        |         531 |          0.1  |          2 | 1.0                    |
| admission_level | cci_connective_tissue_disease   | float64        |         531 |          0.1  |          2 | 0.0                    |
| admission_level | cci_peptic_ulcer                | float64        |         531 |          0.1  |          2 | 0.0                    |
| admission_level | cci_mild_liver_disease          | float64        |         531 |          0.1  |          2 | 1.0                    |
| admission_level | cci_diabetes_uncomplicated      | float64        |         531 |          0.1  |          2 | 0.0                    |
| admission_level | cci_diabetes_complicated        | float64        |         531 |          0.1  |          2 | 0.0                    |
| admission_level | cci_hemiplegia_paraplegia       | float64        |         531 |          0.1  |          2 | 0.0                    |
| admission_level | cci_renal_disease               | float64        |         531 |          0.1  |          2 | 0.0                    |
| admission_level | cci_malignancy                  | float64        |         531 |          0.1  |          2 | 0.0                    |
| admission_level | cci_severe_liver_disease        | float64        |         531 |          0.1  |          2 | 1.0                    |
| admission_level | cci_metastatic_tumor            | float64        |         531 |          0.1  |          2 | 0.0                    |
| admission_level | cci_aids                        | float64        |         531 |          0.1  |          2 | 0.0                    |
| admission_level | charlson_comorbidity_index      | float64        |         531 |          0.1  |         22 | 5.0                    |
| admission_level | dx_diabetes                     | float64        |         531 |          0.1  |          2 | 0.0                    |
| admission_level | dx_hypertension                 | float64        |         531 |          0.1  |          2 | 0.0                    |
| admission_level | dx_ckd                          | float64        |         531 |          0.1  |          2 | 0.0                    |
| admission_level | dx_cad                          | float64        |         531 |          0.1  |          2 | 0.0                    |
| admission_level | dx_copd_flag                    | float64        |         531 |          0.1  |          2 | 1.0                    |
| admission_level | dx_stroke                       | float64        |         531 |          0.1  |          2 | 0.0                    |
| admission_level | icd_embedding_placeholder       | object         |         531 |          0.1  |          1 |                        |
| admission_level | procedure_count                 | float64        |      258524 |         47.35 |         41 | 1.0                    |
| admission_level | unique_procedure_count          | float64        |      258524 |         47.35 |         36 | 1.0                    |
| admission_level | major_procedure_count           | float64        |      258524 |         47.35 |         41 | 1.0                    |
| admission_level | has_major_procedure             | float64        |      258524 |         47.35 |          2 | 1.0                    |
| admission_level | lab_total_count                 | float64        |      325257 |         59.57 |       1370 | 32.0                   |
| admission_level | lab_unique_items                | float64        |      325257 |         59.57 |         19 | 12.0                   |
| admission_level | lab_creatinine_mean             | float64        |      335644 |         61.47 |      17387 | 0.43333330750465393    |
| admission_level | lab_creatinine_median           | float64        |      335644 |         61.47 |        505 | 0.4000000059604645     |
| admission_level | lab_creatinine_min              | float64        |      335644 |         61.47 |        195 | 0.4000000059604645     |
| admission_level | lab_creatinine_max              | float64        |      335644 |         61.47 |        252 | 0.5                    |
| admission_level | lab_creatinine_std              | float64        |      381900 |         69.94 |      31561 | 0.05773502215743065    |
| admission_level | lab_creatinine_first            | float64        |      335644 |         61.47 |        236 | 0.5                    |
| admission_level | lab_creatinine_last             | float64        |      335644 |         61.47 |        210 | 0.4000000059604645     |
| admission_level | lab_creatinine_count            | float64        |      325257 |         59.57 |        200 | 3.0                    |
| admission_level | lab_creatinine_abnormal_count   | float64        |      325257 |         59.57 |        147 | 0.0                    |
| admission_level | lab_creatinine_missing_ratio    | float64        |      325257 |         59.57 |        107 | 0.0                    |
| admission_level | lab_sodium_mean                 | float64        |      338560 |         62    |       8258 | 130.6666717529297      |
| admission_level | lab_sodium_median               | float64        |      338560 |         62    |        110 | 130.0                  |
| admission_level | lab_sodium_min                  | float64        |      338560 |         62    |         84 | 130.0                  |
| admission_level | lab_sodium_max                  | float64        |      338560 |         62    |         67 | 132.0                  |
| admission_level | lab_sodium_std                  | float64        |      382402 |         70.03 |      19773 | 1.154700517654419      |
| admission_level | lab_sodium_first                | float64        |      338560 |         62    |         86 | 132.0                  |
| admission_level | lab_sodium_last                 | float64        |      338560 |         62    |         67 | 130.0                  |
| admission_level | lab_sodium_count                | float64        |      325257 |         59.57 |        202 | 3.0                    |
| admission_level | lab_sodium_abnormal_count       | float64        |      325257 |         59.57 |        106 | 3.0                    |
| admission_level | lab_sodium_missing_ratio        | float64        |      325257 |         59.57 |         64 | 0.0                    |
| admission_level | lab_potassium_mean              | float64        |      337761 |         61.86 |      12062 | 4.9666666984558105     |
| admission_level | lab_potassium_median            | float64        |      337761 |         61.86 |        161 | 5.0                    |
| admission_level | lab_potassium_min               | float64        |      337761 |         61.86 |         82 | 4.699999809265137      |
| admission_level | lab_potassium_max               | float64        |      337761 |         61.86 |         92 | 5.199999809265137      |
| admission_level | lab_potassium_std               | float64        |      381236 |         69.82 |      43496 | 0.25166115164756775    |
| admission_level | lab_potassium_first             | float64        |      337761 |         61.86 |         89 | 4.699999809265137      |
| admission_level | lab_potassium_last              | float64        |      337761 |         61.86 |         82 | 5.199999809265137      |
| admission_level | lab_potassium_count             | float64        |      325257 |         59.57 |        204 | 3.0                    |
| admission_level | lab_potassium_abnormal_count    | float64        |      325257 |         59.57 |         55 | 1.0                    |
| admission_level | lab_potassium_missing_ratio     | float64        |      325257 |         59.57 |        124 | 0.0                    |
| admission_level | lab_bun_mean                    | float64        |      337591 |         61.83 |      14568 | 32.0                   |
| admission_level | lab_bun_median                  | float64        |      337591 |         61.83 |        332 | 33.0                   |
| admission_level | lab_bun_min                     | float64        |      337591 |         61.83 |        169 | 28.0                   |
| admission_level | lab_bun_max                     | float64        |      337591 |         61.83 |        238 | 35.0                   |
| admission_level | lab_bun_std                     | float64        |      382998 |         70.14 |      35131 | 3.605551242828369      |
| admission_level | lab_bun_first                   | float64        |      337591 |         61.83 |        217 | 33.0                   |
| admission_level | lab_bun_last                    | float64        |      337591 |         61.83 |        204 | 35.0                   |
| admission_level | lab_bun_count                   | float64        |      325257 |         59.57 |        195 | 3.0                    |
| admission_level | lab_bun_abnormal_count          | float64        |      325257 |         59.57 |        158 | 3.0                    |
| admission_level | lab_bun_missing_ratio           | float64        |      325257 |         59.57 |        137 | 0.0                    |
| admission_level | lab_wbc_mean                    | float64        |      332806 |         60.95 |      28960 | 4.449999809265137      |
| admission_level | lab_wbc_median                  | float64        |      332806 |         60.95 |       1472 | 4.449999809265137      |
| admission_level | lab_wbc_min                     | float64        |      332806 |         60.95 |        668 | 4.099999904632568      |
| admission_level | lab_wbc_max                     | float64        |      332806 |         60.95 |       1027 | 4.800000190734863      |
| admission_level | lab_wbc_std                     | float64        |      381150 |         69.8  |      83279 | 0.49497494101524353    |
| admission_level | lab_wbc_first                   | float64        |      332806 |         60.95 |        873 | 4.099999904632568      |
| admission_level | lab_wbc_last                    | float64        |      332806 |         60.95 |        799 | 4.800000190734863      |
| admission_level | lab_wbc_count                   | float64        |      325257 |         59.57 |        186 | 2.0                    |
| admission_level | lab_wbc_abnormal_count          | float64        |      325257 |         59.57 |        151 | 0.0                    |
| admission_level | lab_wbc_missing_ratio           | float64        |      325257 |         59.57 |        222 | 0.0                    |
| admission_level | lab_hemoglobin_mean             | float64        |      332439 |         60.88 |      19608 | 11.549999237060547     |
| admission_level | lab_hemoglobin_median           | float64        |      332439 |         60.88 |        396 | 11.549999237060547     |
| admission_level | lab_hemoglobin_min              | float64        |      332439 |         60.88 |        176 | 11.199999809265137     |
| admission_level | lab_hemoglobin_max              | float64        |      332439 |         60.88 |        169 | 11.899999618530273     |
| admission_level | lab_hemoglobin_std              | float64        |      380778 |         69.74 |      62651 | 0.49497461318969727    |
| admission_level | lab_hemoglobin_first            | float64        |      332439 |         60.88 |        182 | 11.899999618530273     |
| admission_level | lab_hemoglobin_last             | float64        |      332439 |         60.88 |        168 | 11.199999809265137     |
| admission_level | lab_hemoglobin_count            | float64        |      325257 |         59.57 |        194 | 2.0                    |
| admission_level | lab_hemoglobin_abnormal_count   | float64        |      325257 |         59.57 |        192 | 2.0                    |
| admission_level | lab_hemoglobin_missing_ratio    | float64        |      325257 |         59.57 |        145 | 0.0                    |
| admission_level | lab_platelets_mean              | float64        |      331648 |         60.74 |      31551 | 94.5                   |
| admission_level | lab_platelets_median            | float64        |      331648 |         60.74 |       1784 | 94.5                   |
| admission_level | lab_platelets_min               | float64        |      331648 |         60.74 |        931 | 94.0                   |
| admission_level | lab_platelets_max               | float64        |      331648 |         60.74 |       1235 | 95.0                   |
| admission_level | lab_platelets_std               | float64        |      380047 |         69.6  |      74854 | 0.7071067690849304     |
| admission_level | lab_platelets_first             | float64        |      331648 |         60.74 |       1059 | 94.0                   |
| admission_level | lab_platelets_last              | float64        |      331648 |         60.74 |       1100 | 95.0                   |
| admission_level | lab_platelets_count             | float64        |      325257 |         59.57 |        202 | 2.0                    |
| admission_level | lab_platelets_abnormal_count    | float64        |      325257 |         59.57 |        180 | 2.0                    |
| admission_level | lab_platelets_missing_ratio     | float64        |      325257 |         59.57 |        299 | 0.0                    |
| admission_level | lab_glucose_mean                | float64        |      340282 |         62.32 |      20300 | 114.33333587646484     |
| admission_level | lab_glucose_median              | float64        |      340282 |         62.32 |        856 | 115.0                  |
| admission_level | lab_glucose_min                 | float64        |      340282 |         62.32 |        514 | 107.0                  |
| admission_level | lab_glucose_max                 | float64        |      340282 |         62.32 |       1307 | 121.0                  |
| admission_level | lab_glucose_std                 | float64        |      384750 |         70.46 |      73425 | 7.023768901824951      |
| admission_level | lab_glucose_first               | float64        |      340282 |         62.32 |        816 | 115.0                  |
| admission_level | lab_glucose_last                | float64        |      340282 |         62.32 |        579 | 121.0                  |
| admission_level | lab_glucose_count               | float64        |      325257 |         59.57 |        198 | 3.0                    |
| admission_level | lab_glucose_abnormal_count      | float64        |      325257 |         59.57 |        173 | 3.0                    |
| admission_level | lab_glucose_missing_ratio       | float64        |      325257 |         59.57 |         62 | 0.0                    |
| admission_level | lab_hematocrit_mean             | float64        |      328430 |         60.15 |      32880 | 33.44999694824219      |
| admission_level | lab_hematocrit_median           | float64        |      328430 |         60.15 |       1065 | 33.44999694824219      |
| admission_level | lab_hematocrit_min              | float64        |      328430 |         60.15 |        491 | 32.099998474121094     |
| admission_level | lab_hematocrit_max              | float64        |      328430 |         60.15 |        462 | 34.79999923706055      |
| admission_level | lab_hematocrit_std              | float64        |      376167 |         68.89 |     100570 | 1.9091888666152954     |
| admission_level | lab_hematocrit_first            | float64        |      328430 |         60.15 |        493 | 34.79999923706055      |
| admission_level | lab_hematocrit_last             | float64        |      328430 |         60.15 |        457 | 32.099998474121094     |
| admission_level | lab_hematocrit_count            | float64        |      325257 |         59.57 |        195 | 2.0                    |
| admission_level | lab_hematocrit_abnormal_count   | float64        |      325257 |         59.57 |        194 | 2.0                    |
| admission_level | lab_hematocrit_missing_ratio    | float64        |      325257 |         59.57 |        148 | 0.0                    |
| admission_level | lab_bicarbonate_mean            | float64        |      339966 |         62.26 |       7680 | 24.0                   |
| admission_level | lab_bicarbonate_median          | float64        |      339966 |         62.26 |         94 | 24.0                   |
| admission_level | lab_bicarbonate_min             | float64        |      339966 |         62.26 |         60 | 21.0                   |
| admission_level | lab_bicarbonate_max             | float64        |      339966 |         62.26 |         51 | 27.0                   |
| admission_level | lab_bicarbonate_std             | float64        |      384220 |         70.37 |      18721 | 3.0                    |
| admission_level | lab_bicarbonate_first           | float64        |      339966 |         62.26 |         55 | 21.0                   |
| admission_level | lab_bicarbonate_last            | float64        |      339966 |         62.26 |         51 | 27.0                   |
| admission_level | lab_bicarbonate_count           | float64        |      325257 |         59.57 |        199 | 3.0                    |
| admission_level | lab_bicarbonate_abnormal_count  | float64        |      325257 |         59.57 |        104 | 1.0                    |
| admission_level | lab_bicarbonate_missing_ratio   | float64        |      325257 |         59.57 |         87 | 0.0                    |
| admission_level | lab_chloride_mean               | float64        |      338801 |         62.05 |       8972 | 100.33333587646484     |
| admission_level | lab_chloride_median             | float64        |      338801 |         62.05 |        116 | 102.0                  |
| admission_level | lab_chloride_min                | float64        |      338801 |         62.05 |         80 | 97.0                   |
| admission_level | lab_chloride_max                | float64        |      338801 |         62.05 |         70 | 102.0                  |
| admission_level | lab_chloride_std                | float64        |      382706 |         70.09 |      21887 | 2.886751174926758      |
| admission_level | lab_chloride_first              | float64        |      338801 |         62.05 |         87 | 102.0                  |
| admission_level | lab_chloride_last               | float64        |      338801 |         62.05 |         68 | 97.0                   |
| admission_level | lab_chloride_count              | float64        |      325257 |         59.57 |        196 | 3.0                    |
| admission_level | lab_chloride_abnormal_count     | float64        |      325257 |         59.57 |        111 | 0.0                    |
| admission_level | lab_chloride_missing_ratio      | float64        |      325257 |         59.57 |         71 | 0.0                    |
| admission_level | lab_anion_gap_mean              | float64        |      339981 |         62.26 |       6531 | 11.333333015441895     |
| admission_level | lab_anion_gap_median            | float64        |      339981 |         62.26 |         84 | 11.0                   |
| admission_level | lab_anion_gap_min               | float64        |      339981 |         62.26 |         61 | 9.0                    |
| admission_level | lab_anion_gap_max               | float64        |      339981 |         62.26 |         63 | 14.0                   |
| admission_level | lab_anion_gap_std               | float64        |      384261 |         70.37 |      16700 | 2.5166115760803223     |