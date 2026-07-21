# Data Cleaning Report

Generated: 2026-07-19T19:35:01.814489Z

## Summary

> **Note on Duplicates:** The summary table (`dup_before`/`dup_after`) counts only the redundant extra copies (`keep='first'`), showing how many rows would be dropped to make the table unique. The Documented Actions table below (`flag_duplicates`) counts *all* rows involved in a duplicate group (`keep=False`), because all copies receive the `_is_duplicate` flag.

| table            |   n_actions | output_path                                                                            |   rows_before |   rows_after |   dup_before |   dup_after |
|:-----------------|------------:|:---------------------------------------------------------------------------------------|--------------:|-------------:|-------------:|------------:|
| patients         |           1 | /Users/apple/Desktop/Clinical Digital Twin/data/interim/patients_clean.parquet         |        364627 |       364627 |            0 |           0 |
| admissions       |          15 | /Users/apple/Desktop/Clinical Digital Twin/data/interim/admissions_clean.parquet       |        546028 |       546028 |            0 |           0 |
| diagnoses_icd    |           0 | /Users/apple/Desktop/Clinical Digital Twin/data/interim/diagnoses_icd_clean.parquet    |       6364488 |      6364488 |            0 |           0 |
| d_icd_diagnoses  |           0 | /Users/apple/Desktop/Clinical Digital Twin/data/interim/d_icd_diagnoses_clean.parquet  |        112107 |       112107 |            0 |           0 |
| procedures_icd   |           0 | /Users/apple/Desktop/Clinical Digital Twin/data/interim/procedures_icd_clean.parquet   |        859655 |       859655 |            0 |           0 |
| d_icd_procedures |           0 | /Users/apple/Desktop/Clinical Digital Twin/data/interim/d_icd_procedures_clean.parquet |         86423 |        86423 |            0 |           0 |
| d_labitems       |           1 | /Users/apple/Desktop/Clinical Digital Twin/data/interim/d_labitems_clean.parquet       |          1650 |         1650 |            0 |           0 |
| icustays         |           2 | /Users/apple/Desktop/Clinical Digital Twin/data/interim/icustays_clean.parquet         |         94458 |        94458 |            0 |           0 |
| d_items          |           4 | /Users/apple/Desktop/Clinical Digital Twin/data/interim/d_items_clean.parquet          |          4095 |         4095 |            0 |           0 |
| labevents        |          13 | /Users/apple/Desktop/Clinical Digital Twin/data/interim/labevents_clean.parquet        |      50390218 |     50390218 |            0 |           0 |
| prescriptions    |          24 | /Users/apple/Desktop/Clinical Digital Twin/data/interim/prescriptions_clean.parquet    |      20292611 |     20292611 |            0 |           0 |
| pharmacy         |          30 | /Users/apple/Desktop/Clinical Digital Twin/data/interim/pharmacy_clean.parquet         |      17847567 |     17847567 |            0 |           0 |
| emar             |          10 | /Users/apple/Desktop/Clinical Digital Twin/data/interim/emar_clean.parquet             |      42808593 |     42808593 |            0 |           0 |
| emar_detail      |          34 | /Users/apple/Desktop/Clinical Digital Twin/data/interim/emar_detail_clean.parquet      |      87371064 |     87371064 |         4861 |        4861 |
| chartevents      |           3 | /Users/apple/Desktop/Clinical Digital Twin/data/interim/chartevents_clean.parquet      |      62862809 |     62862809 |            0 |           0 |
| inputevents      |          11 | /Users/apple/Desktop/Clinical Digital Twin/data/interim/inputevents_clean.parquet      |      10953713 |     10953713 |            0 |           0 |
| outputevents     |           1 | /Users/apple/Desktop/Clinical Digital Twin/data/interim/outputevents_clean.parquet     |       5359395 |      5359395 |            0 |           0 |
| discharge        |           1 | /Users/apple/Desktop/Clinical Digital Twin/data/interim/discharge_clean.parquet        |        331793 |       331793 |            0 |           0 |
| radiology_detail |           0 | /Users/apple/Desktop/Clinical Digital Twin/data/interim/radiology_detail_clean.parquet |       6046121 |      6046121 |            0 |           0 |

## Documented Actions

| table         | action                  | column                               |   n_affected | decision                                                                             | details        |
|:--------------|:------------------------|:-------------------------------------|-------------:|:-------------------------------------------------------------------------------------|:---------------|
| patients      | document_missing        | dod                                  |       326326 | retained missing values; no imputation at cleaning stage                             | 89.50% missing |
| admissions    | document_missing        | deathtime                            |       534238 | retained missing values; no imputation at cleaning stage                             | 97.84% missing |
| admissions    | document_missing        | admit_provider_id                    |            4 | retained missing values; no imputation at cleaning stage                             | 0.00% missing  |
| admissions    | document_missing        | admission_location                   |            1 | retained missing values; no imputation at cleaning stage                             | 0.00% missing  |
| admissions    | document_missing        | discharge_location                   |       149818 | retained missing values; no imputation at cleaning stage                             | 27.44% missing |
| admissions    | document_missing        | insurance                            |         9355 | retained missing values; no imputation at cleaning stage                             | 1.71% missing  |
| admissions    | document_missing        | language                             |          775 | retained missing values; no imputation at cleaning stage                             | 0.14% missing  |
| admissions    | document_missing        | marital_status                       |        13619 | retained missing values; no imputation at cleaning stage                             | 2.49% missing  |
| admissions    | document_missing        | edregtime                            |       166788 | retained missing values; no imputation at cleaning stage                             | 30.55% missing |
| admissions    | document_missing        | edouttime                            |       166788 | retained missing values; no imputation at cleaning stage                             | 30.55% missing |
| admissions    | invalid_timestamp_order | admittime/dischtime                  |          175 | flagged with _invalid_time_order=1 and kept in dataset (no values dropped or nulled) |                |
| admissions    | normalize_categorical   | admission_location                   |            1 | empty strings normalized to NaN                                                      |                |
| admissions    | normalize_categorical   | discharge_location                   |       149818 | empty strings normalized to NaN                                                      |                |
| admissions    | normalize_categorical   | insurance                            |         9355 | empty strings normalized to NaN                                                      |                |
| admissions    | normalize_categorical   | language                             |          775 | empty strings normalized to NaN                                                      |                |
| admissions    | normalize_categorical   | marital_status                       |        13619 | empty strings normalized to NaN                                                      |                |
| d_labitems    | document_missing        | label                                |            4 | retained missing values; no imputation at cleaning stage                             | 0.24% missing  |
| icustays      | document_missing        | outtime                              |           14 | retained missing values; no imputation at cleaning stage                             | 0.01% missing  |
| icustays      | document_missing        | los                                  |           14 | retained missing values; no imputation at cleaning stage                             | 0.01% missing  |
| d_items       | document_missing        | unitname                             |         2972 | retained missing values; no imputation at cleaning stage                             | 72.58% missing |
| d_items       | document_missing        | lownormalvalue                       |         4076 | retained missing values; no imputation at cleaning stage                             | 99.54% missing |
| d_items       | document_missing        | highnormalvalue                      |         4073 | retained missing values; no imputation at cleaning stage                             | 99.46% missing |
| d_items       | normalize_categorical   | unitname                             |         2972 | empty strings normalized to NaN                                                      |                |
| labevents     | document_missing        | hadm_id                              |     18901693 | retained missing values; no imputation at cleaning stage                             | 37.51% missing |
| labevents     | document_missing        | order_provider_id                    |     38901253 | retained missing values; no imputation at cleaning stage                             | 77.20% missing |
| labevents     | document_missing        | storetime                            |           54 | retained missing values; no imputation at cleaning stage                             | 0.00% missing  |
| labevents     | document_missing        | value                                |        58098 | retained missing values; no imputation at cleaning stage                             | 0.12% missing  |
| labevents     | document_missing        | valuenum                             |        58108 | retained missing values; no imputation at cleaning stage                             | 0.12% missing  |
| labevents     | document_missing        | valueuom                             |           12 | retained missing values; no imputation at cleaning stage                             | 0.00% missing  |
| labevents     | document_missing        | ref_range_lower                      |       144072 | retained missing values; no imputation at cleaning stage                             | 0.29% missing  |
| labevents     | document_missing        | ref_range_upper                      |       144072 | retained missing values; no imputation at cleaning stage                             | 0.29% missing  |
| labevents     | document_missing        | flag                                 |     32665620 | retained missing values; no imputation at cleaning stage                             | 64.83% missing |
| labevents     | document_missing        | priority                             |      1185495 | retained missing values; no imputation at cleaning stage                             | 2.35% missing  |
| labevents     | document_missing        | comments                             |     45208080 | retained missing values; no imputation at cleaning stage                             | 89.72% missing |
| labevents     | normalize_categorical   | flag                                 |     32665620 | empty strings normalized to NaN                                                      |                |
| labevents     | normalize_categorical   | priority                             |      1185495 | empty strings normalized to NaN                                                      |                |
| prescriptions | strip_whitespace        | drug                                 |       278044 | strip leading/trailing whitespace                                                    |                |
| prescriptions | document_missing        | poe_id                               |       184441 | retained missing values; no imputation at cleaning stage                             | 0.91% missing  |
| prescriptions | document_missing        | poe_seq                              |       184441 | retained missing values; no imputation at cleaning stage                             | 0.91% missing  |
| prescriptions | document_missing        | order_provider_id                    |        66367 | retained missing values; no imputation at cleaning stage                             | 0.33% missing  |
| prescriptions | document_missing        | starttime                            |        21890 | retained missing values; no imputation at cleaning stage                             | 0.11% missing  |
| prescriptions | document_missing        | stoptime                             |        31436 | retained missing values; no imputation at cleaning stage                             | 0.15% missing  |
| prescriptions | document_missing        | drug                                 |            1 | retained missing values; no imputation at cleaning stage                             | 0.00% missing  |
| prescriptions | document_missing        | formulary_drug_cd                    |        24565 | retained missing values; no imputation at cleaning stage                             | 0.12% missing  |
| prescriptions | document_missing        | gsn                                  |      2407028 | retained missing values; no imputation at cleaning stage                             | 11.86% missing |
| prescriptions | document_missing        | ndc                                  |        33456 | retained missing values; no imputation at cleaning stage                             | 0.16% missing  |
| prescriptions | document_missing        | prod_strength                        |         9398 | retained missing values; no imputation at cleaning stage                             | 0.05% missing  |
| prescriptions | document_missing        | form_rx                              |     20267167 | retained missing values; no imputation at cleaning stage                             | 99.87% missing |
| prescriptions | document_missing        | dose_val_rx                          |         9349 | retained missing values; no imputation at cleaning stage                             | 0.05% missing  |
| prescriptions | document_missing        | dose_unit_rx                         |         9348 | retained missing values; no imputation at cleaning stage                             | 0.05% missing  |
| prescriptions | document_missing        | form_val_disp                        |         9372 | retained missing values; no imputation at cleaning stage                             | 0.05% missing  |
| prescriptions | document_missing        | form_unit_disp                       |         9389 | retained missing values; no imputation at cleaning stage                             | 0.05% missing  |
| prescriptions | document_missing        | doses_per_24_hrs                     |      7903921 | retained missing values; no imputation at cleaning stage                             | 38.95% missing |
| prescriptions | document_missing        | route                                |         6439 | retained missing values; no imputation at cleaning stage                             | 0.03% missing  |
| prescriptions | invalid_timestamp_order | starttime/stoptime                   |       816994 | flagged with _invalid_time_order=1 and kept in dataset (no values dropped or nulled) |                |
| prescriptions | normalize_categorical   | drug                                 |            1 | empty strings normalized to NaN                                                      |                |
| prescriptions | normalize_categorical   | route                                |         6439 | empty strings normalized to NaN                                                      |                |
| prescriptions | normalize_categorical   | form_rx                              |     20267167 | empty strings normalized to NaN                                                      |                |
| prescriptions | normalize_categorical   | dose_unit_rx                         |         9348 | empty strings normalized to NaN                                                      |                |
| prescriptions | normalize_categorical   | form_unit_disp                       |         9389 | empty strings normalized to NaN                                                      |                |
| pharmacy      | strip_whitespace        | medication                           |       271552 | strip leading/trailing whitespace                                                    |                |
| pharmacy      | strip_whitespace        | frequency                            |           17 | strip leading/trailing whitespace                                                    |                |
| pharmacy      | fix_datetime            | starttime                            |            2 | coerce invalid timestamps to NaT (retained as missing)                               |                |
| pharmacy      | fix_datetime            | expirationdate                       |            1 | coerce invalid timestamps to NaT (retained as missing)                               |                |
| pharmacy      | document_missing        | poe_id                               |       145597 | retained missing values; no imputation at cleaning stage                             | 0.82% missing  |
| pharmacy      | document_missing        | starttime                            |        21896 | retained missing values; no imputation at cleaning stage                             | 0.12% missing  |
| pharmacy      | document_missing        | stoptime                             |        92169 | retained missing values; no imputation at cleaning stage                             | 0.52% missing  |
| pharmacy      | document_missing        | medication                           |      1137574 | retained missing values; no imputation at cleaning stage                             | 6.37% missing  |
| pharmacy      | document_missing        | verifiedtime                         |         6571 | retained missing values; no imputation at cleaning stage                             | 0.04% missing  |
| pharmacy      | document_missing        | route                                |        74881 | retained missing values; no imputation at cleaning stage                             | 0.42% missing  |
| pharmacy      | document_missing        | frequency                            |        96223 | retained missing values; no imputation at cleaning stage                             | 0.54% missing  |
| pharmacy      | document_missing        | disp_sched                           |      7172767 | retained missing values; no imputation at cleaning stage                             | 40.19% missing |
| pharmacy      | document_missing        | infusion_type                        |     16635584 | retained missing values; no imputation at cleaning stage                             | 93.21% missing |
| pharmacy      | document_missing        | sliding_scale                        |     15964957 | retained missing values; no imputation at cleaning stage                             | 89.45% missing |
| pharmacy      | document_missing        | lockout_interval                     |     17541248 | retained missing values; no imputation at cleaning stage                             | 98.28% missing |
| pharmacy      | document_missing        | basal_rate                           |     17540691 | retained missing values; no imputation at cleaning stage                             | 98.28% missing |
| pharmacy      | document_missing        | one_hr_max                           |     17785592 | retained missing values; no imputation at cleaning stage                             | 99.65% missing |
| pharmacy      | document_missing        | doses_per_24_hrs                     |      6884721 | retained missing values; no imputation at cleaning stage                             | 38.58% missing |
| pharmacy      | document_missing        | duration                             |     13539612 | retained missing values; no imputation at cleaning stage                             | 75.86% missing |
| pharmacy      | document_missing        | duration_interval                    |        96540 | retained missing values; no imputation at cleaning stage                             | 0.54% missing  |
| pharmacy      | document_missing        | expiration_value                     |      2263523 | retained missing values; no imputation at cleaning stage                             | 12.68% missing |
| pharmacy      | document_missing        | expiration_unit                      |       101837 | retained missing values; no imputation at cleaning stage                             | 0.57% missing  |
| pharmacy      | document_missing        | expirationdate                       |     17825816 | retained missing values; no imputation at cleaning stage                             | 99.88% missing |
| pharmacy      | document_missing        | dispensation                         |        79257 | retained missing values; no imputation at cleaning stage                             | 0.44% missing  |
| pharmacy      | document_missing        | fill_quantity                        |     17825816 | retained missing values; no imputation at cleaning stage                             | 99.88% missing |
| pharmacy      | invalid_timestamp_order | starttime/stoptime                   |       717513 | flagged with _invalid_time_order=1 and kept in dataset (no values dropped or nulled) |                |
| pharmacy      | normalize_categorical   | medication                           |      1137574 | empty strings normalized to NaN                                                      |                |
| pharmacy      | normalize_categorical   | route                                |        74881 | empty strings normalized to NaN                                                      |                |
| pharmacy      | normalize_categorical   | frequency                            |        96223 | empty strings normalized to NaN                                                      |                |
| pharmacy      | normalize_categorical   | infusion_type                        |     16635584 | empty strings normalized to NaN                                                      |                |
| emar          | strip_whitespace        | medication                           |       350677 | strip leading/trailing whitespace                                                    |                |
| emar          | strip_whitespace        | event_txt                            |        62128 | strip leading/trailing whitespace                                                    |                |
| emar          | document_missing        | hadm_id                              |      1417390 | retained missing values; no imputation at cleaning stage                             | 3.31% missing  |
| emar          | document_missing        | pharmacy_id                          |      8314698 | retained missing values; no imputation at cleaning stage                             | 19.42% missing |
| emar          | document_missing        | enter_provider_id                    |     36753267 | retained missing values; no imputation at cleaning stage                             | 85.85% missing |
| emar          | document_missing        | medication                           |      2120873 | retained missing values; no imputation at cleaning stage                             | 4.95% missing  |
| emar          | document_missing        | event_txt                            |       446102 | retained missing values; no imputation at cleaning stage                             | 1.04% missing  |
| emar          | document_missing        | scheduletime                         |        23249 | retained missing values; no imputation at cleaning stage                             | 0.05% missing  |
| emar          | normalize_categorical   | medication                           |      2120873 | empty strings normalized to NaN                                                      |                |
| emar          | normalize_categorical   | event_txt                            |       446102 | empty strings normalized to NaN                                                      |                |
| emar_detail   | document_missing        | parent_field_ordinal                 |     42808593 | retained missing values; no imputation at cleaning stage                             | 49.00% missing |
| emar_detail   | document_missing        | administration_type                  |     44569078 | retained missing values; no imputation at cleaning stage                             | 51.01% missing |
| emar_detail   | document_missing        | pharmacy_id                          |     45515584 | retained missing values; no imputation at cleaning stage                             | 52.09% missing |
| emar_detail   | document_missing        | barcode_type                         |     44509140 | retained missing values; no imputation at cleaning stage                             | 50.94% missing |
| emar_detail   | document_missing        | reason_for_no_barcode                |     85682865 | retained missing values; no imputation at cleaning stage                             | 98.07% missing |
| emar_detail   | document_missing        | complete_dose_not_given              |     80372558 | retained missing values; no imputation at cleaning stage                             | 91.99% missing |
| emar_detail   | document_missing        | dose_due                             |     45761669 | retained missing values; no imputation at cleaning stage                             | 52.38% missing |
| emar_detail   | document_missing        | dose_due_unit                        |     44955793 | retained missing values; no imputation at cleaning stage                             | 51.45% missing |
| emar_detail   | document_missing        | dose_given                           |     46269155 | retained missing values; no imputation at cleaning stage                             | 52.96% missing |
| emar_detail   | document_missing        | dose_given_unit                      |     45453448 | retained missing values; no imputation at cleaning stage                             | 52.02% missing |
| emar_detail   | document_missing        | will_remainder_of_dose_be_given      |     75688157 | retained missing values; no imputation at cleaning stage                             | 86.63% missing |
| emar_detail   | document_missing        | product_amount_given                 |     48893438 | retained missing values; no imputation at cleaning stage                             | 55.96% missing |
| emar_detail   | document_missing        | product_unit                         |     48097464 | retained missing values; no imputation at cleaning stage                             | 55.05% missing |
| emar_detail   | document_missing        | product_code                         |     45211473 | retained missing values; no imputation at cleaning stage                             | 51.75% missing |
| emar_detail   | document_missing        | product_description                  |     52416526 | retained missing values; no imputation at cleaning stage                             | 59.99% missing |
| emar_detail   | document_missing        | product_description_other            |     84075573 | retained missing values; no imputation at cleaning stage                             | 96.23% missing |
| emar_detail   | document_missing        | prior_infusion_rate                  |     85774630 | retained missing values; no imputation at cleaning stage                             | 98.17% missing |
| emar_detail   | document_missing        | infusion_rate                        |     85266628 | retained missing values; no imputation at cleaning stage                             | 97.59% missing |
| emar_detail   | document_missing        | infusion_rate_adjustment             |     84698263 | retained missing values; no imputation at cleaning stage                             | 96.94% missing |
| emar_detail   | document_missing        | infusion_rate_adjustment_amount      |     87318523 | retained missing values; no imputation at cleaning stage                             | 99.94% missing |
| emar_detail   | document_missing        | infusion_rate_unit                   |     84563807 | retained missing values; no imputation at cleaning stage                             | 96.79% missing |
| emar_detail   | document_missing        | route                                |     73934074 | retained missing values; no imputation at cleaning stage                             | 84.62% missing |
| emar_detail   | document_missing        | infusion_complete                    |     87032694 | retained missing values; no imputation at cleaning stage                             | 99.61% missing |
| emar_detail   | document_missing        | completion_interval                  |     87120518 | retained missing values; no imputation at cleaning stage                             | 99.71% missing |
| emar_detail   | document_missing        | new_iv_bag_hung                      |     86952622 | retained missing values; no imputation at cleaning stage                             | 99.52% missing |
| emar_detail   | document_missing        | continued_infusion_in_other_location |     87353201 | retained missing values; no imputation at cleaning stage                             | 99.98% missing |
| emar_detail   | document_missing        | restart_interval                     |     87251759 | retained missing values; no imputation at cleaning stage                             | 99.86% missing |
| emar_detail   | document_missing        | side                                 |     87122310 | retained missing values; no imputation at cleaning stage                             | 99.72% missing |
| emar_detail   | document_missing        | site                                 |     87122323 | retained missing values; no imputation at cleaning stage                             | 99.72% missing |
| emar_detail   | document_missing        | non_formulary_visual_verification    |     87222045 | retained missing values; no imputation at cleaning stage                             | 99.83% missing |
| emar_detail   | flag_duplicates         |                                      |         9722 | duplicates retained with _is_duplicate flag; not removed                             |                |
| emar_detail   | normalize_categorical   | administration_type                  |     44569078 | empty strings normalized to NaN                                                      |                |
| emar_detail   | normalize_categorical   | barcode_type                         |     44509140 | empty strings normalized to NaN                                                      |                |
| emar_detail   | normalize_categorical   | route                                |     73934074 | empty strings normalized to NaN                                                      |                |
| chartevents   | document_missing        | value                                |       289687 | retained missing values; no imputation at cleaning stage                             | 0.46% missing  |
| chartevents   | document_missing        | valueuom                             |      6614250 | retained missing values; no imputation at cleaning stage                             | 10.52% missing |
| chartevents   | normalize_categorical   | valueuom                             |      6614250 | empty strings normalized to NaN                                                      |                |
| inputevents   | document_missing        | rate                                 |      4897231 | retained missing values; no imputation at cleaning stage                             | 44.71% missing |
| inputevents   | document_missing        | rateuom                              |      4897231 | retained missing values; no imputation at cleaning stage                             | 44.71% missing |
| inputevents   | document_missing        | secondaryordercategoryname           |      3185024 | retained missing values; no imputation at cleaning stage                             | 29.08% missing |
| inputevents   | document_missing        | totalamount                          |      1521884 | retained missing values; no imputation at cleaning stage                             | 13.89% missing |
| inputevents   | document_missing        | totalamountuom                       |      1518590 | retained missing values; no imputation at cleaning stage                             | 13.86% missing |
| inputevents   | cap_outliers            | amount                               |           18 | set clinically impossible values to NaN (range [0, 10000000.0])                      |                |
| inputevents   | cap_outliers            | rate                                 |           72 | set clinically impossible values to NaN (range [0, 100000.0])                        |                |
| inputevents   | cap_outliers            | patientweight                        |         1722 | set clinically impossible values to NaN (range [1, 500])                             |                |
| inputevents   | invalid_timestamp_order | starttime/endtime                    |           10 | flagged with _invalid_time_order=1 and kept in dataset (no values dropped or nulled) |                |
| inputevents   | normalize_categorical   | secondaryordercategoryname           |      3185024 | empty strings normalized to NaN                                                      |                |
| inputevents   | normalize_categorical   | rateuom                              |      4897231 | empty strings normalized to NaN                                                      |                |
| outputevents  | cap_outliers            | value                                |          606 | set clinically impossible values to NaN (range [0, 1000000.0])                       |                |
| discharge     | document_missing        | storetime                            |           17 | retained missing values; no imputation at cleaning stage                             | 0.01% missing  |