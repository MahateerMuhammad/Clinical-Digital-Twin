# Data Cleaning Report

Generated: 2026-07-20T12:46:55.213902Z

## Summary

> **Note on Duplicates:** The summary table (`dup_before`/`dup_after`) counts only the redundant extra copies (`keep='first'`), showing how many rows would be dropped to make the table unique. The Documented Actions table below (`flag_duplicates`) counts *all* rows involved in a duplicate group (`keep=False`), because all copies receive the `_is_duplicate` flag.

| table            |   n_actions | output_path                                                                          |   rows_before |   rows_after |   dup_before |   dup_after |
|:-----------------|------------:|:-------------------------------------------------------------------------------------|--------------:|-------------:|-------------:|------------:|
| patients         |           1 | /Users/mc/Projects/Clinical-Digital-Twin/data/interim/patients_clean.parquet         |        364627 |       364627 |            0 |           0 |
| admissions       |          15 | /Users/mc/Projects/Clinical-Digital-Twin/data/interim/admissions_clean.parquet       |        546028 |       546028 |            0 |           0 |
| diagnoses_icd    |           0 | /Users/mc/Projects/Clinical-Digital-Twin/data/interim/diagnoses_icd_clean.parquet    |       6364488 |      6364488 |            0 |           0 |
| d_icd_diagnoses  |           0 | /Users/mc/Projects/Clinical-Digital-Twin/data/interim/d_icd_diagnoses_clean.parquet  |        112107 |       112107 |            0 |           0 |
| procedures_icd   |           0 | /Users/mc/Projects/Clinical-Digital-Twin/data/interim/procedures_icd_clean.parquet   |        859655 |       859655 |            0 |           0 |
| d_icd_procedures |           0 | /Users/mc/Projects/Clinical-Digital-Twin/data/interim/d_icd_procedures_clean.parquet |         86423 |        86423 |            0 |           0 |
| d_labitems       |           1 | /Users/mc/Projects/Clinical-Digital-Twin/data/interim/d_labitems_clean.parquet       |          1650 |         1650 |            0 |           0 |
| icustays         |           2 | /Users/mc/Projects/Clinical-Digital-Twin/data/interim/icustays_clean.parquet         |         94458 |        94458 |            0 |           0 |
| d_items          |           3 | /Users/mc/Projects/Clinical-Digital-Twin/data/interim/d_items_clean.parquet          |          4095 |         4095 |            0 |           0 |

## Documented Actions

| table      | action                  | column              |   n_affected | decision                                                                             | details        |
|:-----------|:------------------------|:--------------------|-------------:|:-------------------------------------------------------------------------------------|:---------------|
| patients   | document_missing        | dod                 |       326326 | retained missing values; no imputation at cleaning stage                             | 89.50% missing |
| admissions | strip_whitespace        | admission_location  |            1 | strip leading/trailing whitespace                                                    |                |
| admissions | strip_whitespace        | discharge_location  |       149818 | strip leading/trailing whitespace                                                    |                |
| admissions | strip_whitespace        | insurance           |         9355 | strip leading/trailing whitespace                                                    |                |
| admissions | strip_whitespace        | language            |          775 | strip leading/trailing whitespace                                                    |                |
| admissions | strip_whitespace        | marital_status      |        13619 | strip leading/trailing whitespace                                                    |                |
| admissions | document_missing        | deathtime           |       534238 | retained missing values; no imputation at cleaning stage                             | 97.84% missing |
| admissions | document_missing        | admit_provider_id   |            4 | retained missing values; no imputation at cleaning stage                             | 0.00% missing  |
| admissions | document_missing        | admission_location  |            1 | retained missing values; no imputation at cleaning stage                             | 0.00% missing  |
| admissions | document_missing        | discharge_location  |       149818 | retained missing values; no imputation at cleaning stage                             | 27.44% missing |
| admissions | document_missing        | insurance           |         9355 | retained missing values; no imputation at cleaning stage                             | 1.71% missing  |
| admissions | document_missing        | language            |          775 | retained missing values; no imputation at cleaning stage                             | 0.14% missing  |
| admissions | document_missing        | marital_status      |        13619 | retained missing values; no imputation at cleaning stage                             | 2.49% missing  |
| admissions | document_missing        | edregtime           |       166788 | retained missing values; no imputation at cleaning stage                             | 30.55% missing |
| admissions | document_missing        | edouttime           |       166788 | retained missing values; no imputation at cleaning stage                             | 30.55% missing |
| admissions | invalid_timestamp_order | admittime/dischtime |          175 | flagged with _invalid_time_order=1 and kept in dataset (no values dropped or nulled) |                |
| d_labitems | document_missing        | label               |            4 | retained missing values; no imputation at cleaning stage                             | 0.24% missing  |
| icustays   | document_missing        | outtime             |           14 | retained missing values; no imputation at cleaning stage                             | 0.01% missing  |
| icustays   | document_missing        | los                 |           14 | retained missing values; no imputation at cleaning stage                             | 0.01% missing  |
| d_items    | document_missing        | unitname            |         2972 | retained missing values; no imputation at cleaning stage                             | 72.58% missing |
| d_items    | document_missing        | lownormalvalue      |         4076 | retained missing values; no imputation at cleaning stage                             | 99.54% missing |
| d_items    | document_missing        | highnormalvalue     |         4073 | retained missing values; no imputation at cleaning stage                             | 99.46% missing |