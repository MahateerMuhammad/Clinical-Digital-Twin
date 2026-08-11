# Emergency Department Feature Coverage

Generated from `/Users/mc/Projects/Clinical-Digital-Twin/data/interim/features/emergency_features.parquet` (window: admittime + 0h).

- Cohort admissions: **546,028**
- With a linked ED stay: **202,415** (37.1%)
- Features produced: **66**

Admissions without an ED stay hold NaN, never 0.0 — the ED module is a partial capture of this cohort, and a filled zero would assert a measurement that was never taken.

| feature                    |   non_null_cohort_pct |   non_null_among_ed_pct |
|:---------------------------|----------------------:|------------------------:|
| ed_n_stays                 |                 37.07 |                  100    |
| ed_arrival_ambulance       |                 37.07 |                  100    |
| ed_available               |                 37.07 |                  100    |
| ed_los_hours               |                 37.02 |                   99.86 |
| ed_triage_acuity           |                 36.12 |                   97.45 |
| ed_triage_heartrate        |                 34.62 |                   93.39 |
| ed_triage_sbp              |                 34.52 |                   93.13 |
| ed_triage_shock_index      |                 34.42 |                   92.86 |
| ed_triage_dbp              |                 34.42 |                   92.85 |
| ed_triage_pulse_pressure   |                 34.4  |                   92.81 |
| ed_triage_resprate         |                 34.25 |                   92.4  |
| ed_triage_o2sat            |                 34.24 |                   92.37 |
| ed_triage_temperature      |                 33.85 |                   91.33 |
| ed_vital_heartrate_count   |                 32.7  |                   88.2  |
| ed_vital_resprate_count    |                 32.7  |                   88.2  |
| ed_vital_dbp_count         |                 32.7  |                   88.2  |
| ed_vital_o2sat_count       |                 32.7  |                   88.2  |
| ed_vital_temperature_count |                 32.7  |                   88.2  |
| ed_vital_sbp_count         |                 32.7  |                   88.2  |
| ed_vital_heartrate_max     |                 32.31 |                   87.15 |
| ed_vital_heartrate_first   |                 32.31 |                   87.15 |
| ed_vital_heartrate_delta   |                 32.31 |                   87.15 |
| ed_vital_heartrate_min     |                 32.31 |                   87.15 |
| ed_vital_heartrate_mean    |                 32.31 |                   87.15 |
| ed_vital_heartrate_last    |                 32.31 |                   87.15 |
| ed_vital_resprate_max      |                 32.24 |                   86.97 |
| ed_vital_resprate_last     |                 32.24 |                   86.97 |
| ed_vital_resprate_mean     |                 32.24 |                   86.97 |
| ed_vital_resprate_delta    |                 32.24 |                   86.97 |
| ed_vital_resprate_first    |                 32.24 |                   86.97 |
| ed_vital_resprate_min      |                 32.24 |                   86.97 |
| ed_vital_sbp_last          |                 32.21 |                   86.89 |
| ed_vital_sbp_mean          |                 32.21 |                   86.89 |
| ed_vital_dbp_min           |                 32.21 |                   86.89 |
| ed_vital_dbp_delta         |                 32.21 |                   86.89 |
| ed_vital_dbp_max           |                 32.21 |                   86.89 |
| ed_vital_dbp_last          |                 32.21 |                   86.89 |
| ed_vital_dbp_first         |                 32.21 |                   86.89 |
| ed_vital_sbp_max           |                 32.21 |                   86.89 |
| ed_vital_sbp_first         |                 32.21 |                   86.89 |
| ed_vital_sbp_min           |                 32.21 |                   86.89 |
| ed_vital_dbp_mean          |                 32.21 |                   86.89 |
| ed_vital_sbp_delta         |                 32.21 |                   86.89 |
| ed_triage_pain             |                 31.93 |                   86.13 |
| ed_vital_o2sat_mean        |                 31.44 |                   84.81 |
| ed_vital_o2sat_delta       |                 31.44 |                   84.81 |
| ed_vital_o2sat_last        |                 31.44 |                   84.81 |
| ed_vital_o2sat_min         |                 31.44 |                   84.81 |
| ed_vital_o2sat_first       |                 31.44 |                   84.81 |
| ed_vital_o2sat_max         |                 31.44 |                   84.81 |
| ed_medrecon_n_classes      |                 29.56 |                   79.74 |
| ed_medrecon_polypharmacy   |                 29.56 |                   79.74 |
| ed_medrecon_n_drugs        |                 29.56 |                   79.74 |
| ed_medrecon_n_unique_drugs |                 29.56 |                   79.74 |
| ed_vital_temperature_min   |                 28.5  |                   76.88 |
| ed_vital_temperature_max   |                 28.5  |                   76.88 |
| ed_vital_temperature_first |                 28.5  |                   76.88 |
| ed_vital_temperature_mean  |                 28.5  |                   76.88 |
| ed_vital_temperature_delta |                 28.5  |                   76.88 |
| ed_vital_temperature_last  |                 28.5  |                   76.88 |
| ed_vital_heartrate_std     |                 24.77 |                   66.81 |
| ed_vital_resprate_std      |                 24.62 |                   66.42 |
| ed_vital_dbp_std           |                 24.58 |                   66.31 |
| ed_vital_sbp_std           |                 24.57 |                   66.29 |
| ed_vital_o2sat_std         |                 23.9  |                   64.46 |
| ed_vital_temperature_std   |                 15.82 |                   42.67 |
