# Clinical Deterioration Model Comparison & Results

## Proxy Definition & Prediction Window Documentation

- **Primary Proxy Event**: Ward-to-ICU transfer (`time_to_icu > 6 hours` or ward-origin admissions requiring ICU admission).
- **Limitations**: Captures severe deterioration requiring ICU-level care; misses ward-only deterioration without transfer (e.g. CMO/DNR) and direct ward mortality.
- **Prediction Window**: 6-hour pre-transfer window. Shorter windows score higher technically but offer less clinical lead time; longer windows offer more warning but exhibit signal degradation. 6 hours is a tunable starting point.

## Model Performance Comparison Table

| Model              |   AUROC |   AUPRC |   Base Rate |   Brier Score Pre-Calib |   Brier Score Post-Calib |
|:-------------------|--------:|--------:|------------:|------------------------:|-------------------------:|
| LogisticRegression |  0.8625 |  0.3551 |      0.0595 |                  0.1479 |                 nan      |
| XGBoost            |  0.8865 |  0.424  |      0.0595 |                  0.1361 |                 nan      |
| LightGBM           |  0.8878 |  0.4302 |      0.0595 |                  0.1338 |                   0.0428 |

---
*Report generated automatically by DeteriorationModelPipeline*
