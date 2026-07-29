# In-Hospital Mortality Prediction — Model Comparison & Leakage Audit

## 1. Executive Summary & Leakage Protocol Audit

> [!WARNING]
> **Leakage Audit Flagged:** Run A (including diagnosis/Charlson ICD codes) achieved an AUROC > significantly higher (>0.05 gap) than the leak-free runs. In MIMIC-IV, ICD codes are assigned > post-hoc at discharge, so Run A is reported for audit purposes only and must not be quoted > as a performance result.

**Headline result — Run C (24h Window):** `XGBoost`

> Run A retains post-hoc ICD codes and is an upper bound under leakage, not a result. Run B removes diagnosis-derived features. **Run C additionally enforces the strict 24-hour observation window and is the figure quoted throughout the rest of the project.**

## 2. Test Set Performance Comparison Table

| Model | Feature Set | AUROC | AUPRC | Base Rate AUPRC | Brier Score | Decision Threshold | F1 | Precision | Recall |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **XGBoost** | Run A (With ICD) | **0.9961** | **0.8920** | 0.0216 | 0.0169 | 0.9302 | 0.8288 | 0.8385 | 0.8193 |
| **LightGBM** | Run A (With ICD) | **0.9965** | **0.9049** | 0.0216 | 0.0110 | 0.9350 | 0.8308 | 0.8536 | 0.8092 |
| **Logistic Regression** | Run A (With ICD) | **0.9918** | **0.7883** | 0.0216 | 0.0292 | 0.9414 | 0.7343 | 0.6740 | 0.8064 |
| **XGBoost** | Run B (Leak-Free) | **0.9907** | **0.8425** | 0.0216 | 0.0223 | 0.8558 | 0.7760 | 0.7412 | 0.8142 |
| **LightGBM** | Run B (Leak-Free) | **0.9917** | **0.8563** | 0.0216 | 0.0165 | 0.8736 | 0.7977 | 0.7768 | 0.8198 |
| **Logistic Regression** | Run B (Leak-Free) | **0.9824** | **0.7107** | 0.0216 | 0.0415 | 0.8838 | 0.6503 | 0.5413 | 0.8142 |
| **XGBoost** | Run C (24h Window) | **0.8273** | **0.1483** | 0.0216 | 0.1689 | 0.4976 | 0.1014 | 0.0542 | 0.7890 |
| **LightGBM** | Run C (24h Window) | **0.8260** | **0.1484** | 0.0216 | 0.1598 | 0.4686 | 0.1002 | 0.0535 | 0.7885 |
| **Logistic Regression** | Run C (24h Window) | **0.8252** | **0.1131** | 0.0216 | 0.1788 | 0.4952 | 0.1001 | 0.0534 | 0.7952 |
| **XGBoost (Calibrated)** | Run C (24h Window) | **0.8267** | **0.1420** | 0.0216 | 0.0197 | 0.0244 | 0.0957 | 0.0508 | 0.8304 |

## 3. Clinical Decision Threshold Rationale

The decision threshold was tuned on validation out-of-sample predictions aiming for a **target recall of ~80%** (high sensitivity) to prioritize flagging deteriorating patients early for clinical intervention.
