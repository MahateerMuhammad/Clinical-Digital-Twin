# In-Hospital Mortality Prediction — Model Comparison & Leakage Audit

## 1. Executive Summary & Leakage Protocol Audit

> [!WARNING]
> **Leakage Audit Flagged:** Run A (including diagnosis/Charlson ICD codes) achieved an AUROC > significantly higher (>0.05 gap) than the leak-free runs. In MIMIC-IV, ICD codes are assigned > post-hoc at discharge, so Run A is reported for audit purposes only and must not be quoted > as a performance result.

**Headline result — Run C (24h Window):** `LightGBM`

> Run A retains post-hoc ICD codes and is an upper bound under leakage, not a result. Run B removes diagnosis-derived features. **Run C additionally enforces the strict 24-hour observation window and is the figure quoted throughout the rest of the project.**

## 2. Test Set Performance Comparison Table

| Model | Feature Set | AUROC | AUPRC | Base Rate AUPRC | Brier Score | Decision Threshold | F1 | Precision | Recall |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **XGBoost** | Run A (With ICD) | **0.9961** | **0.8940** | 0.0216 | 0.0167 | 0.9322 | 0.8274 | 0.8427 | 0.8125 |
| **LightGBM** | Run A (With ICD) | **0.9966** | **0.9048** | 0.0216 | 0.0108 | 0.9357 | 0.8357 | 0.8584 | 0.8142 |
| **Logistic Regression** | Run A (With ICD) | **0.9919** | **0.7876** | 0.0216 | 0.0292 | 0.9394 | 0.7322 | 0.6716 | 0.8047 |
| **XGBoost** | Run B (Leak-Free) | **0.9907** | **0.8428** | 0.0216 | 0.0221 | 0.8611 | 0.7784 | 0.7476 | 0.8120 |
| **LightGBM** | Run B (Leak-Free) | **0.9915** | **0.8550** | 0.0216 | 0.0164 | 0.8773 | 0.7933 | 0.7765 | 0.8109 |
| **Logistic Regression** | Run B (Leak-Free) | **0.9827** | **0.7116** | 0.0216 | 0.0414 | 0.8855 | 0.6529 | 0.5445 | 0.8153 |
| **XGBoost** | Run C (24h Window) | **0.9421** | **0.3654** | 0.0216 | 0.0853 | 0.6024 | 0.2648 | 0.1586 | 0.8019 |
| **LightGBM** | Run C (24h Window) | **0.9442** | **0.3800** | 0.0216 | 0.0768 | 0.5743 | 0.2671 | 0.1602 | 0.8030 |
| **Logistic Regression** | Run C (24h Window) | **0.9296** | **0.2568** | 0.0216 | 0.1064 | 0.5852 | 0.2325 | 0.1356 | 0.8142 |
| **LightGBM (Calibrated)** | Run C (24h Window) | **0.9438** | **0.3608** | 0.0216 | 0.0164 | 0.0437 | 0.2437 | 0.1424 | 0.8444 |

## 3. Clinical Decision Threshold Rationale

The decision threshold was tuned on validation out-of-sample predictions aiming for a **target recall of ~80%** (high sensitivity) to prioritize flagging deteriorating patients early for clinical intervention.
