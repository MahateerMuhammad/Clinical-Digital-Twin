# In-Hospital Mortality Prediction — Model Comparison & Leakage Audit

## 1. Executive Summary & Leakage Protocol Audit

> [!WARNING]
> **Leakage Audit Flagged:** Run A (including diagnosis/Charlson ICD codes) achieved an AUROC > significantly higher (>0.05 gap) than the leak-free runs. In MIMIC-IV, ICD codes are assigned > post-hoc at discharge, so Run A is reported for audit purposes only and must not be quoted > as a performance result.

**Headline result — Run C (24h Window):** `LightGBM`

> Run A retains post-hoc ICD codes and is an upper bound under leakage, not a result. Run B removes diagnosis-derived features. **Run C additionally enforces the strict 24-hour observation window and is the figure quoted throughout the rest of the project.**

## 2. Test Set Performance Comparison Table

| Model | Feature Set | AUROC | AUPRC | Base Rate AUPRC | Brier Score | Decision Threshold | F1 | Precision | Recall |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **XGBoost** | Run A (With ICD) | **0.9961** | **0.8917** | 0.0216 | 0.0169 | 0.9308 | 0.8239 | 0.8351 | 0.8131 |
| **LightGBM** | Run A (With ICD) | **0.9966** | **0.9035** | 0.0216 | 0.0109 | 0.9373 | 0.8313 | 0.8591 | 0.8053 |
| **Logistic Regression** | Run A (With ICD) | **0.9918** | **0.7883** | 0.0216 | 0.0293 | 0.9406 | 0.7320 | 0.6713 | 0.8047 |
| **XGBoost** | Run B (Leak-Free) | **0.9907** | **0.8424** | 0.0216 | 0.0223 | 0.8581 | 0.7782 | 0.7448 | 0.8148 |
| **LightGBM** | Run B (Leak-Free) | **0.9917** | **0.8556** | 0.0216 | 0.0165 | 0.8733 | 0.7938 | 0.7734 | 0.8153 |
| **Logistic Regression** | Run B (Leak-Free) | **0.9825** | **0.7114** | 0.0216 | 0.0415 | 0.8855 | 0.6494 | 0.5426 | 0.8086 |
| **XGBoost** | Run C (24h Window) | **0.9035** | **0.3122** | 0.0216 | 0.1060 | 0.5112 | 0.1779 | 0.1005 | 0.7722 |
| **LightGBM** | Run C (24h Window) | **0.9062** | **0.3281** | 0.0216 | 0.0957 | 0.4794 | 0.1812 | 0.1026 | 0.7767 |
| **Logistic Regression** | Run C (24h Window) | **0.8938** | **0.2265** | 0.0216 | 0.1281 | 0.5251 | 0.1637 | 0.0914 | 0.7846 |
| **LightGBM (Calibrated)** | Run C (24h Window) | **0.9059** | **0.3131** | 0.0216 | 0.0172 | 0.0261 | 0.1749 | 0.0984 | 0.7868 |

## 3. Clinical Decision Threshold Rationale

The decision threshold was tuned on validation out-of-sample predictions aiming for a **target recall of ~80%** (high sensitivity) to prioritize flagging deteriorating patients early for clinical intervention.
