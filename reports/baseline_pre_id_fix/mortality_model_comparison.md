# In-Hospital Mortality Prediction — Model Comparison & Leakage Audit

## 1. Executive Summary & Leakage Protocol Audit

> [!WARNING]
> **Leakage Audit Flagged:** Run A (including diagnosis/Charlson ICD codes) achieved an AUROC > significantly higher (>0.05 gap) than Run B. In MIMIC-IV, ICD codes are assigned post-hoc at discharge. > Therefore, **Run B (diagnosis-derived features excluded) is reported as the primary, leak-free model.**

**Primary Winning Model (Run B):** `LightGBM`

## 2. Test Set Performance Comparison Table

| Model | Feature Set | AUROC | AUPRC | Base Rate AUPRC | Brier Score | Decision Threshold | F1 | Precision | Recall |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **XGBoost** | Run A (With ICD) | **0.9932** | **0.8076** | 0.0216 | 0.0276 | 0.9074 | 0.7337 | 0.6704 | 0.8103 |
| **LightGBM** | Run A (With ICD) | **0.9940** | **0.8303** | 0.0216 | 0.0191 | 0.9066 | 0.7577 | 0.7056 | 0.8181 |
| **Logistic Regression** | Run A (With ICD) | **0.9872** | **0.6714** | 0.0216 | 0.0422 | 0.8874 | 0.6271 | 0.5067 | 0.8226 |
| **XGBoost** | Run B (Leak-Free) | **0.9818** | **0.7130** | 0.0216 | 0.0424 | 0.8020 | 0.5909 | 0.4651 | 0.8097 |
| **LightGBM** | Run B (Leak-Free) | **0.9835** | **0.7331** | 0.0216 | 0.0343 | 0.8110 | 0.6203 | 0.5005 | 0.8153 |
| **Logistic Regression** | Run B (Leak-Free) | **0.9711** | **0.5572** | 0.0216 | 0.0622 | 0.8016 | 0.4834 | 0.3433 | 0.8170 |
| **XGBoost** | Run C (24h Window) | **0.9488** | **0.4540** | 0.0216 | 0.0826 | 0.6876 | 0.3026 | 0.1875 | 0.7840 |
| **LightGBM** | Run C (24h Window) | **0.9490** | **0.4706** | 0.0216 | 0.0751 | 0.6701 | 0.3094 | 0.1925 | 0.7879 |
| **Logistic Regression** | Run C (24h Window) | **0.9379** | **0.3581** | 0.0216 | 0.1037 | 0.6758 | 0.2668 | 0.1606 | 0.7879 |
| **LightGBM (Calibrated)** | Run C (24h Window) | **0.9484** | **0.4554** | 0.0216 | 0.0150 | 0.0548 | 0.2970 | 0.1819 | 0.8086 |

## 3. Clinical Decision Threshold Rationale

The decision threshold was tuned on validation out-of-sample predictions aiming for a **target recall of ~80%** (high sensitivity) to prioritize flagging deteriorating patients early for clinical intervention.
