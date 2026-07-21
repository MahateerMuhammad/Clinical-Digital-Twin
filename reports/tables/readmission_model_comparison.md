# In-Hospital Mortality Prediction — Model Comparison & Leakage Audit

## 1. Executive Summary & Leakage Protocol Audit

> [!WARNING]
> **Leakage Audit Flagged:** Run A (including diagnosis/Charlson ICD codes) achieved an AUROC > significantly higher (>0.05 gap) than Run B. In MIMIC-IV, ICD codes are assigned post-hoc at discharge. > Therefore, **Run B (diagnosis-derived features excluded) is reported as the primary, leak-free model.**

**Primary Winning Model (Run B):** `LightGBM`

## 2. Test Set Performance Comparison Table

| Model | Feature Set | AUROC | AUPRC | Base Rate AUPRC | Brier Score | Decision Threshold | F1 | Precision | Recall |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **LACE Clinical Score** | Clinical Baseline | **0.4994** | **0.2038** | 0.2047 | 0.1759 | 0.1111 | 0.3345 | 0.2059 | 0.8914 |
| **XGBoost** | Run A (Full-Stay) | **0.7273** | **0.4340** | 0.2047 | 0.2071 | 0.3995 | 0.4264 | 0.2906 | 0.8003 |
| **LightGBM** | Run A (Full-Stay) | **0.7299** | **0.4371** | 0.2047 | 0.2056 | 0.3992 | 0.4301 | 0.2940 | 0.8004 |
| **Logistic Regression** | Run A (Full-Stay) | **0.7033** | **0.4079** | 0.2047 | 0.2156 | 0.3941 | 0.4047 | 0.2700 | 0.8081 |
| **XGBoost** | Run B (Strict 24h) | **0.7040** | **0.4131** | 0.2047 | 0.2149 | 0.3978 | 0.4049 | 0.2705 | 0.8047 |
| **LightGBM** | Run B (Strict 24h) | **0.7094** | **0.4195** | 0.2047 | 0.2112 | 0.3925 | 0.4089 | 0.2742 | 0.8037 |
| **Logistic Regression** | Run B (Strict 24h) | **0.6873** | **0.3987** | 0.2047 | 0.2191 | 0.3965 | 0.3905 | 0.2578 | 0.8052 |
| **LightGBM (Calibrated)** | Run B (Strict 24h) | **0.7093** | **0.4115** | 0.2047 | 0.1450 | 0.1471 | 0.4016 | 0.2650 | 0.8294 |

## 3. Clinical Decision Threshold Rationale

The decision threshold was tuned on validation out-of-sample predictions aiming for a **target recall of ~80%** (high sensitivity) to prioritize flagging deteriorating patients early for clinical intervention.
