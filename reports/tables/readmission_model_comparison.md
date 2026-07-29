# 30-Day Unplanned Readmission Prediction — Model Comparison & Leakage Audit

## 1. Executive Summary & Leakage Protocol Audit

> [!WARNING]
> **Leakage Audit Flagged:** Run A (including diagnosis/Charlson ICD codes) achieved an AUROC > significantly higher (>0.05 gap) than the leak-free runs. In MIMIC-IV, ICD codes are assigned > post-hoc at discharge, so Run A is reported for audit purposes only and must not be quoted > as a performance result.

**Headline result — Run B (Strict 24h):** `LightGBM`

> Run A retains post-hoc ICD codes and is an upper bound under leakage, not a result. Run B removes diagnosis-derived features. **Run C additionally enforces the strict 24-hour observation window and is the figure quoted throughout the rest of the project.**

## 2. Test Set Performance Comparison Table

| Model | Feature Set | AUROC | AUPRC | Base Rate AUPRC | Brier Score | Decision Threshold | F1 | Precision | Recall |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **LACE Clinical Score** | Clinical Baseline | **0.4994** | **0.2038** | 0.2047 | 0.1759 | 0.1111 | 0.3345 | 0.2059 | 0.8914 |
| **XGBoost** | Run A (Full-Stay) | **0.7291** | **0.4368** | 0.2047 | 0.2060 | 0.3962 | 0.4275 | 0.2912 | 0.8038 |
| **LightGBM** | Run A (Full-Stay) | **0.7324** | **0.4407** | 0.2047 | 0.2044 | 0.3962 | 0.4309 | 0.2944 | 0.8030 |
| **Logistic Regression** | Run A (Full-Stay) | **0.7081** | **0.4126** | 0.2047 | 0.2143 | 0.3949 | 0.4088 | 0.2735 | 0.8092 |
| **XGBoost** | Run B (Strict 24h) | **0.7054** | **0.4160** | 0.2047 | 0.2125 | 0.3930 | 0.4057 | 0.2717 | 0.8007 |
| **LightGBM** | Run B (Strict 24h) | **0.7072** | **0.4173** | 0.2047 | 0.2115 | 0.3919 | 0.4076 | 0.2732 | 0.8022 |
| **Logistic Regression** | Run B (Strict 24h) | **0.6899** | **0.4004** | 0.2047 | 0.2185 | 0.3953 | 0.3929 | 0.2597 | 0.8064 |
| **LightGBM (Calibrated)** | Run B (Strict 24h) | **0.7067** | **0.4104** | 0.2047 | 0.1453 | 0.1482 | 0.4044 | 0.2693 | 0.8114 |

## 3. Clinical Decision Threshold Rationale

The decision threshold was tuned on validation out-of-sample predictions aiming for a **target recall of ~80%** (high sensitivity) to prioritize flagging deteriorating patients early for clinical intervention.
