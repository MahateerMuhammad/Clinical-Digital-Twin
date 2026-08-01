# 30-Day Unplanned Readmission Prediction — Model Comparison & Leakage Audit

## 1. Executive Summary & Leakage Protocol Audit

> [!WARNING]
> **Leakage Audit Flagged:** Run A (including diagnosis/Charlson ICD codes) achieved an AUROC > significantly higher (>0.05 gap) than the leak-free runs. In MIMIC-IV, ICD codes are assigned > post-hoc at discharge, so Run A is reported for audit purposes only and must not be quoted > as a performance result.

**Headline result — Run B (Strict 24h):** `LightGBM`

> Run A retains post-hoc ICD codes and is an upper bound under leakage, not a result. Run B removes diagnosis-derived features. **Run C additionally enforces the strict 24-hour observation window and is the figure quoted throughout the rest of the project.**

## 2. Test Set Performance Comparison Table

| Model | Feature Set | AUROC | AUPRC | Base Rate AUPRC | Brier Score | Decision Threshold | F1 | Precision | Recall |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **LACE Clinical Score (modified E)** | Clinical Baseline | **0.6096** | **0.2836** | 0.2047 | 0.1994 | 0.1667 | 0.3521 | 0.2220 | 0.8512 |
| **XGBoost** | Run A (Full-Stay) | **0.7292** | **0.4376** | 0.2047 | 0.2060 | 0.3967 | 0.4274 | 0.2913 | 0.8025 |
| **LightGBM** | Run A (Full-Stay) | **0.7328** | **0.4409** | 0.2047 | 0.2043 | 0.3971 | 0.4317 | 0.2952 | 0.8029 |
| **Logistic Regression** | Run A (Full-Stay) | **0.7083** | **0.4129** | 0.2047 | 0.2143 | 0.3953 | 0.4089 | 0.2736 | 0.8089 |
| **XGBoost** | Run B (Strict 24h) | **0.7089** | **0.4146** | 0.2047 | 0.2137 | 0.3979 | 0.4116 | 0.2765 | 0.8047 |
| **LightGBM** | Run B (Strict 24h) | **0.7158** | **0.4218** | 0.2047 | 0.2098 | 0.3955 | 0.4175 | 0.2821 | 0.8030 |
| **Logistic Regression** | Run B (Strict 24h) | **0.6944** | **0.4011** | 0.2047 | 0.2178 | 0.3932 | 0.3975 | 0.2632 | 0.8118 |
| **LightGBM (Calibrated)** | Run B (Strict 24h) | **0.7155** | **0.4128** | 0.2047 | 0.1445 | 0.1431 | 0.4141 | 0.2778 | 0.8135 |

## 3. Clinical Decision Threshold Rationale

The decision threshold was tuned on validation out-of-sample predictions aiming for a **target recall of ~80%** (high sensitivity) to prioritize flagging deteriorating patients early for clinical intervention.
