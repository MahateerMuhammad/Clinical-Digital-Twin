# Two-Stage Length of Stay (LOS) Prediction — Model Results & Methodological Audit

## 1. Methodological Rationale & Two-Stage Framework

> [!NOTE]
> **Literature Rationale:** Multiple MIMIC-IV LOS studies found that direct regression using only > early/admission-time features performs poorly across the full LOS range, because LOS has a long > right tail (a small number of very long stays) that early features cannot predict well. The consistently > recommended framework across this literature is: (1) classify short vs. long stay first, (2) only apply > regression to predict exact duration within the 'short' bucket, and explicitly acknowledge the limitation > that this framework is not designed to precisely predict long-stay durations.

**Empirical 75th Percentile Thresholds (Training Set Split):**
- **Hospital LOS (`los_days`):** `5.63` days
- **ICU LOS (`icu_los_days`):** `4.18` days (evaluated on ICU admission cohort)

## 2. Stage A — Short vs. Long Stay Classification Performance

| Target | Model Name | Run Protocol | AUROC | AUPRC | Base Rate AUPRC | Brier Score | Decision Threshold | F1 | Precision | Recall |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Hospital LOS | **Logistic Regression** | Stage A (Admission-Time) | **0.8179** | **0.5570** | 0.2486 | 0.1809 | 0.5163 | 0.5738 | 0.4477 | 0.7985 |
| Hospital LOS | **XGBoost** | Stage A (Admission-Time) | **0.8311** | **0.5878** | 0.2486 | 0.1746 | 0.5289 | 0.5876 | 0.4659 | 0.7953 |
| Hospital LOS | **LightGBM** | Stage A (Admission-Time) | **0.8350** | **0.5958** | 0.2486 | 0.1723 | 0.5341 | 0.5931 | 0.4719 | 0.7981 |
| Hospital LOS | **LightGBM (Calibrated)** | Stage A (Admission-Time) | **0.8349** | **0.5879** | 0.2486 | 0.1368 | 0.2689 | 0.5931 | 0.4716 | 0.7991 |
| ICU LOS | **Logistic Regression** | Stage A (Admission-Time) | **0.6706** | **0.4025** | 0.2498 | 0.2271 | 0.4158 | 0.4476 | 0.3100 | 0.8050 |
| ICU LOS | **XGBoost** | Stage A (Admission-Time) | **0.7012** | **0.4478** | 0.2498 | 0.2132 | 0.4093 | 0.4663 | 0.3286 | 0.8025 |
| ICU LOS | **LightGBM** | Stage A (Admission-Time) | **0.7015** | **0.4500** | 0.2498 | 0.2082 | 0.3888 | 0.4655 | 0.3270 | 0.8078 |
| ICU LOS | **LightGBM (Calibrated)** | Stage A (Admission-Time) | **0.7004** | **0.4372** | 0.2498 | 0.1685 | 0.1935 | 0.4619 | 0.3175 | 0.8473 |

## 3. Stage B — Short-Bucket Duration Regression Performance

> [!IMPORTANT]
> **Evaluation Scope Discipline:** Stage B regression metrics (MAE, RMSE, R²) are evaluated **strictly within the restricted short-stay bucket** > (`<= 75th percentile threshold`). Primary deployment metrics reflect performance on the **predicted short bucket** (Stage A classifier output), > while actual-bucket metrics serve as an optimistic upper bound.

| Target | Model Name | Evaluation Protocol (Scope) | Sample Size (N) | MAE (days) | RMSE (days) | R² Score |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| Hospital LOS | **LightGBM Regressor** | Predicted Short Bucket (Deployment Primary) | 47,924 | **1.3366** | **3.3196** | **0.1189** |
| Hospital LOS | **XGBoost Regressor** | Predicted Short Bucket (Deployment Primary) | 47,924 | **1.3405** | **3.3202** | **0.1186** |
| Hospital LOS | **LightGBM Regressor** | Actual Short Bucket (Optimistic Upper Bound) | 62,221 | **0.8306** | **1.0763** | **0.4825** |
| Hospital LOS | **XGBoost Regressor** | Actual Short Bucket (Optimistic Upper Bound) | 62,221 | **0.8340** | **1.0796** | **0.4793** |
| ICU LOS | **LightGBM Regressor** | Predicted Short Bucket (Deployment Primary) | 4,294 | **1.5351** | **4.2500** | **-0.0429** |
| ICU LOS | **XGBoost Regressor** | Predicted Short Bucket (Deployment Primary) | 4,294 | **1.5350** | **4.2500** | **-0.0429** |
| ICU LOS | **LightGBM Regressor** | Actual Short Bucket (Optimistic Upper Bound) | 9,660 | **0.7655** | **0.9339** | **0.0783** |
| ICU LOS | **XGBoost Regressor** | Actual Short Bucket (Optimistic Upper Bound) | 9,660 | **0.7660** | **0.9339** | **0.0784** |

## 4. Key Observations & Framework Limitations

1. **Right-Tail Isolation:** Classifying long stays in Stage A effectively isolates extreme outliers (>75th percentile), preventing regression skew.
2. **Deployment Realism:** Evaluating Stage B on predicted short-stay patients captures real-world error propagation from Stage A classification.
3. **Explicit Scope Limitation:** Exact duration predictions are provided ONLY for the short-stay bucket; long-stay cases are flagged for clinical review without artificial exact-day estimates.
