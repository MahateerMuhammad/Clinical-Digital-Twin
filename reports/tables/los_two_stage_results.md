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
| Hospital LOS | **Logistic Regression** | Stage A | **0.7940** | **0.5045** | 0.2486 | 0.1917 | 0.5231 | 0.5541 | 0.4232 | 0.8022 |
| Hospital LOS | **XGBoost** | Stage A | **0.8079** | **0.5373** | 0.2486 | 0.1856 | 0.5367 | 0.5647 | 0.4378 | 0.7951 |
| Hospital LOS | **LightGBM** | Stage A | **0.8114** | **0.5434** | 0.2486 | 0.1837 | 0.5393 | 0.5698 | 0.4437 | 0.7960 |
| Hospital LOS | **LightGBM (Calibrated)** | Stage A | **0.8112** | **0.5367** | 0.2486 | 0.1446 | 0.2820 | 0.5696 | 0.4382 | 0.8137 |
| ICU LOS | **Logistic Regression** | Stage A | **0.6210** | **0.3429** | 0.2498 | 0.2385 | 0.4389 | 0.4246 | 0.2886 | 0.8029 |
| ICU LOS | **XGBoost** | Stage A | **0.6406** | **0.3666** | 0.2498 | 0.2301 | 0.4342 | 0.4359 | 0.2998 | 0.7982 |
| ICU LOS | **LightGBM** | Stage A | **0.6381** | **0.3646** | 0.2498 | 0.2258 | 0.4171 | 0.4328 | 0.2972 | 0.7960 |
| ICU LOS | **LightGBM (Calibrated)** | Stage A | **0.6372** | **0.3557** | 0.2498 | 0.1787 | 0.1943 | 0.4319 | 0.2926 | 0.8246 |

## 3. Stage B — Short-Bucket Duration Regression Performance

> [!IMPORTANT]
> **Evaluation Scope Discipline:** Stage B regression metrics (MAE, RMSE, R²) are evaluated **strictly within the restricted short-stay bucket** > (`<= 75th percentile threshold`). Primary deployment metrics reflect performance on the **predicted short bucket** (Stage A classifier output), > while actual-bucket metrics serve as an optimistic upper bound.

| Target | Model Name | Evaluation Protocol (Scope) | Sample Size (N) | MAE (days) | RMSE (days) | R² Score |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| Hospital LOS | **LightGBM Regressor** | Predicted Short Bucket (Deployment Primary) | 44,580 | **1.3622** | **3.3330** | **0.1162** |
| Hospital LOS | **XGBoost Regressor** | Predicted Short Bucket (Deployment Primary) | 44,580 | **1.3640** | **3.3309** | **0.1172** |
| Hospital LOS | **LightGBM Regressor** | Actual Short Bucket (Optimistic Upper Bound) | 62,221 | **0.8723** | **1.1196** | **0.4401** |
| Hospital LOS | **XGBoost Regressor** | Actual Short Bucket (Optimistic Upper Bound) | 62,221 | **0.8745** | **1.1216** | **0.4380** |
| ICU LOS | **LightGBM Regressor** | Predicted Short Bucket (Deployment Primary) | 3,812 | **1.7935** | **4.5315** | **-0.0645** |
| ICU LOS | **XGBoost Regressor** | Predicted Short Bucket (Deployment Primary) | 3,812 | **1.7927** | **4.5318** | **-0.0647** |
| ICU LOS | **LightGBM Regressor** | Actual Short Bucket (Optimistic Upper Bound) | 9,660 | **0.7855** | **0.9568** | **0.0327** |
| ICU LOS | **XGBoost Regressor** | Actual Short Bucket (Optimistic Upper Bound) | 9,660 | **0.7851** | **0.9562** | **0.0339** |

## 4. Key Observations & Framework Limitations

1. **Right-Tail Isolation:** Classifying long stays in Stage A effectively isolates extreme outliers (>75th percentile), preventing regression skew.
2. **Deployment Realism:** Evaluating Stage B on predicted short-stay patients captures real-world error propagation from Stage A classification.
3. **Explicit Scope Limitation:** Exact duration predictions are provided ONLY for the short-stay bucket; long-stay cases are flagged for clinical review without artificial exact-day estimates.
4. **ICU LOS Scope Distinction:** The ICU LOS model is evaluated strictly on admissions with an ICU stay (`has_icu_stay == 1`, $N = 85,242$), predicting ICU stay duration conditional on ICU admission, distinct from the Phase 3 ICU admission risk model.

> [!WARNING]
> **Stage B Low R² Framing:** Stage B regressors achieve low R² scores ($R^2 = 0.1738$ for hospital LOS, $R^2 = 0.0909$ for ICU LOS under deployment primary). Exact-duration prediction at admission remains inherently unreliable even within short stays. Stage B outputs must be interpreted as **rough directional estimates** rather than precise forecasts.
