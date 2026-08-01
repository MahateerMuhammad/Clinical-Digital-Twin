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
| Hospital LOS | **Logistic Regression** | Stage A (Admission-Time) | **0.8873** | **0.7356** | 0.2486 | 0.1403 | 0.5089 | 0.6628 | 0.5641 | 0.8034 |
| Hospital LOS | **XGBoost** | Stage A (Admission-Time) | **0.8973** | **0.7527** | 0.2486 | 0.1343 | 0.5349 | 0.6768 | 0.5877 | 0.7978 |
| Hospital LOS | **LightGBM** | Stage A (Admission-Time) | **0.9001** | **0.7576** | 0.2486 | 0.1323 | 0.5524 | 0.6845 | 0.5989 | 0.7985 |
| Hospital LOS | **LightGBM (Calibrated)** | Stage A (Admission-Time) | **0.9001** | **0.7503** | 0.2486 | 0.1072 | 0.2852 | 0.6791 | 0.5775 | 0.8241 |
| ICU LOS | **Logistic Regression** | Stage A (Admission-Time) | **0.8374** | **0.6689** | 0.2498 | 0.1618 | 0.3912 | 0.5895 | 0.4589 | 0.8237 |
| ICU LOS | **XGBoost** | Stage A (Admission-Time) | **0.8527** | **0.6970** | 0.2498 | 0.1521 | 0.3999 | 0.6058 | 0.4781 | 0.8265 |
| ICU LOS | **LightGBM** | Stage A (Admission-Time) | **0.8527** | **0.6962** | 0.2498 | 0.1489 | 0.3886 | 0.6095 | 0.4835 | 0.8243 |
| ICU LOS | **LightGBM (Calibrated)** | Stage A (Admission-Time) | **0.8523** | **0.6830** | 0.2498 | 0.1239 | 0.1919 | 0.6037 | 0.4714 | 0.8392 |

## 3. Stage B — Short-Bucket Duration Regression Performance

> [!IMPORTANT]
> **Evaluation Scope Discipline:** Stage B regression metrics (MAE, RMSE, R²) are evaluated **strictly within the restricted short-stay bucket** > (`<= 75th percentile threshold`). Primary deployment metrics reflect performance on the **predicted short bucket** (Stage A classifier output), > while actual-bucket metrics serve as an optimistic upper bound.

| Target | Model Name | Evaluation Protocol (Scope) | Sample Size (N) | MAE (days) | RMSE (days) | R² Score |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| Hospital LOS | **LightGBM Regressor** | Predicted Short Bucket (Deployment Primary) | 53,429 | **1.0548** | **2.0101** | **0.2563** |
| Hospital LOS | **XGBoost Regressor** | Predicted Short Bucket (Deployment Primary) | 53,429 | **1.0607** | **2.0140** | **0.2534** |
| Hospital LOS | **LightGBM Regressor** | Actual Short Bucket (Optimistic Upper Bound) | 62,221 | **0.7963** | **1.0365** | **0.5201** |
| Hospital LOS | **XGBoost Regressor** | Actual Short Bucket (Optimistic Upper Bound) | 62,221 | **0.8017** | **1.0411** | **0.5158** |
| ICU LOS | **LightGBM Regressor** | Predicted Short Bucket (Deployment Primary) | 7,150 | **0.9669** | **1.7026** | **0.0170** |
| ICU LOS | **XGBoost Regressor** | Predicted Short Bucket (Deployment Primary) | 7,150 | **0.9676** | **1.7037** | **0.0157** |
| ICU LOS | **LightGBM Regressor** | Actual Short Bucket (Optimistic Upper Bound) | 9,660 | **0.7298** | **0.8990** | **0.1460** |
| ICU LOS | **XGBoost Regressor** | Actual Short Bucket (Optimistic Upper Bound) | 9,660 | **0.7299** | **0.8989** | **0.1461** |

## 4. Key Observations & Framework Limitations

1. **Right-Tail Isolation:** Classifying long stays in Stage A effectively isolates extreme outliers (>75th percentile), preventing regression skew.
2. **Deployment Realism:** Evaluating Stage B on predicted short-stay patients captures real-world error propagation from Stage A classification.
3. **Explicit Scope Limitation:** Exact duration predictions are provided ONLY for the short-stay bucket; long-stay cases are flagged for clinical review without artificial exact-day estimates.
