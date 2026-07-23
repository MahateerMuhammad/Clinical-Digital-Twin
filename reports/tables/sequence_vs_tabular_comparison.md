# Phase 6 — Sequence vs. Tabular Model Comparison

## 1. Executive Summary & Methodological Alignment

This report evaluates PyTorch sequential models (**LSTM/GRU baseline** and an optimized **Transformer Encoder**) trained on multi-event 24-hour clinical trajectories (`time_series.parquet`) concatenated with 24-hour static presentation features (`admission_level_selected.parquet`) for in-hospital mortality prediction.

### Methodological Confirmations:
1. **Identical Test Cohort**: Both sequence models were trained and evaluated on the **exact same held-out test subjects** ($N = 81,905$ test admissions across $15\%$ holdout patients) as Phase 1, loaded directly via [`data/processed/patient_split.parquet`](file:///Users/apple/Desktop/Clinical%20Digital%20Twin/data/processed/patient_split.parquet).
2. **Phase 1 Run C Leak-Free Exclusion Standard**: Static tabular features concatenated with sequence vectors were filtered using **Phase 1's Run C leak-free exclusion list** (`MORTALITY_EXCLUDE_RUN_C`), dropping 112 post-hoc ICD codes (`dx_*`, `cci_*`), discharge duration proxies (`los_*`), and full-stay trajectory aggregates to ensure zero target leakage.
3. **Worked Observation Window Verification**: Cell 3 in [`notebooks/07_sequence_model_kaggle.ipynb`](file:///Users/apple/Desktop/Clinical%20Digital%20Twin/notebooks/07_sequence_model_kaggle.ipynb) executed successfully, printing a worked patient observation example ($t \le 24.0\text{h}$) with relative event timestamps and cutoff verification.

---

## 2. Test Set Performance Comparison Table (Identical Test Cohort, $N = 81,905$)

| Model Family | Model Architecture | Feature Representation | Test AUROC | Test AUPRC | Test Brier Score |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Tabular Baseline (Phase 1)** | **LightGBM (Run C 24h)** | **24h Summary Aggregates + Static** | **0.9490** | **0.4706** | **0.0150** |
| **Tabular Baseline (Phase 1)** | **Calibrated LightGBM** | **24h Summary Aggregates + Static** | **0.9484** | **0.4554** | **0.0150** |
| **Sequential Deep Learning (Phase 6)** | **PyTorch Transformer Encoder** | **24h Event Trajectory + Static** | **0.9308** | **0.4183** | **0.0159** |
| **Sequential Deep Learning (Phase 6)** | **PyTorch LSTM / GRU** | **24h Event Trajectory + Static** | **0.9283** | **0.4143** | **0.0160** |

---

## 3. Plain-Language Clinical & Methodological Conclusion

1. **Literature Alignment:** Gradient Boosted Decision Trees (LightGBM/XGBoost) trained on engineered summary statistics (`min`, `max`, `mean`, `slope`, `first`, `last`) slightly outperform end-to-end sequential neural networks (**0.9490 vs. 0.9308 AUROC**), aligning directly with published critical care ML literature where acute derangement bounds drive mortality risk.
2. **Computational & Operational Efficiency:** Tabular GBDT baselines train in seconds on CPU without GPU hardware requirements, sequence padding, or positional embedding tuning.
3. **Key Finding:** Both GBDT summary aggregates and sequence Transformers capture acute deterioration signals effectively without target leakage, validating the robustness of the Phase 1 benchmark.
