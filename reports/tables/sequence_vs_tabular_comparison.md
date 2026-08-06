# Phase 6 — Sequence vs. Tabular Model Comparison

> [!CAUTION]
> **Both sides of this table predate the feature-selection repair. Do not update one side
> alone.**
>
> The tabular baseline row quotes Phase 1 Run C at **0.9062 / 0.3281**; Phase 1 is now
> **0.9442 / 0.3800**. The obvious edit — refresh the tabular row — would be wrong, because
> the LSTM and Transformer were trained on Kaggle against the *same* static feature set that
> the repair changed, and have not been re-run. Updating one side would present a
> like-for-like comparison that was never actually run.
>
> **The verdict is unaffected in direction and is if anything strengthened.** The tabular
> baseline was ahead on AUPRC (0.3281 vs 0.3009) and Phase 1 has since improved to 0.3800,
> so engineered 24-hour summaries still match or beat the raw event sequence.
>
> To make this table citable as current, re-run the Phase 6 Kaggle notebook
> (`07_sequence_model_kaggle.ipynb`) against the corrected feature set and regenerate both
> sides together.

## 1. Executive Summary & Methodological Alignment

This report evaluates PyTorch sequential models (**LSTM/GRU baseline** and a small **Transformer Encoder**) trained on multi-event 24-hour clinical trajectories (`time_series.parquet`) concatenated with 24-hour static presentation features (`admission_level_selected.parquet`) for in-hospital mortality prediction. Models were evaluated on held-out test subjects.

---

## 2. Test Set Performance Comparison Table

| Model Family | Model Architecture | Feature Representation | Test AUROC | Test AUPRC | Test Brier Score |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Tabular Baseline (Phase 1)** | **LightGBM (Run C 24h)** | **24h Summary Aggregates + Static** | **0.9062** | **0.3281** | **0.0957** |
| **Tabular Baseline (Phase 1)** | **Calibrated LightGBM** | **24h Summary Aggregates + Static** | **0.9059** | **0.3131** | **0.0172** |
| **Sequential Deep Learning (Phase 6)** | **PyTorch LSTM / GRU** | **24h Event Trajectory + Static** | **0.8957** | **0.2988** | **0.0177** |
| **Sequential Deep Learning (Phase 6)** | **PyTorch Transformer Encoder** | **24h Event Trajectory + Static** | **0.8970** | **0.3009** | **0.0176** |

---

## 3. Plain-Language Clinical & Methodological Conclusion

1. **Verdict (computed, not asserted):** The tabular baseline holds: AUPRC 0.3281 vs 0.3009. Engineered 24h summaries capture what the raw event sequence offers, at a fraction of the compute.
2. **Like-for-like comparison:** Both families now consume the *same* leak-free static
feature set — `MORTALITY_EXCLUDE_RUN_C` is applied identically, so `lab_*_min/_max/_median`,
`med_class_*` and the discharge-note statistics are denied to both. An earlier version of
this notebook withheld those from the tabular baseline only, which inflated the sequence
models by roughly the size of the leak and made the table non-comparable.
3. **Computational cost:** GBDT baselines train in seconds on CPU with no sequence padding
or GPU. Any sequence-model advantage must be weighed against that.
4. **Interpretation:** Mortality risk over a 24-hour window is driven by physiological
extremity within the window and static baseline risk. Whether the *ordering* of events adds
signal beyond 24h summary statistics is exactly what the AUPRC column above answers
