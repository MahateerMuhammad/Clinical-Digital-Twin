# Phase 6 — Sequence vs. Tabular Model Comparison

> [!NOTE]
> **Sequence models re-run on Kaggle 2026-08-06** against the corrected feature set.
> The tabular baseline row is read from
> [`mortality_model_comparison.md`](mortality_model_comparison.md), which is regenerated
> from the Phase 1 model on disk.
>
> **The verdict reversed relative to the version the notebook emitted.** See §3.

## 1. Executive Summary & Methodological Alignment

This report evaluates PyTorch sequential models (**LSTM/GRU baseline** and a small **Transformer Encoder**) trained on multi-event 24-hour clinical trajectories (`time_series.parquet`) concatenated with 24-hour static presentation features (`admission_level_selected.parquet`) for in-hospital mortality prediction. Models were evaluated on held-out test subjects.

Both families are scored on the **same held-out test split** (82,806 admissions). Admissions
with no time-series events are retained and zero-padded rather than dropped, so the cohorts
are identical and the comparison is not confounded by which patients each model could score.

---

## 2. Test Set Performance Comparison Table

| Model Family | Model Architecture | Feature Representation | Test AUROC | Test AUPRC | Test Brier Score |
| :--- | :--- | :--- | :---: | :---: | :---: |
| ★ **Tabular Baseline (Phase 1)** | **LightGBM (Run C 24h)** | **24h Summary Aggregates + Static** | **0.9442** | **0.3800** | **0.0768** |
| **Tabular Baseline (Phase 1)** | **Calibrated LightGBM** | **24h Summary Aggregates + Static** | **0.9438** | **0.3608** | **0.0164** |
| **Sequential Deep Learning (Phase 6)** | **PyTorch LSTM / GRU** | **24h Event Trajectory + Static** | **0.9345** | **0.3569** | **0.0168** |
| **Sequential Deep Learning (Phase 6)** | **PyTorch Transformer Encoder** | **24h Event Trajectory + Static** | **0.9337** | **0.3533** | **0.0168** |

Base mortality rate: 2.16%.

---

## 3. Plain-Language Clinical & Methodological Conclusion

1. **Verdict (computed, not asserted): the tabular baseline holds.** LightGBM leads on both
   metrics — AUPRC **0.3800 vs 0.3569** (1.06x) and AUROC **0.9442 vs 0.9345**. Event
   ordering does not, on this cohort, add signal beyond what 24-hour summary statistics
   already capture.

   > [!WARNING]
   > **The notebook's own output concluded the opposite**, reporting "sequence models
   > outperform the tabular baseline: AUPRC 0.3569 vs 0.3281 (1.09x)". That comparison was
   > against a **hardcoded, superseded** tabular row (0.9062 / 0.3281) captured before the
   > feature-selection repair, while the sequence models had just been retrained on the
   > corrected features. New models were being compared against an old baseline.
   >
   > Against the current Phase 1 figure the sign flips: the claimed +0.0288 sequence
   > advantage is actually a **−0.0231 deficit**. The notebook should read this row from
   > `mortality_model_comparison.md` rather than embedding a literal, which is the same
   > defect class that made the Phase 7 verdict wrong.

2. **Like-for-like comparison:** Both families consume the *same* leak-free static
feature set — `MORTALITY_EXCLUDE_RUN_C` is applied identically, so `lab_*_min/_max/_median`,
`med_class_*` and the discharge-note statistics are denied to both. An earlier version of
this notebook withheld those from the tabular baseline only, which inflated the sequence
models by roughly the size of the leak and made the table non-comparable.

3. **Calibration is the one place sequence models win.** Brier score 0.0168 against the
uncalibrated LightGBM's 0.0768. That advantage disappears once isotonic calibration is
applied to the tabular model (0.0164), which is what the serving layer uses — so it
reflects the raw booster being uncalibrated, not a property of the architecture.

4. **Computational cost:** GBDT baselines train in seconds on CPU with no sequence padding
or GPU. The sequence models require a GPU, and this run additionally cost a CUDA
architecture mismatch and a full session restart. A model that is behind on accuracy and
ahead on cost is not a deployment candidate.

5. **Interpretation:** Mortality risk over a 24-hour window is driven by physiological
extremity within the window and static baseline risk. Whether the *ordering* of events adds
signal beyond 24h summary statistics is exactly what the AUPRC column answers — and on this
cohort, at this scale, it does not. That is a legitimate negative result and worth
reporting as one.

---

## 4. What would change this

The sequence models are close (within 0.023 AUPRC), not beaten decisively. Three things
could plausibly reverse it, in descending order of expected value:

- **Vital signs.** The static feature set contains none — `chartevents` is ICU-only, so 83%
  of admissions have no charted vitals. Sequence models are exactly the architecture that
  should benefit from dense physiological trajectories, and they are being denied them.
  Adding MIMIC-IV-ED triage vitals is the highest-value experiment available here.
- **Sequence length.** Truncation is 70 events with padding to the 95th percentile. Sicker
  patients generate more events, so truncation removes signal preferentially from the
  positive class.
- **Class weighting.** The tabular model uses `class_weight='balanced'`; the sequence models
  use unweighted `BCEWithLogitsLoss` at a 2.16% base rate.

Until at least the first of those is addressed, the tabular baseline is the correct
production choice and this is the correct verdict.
