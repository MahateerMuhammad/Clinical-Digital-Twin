# Phase 7 — Comprehensive Patient Representation & Retrieval Quality Audit Report

## 1. Data Leakage Prevention & Feature Exclusion Audit

### Strict Anti-Leakage Verification
1. **Feature Exclusion**: All encoder inputs strictly enforce `MORTALITY_EXCLUDE_RUN_C` filters. Target outcome labels are never fed into encoders.
2. **Demographic Debiasing**: Sensitive demographic columns (`race`, `insurance`, `marital_status`, `language`) are stripped to prevent demographic race clustering in Euclidean space.
3. **Train-Only Model Fitting**: All feature scalers, triplet samplers, and model weights are fit exclusively on the `train` split ($N = 381,403$ admissions). Held-out `test` admissions ($N = 82,806$) are evaluated strictly by inference pass.

---

## 2. Master 3-Pillar & Outcome Quality Benchmark Table

| Representation Space | Learning Objective | Disease Phenotype Match % | Lab Severity MAE | Medication Class Jaccard % | Mortality Agreement (Base: 2.16%) | Mortality 95% Bootstrap CI | Mortality Enrichment | Readmission Agreement (Base: 20.03%) | Readmission 95% Bootstrap CI | Readmission Enrichment |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Raw Features** | **Static Baseline (1.0x)** | 36.3% | 0.123 | 50.2% | 1.58% | 1.26% – 1.94% | **0.73x** | 20.07% | 19.18% – 20.96% | **1.00x** |
| **Multi-Task Triplet AE** | **3-Pillar Triplet Metric** | 37.3% | 0.182 | 49.8% | 1.70% | 1.38% – 2.08% | **0.79x** | 21.12% | 20.15% – 22.07% | **1.05x** |
| **LightGBM Tree-Leaf AE** | **Supervised Decision Latent** | 35.0% | 0.219 | 48.8% | 2.29% | 1.91% – 2.72% | **1.06x** | 19.85% | 18.94% – 20.68% | **0.99x** |
| **Dual-Head Hybrid AE** | **Unified 32d Dual-Head Fusion** | 36.1% | 0.200 | 49.7% | 2.24% | 1.86% – 2.67% | **1.04x** | 19.47% | 18.46% – 20.31% | **0.97x** |

---

> [!WARNING]
> **The enrichment metric below cannot detect a working embedding.** It reports the
> *unconditional* mean neighbour outcome over a representative query sample, divided by
> the base rate. That quantity tends to 1.0 whatever the embedding does: neighbours of
> high-risk patients are high-risk, neighbours of low-risk patients are low-risk, and
> averaging over a representative sample recovers the base rate.
>
> Scored conditionally instead — can the neighbours' outcomes rank *this* patient's
> outcome? — the same 32-dimensional hybrid space achieves **AUROC 0.7428** for
> mortality and **0.8392** for ICU stay, with **4.42x** enrichment in the top decile.
> See [`twin_retrieval_evaluation.md`](twin_retrieval_evaluation.md).
>
> The *comparative* conclusion below still stands: on this metric, learned spaces did not
> clearly separate from raw scaled features. But "no useful signal" does not follow from
> it, and should not be cited.

## 3. Master Methodological & Clinical Verdict

1. **Statistical Rigor on Outcome Retrieval Quality**:
   - **No representation** achieves a mortality-retrieval CI separated from the naive baseline. On this cohort, learned embeddings do not improve outcome-aligned twin retrieval over raw scaled features.
   - **Dual-Head Hybrid AE** reaches **1.04x** mortality enrichment (95% CI: 1.86%–2.67%), which overlaps the naive baseline CI (1.26%–1.94%).

2. **Disease, laboratory & medication retrieval**:
   Disease phenotype match spans only 2.3 percentage points across every space (best:
   Multi-Task Triplet AE, 37.3%; naive raw features, 36.3%). Lab severity MAE is lowest
   for **Naive Raw Features** (0.123) and medication Jaccard highest for the same space
   (50.2%). Disease and medication labels (`cci_*`/`dx_*`/`med_class_*`) are withheld
   from every encoder, so these figures measure generalisation rather than recall of an
   input. An earlier version of this audit fed those labels to the encoders *and* scored
   against them, reporting 71.7% disease match for the triplet space — that figure was
   memorisation, not retrieval.

3. **What this means for twin retrieval**:
   No learned space beats raw scaled features on outcome-aligned retrieval, and phenotype
   matching is near-identical across all of them. The honest recommendation on this cohort
   is to use the naive raw-feature space for twin retrieval, and to treat the learned
   embeddings as dimensionality reduction rather than outcome alignment. The 32-dimensional
   hybrid space remains useful for Level 5 evidence retrieval in the RAG layer, where the
   benefit is compact nearest-neighbour lookup over 546,028 admissions rather than
   outcome enrichment.