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

> [!CAUTION]
> **The mortality- and readmission-enrichment columns above are superseded, and the
> verdict originally published beneath them was wrong.**
>
> Those columns report *unconditional* enrichment: the mean neighbour outcome over a
> representative query sample, divided by the base rate. That quantity tends to 1.0
> whatever the embedding does — neighbours of high-risk patients are high-risk,
> neighbours of low-risk patients are low-risk, and averaging over a representative
> sample recovers the base rate. It cannot separate a good embedding from a random one,
> so it could not have supported a comparative claim in *either* direction.
>
> Re-scored conditionally — can the neighbours' outcomes rank *this* patient's outcome?
> — all four spaces separate, and the served 32-dimensional hybrid space beats the naive
> baseline by **+0.0734 AUROC (95% CI +0.0496 to +0.0978)** on a paired bootstrap over
> 20,000 queries. See
> [`representation_comparison.md`](representation_comparison.md).
>
> Read §3 below as corrected, not as originally published. The disease, laboratory and
> medication columns are unaffected: those are direct match rates rather than enrichment
> ratios.

## 3. Master Methodological & Clinical Verdict

1. **Statistical Rigor on Outcome Retrieval Quality** — *corrected*:
   - The original finding here — "no representation achieves a mortality-retrieval CI
     separated from the naive baseline" — was an artefact of the unconditional
     enrichment metric, which is constant in embedding quality. It has been withdrawn.
   - On the conditional metric, **Dual-Head Hybrid AE** ranks mortality at
     **AUROC 0.8321** against **0.7587** for naive raw features, a paired difference of
     **+0.0734 (95% CI +0.0496 to +0.0978)** that excludes zero. **LightGBM Tree-Leaf
     AE** also separates (**+0.0512**, CI +0.0264 to +0.0786). **Multi-Task Triplet AE**
     does not (**−0.0134**, CI −0.0388 to +0.0104).
   - The unconditional figures in the table above are retained for provenance: they
     reproduce, and they are what the withdrawn verdict was based on.

2. **Disease, laboratory & medication retrieval**:
   Disease phenotype match spans only 2.3 percentage points across every space (best:
   Multi-Task Triplet AE, 37.3%; naive raw features, 36.3%). Lab severity MAE is lowest
   for **Naive Raw Features** (0.123) and medication Jaccard highest for the same space
   (50.2%). Disease and medication labels (`cci_*`/`dx_*`/`med_class_*`) are withheld
   from every encoder, so these figures measure generalisation rather than recall of an
   input. An earlier version of this audit fed those labels to the encoders *and* scored
   against them, reporting 71.7% disease match for the triplet space — that figure was
   memorisation, not retrieval.

3. **What this means for twin retrieval** — *corrected*:
   The earlier recommendation was to use the naive raw-feature space and to treat the
   learned embeddings as dimensionality reduction rather than outcome alignment. That
   followed from the withdrawn finding in §1 and does not survive it.

   On the conditional metric the served 32-dimensional hybrid space is the strongest of
   the four for outcome-aligned retrieval, and its advantage over raw scaled features is
   larger than the interval around it. It also does this in 32 dimensions against the
   baseline's 100, so the compactness that motivated it costs nothing in retrieval
   quality.

   Phenotype matching remains near-identical across all four spaces (2.3 percentage
   points end to end); that part of the original reading stands. Retrieval-based
   prediction also remains weaker than the trained tabular model (Phase 1 Run C: AUROC
   0.9442) — twin retrieval earns its place by supplying interpretable precedent, not by
   improving the risk estimate.