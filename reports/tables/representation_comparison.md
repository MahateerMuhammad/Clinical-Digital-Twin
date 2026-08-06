# Phase 7 — Representation Comparison on a Metric That Can Separate Them

_Generated 2026-08-04 15:38 UTC by `scripts/evaluation/run_representation_comparison.py`._

Reference pool 250,000 admissions, 20,000 scored queries, top-10 neighbours, same-patient admissions excluded, seed 7.

## 1. Why this exists

`embedding_retrieval_quality.md` concluded that no learned space beats raw scaled
features, using **unconditional enrichment** — mean neighbour mortality over a
random query sample, divided by the base rate. That statistic tends to 1.0
whatever the embedding does, so it cannot support a comparative claim in either
direction, including the negative one that was published.

The conditional question — *given the neighbours' outcomes, how well can you rank
this patient's?* — is a ranking problem, so it is scored with AUROC, AUPRC and
top-decile enrichment. Both metrics are computed here on identical queries.

## 2. Results

| Representation Space | Dims | Mortality AUROC | 95% CI | Δ vs naive (95% CI) | Mortality AUPRC | Top-decile enrichment | ICU AUROC | _Unconditional enrichment_ |
| :--- | ---: | ---: | :---: | :---: | ---: | ---: | ---: | ---: |
| Naive Raw Features | 100 | 0.7587 | [0.7355, 0.7836] | baseline | 0.1808 | 5.89x | 0.8542 | _0.73x_ |
| Multi-Task Triplet AE | 16 | 0.7453 | [0.7202, 0.7711] | -0.0134 [-0.0388, +0.0104] | 0.1454 | 5.23x | 0.8303 | _0.79x_ |
| LightGBM Tree-Leaf AE | 16 | 0.8099 | [0.7892, 0.8337] | +0.0512 [+0.0264, +0.0786] | 0.1904 | 6.31x | 0.8545 | _1.06x_ |
| **Dual-Head Hybrid AE** | 32 | **0.8321** | [0.8116, 0.8542] | +0.0734 [+0.0496, +0.0978] | 0.2328 | 6.69x | 0.8610 | _1.03x_ |

Base mortality rate over the 20,000 scored queries: 2.13%. All spaces are scored on the same queries, and the
Δ column is a paired bootstrap over 1,000 resamples — the spaces are
resampled together, so the interval is on the difference itself rather than on
two independent AUROCs whose overlap says little.

## 3. What changed

The final column reproduces Phase 7's metric and its verdict: every space sits
near 1.0x and none separates from any other. That is the metric's behaviour, not
a finding about the embeddings.

On the conditional metric the spaces do separate. **Dual-Head Hybrid AE** ranks
mortality at AUROC 0.8321 against 0.7587 for the naive
baseline — a difference of +0.0734, 95% CI [+0.0496, +0.0978].

Spaces whose advantage over the naive baseline excludes zero: LightGBM Tree-Leaf AE, Dual-Head Hybrid AE.

## 4. Consequence for the Phase 7 verdict

The recommendation to prefer the naive raw-feature space for twin retrieval was
drawn from a statistic that cannot rank representations, and is superseded by
this table.

The serving layer uses the 32-dimensional hybrid space. On this metric it beats the naive baseline by +0.0734 AUROC, 95% CI [+0.0496, +0.0978] — an advantage that excludes zero, so the choice is positively supported rather than merely unrefuted.

The disease/laboratory/medication columns of the Phase 7 audit are unaffected —
those are direct match rates, not enrichment ratios, and are read as published.
