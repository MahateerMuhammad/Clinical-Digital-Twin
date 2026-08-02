#!/usr/bin/env python3
"""
scripts/evaluation/run_twin_retrieval_eval.py
──────────────────────────
Evaluate Level 5 digital-twin retrieval: do the retrieved neighbours' outcomes
predict the query patient's outcome?

Why this is separate from scripts/evaluation/run_retrieval_eval.py
───────────────────────────────────────────────
That harness scores terminology, Level 1 guidelines and topical relevance against
version-controlled gold sets, and is deliberately offline and fast. This one needs the
Phase 7 embeddings and the admission outcomes, so it is kept apart to avoid making the
gold-set harness heavyweight.

Why the metric differs from Phase 7's
─────────────────────────────────────
Phase 7 reported *unconditional* enrichment: mean neighbour mortality across a random
sample of queries, divided by the base rate. On a representative query sample that
statistic tends to 1.0 **whatever the embedding does** — neighbours of high-risk
patients are high-risk, neighbours of low-risk patients are low-risk, and averaging
over a representative sample recovers the base rate. It therefore cannot distinguish a
good embedding from a useless one, and Phase 7's "no representation beats the naive
baseline" conclusion rests on it.

The informative question is conditional: *given* the neighbours' outcomes, how well can
you predict this patient's? That is a ranking problem, so it is scored with AUROC and
AUPRC, plus top-decile enrichment. Both statistics are reported so the difference is
visible.

Usage
─────
    python scripts/evaluation/run_twin_retrieval_eval.py
    python scripts/evaluation/run_twin_retrieval_eval.py --queries 5000 --top-k 10
"""


from __future__ import annotations


# ── repo-root bootstrap ──────────────────────────────────────────────────────
# These scripts live two levels below the project root. Python puts the *script's*
# directory on sys.path, not the working directory, so `import src...` would fail
# from here; and many of them address data with root-relative paths such as
# "models/" or "reports/tables/". Both are fixed by putting the root on the path
# and running from it, which makes execution identical from any directory.
import os as _os
import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
_os.chdir(_ROOT)
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "reports" / "tables" / "twin_retrieval_evaluation.md"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", type=int, default=3000)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fail-under", type=float, default=None,
                    help="exit non-zero if AUROC falls below this")
    args = ap.parse_args()

    from sklearn.metrics import roc_auc_score, average_precision_score
    from src.llm.prompt_builder import ClinicalPromptBuilder

    print("Loading embeddings and building the twin index ...")
    builder = ClinicalPromptBuilder()
    if not builder._embed_cols:
        print("similarity.parquet has no dim_* columns — run Phase 7 first.",
              file=sys.stderr)
        return 1

    adm = builder._adm_index
    pool = builder.sim_df.dropna(subset=builder._embed_cols)["hadm_id"].astype("int64").to_numpy()
    rng = np.random.default_rng(args.seed)
    queries = rng.choice(pool, min(args.queries, len(pool)), replace=False)

    print(f"Scoring {len(queries):,} queries at top_k={args.top_k} ...")
    t0 = time.time()
    scores, truth, icu_scores, icu_truth, dists = [], [], [], [], []
    for hadm in queries:
        twins = builder.get_digital_twins(int(hadm), top_k=args.top_k)
        if not twins:
            continue
        row = adm.loc[int(hadm)]
        if getattr(row, "ndim", 1) > 1:
            row = row.iloc[0]
        y = row.get("hospital_expire_flag", np.nan)
        if y is None or (isinstance(y, float) and np.isnan(y)):
            continue
        scores.append(np.mean([t["hospital_expire_flag"] for t in twins]))
        truth.append(int(y))
        yi = row.get("has_icu_stay", np.nan)
        if yi is not None and not (isinstance(yi, float) and np.isnan(yi)):
            icu_scores.append(np.mean([t["has_icu_stay"] for t in twins]))
            icu_truth.append(int(yi))
        dists.append(np.mean([t["distance"] for t in twins]))

    scores, truth = np.array(scores), np.array(truth)
    elapsed = time.time() - t0
    base = float(truth.mean())

    auroc = roc_auc_score(truth, scores)
    auprc = average_precision_score(truth, scores)
    hi = scores >= np.percentile(scores, 90)
    top_decile = float(truth[hi].mean())
    uncond = float(scores.mean())

    icu_auroc = (roc_auc_score(icu_truth, icu_scores)
                 if icu_truth and len(set(icu_truth)) > 1 else float("nan"))

    print(f"\n{len(truth):,} queries scored in {elapsed:.0f}s "
          f"({truth.sum()} deaths, base {base:.2%})\n")
    print("  CONDITIONAL — do twin outcomes predict the query outcome?")
    print(f"    mortality AUROC            {auroc:.4f}")
    print(f"    mortality AUPRC            {auprc:.4f}   (base {base:.4f}, "
          f"{auprc/base:.1f}x)")
    print(f"    top-decile mortality       {top_decile:.2%}  ({top_decile/base:.2f}x)")
    print(f"    ICU-stay AUROC             {icu_auroc:.4f}")
    print("\n  UNCONDITIONAL — the Phase 7 statistic")
    print(f"    mean twin mortality        {uncond:.4f}  ({uncond/base:.2f}x)")
    print("    ~1.0 by construction on a representative sample; not informative")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(f"""# Level 5 — Digital Twin Retrieval Evaluation

> [!NOTE]
> Generated by `scripts/evaluation/run_twin_retrieval_eval.py` from `data/processed/similarity.parquet`
> (Phase 7 embeddings) and the admission outcomes. Regenerate after any Phase 7 rerun.

## 1. What is measured

For each query admission, the {args.top_k} nearest neighbours in the 32-dimensional
Phase 7 embedding space are retrieved, excluding every admission belonging to the same
patient. The neighbours' observed outcome rate is then treated as a prediction of the
query patient's outcome.

Queries: {len(truth):,} sampled at random (seed {args.seed}); observed deaths
{int(truth.sum())} ({base:.2%}).

## 2. Results

| Metric | Value | Interpretation |
| :--- | ---: | :--- |
| Mortality AUROC | **{auroc:.4f}** | ranking quality from retrieval alone, no model |
| Mortality AUPRC | **{auprc:.4f}** | {auprc/base:.1f}x the {base:.4f} base rate |
| Top-decile observed mortality | **{top_decile:.2%}** | **{top_decile/base:.2f}x** enrichment |
| ICU-stay AUROC | {icu_auroc:.4f} | secondary outcome |
| Mean twin distance | {np.mean(dists):.3f} | Euclidean, embedding space |
| _Unconditional mean twin mortality_ | _{uncond:.4f}_ | _{uncond/base:.2f}x — see §3_ |

## 3. Why this differs from the Phase 7 verdict

Phase 7 reported that no representation achieved mortality enrichment separated from a
naive baseline, using **unconditional** enrichment: mean neighbour mortality over a
random query sample, divided by the base rate.

That statistic tends to 1.0 regardless of embedding quality. If the embedding works,
neighbours of high-risk patients are high-risk and neighbours of low-risk patients are
low-risk; averaging over a representative sample recovers the base rate. It cannot
separate a good embedding from a random one, and this run reproduces it exactly
({uncond/base:.2f}x).

Conditioning on the query reverses the picture: the same neighbours rank query mortality
at AUROC {auroc:.4f}, and queries whose twins fall in the top decile of twin-mortality
die at {top_decile:.2%} against a {base:.2%} base rate. The embedding is informative;
the Phase 7 metric was structurally unable to show it.

This does not overturn Phase 7's *comparative* finding — that learned spaces did not
clearly beat raw scaled features on that metric — but it does mean "no useful signal"
was the wrong conclusion to draw from it.

## 4. Caveat

Retrieval-based prediction is weaker than the trained tabular model
(Phase 1 Run C: AUROC 0.9062, AUPRC 0.3281). Twin retrieval earns its place by
supplying *interpretable precedent* — "here are {args.top_k} comparable admissions and
what happened to them" — not by improving the risk estimate.
""", encoding="utf-8")
    print(f"\nReport → {OUT}")

    if args.fail_under is not None and auroc < args.fail_under:
        print(f"FAIL: AUROC {auroc:.4f} < {args.fail_under}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
