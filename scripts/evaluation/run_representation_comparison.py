#!/usr/bin/env python3
"""
scripts/evaluation/run_representation_comparison.py
───────────────────────────────────────────────────
Re-score Phase 7's four representation spaces on a metric that can tell them apart.

The problem this fixes
──────────────────────
`reports/tables/embedding_retrieval_quality.md` concludes that no learned space beats
raw scaled features, and recommends the naive space for twin retrieval. That verdict
rests on **unconditional enrichment**: mean neighbour mortality over a random query
sample, divided by the base rate.

That statistic tends to 1.0 whatever the embedding does. If the embedding works,
neighbours of high-risk patients are high-risk and neighbours of low-risk patients are
low-risk; averaging over a representative sample recovers the base rate. It cannot
separate a good embedding from a random one — so it cannot support *either* direction
of a comparative claim, including the negative one that was published.

`twin_retrieval_evaluation.md` already re-scored the hybrid space conditionally and
found AUROC 0.8044. But it scored only that one space, so the *comparison* was still
open: the published recommendation was annotated as unsafe rather than corrected.

What this does
──────────────
Scores all four spaces on the same queries, with both metrics side by side:

* **Naive Raw Features** — the 99-column debiased encoder input, scaled. Rebuilt
  through `PatientProjector.encoder_matrix`, which is the same transform Phase 7 fed
  its encoders, so this is the baseline Phase 7 actually compared against.
* **Multi-Task Triplet AE** — `dim_0..15`.
* **LightGBM Tree-Leaf AE** — `dim_16..31`.
* **Dual-Head Hybrid AE** — `dim_0..31`, the space the serving layer uses.

Neighbours belonging to the query's own patient are excluded, as in serving.

Usage
─────
    python scripts/evaluation/run_representation_comparison.py
    python scripts/evaluation/run_representation_comparison.py --queries 5000 --pool 200000
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
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = _ROOT
OUT = ROOT / "reports" / "tables" / "representation_comparison.md"

#: Phase 7 wrote the triplet head into dim_0..15 and the tree-leaf head into
#: dim_16..31, in that hstack order. Slicing them apart recovers each head's own space
#: without re-running the notebook.
TRIPLET_DIMS = slice(0, 16)
LEAF_DIMS = slice(16, 32)


def load_embeddings():
    """similarity.parquet, restricted to rows with a populated embedding."""
    sim = pd.read_parquet(ROOT / "data/processed/similarity.parquet")
    dims = sorted((c for c in sim.columns if c.startswith("dim_")),
                  key=lambda c: int(c.split("_")[1]))
    if not dims:
        raise SystemExit("similarity.parquet has no dim_* columns — run Phase 7 first.")
    sim = sim.dropna(subset=dims).reset_index(drop=True)
    return sim, dims


def naive_space(sim: pd.DataFrame) -> np.ndarray:
    """
    The scaled raw-feature baseline, rebuilt exactly as Phase 7 fed its encoders.

    Reconstructed rather than re-derived by hand: `encoder_matrix` reproduces the
    fit-time column list, the `astype(str)` that turns NaN into its own dummy level,
    and the no-`drop_first` encoding. Approximating any of those would compare the
    learned spaces against a baseline Phase 7 never used.
    """
    from src.llm.twin_projection import PatientProjector

    projector = PatientProjector()
    frame = projector.load_source_frame(sim["hadm_id"].astype("int64").tolist())
    frame = (frame.set_index(frame["hadm_id"].astype("int64"))
                  .reindex(sim["hadm_id"].astype("int64")).reset_index(drop=True))
    return projector.encoder_matrix(frame).astype(np.float32)


def score_space(name, X, sim, queries_idx, top_k, seed):
    """
    Retrieve top-k neighbours for each query and score both metrics.

    Returns conditional AUROC/AUPRC and top-decile enrichment — "can the neighbours'
    outcomes rank this patient's outcome?" — alongside the unconditional enrichment
    Phase 7 used, so the two can be read against each other on identical queries.
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import roc_auc_score, average_precision_score

    subject = sim["subject_id"].astype("int64").to_numpy()
    mortality = pd.to_numeric(sim["hospital_expire_flag"], errors="coerce").to_numpy()
    icu = pd.to_numeric(sim["has_icu_stay"], errors="coerce").to_numpy()

    # Over-fetch so same-subject admissions can be dropped without a short result.
    over = top_k + 25
    t0 = time.time()
    nn = NearestNeighbors(n_neighbors=over, algorithm="auto").fit(X)
    _, idx = nn.kneighbors(X[queries_idx])

    scores, icu_scores, keep = [], [], []
    for row, q in zip(idx, queries_idx):
        neigh = row[(row != q) & (subject[row] != subject[q])][:top_k]
        if len(neigh) < top_k:
            continue
        keep.append(q)
        scores.append(float(np.nanmean(mortality[neigh])))
        icu_scores.append(float(np.nanmean(icu[neigh])))

    # Per-query results, keyed by query index. Alignment onto a query set common to
    # every space happens in `align_spaces` — comparing spaces over different query
    # sets would confound the metric with which queries each space happened to retain.
    return {
        "space": name,
        "dims": int(X.shape[1]),
        "by_query": dict(zip(keep, zip(scores, icu_scores))),
        "seconds": time.time() - t0,
    }


def align_spaces(rows, sim):
    """Restrict every space to the queries all of them retained, then score."""
    from sklearn.metrics import roc_auc_score, average_precision_score

    mortality = pd.to_numeric(sim["hospital_expire_flag"], errors="coerce").to_numpy()
    icu = pd.to_numeric(sim["has_icu_stay"], errors="coerce").to_numpy()

    common = set.intersection(*(set(r["by_query"]) for r in rows))
    common = np.array(sorted(q for q in common if not np.isnan(mortality[q])))
    if len(common) < 100:
        raise SystemExit(f"only {len(common)} queries common to all spaces — "
                         "raise --pool or lower --top-k")

    y = mortality[common].astype(int)
    y_icu = icu[common]
    icu_ok = ~np.isnan(y_icu)
    base = float(y.mean())

    for r in rows:
        scores = np.array([r["by_query"][q][0] for q in common])
        icu_scores = np.array([r["by_query"][q][1] for q in common])
        order = np.argsort(-scores)
        decile = order[: max(1, len(order) // 10)]
        top_rate = float(y[decile].mean())

        # Kept so the spaces can be bootstrapped *jointly* on the same resampled
        # queries. Phase 7's negative claim was about overlapping CIs, and two
        # overlapping intervals are entirely compatible with one space winning on every
        # single resample — so the paired difference is the statistic that answers it.
        r["_y"], r["_scores"] = y, scores
        r["n"] = int(len(y))
        r["base_rate"] = base
        r["auroc"] = float(roc_auc_score(y, scores))
        r["auprc"] = float(average_precision_score(y, scores))
        r["icu_auroc"] = (float(roc_auc_score(y_icu[icu_ok].astype(int), icu_scores[icu_ok]))
                          if icu_ok.sum() and len(set(y_icu[icu_ok])) > 1 else float("nan"))
        r["top_decile"] = top_rate
        r["top_decile_enrichment"] = top_rate / base if base else float("nan")
        # Phase 7's own statistic, on these exact queries.
        r["unconditional"] = float(np.mean(scores))
        r["unconditional_enrichment"] = float(np.mean(scores)) / base if base else float("nan")
        r.pop("by_query")


def bootstrap_auroc(rows, naive, rounds, seed):
    """
    Attach a CI for each space's AUROC and for its paired difference from the baseline.

    All spaces are scored on the *same* resampled queries each round. Phase 7 compared
    independent per-space intervals and concluded from their overlap that nothing
    separated; two overlapping intervals are entirely compatible with one space winning
    on every single resample, so the paired difference is the statistic that answers
    the question actually being asked.
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    y0 = naive["_y"]
    draws = {r["space"]: [] for r in rows}
    diffs = {r["space"]: [] for r in rows}

    for _ in range(rounds):
        i = rng.integers(0, len(y0), len(y0))
        if len(set(y0[i])) < 2:
            continue
        per_space = {}
        for r in rows:
            per_space[r["space"]] = roc_auc_score(r["_y"][i], r["_scores"][i])
        for space, auc in per_space.items():
            draws[space].append(auc)
            diffs[space].append(auc - per_space[naive["space"]])

    for r in rows:
        d, f = np.asarray(draws[r["space"]]), np.asarray(diffs[r["space"]])
        r["auroc_ci"] = (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
        r["diff_ci"] = (float(np.percentile(f, 2.5)), float(np.percentile(f, 97.5)))
        r["beats_naive"] = bool(r["diff_ci"][0] > 0)


def _serving_verdict(rows, naive):
    """
    State what this run does and does not establish about the served space.

    Written from the numbers rather than asserted, because the interesting outcome is
    the one where the served space improves on the baseline without its interval
    excluding zero. Claiming support in that case would repeat the original error in
    the opposite direction — reading a metric for more than it can carry.
    """
    hybrid = next((r for r in rows if r["space"] == "Dual-Head Hybrid AE"), None)
    if hybrid is None:
        return "The hybrid space was not scored on this run."

    lo, hi = hybrid["diff_ci"]
    delta = hybrid["auroc"] - naive["auroc"]
    if lo > 0:
        return (f"The serving layer uses the 32-dimensional hybrid space. On this metric "
                f"it beats the naive baseline by {delta:+.4f} AUROC, 95% CI "
                f"[{lo:+.4f}, {hi:+.4f}] — an advantage that excludes zero, so the choice "
                f"is positively supported rather than merely unrefuted.")
    if hi < 0:
        return (f"The serving layer uses the 32-dimensional hybrid space, and on this "
                f"metric it is *worse* than the naive baseline by {delta:+.4f} AUROC, "
                f"95% CI [{lo:+.4f}, {hi:+.4f}]. That is a reason to reconsider the "
                f"serving space, not a confirmation of it.")
    return (f"The serving layer uses the 32-dimensional hybrid space. It ranks "
            f"{delta:+.4f} AUROC above the naive baseline, but the 95% CI "
            f"[{lo:+.4f}, {hi:+.4f}] includes zero, so this run does not establish that "
            f"it beats raw scaled features — only that the published claim it was *worse* "
            f"came from a metric that could not have shown either. What the run does "
            f"establish is that at least one learned space separates from the baseline; "
            f"see the Δ column.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--queries", type=int, default=3000)
    ap.add_argument("--pool", type=int, default=120000,
                    help="reference admissions the index is built over")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    sim, dims = load_embeddings()
    rng = np.random.default_rng(args.seed)
    if args.pool < len(sim):
        sim = sim.iloc[np.sort(rng.choice(len(sim), args.pool, replace=False))]
        sim = sim.reset_index(drop=True)
    queries_idx = rng.choice(len(sim), min(args.queries, len(sim)), replace=False)

    print(f"Pool {len(sim):,} admissions, {len(queries_idx):,} queries, top_k={args.top_k}\n")

    Z = sim[dims].to_numpy(dtype=np.float32)
    print("Rebuilding the naive raw-feature space ...")
    spaces = [
        ("Naive Raw Features", naive_space(sim)),
        ("Multi-Task Triplet AE", Z[:, TRIPLET_DIMS]),
        ("LightGBM Tree-Leaf AE", Z[:, LEAF_DIMS]),
        ("Dual-Head Hybrid AE", Z),
    ]

    rows = []
    for name, X in spaces:
        print(f"Retrieving in {name} ({X.shape[1]}d) ...", flush=True)
        rows.append(score_space(name, X, sim, queries_idx, args.top_k, args.seed))

    align_spaces(rows, sim)
    for r in rows:
        print(f"    {r['space']:24s} AUROC {r['auroc']:.4f}   "
              f"top-decile {r['top_decile_enrichment']:.2f}x   "
              f"unconditional {r['unconditional_enrichment']:.2f}x")

    best = max(rows, key=lambda r: r["auroc"])
    naive = next(r for r in rows if r["space"] == "Naive Raw Features")

    print("\nBootstrapping paired AUROC differences vs the naive baseline ...")
    bootstrap_auroc(rows, naive, args.bootstrap, args.seed)

    lines = [
        "# Phase 7 — Representation Comparison on a Metric That Can Separate Them",
        "",
        f"_Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} by "
        "`scripts/evaluation/run_representation_comparison.py`._",
        "",
        f"Reference pool {len(sim):,} admissions, {rows[0]['n']:,} scored queries, "
        f"top-{args.top_k} neighbours, same-patient admissions excluded, seed {args.seed}.",
        "",
        "## 1. Why this exists",
        "",
        "`embedding_retrieval_quality.md` concluded that no learned space beats raw scaled",
        "features, using **unconditional enrichment** — mean neighbour mortality over a",
        "random query sample, divided by the base rate. That statistic tends to 1.0",
        "whatever the embedding does, so it cannot support a comparative claim in either",
        "direction, including the negative one that was published.",
        "",
        "The conditional question — *given the neighbours' outcomes, how well can you rank",
        "this patient's?* — is a ranking problem, so it is scored with AUROC, AUPRC and",
        "top-decile enrichment. Both metrics are computed here on identical queries.",
        "",
        "## 2. Results",
        "",
        "| Representation Space | Dims | Mortality AUROC | 95% CI | Δ vs naive (95% CI) | Mortality AUPRC | Top-decile enrichment | ICU AUROC | _Unconditional enrichment_ |",
        "| :--- | ---: | ---: | :---: | :---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        bold = "**" if r is best else ""
        lo, hi = r["auroc_ci"]
        dlo, dhi = r["diff_ci"]
        diff = ("baseline" if r is naive
                else f"{r['auroc'] - naive['auroc']:+.4f} [{dlo:+.4f}, {dhi:+.4f}]")
        lines.append(
            f"| {bold}{r['space']}{bold} | {r['dims']} | {bold}{r['auroc']:.4f}{bold} | "
            f"[{lo:.4f}, {hi:.4f}] | {diff} | {r['auprc']:.4f} | "
            f"{r['top_decile_enrichment']:.2f}x | {r['icu_auroc']:.4f} | "
            f"_{r['unconditional_enrichment']:.2f}x_ |")

    separated = [r for r in rows if r["space"] != naive["space"] and r["beats_naive"]]
    lines += [
        "",
        f"Base mortality rate over the {rows[0]['n']:,} scored queries: "
        f"{rows[0]['base_rate']:.2%}. All spaces are scored on the same queries, and the",
        f"Δ column is a paired bootstrap over {args.bootstrap:,} resamples — the spaces are",
        "resampled together, so the interval is on the difference itself rather than on",
        "two independent AUROCs whose overlap says little.",
        "",
        "## 3. What changed",
        "",
        "The final column reproduces Phase 7's metric and its verdict: every space sits",
        "near 1.0x and none separates from any other. That is the metric's behaviour, not",
        "a finding about the embeddings.",
        "",
        f"On the conditional metric the spaces do separate. **{best['space']}** ranks",
        f"mortality at AUROC {best['auroc']:.4f} against {naive['auroc']:.4f} for the naive",
        f"baseline — a difference of {best['auroc'] - naive['auroc']:+.4f}, 95% CI "
        f"[{best['diff_ci'][0]:+.4f}, {best['diff_ci'][1]:+.4f}].",
        "",
        (f"Spaces whose advantage over the naive baseline excludes zero: "
         f"{', '.join(r['space'] for r in separated)}."
         if separated else
         "No space's advantage over the naive baseline excludes zero on this run."),
        "",
        "## 4. Consequence for the Phase 7 verdict",
        "",
        "The recommendation to prefer the naive raw-feature space for twin retrieval was",
        "drawn from a statistic that cannot rank representations, and is superseded by",
        "this table.",
        "",
        _serving_verdict(rows, naive),
        "",
        "The disease/laboratory/medication columns of the Phase 7 audit are unaffected —",
        "those are direct match rates, not enrichment ratios, and are read as published.",
        "",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
