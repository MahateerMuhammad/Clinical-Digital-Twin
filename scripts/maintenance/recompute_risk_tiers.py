#!/usr/bin/env python3
"""
scripts/maintenance/recompute_risk_tiers.py
───────────────────────
Recompute the Phase 9 risk-tier cutoffs from the current calibrated mortality model.

The four-tier scheme splits the held-out test set at the 50th, 80th and 95th
percentiles of predicted probability. Those cutoffs are therefore a property of a
*specific trained model*: any retrain invalidates them. Both the cutoffs
(``TIER_CUTOFFS``) and the published per-tier mortality rates (``SYSTEM_CONSTANTS``)
live in ``src/llm/report_composer.py``, which ``model_runner`` imports — ``--patch``
rewrites both from the model on disk, so neither can go stale independently. They did
exactly that across the 2026-07-29 retrain, when the cutoffs were updated by hand and
the rates were not.

This script recomputes them from the model currently on disk and reports the tier
statistics, so the published stratification can be regenerated without guesswork.

Usage
─────
    python scripts/maintenance/recompute_risk_tiers.py             # report only
    python scripts/maintenance/recompute_risk_tiers.py --patch     # also rewrite the constants in place
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
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
#: The single home of the tier cutoffs and the published per-tier mortality rates.
#: model_runner imports both from here rather than carrying its own copies.
COMPOSER = ROOT / "src" / "llm" / "report_composer.py"

#: Percentile boundaries of the published 4-tier scheme.
TIER_PERCENTILES = (50, 80, 95)
TIER_NAMES = ("Tier 1: Low Risk", "Tier 2: Moderate Risk",
              "Tier 3: High Risk", "Tier 4: Extreme Risk")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patch", action="store_true",
                    help="rewrite TIER_CUTOFFS and SYSTEM_CONSTANTS in src/llm/report_composer.py")
    ap.add_argument("--write-report", action="store_true",
                    help="regenerate reports/tables/risk_stratification.md")
    args = ap.parse_args()

    from src.models.mortality import MortalityModelPipeline
    import joblib

    cal_path = ROOT / "models" / "calibrated_mortality.pkl"
    lgb_path = ROOT / "models" / "lightgbm_mortality.pkl"
    for p in (cal_path, lgb_path):
        if not p.exists():
            print(f"Missing {p} — run scripts/pipelines/run_mortality_pipeline.py first.", file=sys.stderr)
            return 1

    print("Rebuilding the Run C test split (this reproduces Phase 1's protocol)...")
    pipe = MortalityModelPipeline()
    (_, _, X_test, _, _, y_test, _, _, _, _) = pipe.prepare_datasets(run_type="C")

    model = joblib.load(lgb_path)
    calibrator = joblib.load(cal_path)
    raw = model.predict_proba(X_test)[:, 1]
    probs = calibrator.predict(raw)

    cuts = [float(np.percentile(probs, q)) for q in TIER_PERCENTILES]
    edges = [-np.inf] + cuts + [np.inf]
    base = float(y_test.mean())

    print(f"\nTest admissions: {len(y_test):,}   observed deaths: {int(y_test.sum()):,} "
          f"({base:.2%})\n")
    hdr = f"{'tier':<24}{'range':<26}{'N':>9}{'deaths':>9}{'rate':>9}{'lift':>8}"
    print(hdr); print("-" * len(hdr))
    for i, name in enumerate(TIER_NAMES):
        m = (probs >= edges[i]) & (probs < edges[i + 1])
        n, d = int(m.sum()), int(y_test[m].sum())
        rate = d / n if n else 0.0
        lo = "0" if i == 0 else f"{edges[i]:.4f}"
        hi = "1" if i == len(TIER_NAMES) - 1 else f"{edges[i+1]:.4f}"
        print(f"{name:<24}[{lo} – {hi})".ljust(50)
              + f"{n:>9,}{d:>9,}{rate:>8.2%}{rate/base if base else 0:>7.2f}x")

    print(f"\nNew cutoffs: {cuts[0]:.4f} / {cuts[1]:.4f} / {cuts[2]:.4f}")

    if args.write_report:
        from sklearn.metrics import roc_auc_score, average_precision_score
        rng = np.random.default_rng(42)
        rows, pcts = [], ["0 – 50th%", "50 – 80th%", "80 – 95th%", "Top 5% (95–100th%)"]
        for i, name in enumerate(TIER_NAMES):
            m = (probs >= edges[i]) & (probs < edges[i + 1])
            n, d = int(m.sum()), int(y_test[m].sum())
            rate = d / n if n else 0.0
            boot = [y_test[m][rng.integers(0, n, n)].mean() for _ in range(1000)] if n else [0.0]
            lo, hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
            lo_c = "0.00%" if i == 0 else f"{edges[i]*100:.2f}%"
            hi_c = "100.00%" if i == len(TIER_NAMES) - 1 else f"{edges[i+1]*100:.2f}%"
            rows.append(
                f"| **{name}** | {pcts[i]} | $[{lo_c} - {hi_c})$ | {n:,} | **{n/len(y_test)*100:.1f}%** | "
                f"{d:,} | {d/int(y_test.sum())*100:.1f}% | **{rate*100:.2f}%** | "
                f"**{lo*100:.2f}% – {hi*100:.2f}%** | **{rate/base:.2f}x** | "
                f"{'General Ward / Routine Floor Care' if i==0 else 'Standard Telemetry & Continuous Vitals' if i==1 else 'Step-Down / Progressive Care Unit' if i==2 else 'Immediate ICU Consultation & Rapid Response'} |")

        auroc, auprc = roc_auc_score(y_test, probs), average_precision_score(y_test, probs)
        out = Path("reports/tables/risk_stratification.md")
        out.write_text(f"""# Phase 9 — In-Hospital Mortality Risk Stratification & Clinical Resource Planning Audit

> [!NOTE]
> Regenerated by `scripts/maintenance/recompute_risk_tiers.py --write-report` from the model currently in
> `models/`. Tier cutoffs are percentiles of a *specific* model's test predictions, so
> any Phase 1 retrain invalidates them; this file must be regenerated alongside.

## 1. Scope

Built from the winning **Phase 1 Calibrated LightGBM** mortality model
(strict 24-hour observation window, AUROC **{auroc:.4f}**, AUPRC **{auprc:.4f}**),
evaluated on the held-out test split (*N* = {len(y_test):,} admissions, base mortality
rate **{base*100:.2f}%**, {int(y_test.sum()):,} observed deaths).

* **No model retraining.** Probabilities are evaluated only to define risk bands.
* **Held-out isolation.** Test admissions were never seen in training or calibration.
* **Bootstrap CIs.** 1,000 resamples per tier (2.5th–97.5th percentiles).
* **Monotonicity.** Ordering across probability-sorted bands is expected by
  construction; it confirms the binning works, it is not independent validation.

## 2. Clinical 4-Tier Stratification

Asymmetric cutoffs (50th / 80th / 95th percentile) isolate the low-risk majority for
routine care while concentrating deaths into actionable tiers.

| Risk Tier | Percentile Range | Predicted Probability | Admissions ($N$) | Cohort Share | Deaths | Share of Deaths | Observed Mortality | 95% Bootstrap CI | Enrichment | Recommended Action |
| :--- | :---: | :---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | :--- |
{chr(10).join(rows)}
| **Total** | 0 – 100th% | $[0.00% - 100.00%]$ | {len(y_test):,} | **100.0%** | {int(y_test.sum()):,} | 100.0% | **{base*100:.2f}%** | — | **1.00x** | Population baseline |

Probability cutoffs: **{cuts[0]:.4f} / {cuts[1]:.4f} / {cuts[2]:.4f}** — held in
`TIER_CUTOFFS` in `src/llm/report_composer.py`, alongside the observed rates above in
`SYSTEM_CONSTANTS`. `model_runner` imports both; `--patch` keeps them in step.

## 3. Interpretation

The top tier concentrates the majority of deaths into a small fraction of admissions,
which is the property that makes the stratification clinically actionable. Note that
enrichment improved after the observation-window leakage was removed even though AUROC
fell — the leak-free model separates the extreme tail *better*, despite scoring lower
on a threshold-free metric. See §6 and §8 of
[`data_correction_notice.md`](../data_correction_notice.md).

At the deployed operating point the model trades precision for recall; tier assignment
is a triage aid, not a diagnosis, and the observed rates above are the honest
expectation for each band.
""", encoding="utf-8")
        print(f"\nWritten → {out}")

    if not args.patch:
        print("\nReport only. Re-run with --patch to rewrite the constants in "
              "src/llm/report_composer.py.")
        return 0

    # Both the cutoffs and the observed per-tier rates live in report_composer, which
    # model_runner imports. This used to rewrite `p_mort < 0.0034` literals inside
    # model_runner and print "remember to update SYSTEM_CONSTANTS by hand" — a manual
    # step that was, predictably, forgotten, leaving the served tiers and the quoted
    # mortality rates describing two different models. Patching one file removes the
    # opportunity to forget.
    src = COMPOSER.read_text(encoding="utf-8")

    rates = []
    for i in range(len(TIER_NAMES)):
        m = (probs >= edges[i]) & (probs < edges[i + 1])
        n = int(m.sum())
        rates.append(round((int(y_test[m].sum()) / n * 100) if n else 0.0, 2))

    cut_pat = re.compile(
        r"(TIER_CUTOFFS: tuple\[float, float, float\] = )\([^)]*\)")
    if not cut_pat.search(src):
        print(f"\nTIER_CUTOFFS not found in {COMPOSER.name} — update by hand.",
              file=sys.stderr)
        return 1
    src = cut_pat.sub(
        rf"\g<1>({cuts[0]:.4f}, {cuts[1]:.4f}, {cuts[2]:.4f})", src, count=1)

    old_rates = []
    for i, rate in enumerate(rates, start=1):
        key = f"phase9_tier{i}_observed_mortality_pct"
        pat = re.compile(rf'("{key}":\s*)([\d.]+)')
        m = pat.search(src)
        if not m:
            print(f"\n{key} not found in {COMPOSER.name} — update by hand.",
                  file=sys.stderr)
            return 1
        old_rates.append(m.group(2))
        src = pat.sub(rf"\g<1>{rate}", src, count=1)

    COMPOSER.write_text(src, encoding="utf-8")
    print(f"\nPatched {COMPOSER.relative_to(ROOT)}")
    print(f"  cutoffs        → {cuts[0]:.4f} / {cuts[1]:.4f} / {cuts[2]:.4f}")
    print(f"  observed rates   {' / '.join(old_rates)}  →  "
          f"{' / '.join(f'{r}' for r in rates)}")
    print("\nmodel_runner reads both from report_composer, so nothing else to update.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
