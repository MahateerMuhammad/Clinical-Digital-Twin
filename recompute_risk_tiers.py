#!/usr/bin/env python3
"""
recompute_risk_tiers.py
───────────────────────
Recompute the Phase 9 risk-tier cutoffs from the current calibrated mortality model.

The four-tier scheme splits the held-out test set at the 50th, 80th and 95th
percentiles of predicted probability. Those cutoffs are therefore a property of a
*specific trained model*: any retrain invalidates them. They are hardcoded in
``src/llm/model_runner.py`` and quoted as system constants in
``src/llm/report_composer.py``, with nothing tying them back to the model, so they
silently go stale — exactly what happened across the 2026-07-29 retrain.

This script recomputes them from the model currently on disk and reports the tier
statistics, so the published stratification can be regenerated without guesswork.

Usage
─────
    python recompute_risk_tiers.py             # report only
    python recompute_risk_tiers.py --patch     # also rewrite the constants in place
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "src" / "llm" / "model_runner.py"

#: Percentile boundaries of the published 4-tier scheme.
TIER_PERCENTILES = (50, 80, 95)
TIER_NAMES = ("Tier 1: Low Risk", "Tier 2: Moderate Risk",
              "Tier 3: High Risk", "Tier 4: Extreme Risk")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patch", action="store_true",
                    help="rewrite the hardcoded cutoffs in src/llm/model_runner.py")
    args = ap.parse_args()

    from src.models.mortality import MortalityModelPipeline
    import joblib

    cal_path = ROOT / "models" / "calibrated_mortality.pkl"
    lgb_path = ROOT / "models" / "lightgbm_mortality.pkl"
    for p in (cal_path, lgb_path):
        if not p.exists():
            print(f"Missing {p} — run run_mortality_pipeline.py first.", file=sys.stderr)
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

    if not args.patch:
        print("\nReport only. Re-run with --patch to rewrite src/llm/model_runner.py.")
        return 0

    src = RUNNER.read_text(encoding="utf-8")
    found = re.findall(r"p_mort < (\d\.\d+)", src)
    if len(found) != 3:
        print(f"\nExpected 3 cutoff comparisons in {RUNNER.name}, found {len(found)}. "
              "Not patching — update by hand.", file=sys.stderr)
        return 1
    for old, new in zip(found, cuts):
        src = src.replace(f"p_mort < {old}", f"p_mort < {new:.4f}", 1)
    RUNNER.write_text(src, encoding="utf-8")
    print(f"\nPatched {RUNNER.relative_to(ROOT)}: {' / '.join(found)} "
          f"→ {cuts[0]:.4f} / {cuts[1]:.4f} / {cuts[2]:.4f}")
    print("Remember to update SYSTEM_CONSTANTS in src/llm/report_composer.py to match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
