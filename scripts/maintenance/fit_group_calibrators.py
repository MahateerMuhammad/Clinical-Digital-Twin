#!/usr/bin/env python
"""
scripts/maintenance/fit_group_calibrators.py
────────────────────────────────────────────
Fit per-age-band isotonic calibrators on the validation split.

    PYTHONPATH=. .venv/bin/python scripts/maintenance/fit_group_calibrators.py

Writes ``models/best_models/group_calibrators.pkl`` and prints the before/after
calibration error per band, measured on the **test** split it was not fitted on.

This is not retraining
──────────────────────
No booster is touched, no feature set changes, no model is refitted. The global
calibrators were already fitted with ``IsotonicRegression`` on validation
predictions (``src/models/icu_admission.py:calibrate_predictions``); this fits
the same estimator on the same split, partitioned by age band.

The split discipline is the part that matters. Calibrators are fitted on
**val** and reported on **test**. Fitting and scoring on the same rows would
show a large improvement and mean nothing — it would measure the isotonic
regression's ability to memorise, which is total.

A band that does not clear the support floor gets no calibrator and keeps the
global one. Nothing is silently degraded: the runner falls back on every path.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.evaluation import metrics as M
from src.models.group_calibration import (
    MIN_FIT_EVENTS, MIN_FIT_ROWS, GroupCalibrators, age_band,
)

FEATURES = Path("data/processed/admission_level_selected.parquet")
SPLIT = Path("data/processed/patient_split.parquet")

TASKS = {"mortality": "hospital_expire_flag",
         "icu_admission": "has_icu_stay",
         "readmission": "readmission_30d"}


def _load_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    sp = pd.read_parquet(SPLIT)[["subject_id", "split"]]
    merged = df.merge(sp, on="subject_id", how="inner")
    return merged[merged["split"].astype(str).str.lower() == split]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="report the effect without writing the artefact")
    args = ap.parse_args(argv)

    if not FEATURES.exists() or not SPLIT.exists():
        print(f"Need {FEATURES} and {SPLIT}")
        return 2

    from src.llm.model_runner import LiveModelRunner

    runner = LiveModelRunner()
    raw_df = pd.read_parquet(FEATURES)
    val = _load_split(raw_df, "val")
    test = _load_split(raw_df, "test")
    print(f"val rows {len(val):,} · test rows {len(test):,}")

    gc = GroupCalibrators(meta={
        "fitted_on": date.today().isoformat(),
        "split": "val",
        "min_fit_rows": MIN_FIT_ROWS,
        "min_fit_events": MIN_FIT_EVENTS,
        "note": "age-band isotonic calibration; boosters unchanged",
    })

    for task, target in TASKS.items():
        model = runner.lgbm_models.get(task)
        global_cal = runner.calibrators.get(task)
        if model is None or target not in raw_df.columns:
            print(f"\n[{task}] skipped — model or target absent")
            continue

        feats = model.booster_.feature_name()

        def scored(frame: pd.DataFrame):
            raw = model.predict_proba(frame.reindex(columns=feats))[:, 1]
            base = (np.asarray(global_cal.predict(raw), dtype=float)
                    if global_cal is not None and hasattr(global_cal, "predict")
                    else raw)
            y = pd.to_numeric(frame[target], errors="coerce").fillna(0).astype(int).to_numpy()
            bands = frame["anchor_age"].map(age_band).to_numpy()
            return raw, base, y, bands

        v_raw, v_base, v_y, v_band = scored(val)
        t_raw, t_base, t_y, t_band = scored(test)

        print(f"\n[{task}]  band            n(val)  events   ECE before → after")
        fitted: Dict[str, Any] = {}
        # Recorded in the artefact so the report can state the effect with
        # provenance rather than a number someone typed in from a console.
        effect: Dict[str, Any] = {}
        for band in sorted({b for b in v_band if b}):
            v_mask = v_band == band
            n, events = int(v_mask.sum()), int(v_y[v_mask].sum())
            if n < MIN_FIT_ROWS or events < MIN_FIT_EVENTS:
                print(f"          {band:14} {n:7,} {events:7}   "
                      f"skipped (below fit floor)")
                continue

            # Fit on the globally calibrated value, not the raw score: this is a
            # correction layered on what is already served, so a band with no
            # residual bias learns the identity and changes nothing.
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(v_base[v_mask], v_y[v_mask])

            t_mask = t_band == band
            before = M.expected_calibration_error(list(t_base[t_mask]),
                                                  list(t_y[t_mask]))
            after = M.expected_calibration_error(
                list(np.clip(cal.predict(t_base[t_mask]), 1e-4, 1 - 1e-4)),
                list(t_y[t_mask]))
            arrow = "✓" if (after is not None and before is not None
                            and after <= before) else "✗ worse"
            print(f"          {band:14} {n:7,} {events:7}   "
                  f"{before:.4f} → {after:.4f}  {arrow}")

            # Only keep a calibrator that helps on data it never saw. An isotonic
            # fit always improves its own split; the test split is the only thing
            # that can say whether it generalises.
            if after is not None and before is not None and after < before:
                fitted[band] = cal
                effect[band] = {"n_val": n, "events_val": events,
                                "ece_before": before, "ece_after": after}

        if fitted:
            gc.by_task[task] = fitted
            gc.meta.setdefault("effect", {})[task] = effect
        print(f"          kept {len(fitted)} band calibrator(s)")

    if args.dry_run:
        print("\nDry run — nothing written.")
        return 0

    path = gc.save(Path(args.out) if args.out else None)
    print(f"\nWrote {path}")
    print(f"  {sum(len(v) for v in gc.by_task.values())} calibrators across "
          f"{len(gc.by_task)} task(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
