#!/usr/bin/env python
"""
Build the emergency-department feature set and persist it.

Runs standalone, against the raw ED CSVs and the existing admission table, and
writes one hadm_id-keyed Parquet. Nothing upstream is re-derived: the cohort, the
split and every already-trained model are untouched by this script, because ED
data joins onto admissions and never creates them.

    python scripts/pipelines/build_ed_features.py
    python scripts/pipelines/build_ed_features.py --window-hours 24
    python scripts/pipelines/build_ed_features.py --report

The default window is 0.0 — observations at or before ``admittime`` only, which
is safe under every protocol in this project. ``--window-hours 24`` produces the
counterpart consistent with the ``lab_*_24h`` family for models scored under the
24-hour protocol instead of at admission.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.features.emergency import build_ed_features
from src.utils.config import CFG
from src.utils.io_utils import save_parquet
from src.utils.logger import get_logger

log = get_logger(__name__)

ADMISSIONS = Path("data/processed/admission_level.parquet")


def _read(table: str, **kwargs) -> pd.DataFrame:
    path = CFG.table_file(table)
    if not path.exists():
        log.warning("%s not found at %s — skipping", table, path)
        return pd.DataFrame()
    df = pd.read_csv(path, **kwargs)
    log.info("Read %s: %d rows", table, len(df))
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window-hours", type=float, default=0.0,
                    help="keep ED vitals charted up to admittime + this many hours")
    ap.add_argument("--out", type=Path, default=None,
                    help="output Parquet path (default derives from --window-hours)")
    ap.add_argument("--report", action="store_true",
                    help="also write reports/tables/ed_feature_coverage.md")
    args = ap.parse_args()

    if not ADMISSIONS.exists():
        log.error("%s missing; run the main pipeline first", ADMISSIONS)
        return 1

    adm = pd.read_parquet(ADMISSIONS, columns=["hadm_id", "subject_id", "admittime"])
    adm["hadm_id"] = adm["hadm_id"].astype("int64")
    log.info("Cohort: %d admissions, %d patients",
             len(adm), adm["subject_id"].nunique())

    feats = build_ed_features(
        _read("edstays"),
        adm,
        triage=_read("triage"),
        vitalsign=_read("ed_vitalsign"),
        medrecon=_read("medrecon"),
        window_hours=args.window_hours,
    )
    if feats.empty:
        log.error("No ED features produced")
        return 1

    # The guarantee this whole integration rests on.
    assert feats["hadm_id"].is_unique, "ED features are not one row per admission"
    assert set(feats["hadm_id"]) <= set(adm["hadm_id"]), \
        "ED features reference admissions outside the cohort"

    suffix = "" if args.window_hours == 0.0 else f"_{int(args.window_hours)}h"
    out = args.out or Path(
        CFG.resolve(CFG.paths.interim)) / "features" / f"emergency_features{suffix}.parquet"
    save_parquet(feats, out)

    coverage = 100.0 * len(feats) / len(adm)
    log.info("Wrote %s — %d admissions (%.1f%% of cohort) × %d features",
             out, len(feats), coverage, feats.shape[1] - 1)

    if args.report:
        merged = adm[["hadm_id"]].merge(feats, on="hadm_id", how="left")
        cols = [c for c in feats.columns if c != "hadm_id"]
        rows = pd.DataFrame({
            "feature": cols,
            "non_null_cohort_pct": [
                round(100.0 * merged[c].notna().mean(), 2) for c in cols],
            "non_null_among_ed_pct": [
                round(100.0 * feats[c].notna().mean(), 2) for c in cols],
        }).sort_values("non_null_cohort_pct", ascending=False)

        report = Path(CFG.resolve(CFG.paths.tables)) / "ed_feature_coverage.md"
        with report.open("w", encoding="utf-8") as fh:
            fh.write("# Emergency Department Feature Coverage\n\n")
            fh.write(f"Generated from `{out}` "
                     f"(window: admittime + {args.window_hours:.0f}h).\n\n")
            fh.write(f"- Cohort admissions: **{len(adm):,}**\n")
            fh.write(f"- With a linked ED stay: **{len(feats):,}** "
                     f"({coverage:.1f}%)\n")
            fh.write(f"- Features produced: **{len(cols)}**\n\n")
            fh.write("Admissions without an ED stay hold NaN, never 0.0 — the ED "
                     "module is a partial capture of this cohort, and a filled "
                     "zero would assert a measurement that was never taken.\n\n")
            fh.write(rows.to_markdown(index=False))
            fh.write("\n")
        log.info("Wrote %s", report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
