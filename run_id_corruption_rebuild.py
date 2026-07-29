#!/usr/bin/env python3
"""
run_id_corruption_rebuild.py
────────────────────────────
Targeted rebuild of the artifacts damaged by the float32 identifier corruption.

Background
----------
``optimise_dtypes`` downcast every float64 column to float32, including nullable
identifier columns. float32 holds consecutive integers exactly only up to 2**24
(16,777,216); MIMIC-IV ``hadm_id`` values are ~2.2e7, so every odd id was rounded
to an even neighbour. ~50% of admissions consequently lost all laboratory
features, and some inherited labs belonging to a different admission.

The code defect is fixed in ``src/utils/io_utils.optimise_dtypes``. This script
repairs the *data* produced before that fix.

Scope
-----
Only ``labevents`` and ``emar`` were affected — they are the only tables where
``hadm_id`` is nullable, and therefore the only ones that arrived as float64.
Every other table already carried int32 identifiers and is untouched. Of those
two, only ``labevents`` feeds the feature pipeline, so it is the sole cause of the
downstream damage.

``data/interim/raw_cache/`` was written by the same defective code and is poisoned
for those two tables, so labevents must be re-read from CSV. Everything else can
be restored from cache, which avoids re-reading the 42 GB chartevents file.

Usage
-----
    python run_id_corruption_rebuild.py --audit          # what is damaged (read-only)
    python run_id_corruption_rebuild.py --verify         # did the repair work
    python run_id_corruption_rebuild.py --stage all      # run the repair
    python run_id_corruption_rebuild.py --stage labs     # one stage at a time

Stages are independent and resumable; each verifies its own output before moving on.
``patient_split.parquet`` is deliberately NOT regenerated: it is keyed on
``subject_id``, which was never corrupted, and holding it fixed makes before/after
model metrics directly comparable.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.utils.config import CFG                     # noqa: E402
from src.utils.io_utils import save_parquet          # noqa: E402
from src.utils.logger import get_logger, setup_root_logger  # noqa: E402

log = get_logger("rebuild")

INTERIM = Path(CFG.resolve(CFG.paths.interim))
PROCESSED = Path(CFG.resolve(CFG.paths.processed))
FEATURES = INTERIM / "features"
RAW_CACHE = INTERIM / "raw_cache"

ID_COLS = ("subject_id", "hadm_id", "stay_id")

#: Artifacts that must be rebuilt, in dependency order.
DAMAGED_ARTIFACTS = [
    INTERIM / "labevents_clean.parquet",
    INTERIM / "emar_clean.parquet",
    FEATURES / "laboratory_features.parquet",
    PROCESSED / "admission_level.parquet",
    PROCESSED / "admission_level_selected.parquet",
    PROCESSED / "time_series.parquet",
    PROCESSED / "similarity.parquet",
    PROCESSED / "patient_level.parquet",
    PROCESSED / "icu_level.parquet",
]

#: Tables replaced by synthetic fixtures when the old test suite ran.
STUBBED = {
    "admissions": INTERIM / "admissions_clean.parquet",
    "chartevents": INTERIM / "chartevents_clean.parquet",
    "radiology_detail": INTERIM / "radiology_detail_clean.parquet",
}


def _looks_stubbed(table: str, path: Path) -> bool:
    """A cleaned table with far fewer rows than its source was overwritten.

    Size alone is a poor signal: ``d_labitems`` is legitimately ~26 KB. Compare
    row counts against the cached raw load instead.
    """
    if not path.exists():
        return False
    try:
        rows = pq.ParquetFile(path).metadata.num_rows
    except Exception:
        return False
    cache = RAW_CACHE / f"{table}.parquet"
    if cache.exists():
        try:
            src_rows = pq.ParquetFile(cache).metadata.num_rows
            return src_rows > 0 and rows < 0.5 * src_rows
        except Exception:
            pass
    return rows < 1000


# ── diagnostics ───────────────────────────────────────────────────────────

def _id_dtypes(path: Path) -> Dict[str, str]:
    try:
        schema = pq.read_schema(path)
    except Exception:
        return {}
    return {n: str(schema.field(n).type) for n in schema.names if n in ID_COLS}


def _odd_fraction(path: Path, col: str = "hadm_id", limit: int = 2_000_000
                  ) -> Optional[float]:
    """Fraction of odd values in an id column — the corruption fingerprint.

    Genuine MIMIC ids are ~50% odd. Corrupted columns are 0% odd, because float32
    rounding at this magnitude snaps every value to an even number.
    """
    try:
        f = pq.ParquetFile(path)
        if col not in f.schema_arrow.names:
            return None
        batch = next(f.iter_batches(batch_size=limit, columns=[col]))
        s = pd.Series(batch.column(0).to_pandas()).dropna()
        if s.empty:
            return None
        return float((s.astype("int64") % 2 != 0).mean())
    except Exception:
        return None


def audit() -> List[Tuple[str, str]]:
    """Report the state of every artifact. Read-only."""
    findings: List[Tuple[str, str]] = []
    print("\n" + "=" * 78)
    print(" ID CORRUPTION AUDIT")
    print("=" * 78)

    print("\n--- source tables (interim) ---")
    for p in sorted(INTERIM.glob("*_clean.parquet")):
        ids = _id_dtypes(p)
        floats = [n for n, t in ids.items() if "float" in t]
        odd = _odd_fraction(p)
        size_mb = p.stat().st_size / 1_048_576
        flag = ""
        if floats:
            flag = f"  <-- FLOAT ID {floats}"
            findings.append((p.name, "float_id"))
        if odd is not None and odd < 0.01:
            flag += "  <-- 0% ODD (corrupted)"
            if (p.name, "float_id") not in findings:
                findings.append((p.name, "no_odd_ids"))
        table = p.name.replace("_clean.parquet", "")
        if _looks_stubbed(table, p):
            flag += "  <-- STUB (overwritten by old test suite)"
            findings.append((p.name, "stub"))
        oddtxt = "n/a" if odd is None else f"{odd:6.1%}"
        print(f"  {p.name:36} {size_mb:9.1f} MB  odd={oddtxt}{flag}")

    print("\n--- features ---")
    for p in sorted(FEATURES.glob("*.parquet")):
        ids = _id_dtypes(p)
        floats = [n for n, t in ids.items() if "float" in t]
        odd = _odd_fraction(p)
        oddtxt = "n/a" if odd is None else f"{odd:6.1%}"
        flag = "  <-- CORRUPTED" if floats or (odd is not None and odd < 0.01) else ""
        if flag:
            findings.append((p.name, "corrupt_feature"))
        print(f"  {p.name:36} {p.stat().st_size/1_048_576:9.1f} MB  odd={oddtxt}{flag}")

    print("\n--- processed datasets ---")
    for p in sorted(PROCESSED.glob("*.parquet")):
        ids = _id_dtypes(p)
        floats = [n for n, t in ids.items() if "float" in t]
        flag = f"  <-- FLOAT ID {floats}" if floats else ""
        if floats:
            findings.append((p.name, "float_id"))
        print(f"  {p.name:36} {p.stat().st_size/1_048_576:9.1f} MB{flag}")

    # the decisive downstream test
    sel = PROCESSED / "admission_level_selected.parquet"
    if sel.exists():
        print("\n--- downstream impact: lab coverage by hadm_id parity ---")
        try:
            schema = pq.read_schema(sel)
            labcol = next((n for n in schema.names if n.startswith("lab_")), None)
            if labcol:
                df = pd.read_parquet(sel, columns=["hadm_id", labcol])
                df["odd"] = df["hadm_id"].astype("int64") % 2 != 0
                g = df.groupby("odd")[labcol].apply(lambda x: x.notna().mean())
                print(f"  column: {labcol}")
                print(f"    even hadm_id : {g.get(False, float('nan')):.1%} have lab data")
                print(f"    odd  hadm_id : {g.get(True, float('nan')):.1%} have lab data")
                if g.get(True, 1.0) < 0.01:
                    print("    ==> CONFIRMED: odd-id admissions have no laboratory features")
                    findings.append((sel.name, "odd_ids_have_no_labs"))
                else:
                    print("    ==> parity balanced — looks repaired")
        except Exception as e:
            print(f"  (could not evaluate: {e})")

    print("\n" + "=" * 78)
    print(f" {len(findings)} damaged artifact(s)" if findings else " No damage detected")
    print("=" * 78 + "\n")
    return findings


# ── stages ────────────────────────────────────────────────────────────────

def _clean_and_save(df: pd.DataFrame, table: str) -> pd.DataFrame:
    from src.data.cleaner import DataCleaner
    t_cfg = CFG.tables.get(table, {})
    cleaned, _ = DataCleaner().clean_table(
        df, table, id_cols=t_cfg.get("id_cols"), save=True
    )
    return cleaned


def stage_stubs(backup: bool = True) -> None:
    """Restore tables that the old test suite overwrote with fixtures."""
    from src.data.loader import DataLoader
    loader = DataLoader()

    for table, path in STUBBED.items():
        if not _looks_stubbed(table, path):
            log.info("stage_stubs: %s looks intact — skipping", table)
            continue

        cache = RAW_CACHE / f"{table}.parquet"
        if cache.exists() and not _id_dtypes(cache).get("hadm_id", "").startswith("float"):
            log.info("stage_stubs: restoring %s from clean cache (%.0f MB)",
                     table, cache.stat().st_size / 1_048_576)
            df = pd.read_parquet(cache)
        else:
            log.info("stage_stubs: re-reading %s from CSV", table)
            df, _ = getattr(loader, f"load_{table}")()

        if backup and path.exists():
            shutil.copy2(path, path.with_suffix(".parquet.stub_backup"))
        _clean_and_save(df, table)
        log.info("stage_stubs: %s rebuilt (%d rows)", table, len(df))
        del df


def stage_labs() -> None:
    """Re-read labevents from CSV — raw_cache is poisoned for this table."""
    from src.data.loader import DataLoader
    log.info("stage_labs: re-reading labevents from CSV (this is the slow step)")
    t = time.time()
    df, summary = DataLoader().load_labevents()
    log.info("stage_labs: loaded %d rows in %.1f min", len(df), (time.time() - t) / 60)

    odd = (df["hadm_id"].dropna().astype("int64") % 2 != 0).mean()
    log.info("stage_labs: odd hadm_id fraction at load = %.1f%%", odd * 100)
    if odd < 0.10:
        raise RuntimeError(
            f"labevents still shows {odd:.1%} odd hadm_id after re-read — the "
            "identifier fix is not active. Aborting before writing."
        )

    _clean_and_save(df, "labevents")
    del df
    log.info("stage_labs: labevents_clean.parquet rebuilt")


def stage_emar(skip: bool = True) -> None:
    """emar is corrupted but consumed by no feature builder — optional."""
    if skip:
        log.info("stage_emar: skipped (emar feeds no feature builder; "
                 "pass --with-emar to rebuild it anyway)")
        return
    from src.data.loader import DataLoader
    log.info("stage_emar: re-reading emar from CSV (6.2 GB)")
    df, _ = DataLoader().load_emar()
    _clean_and_save(df, "emar")
    del df


def stage_features() -> None:
    """Rebuild laboratory features from the repaired labevents."""
    from src.features.build_features import FeatureBuilder
    from src.features.laboratory import (
        build_lab_features_from_df,
        build_lab_features_windowed,
    )

    path = INTERIM / "labevents_clean.parquet"
    odd = _odd_fraction(path)
    if odd is None or odd < 0.10:
        raise RuntimeError(
            f"refusing to build features: {path.name} shows odd fraction {odd}. "
            "Run --stage labs first."
        )

    log.info("stage_features: loading repaired labevents")
    labs = pd.read_parquet(path)
    log.info("stage_features: building lab features from %d rows", len(labs))
    feats = build_lab_features_from_df(labs)
    FeatureBuilder().save_features({"laboratory": feats})
    log.info("stage_features: laboratory_features rebuilt (%d admissions)", len(feats))
    del feats

    # 24-hour observation window variant. The full-stay aggregates above remain
    # correct for the full-stay protocols; these suffixed columns are what the
    # strict early-window protocols consume, replacing whole-admission extremes
    # that were previously presented as 24h observations.
    log.info("stage_features: building 24h-windowed lab features")
    adm = pd.read_parquet(INTERIM / "admissions_clean.parquet",
                          columns=["hadm_id", "admittime"])
    feats_24h = build_lab_features_windowed(labs, adm, hours=24.0)
    del labs, adm
    FeatureBuilder().save_features({"laboratory_24h": feats_24h})
    log.info("stage_features: laboratory_24h_features rebuilt (%d admissions)",
             len(feats_24h))
    del feats_24h


def stage_datasets() -> None:
    """Rebuild the processed datasets from repaired features."""
    from src.data.pipeline import ClinicalDigitalTwinPipeline
    log.info("stage_datasets: rebuilding processed datasets")
    p = ClinicalDigitalTwinPipeline(steps=["datasets"])
    p.step_datasets()
    log.info("stage_datasets: complete")


def verify() -> bool:
    """Acceptance test: did the repair actually work?"""
    print("\n" + "=" * 78)
    print(" REPAIR VERIFICATION")
    print("=" * 78)
    ok = True

    checks: List[Tuple[str, Path, float]] = [
        ("labevents_clean", INTERIM / "labevents_clean.parquet", 0.40),
        ("laboratory_features", FEATURES / "laboratory_features.parquet", 0.40),
    ]
    for name, path, floor in checks:
        if not path.exists():
            print(f"  [SKIP] {name}: not present")
            continue
        odd = _odd_fraction(path)
        good = odd is not None and odd >= floor
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {name}: odd hadm_id = "
              f"{'n/a' if odd is None else f'{odd:.1%}'} (expect ~50%)")

    sel = PROCESSED / "admission_level_selected.parquet"
    if sel.exists():
        schema = pq.read_schema(sel)
        labcol = next((n for n in schema.names if n.startswith("lab_")), None)
        if labcol:
            df = pd.read_parquet(sel, columns=["hadm_id", labcol])
            df["odd"] = df["hadm_id"].astype("int64") % 2 != 0
            g = df.groupby("odd")[labcol].apply(lambda x: x.notna().mean())
            even_r, odd_r = g.get(False, 0.0), g.get(True, 0.0)
            balanced = abs(even_r - odd_r) < 0.05
            ok &= balanced
            print(f"  [{'PASS' if balanced else 'FAIL'}] lab coverage parity: "
                  f"even {even_r:.1%} vs odd {odd_r:.1%} (must be within 5 pts)")

    print("=" * 78)
    print(" REPAIR VERIFIED — safe to retrain" if ok else
          " REPAIR INCOMPLETE — do not retrain yet")
    print("=" * 78 + "\n")
    return ok


STAGES = {
    "stubs": stage_stubs,
    "labs": stage_labs,
    "features": stage_features,
    "datasets": stage_datasets,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", action="store_true", help="report damage, change nothing")
    ap.add_argument("--verify", action="store_true", help="check whether the repair worked")
    ap.add_argument("--stage", choices=list(STAGES) + ["all"], help="stage to run")
    ap.add_argument("--with-emar", action="store_true", help="also rebuild emar")
    ap.add_argument("--no-backup", action="store_true", help="skip stub backups")
    args = ap.parse_args()

    setup_root_logger(log_file=CFG.logging.log_file, level=CFG.logging.level,
                      max_bytes=CFG.logging.max_bytes, backup_count=CFG.logging.backup_count)

    if args.audit or not (args.stage or args.verify):
        audit()
        if not args.stage:
            return 0
    if args.verify:
        return 0 if verify() else 1

    order = ["stubs", "labs", "features", "datasets"] if args.stage == "all" else [args.stage]
    started = time.time()
    for name in order:
        log.info("── stage: %s ──", name)
        t = time.time()
        if name == "stubs":
            stage_stubs(backup=not args.no_backup)
        else:
            STAGES[name]()
        log.info("── stage %s done in %.1f min ──", name, (time.time() - t) / 60)

    if args.with_emar:
        stage_emar(skip=False)

    log.info("rebuild finished in %.1f min", (time.time() - started) / 60)
    verify()
    print("\nNext: retrain Phases 1-5, then regenerate the reports.\n"
          "  python run_mortality_pipeline.py\n"
          "  python run_readmission_pipeline.py\n"
          "  python run_icu_admission_pipeline.py\n"
          "  python run_los_pipeline.py\n"
          "  python run_deterioration_pipeline.py\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
