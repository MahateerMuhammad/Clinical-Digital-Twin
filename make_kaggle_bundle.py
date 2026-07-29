#!/usr/bin/env python3
"""
make_kaggle_bundle.py
─────────────────────
Assemble the minimal dataset needed to run Phase 6 (sequence models) and Phase 7
(patient embeddings) on Kaggle.

``admission_level_selected.parquet`` is ~3 GB, but 96% of that is two raw discharge
note columns — ``text_clean`` and ``text_tfidf_ready`` — which neither Kaggle
notebook consumes: both appear in every pipeline's ``non_features`` list, and the
Phase 7 autoencoder is trained on tabular features only. Dropping them takes the
upload from 3.4 GB to roughly 0.5 GB and, more importantly, keeps the frame inside
a Kaggle kernel's memory budget.

Output → ``kaggle_bundle/``

Usage
─────
    python make_kaggle_bundle.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "kaggle_bundle"

#: Columns carrying raw note text. Excluded from every model's feature matrix.
TEXT_COLS = ["text_clean", "text_tfidf_ready"]

#: Copied verbatim — already small.
VERBATIM = [
    (PROC / "time_series.parquet",   "Phase 6: event sequences"),
    (PROC / "patient_split.parquet", "Phases 6+7: patient-level split"),
    (PROC / "similarity.parquet",    "Phase 7: target for dim_0..dim_31"),
    (ROOT / "models" / "lightgbm_mortality.pkl",
     "Phase 7: supervised signal for the LGB-guided autoencoder"),
]


def _mb(p: Path) -> float:
    return p.stat().st_size / 1e6


def main() -> int:
    src = PROC / "admission_level_selected.parquet"
    if not src.exists():
        print(f"Missing {src}", file=sys.stderr)
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    print("Reading admission_level_selected.parquet (skipping note text)...")
    cols = pd.read_parquet(src, columns=[]).columns.tolist() or None
    import pyarrow.parquet as pq
    schema = pq.ParquetFile(src).schema_arrow
    keep = [c for c in schema.names if c not in TEXT_COLS]
    dropped = [c for c in TEXT_COLS if c in schema.names]

    df = pd.read_parquet(src, columns=keep)
    dst = OUT / "admission_level_selected.parquet"
    df.to_parquet(dst, compression="snappy", index=False)
    rows.append((dst.name, _mb(src), _mb(dst),
                 f"dropped {len(dropped)} text column(s): {', '.join(dropped) or 'none'}"))
    del df

    for p, note in VERBATIM:
        if not p.exists():
            rows.append((p.name, 0.0, 0.0, "MISSING — skipped"))
            continue
        d = OUT / p.name
        shutil.copy2(p, d)
        rows.append((d.name, _mb(p), _mb(d), note))

    w = max(len(r[0]) for r in rows)
    print(f"\n{'file':<{w}}{'source MB':>12}{'bundle MB':>12}  note")
    print("-" * (w + 30))
    total = 0.0
    for name, s, d, note in rows:
        total += d
        print(f"{name:<{w}}{s:>12.1f}{d:>12.1f}  {note}")
    print("-" * (w + 30))
    print(f"{'TOTAL':<{w}}{'':>12}{total:>12.1f} MB\n")
    print(f"Bundle ready → {OUT}")
    print("Upload this directory as a Kaggle Dataset, then attach it to the notebooks:")
    print("  notebooks/07_sequence_model_kaggle.ipynb   (Phase 6)")
    print("  notebooks/12_patient_embeddings_kaggle.ipynb (Phase 7)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
