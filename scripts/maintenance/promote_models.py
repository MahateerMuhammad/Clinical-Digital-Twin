#!/usr/bin/env python3
"""
scripts/maintenance/promote_models.py
─────────────────
Promote freshly trained model artifacts from ``models/`` into ``models/best_models/``,
which is the directory the LLM serving layer (``src/llm/model_runner.py``) loads from.

Why this exists
───────────────
The training pipelines write to ``models/`` under one naming scheme
(``lightgbm_mortality.pkl``); the serving layer reads from ``models/best_models/``
under another (``phase1_mortality_winning.pkl``). Nothing in the codebase
bridged the two, so promotion was a manual copy-and-rename.

The consequence was silent staleness: after Phases 1-5 were retrained on 2026-07-29
to remove observation-window leakage, ``best_models/`` still held the 2026-07-21
artifacts. Phase 9 risk stratification and every LLM-layer prediction continued to
serve models built on corrupted laboratory joins and discharge-note leakage, with
no error and no warning — the only symptom was file mtimes eight days apart.

Deep-learning artifacts (``*.pt`` — LSTM, transformer, autoencoders) are trained
externally on Kaggle and are NOT touched by this script.

Usage
─────
    python scripts/maintenance/promote_models.py              # dry run: show what would change
    python scripts/maintenance/promote_models.py --apply      # perform the promotion
    python scripts/maintenance/promote_models.py --apply --no-backup
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
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "models"
DST = ROOT / "models" / "best_models"

#: (source filename in models/, destination filename in models/best_models/)
#:
#: Phase 1 promotes the **Run C** artifacts: scripts/pipelines/run_mortality_pipeline.py calls
#: ``save_models(logreg_c, xgb_c, lgb_c, calibrator)``, so these pickles are the
#: strict 24-hour observation-window models, which is what should be served.
PROMOTIONS: List[Tuple[str, str]] = [
    ("lightgbm_mortality.pkl",              "phase1_mortality_winning.pkl"),
    ("calibrated_mortality.pkl",            "phase1_mortality_calibrated.pkl"),
    ("lightgbm_readmission.pkl",            "phase2_readmission_winning.pkl"),
    ("calibrated_readmission.pkl",          "phase2_readmission_calibrated.pkl"),
    ("lightgbm_icu_admission.pkl",          "phase3_icu_admission_winning.pkl"),
    ("calibrated_icu_admission.pkl",        "phase3_icu_admission_calibrated.pkl"),
    ("los_stageA_classifier_lightgbm.pkl",  "phase4_hosp_los_stageA_winning.pkl"),
    ("los_stageA_calibrated.pkl",           "phase4_hosp_los_stageA_calibrated.pkl"),
    ("los_stageB_regressor_lightgbm.pkl",   "phase4_hosp_los_stageB_winning.pkl"),
    ("icu_los_stageA_classifier_lightgbm.pkl", "phase4_icu_los_stageA_winning.pkl"),
    ("icu_los_stageA_calibrated.pkl",       "phase4_icu_los_stageA_calibrated.pkl"),
    ("icu_los_stageB_regressor_lightgbm.pkl", "phase4_icu_los_stageB_winning.pkl"),
    ("lightgbm_deterioration.pkl",          "phase5_deterioration_winning.pkl"),
    ("calibrated_deterioration.pkl",        "phase5_deterioration_calibrated.pkl"),
]


def _resolve_deterioration_winner(promotions: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Promote whichever deterioration model actually won, not a hardcoded one.

    Phase 5 picks its winner at runtime by test AUPRC. This list assumed LightGBM,
    but XGBoost has won since the 2026-08-01 retrain (AUPRC 0.3771 against 0.3739) —
    so the promoted artifact was not the one the pipeline evaluated, calibrated or
    reported on. `deterioration_winner.json` records the choice; if it is absent the
    LightGBM default stands and a warning is printed rather than silently promoting
    a possibly-wrong file.
    """
    record = SRC / "deterioration_winner.json"
    if not record.exists():
        print("  ! deterioration_winner.json absent — defaulting to LightGBM. "
              "Re-run scripts/pipelines/run_deterioration_pipeline.py to record the winner.")
        return promotions

    info = json.loads(record.read_text(encoding="utf-8"))
    pickle_name = info.get("pickle")
    if not pickle_name:
        return promotions

    out = []
    for src, dst in promotions:
        if dst == "phase5_deterioration_winning.pkl":
            print(f"  deterioration winner: {info.get('winning_model')} -> {pickle_name}")
            out.append((pickle_name, dst))
        else:
            out.append((src, dst))
    return out


def _stamp(path: Path) -> str:
    if not path.exists():
        return "absent"
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="perform the promotion (default is a dry run)")
    ap.add_argument("--no-backup", action="store_true",
                    help="skip archiving the artifacts being replaced")
    args = ap.parse_args()

    if not SRC.exists():
        print(f"No models directory at {SRC}", file=sys.stderr)
        return 1
    DST.mkdir(parents=True, exist_ok=True)

    plan, missing = [], []
    promotions = _resolve_deterioration_winner(PROMOTIONS)
    for src_name, dst_name in promotions:
        s, d = SRC / src_name, DST / dst_name
        (plan if s.exists() else missing).append((s, d))

    width = max(len(d.name) for _, d in plan) if plan else 40
    print(f"\n{'destination':<{width}}  {'current':<16}  {'incoming':<16}")
    print("-" * (width + 36))
    for s, d in plan:
        print(f"{d.name:<{width}}  {_stamp(d):<16}  {_stamp(s):<16}")

    if missing:
        print("\nNot found in models/ — left untouched:")
        for s, d in missing:
            print(f"  {s.name}  ->  {d.name}")

    if not args.apply:
        print(f"\nDry run. {len(plan)} artifact(s) would be promoted. "
              f"Re-run with --apply to perform it.")
        return 0

    backup = None
    if not args.no_backup:
        backup = DST / f"_superseded_{datetime.now():%Y%m%d_%H%M%S}"
        backup.mkdir(parents=True, exist_ok=True)

    promoted = 0
    for s, d in plan:
        if backup is not None and d.exists():
            shutil.copy2(d, backup / d.name)
        shutil.copy2(s, d)
        promoted += 1

    print(f"\nPromoted {promoted} artifact(s) → {DST}")
    if backup is not None:
        print(f"Replaced artifacts archived → {backup}")
    print("\nNOTE: *.pt deep-learning artifacts (Phases 6-7, Kaggle-trained) were not touched.")
    print("NOTE: Phase 9 tier cutoffs in src/llm/model_runner.py are derived from the")
    print("      calibrated model's test-set percentiles and must be recomputed after")
    print("      promotion — run: python scripts/maintenance/recompute_risk_tiers.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
