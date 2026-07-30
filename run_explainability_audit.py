#!/usr/bin/env python3
"""
run_explainability_audit.py
───────────────────────────
Phase 8 — regenerate the SHAP explainability audit from the models currently on disk.

Why this exists
───────────────
``reports/tables/explainability_audit.md`` had no generator. It was written by hand
in July 2026 and then went stale silently: after the 2026-07-29 retrain it still
discussed ``med_class_opioid``, ``sentence_count`` and ``lab_bicarbonate_min`` as
leading predictors, none of which survive the corrected exclusion lists. A document
describing features the models no longer contain is worse than no document, because
it reads as current.

Beyond listing rankings, this script runs a **leakage screen**: every feature in each
model's SHAP top-N is tested against the families that were removed in the
observation-window correction. If one reappears, the audit says so in bold rather
than leaving a reader to notice. That screen is the point — Phase 8 is what caught
the leaks originally, so it should keep catching them.

Usage
─────
    python run_explainability_audit.py                 # all phases
    python run_explainability_audit.py --phases 1,3    # a subset
    python run_explainability_audit.py --sample 3000   # smaller SHAP sample
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports" / "tables" / "explainability_audit.md"

#: Families removed in the observation-window correction. A feature matching any of
#: these in a strict-protocol model means a leak has been reintroduced.
#: ``_24h`` variants are explicitly permitted — they are the corrected form.
LEAK_SCREEN: Dict[str, str] = {
    "med_class_": "whole-stay drug flag (end-of-life prescribing proxy)",
    "sentence_count": "discharge-note statistic (authored at discharge)",
    "word_count": "discharge-note statistic",
    "char_count": "discharge-note statistic",
    "medical_keyword_count": "discharge-note statistic",
    "negation_count": "discharge-note statistic",
    "note_": "discharge-note derived",
    "lab_unique_items": "full-stay lab ordering intensity",
    "lab_total_count": "full-stay lab ordering intensity",
    "cci_": "post-hoc ICD comorbidity coding",
    "dx_": "post-hoc ICD diagnosis coding",
    "icu_": "post-hoc ICU accumulation",
    "los_": "outcome-adjacent duration",
}

#: Whole-admission lab aggregates: a leak only when the ``_24h`` suffix is absent.
_LAB_AGG_SUFFIXES = ("_min", "_max", "_median", "_first", "_last", "_slope", "_std")


def screen_feature(name: str) -> Optional[str]:
    """Return a description of the leak family this feature belongs to, or None."""
    if name.endswith("_24h") or "_24h" in name:
        return None                                   # corrected windowed form
    for prefix, why in LEAK_SCREEN.items():
        if name.startswith(prefix):
            return why
    if name.startswith("lab_") and name.endswith(_LAB_AGG_SUFFIXES):
        return "whole-admission lab aggregate (charttime never filtered)"
    return None


def shap_top(model, X: pd.DataFrame, k: int, sample: int, seed: int = 42
             ) -> List[Tuple[str, float]]:
    """Mean |SHAP| ranking over a random sample of rows."""
    import shap
    if len(X) > sample:
        X = X.sample(sample, random_state=seed)
    values = shap.TreeExplainer(model).shap_values(X)
    if isinstance(values, list):                      # older API: one array per class
        values = values[1] if len(values) > 1 else values[0]
    values = np.asarray(values)
    if values.ndim == 3:                              # (rows, features, classes)
        values = values[:, :, 1] if values.shape[2] > 1 else values[:, :, 0]
    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:k]
    return [(str(X.columns[i]), float(mean_abs[i])) for i in order]


# ── per-phase loaders ────────────────────────────────────────────────────────
def load_phase(n: int):
    """Return (title, protocol, model, X_test) for a phase, or None if unavailable."""
    import joblib
    M = ROOT / "models"

    if n == 1:
        from src.models.mortality import MortalityModelPipeline
        X = MortalityModelPipeline().prepare_datasets(run_type="C")[2]
        return ("Phase 1 — In-hospital mortality", "Run C (strict 24h window)",
                joblib.load(M / "lightgbm_mortality.pkl"), X)
    if n == 2:
        from src.models.readmission import ReadmissionModelPipeline
        X = ReadmissionModelPipeline().prepare_datasets(run_type="B")[2]
        return ("Phase 2 — 30-day unplanned readmission", "Run B (strict 24h window)",
                joblib.load(M / "lightgbm_readmission.pkl"), X)
    if n == 3:
        from src.models.icu_admission import ICUAdmissionModelPipeline
        X = ICUAdmissionModelPipeline().prepare_datasets()[2]
        return ("Phase 3 — ICU admission risk", "Admission-time (strict)",
                joblib.load(M / "lightgbm_icu_admission.pkl"), X)
    if n == 4:
        from src.models.los import LengthOfStayModelPipeline
        X = LengthOfStayModelPipeline().prepare_datasets()[2]
        return ("Phase 4 — Hospital length of stay (Stage A)", "Admission-time (strict)",
                joblib.load(M / "los_stageA_classifier_lightgbm.pkl"), X)
    if n == 5:
        from src.models.deterioration import DeteriorationModelPipeline
        X = DeteriorationModelPipeline().prepare_datasets()[2]
        return ("Phase 5 — Clinical deterioration", "Strict (6h horizon)",
                joblib.load(M / "lightgbm_deterioration.pkl"), X)
    raise ValueError(n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phases", default="1,2,3,4,5", help="comma-separated phase numbers")
    ap.add_argument("--top", type=int, default=15, help="features to report per model")
    ap.add_argument("--sample", type=int, default=5000, help="rows sampled for SHAP")
    args = ap.parse_args()

    try:
        import shap  # noqa: F401
    except ImportError:
        print("shap is not installed: pip install shap", file=sys.stderr)
        return 1

    phases = [int(x) for x in args.phases.split(",") if x.strip()]
    sections, all_flags = [], []

    for n in phases:
        try:
            title, protocol, model, X = load_phase(n)
        except Exception as exc:                       # noqa: BLE001
            print(f"  [SKIP] phase {n}: {exc}")
            sections.append(f"### {title if 'title' in dir() else f'Phase {n}'}\n\n"
                            f"_Not audited: {exc}_\n")
            continue

        print(f"  [phase {n}] SHAP over {min(len(X), args.sample):,} of {len(X):,} rows, "
              f"{X.shape[1]} features ...")
        ranking = shap_top(model, X, args.top, args.sample)

        rows, flags = [], []
        for rank, (feat, val) in enumerate(ranking, 1):
            leak = screen_feature(feat)
            note = f"**LEAK — {leak}**" if leak else ""
            if leak:
                flags.append((n, feat, leak, rank))
            rows.append(f"| {rank} | `{feat}` | {val:.4f} | {note} |")
        all_flags.extend(flags)

        verdict = ("**CLEAN** — no feature in the top "
                   f"{args.top} matches a removed leak family."
                   if not flags else
                   f"**{len(flags)} FLAGGED** — see the table.")
        sections.append(
            f"### {title}\n\n"
            f"*Protocol:* {protocol} · *Features available:* {X.shape[1]} · "
            f"*Test rows:* {len(X):,}\n\n"
            f"Leakage screen: {verdict}\n\n"
            f"| Rank | Feature | Mean \\|SHAP\\| | Screen |\n"
            f"| ---: | :--- | ---: | :--- |\n" + "\n".join(rows) + "\n")

    header_verdict = (
        "**All audited models pass the leakage screen.** No feature in any model's "
        f"top {args.top} belongs to a family removed by the observation-window "
        "correction."
        if not all_flags else
        f"**{len(all_flags)} flagged feature(s) across {len({f[0] for f in all_flags})} "
        "model(s).** A removed leak family has reappeared — investigate before citing "
        "any downstream figure.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(f"""# Phase 8 — Model Explainability & Leakage Screen

> [!NOTE]
> Generated by `run_explainability_audit.py` from the models in `models/`. SHAP
> rankings are a property of a specific trained model, so this file must be
> regenerated after any retrain. The previous version was hand-written and went
> stale, describing features the corrected models no longer contain.

## 1. Verdict

{header_verdict}

## 2. Method

Mean absolute SHAP value per feature, computed with `shap.TreeExplainer` over a
random sample of up to {args.sample:,} held-out test rows per model. Ranking by mean
|SHAP| measures *influence on the prediction*, not causal effect: a feature can rank
highly because it proxies severity, because it encodes clinical process, or because
it leaks the outcome. Distinguishing those is the purpose of §3.

Every ranked feature is tested against the families removed in the observation-window
correction — whole-stay drug flags, discharge-note statistics, full-stay lab
aggregates and counters, post-hoc ICD coding, and outcome-adjacent duration fields.
`_24h`-suffixed features are permitted: they are the corrected, windowed form.

## 3. Per-model rankings

{chr(10).join(sections)}
## 4. Reading these rankings

A feature ranking first does not make it a cause. Three patterns recur:

* **Physiological severity** — `lab_bun_max_24h`, `lab_wbc_max_24h`,
  `lab_bicarbonate_min_24h`. Derangement within the observation window. These are the
  rankings you want to see at the top.
* **Care-intensity proxies** — `lab_total_count_24h`, `lab_unique_items_24h`. How many
  assays the team ordered in the first 24 hours. Legitimately observable inside the
  window and genuinely predictive, but they encode clinician concern rather than
  physiology. Disclose them; a model intended to stand on physiology alone should
  exclude them.
* **Static baseline risk** — `anchor_age`, `admission_type_*`, `admission_location_*`.
  Stable, interpretable, and unsurprising.

What should *not* appear is anything describing the admission as a whole. That is what
the screen in §1 tests, and it is what the July 2026 audit caught: `med_class_opioid`
ranked first in mortality Run C at mean |SHAP| 1.035 — present for 94.8% of deaths
versus 56.6% of survivors, because opioids are given *because* care is being withdrawn.
`sentence_count` ranked fourth for the same reason: the discharge summary is written at
discharge. See §6 of [`data_correction_notice.md`](../data_correction_notice.md).

## 5. Regenerating

```bash
python run_explainability_audit.py                # all phases
python run_explainability_audit.py --phases 1,3   # a subset
```
""", encoding="utf-8")

    print(f"\n{'FLAGGED: ' + str(len(all_flags)) if all_flags else 'Leakage screen: CLEAN'}")
    for n, feat, why, rank in all_flags:
        print(f"  phase {n} rank {rank}: {feat} — {why}")
    print(f"Written → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
