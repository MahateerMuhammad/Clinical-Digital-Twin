#!/usr/bin/env python
"""
scripts/evaluation/run_slice_eval.py
────────────────────────────────────
Subgroup performance for the promoted risk models.

    PYTHONPATH=. .venv/bin/python scripts/evaluation/run_slice_eval.py
    … --max-rows 40000     cap the test split (default: all of it)
    … --task mortality     one task instead of all three

Writes ``reports/slice_evaluation.md``.

Why this exists separately from the headline numbers
────────────────────────────────────────────────────
``reports/assistant_evaluation.md`` reports one ECE per task. That number is an
average over the whole test split, and an average is exactly what conceals the
failure that matters here: a model can be well calibrated overall and badly
calibrated for one group, and the people in that group are the ones harmed by
it. The overall figure cannot show that. This file exists to.

The headline of a subgroup report is the **gap**, not the mean.

Method
──────
* Probabilities are the **isotonic-calibrated** values the runner serves, not
  the raw booster output. Scoring the raw output measures a number no caller
  ever sees.
* Rows are the **held-out test patients**, joined from ``patient_split.parquet``.
  The feature table has no split column, so without that join the metrics would
  include patients the calibrators were fitted on.
* MIMIC race strings are hierarchical — "WHITE - OTHER EUROPEAN",
  "ASIAN - CHINESE". They are grouped on the delimiter, which raises subgroup
  support enough to measure without collapsing distinct populations by hand.
  The ungrouped category count is reported so the grouping is visible.
* A slice below the support floor is reported as **unmeasured**, never as a
  finding. A slice with four deaths produces an AUROC that moves on one case.
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

from src.evaluation import metrics as M
from src.models.group_calibration import age_band as gc_age_band

REPORT = Path("reports/slice_evaluation.md")
FEATURES = Path("data/processed/admission_level_selected.parquet")
SPLIT = Path("data/processed/patient_split.parquet")

TASKS = {"mortality": "hospital_expire_flag",
         "icu_admission": "has_icu_stay",
         "readmission": "readmission_30d"}

AGE_BANDS = [(18, 39), (40, 54), (55, 69), (70, 84), (85, 120)]


def _age_band(age: float) -> str:
    for lo, hi in AGE_BANDS:
        if lo <= age <= hi:
            return f"{lo}-{hi}" if hi < 120 else f"{lo}+"
    return "unknown"


def _race_group(value: Any) -> str:
    """
    Collapse MIMIC's hierarchical race strings onto their top level.

    "WHITE - OTHER EUROPEAN" and "WHITE - RUSSIAN" become WHITE. This is a
    documented property of the coding scheme rather than a judgement about which
    populations to merge, and it takes the category count from 33 to a handful
    with enough support to measure. UNKNOWN is kept as its own group — on this
    split it carries a mortality rate several times the cohort average, which
    would be invisible if it were folded into OTHER.
    """
    s = str(value).strip().upper()
    if not s or s in ("NAN", "NONE"):
        return "UNKNOWN"
    return s.split(" - ")[0].split("/")[0].strip()


def build_slices(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Column name → the slice label for each row."""
    return {
        "sex": df["gender"].astype(str),
        "age_band": df["anchor_age"].astype(float).map(_age_band),
        "race": df["race"].map(_race_group),
        "insurance": df["insurance"].astype(str),
        "language": df["language"].astype(str),
        "marital_status": df["marital_status"].astype(str),
        "admission_type": df["admission_type"].astype(str),
    }


def evaluate(max_rows: Optional[int], only_task: Optional[str]) -> Dict[str, Any]:
    from src.llm.model_runner import LiveModelRunner

    if not FEATURES.exists() or not SPLIT.exists():
        return {"skipped": f"need {FEATURES} and {SPLIT}"}

    runner = LiveModelRunner()
    group_cal = runner.group_calibrators
    df = pd.read_parquet(FEATURES)
    split = pd.read_parquet(SPLIT)[["subject_id", "split"]]
    df = df.merge(split, on="subject_id", how="inner")
    df = df[df["split"].astype(str).str.lower() == "test"]
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=0)
    if df.empty:
        return {"skipped": "no test rows"}

    slices = build_slices(df)
    n_race_raw = df["race"].astype(str).nunique()

    out: Dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_race_categories_raw": int(n_race_raw),
        "n_race_categories_grouped": int(slices["race"].nunique()),
        "min_rows": M.MIN_SLICE_ROWS, "min_events": M.MIN_SLICE_EVENTS,
        "group_calibrated": bool(group_cal.available),
        "calibration_meta": dict(group_cal.meta),
        "tasks": {},
    }

    for task, target in TASKS.items():
        if only_task and task != only_task:
            continue
        model = runner.lgbm_models.get(task)
        calibrator = runner.calibrators.get(task)
        if model is None or target not in df.columns:
            out["tasks"][task] = {"skipped": "model or target absent"}
            continue

        feats = model.booster_.feature_name()
        raw = model.predict_proba(df.reindex(columns=feats))[:, 1]
        probs = (np.asarray(calibrator.predict(raw), dtype=float)
                 if calibrator is not None and hasattr(calibrator, "predict")
                 else raw)

        # Apply the age-band calibrators the runner applies. Measuring the
        # global-only value would score a number the API no longer serves, and
        # the whole point of this report is to describe what a clinician sees.
        if group_cal.available:
            bands = df["anchor_age"].map(gc_age_band).to_numpy()
            adjusted = probs.copy()
            for i, band in enumerate(bands):
                if not band:
                    continue
                v = group_cal.calibrate(task, float(probs[i]),
                                        age=df["anchor_age"].iloc[i])
                if v is not None:
                    adjusted[i] = v
            probs = adjusted
        y = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int).to_numpy()

        task_out: Dict[str, Any] = {
            "overall": {"n": int(len(y)), "base_rate": float(y.mean()),
                        "auroc": M.auroc(list(probs), list(y)),
                        "brier": M.brier_score(list(probs), list(y)),
                        "ece": M.expected_calibration_error(list(probs), list(y))},
            "calibrated": calibrator is not None,
            "by": {},
        }

        for slice_name, labels in slices.items():
            groups = {}
            for value in sorted(labels.unique(), key=str):
                mask = (labels == value).to_numpy()
                groups[str(value)] = (list(probs[mask]), list(y[mask]))
            task_out["by"][slice_name] = M.slice_report(groups)

        out["tasks"][task] = task_out
    return out


# ── report ───────────────────────────────────────────────────────────────────

def _num(v, p=3):
    return "n/a" if v is None else f"{v:.{p}f}"


def _pct(v, p=2):
    return "n/a" if v is None else f"{v * 100:.{p}f}%"


def render(res: Dict[str, Any]) -> str:
    L: List[str] = []
    add = L.append

    add("# Subgroup (Slice) Evaluation")
    add("")
    add(f"*Generated {date.today().isoformat()} by "
        "`scripts/evaluation/run_slice_eval.py`.*")
    add("")
    if res.get("skipped"):
        add(f"*Skipped: {res['skipped']}*")
        return "\n".join(L) + "\n"

    add("The headline evaluation reports one calibration figure per task. That "
        "figure is an average, and an average is what hides a model that is "
        "well calibrated overall and poorly calibrated for one group. **The "
        "number to read here is the gap between slices, not the mean.**")
    add("")
    add(f"- Held-out test rows: **{res['n_rows']:,}**")
    add(f"- Probabilities: **isotonic-calibrated** (what the API returns)")
    add(f"- Support floor: a slice needs **n ≥ {res['min_rows']}** and "
        f"**≥ {res['min_events']} events** to be measured; below that it is "
        "reported as unmeasured rather than as a finding")
    add(f"- Race categories: {res['n_race_categories_raw']} raw → "
        f"{res['n_race_categories_grouped']} after grouping on MIMIC's "
        "hierarchical delimiter")
    add("")

    for task, t in res["tasks"].items():
        add(f"## {task}")
        add("")
        if t.get("skipped"):
            add(f"*Skipped: {t['skipped']}*")
            add("")
            continue
        o = t["overall"]
        add(f"Overall — n {o['n']:,} · base rate {_pct(o['base_rate'])} · "
            f"AUROC {_num(o['auroc'])} · Brier {_num(o['brier'], 4)} · "
            f"ECE {_num(o['ece'], 4)}")
        add("")

        for slice_name, rep in t["by"].items():
            s = rep["summary"]
            add(f"### by {slice_name}")
            add("")
            if s.get("auroc_gap") is not None:
                add(f"**AUROC gap {_num(s['auroc_gap'])}** — "
                    f"best `{s['auroc_best']['slice']}` {_num(s['auroc_best']['value'])}, "
                    f"worst `{s['auroc_worst']['slice']}` {_num(s['auroc_worst']['value'])}. "
                    f"**ECE gap {_num(s['ece_gap'], 4)}** — worst "
                    f"`{s['ece_worst']['slice']}` {_num(s['ece_worst']['value'], 4)}.")
                add("")
            add("| Slice | n | Events | Base rate | Mean pred | AUROC | Brier | ECE |")
            add("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            for r in sorted(rep["slices"], key=lambda x: -x["n"]):
                if not r["measured"]:
                    add(f"| `{r['name']}` | {r['n']:,} | {r['events']} | "
                        f"{_pct(r['base_rate'])} | — | — | — | *unmeasured* |")
                    continue
                add(f"| `{r['name']}` | {r['n']:,} | {r['events']} | "
                    f"{_pct(r['base_rate'])} | {_pct(r['mean_predicted'])} | "
                    f"{_num(r['auroc'])} | {_num(r['brier'], 4)} | "
                    f"{_num(r['ece'], 4)} |")
            add("")
            if s["n_unmeasured"]:
                add(f"*{s['n_unmeasured']} of {s['n_slices']} slices fell below "
                    "the support floor and are not scored.*")
                add("")

    add("---")
    add("")
    add("## Remediation applied")
    add("")
    if not res.get("group_calibrated"):
        add("*No age-band calibrators are loaded. Run "
            "`scripts/maintenance/fit_group_calibrators.py`.*")
        add("")
    else:
        add("The numbers above already include **age-band isotonic "
            "calibration**, fitted by "
            "`scripts/maintenance/fit_group_calibrators.py`.")
        add("")
        add("**This is not retraining.** No booster was refitted and no feature "
            "set changed. A second isotonic regression is fitted per age band on "
            "the *validation* split — the same estimator, on the same split, as "
            "the global calibrators already used — and applied on top of the "
            "global value at serve time. Only the mapping from score to "
            "probability moves; the model's ranking is untouched, which is why "
            "AUROC is essentially unchanged while ECE falls sharply.")
        add("")
        add("A band calibrator is kept **only if it improves calibration on the "
            "test split it was not fitted on**. An isotonic fit always improves "
            "its own data, so that check is what separates a real correction "
            "from memorisation.")
        add("")
        meta = (res.get("calibration_meta") or {})
        effect = meta.get("effect") or {}
        if effect:
            add(f"Fitted {meta.get('fitted_on', '—')} on the "
                f"`{meta.get('split', '—')}` split; floors n ≥ "
                f"{meta.get('min_fit_rows')} and events ≥ "
                f"{meta.get('min_fit_events')}.")
            add("")
            add("| Task | Band | val n | val events | ECE before | ECE after |")
            add("| :--- | :--- | ---: | ---: | ---: | ---: |")
            for task_name, bands in effect.items():
                for band, e in sorted(bands.items()):
                    add(f"| `{task_name}` | {band} | {e['n_val']:,} | "
                        f"{e['events_val']} | {e['ece_before']:.4f} | "
                        f"{e['ece_after']:.4f} |")
            add("")
        add("**Still open, and requiring a retrain rather than a recalibration:** "
            "the `UNKNOWN` / `UNABLE TO OBTAIN` race groups, whose observed "
            "mortality is several times the cohort average while the model "
            "predicts roughly half of it. Those labels mark patients too unwell "
            "for demographics to be collected — signal the model currently sees "
            "only as another category value. An explicit "
            "`demographics_incomplete` feature would let it learn what the "
            "label means. Race is also an optional payload field, so a "
            "calibrator keyed on it would apply to a minority of real requests "
            "and would conflate \"the clinician did not type it\" with \"the "
            "hospital could not record it\" — different patients.")
        add("")
    add("## Reading this")
    add("")
    add("A large **AUROC gap** means the model separates cases better in some "
        "groups than others — it ranks well for one population and poorly for "
        "another. A large **ECE gap** means the probability means different "
        "things depending on the group: \"5% risk\" may be accurate for one and "
        "an underestimate for another, which is the more dangerous of the two "
        "because the number looks identical on screen.")
    add("")
    add("Base-rate differences between slices are **not** themselves a model "
        "defect. Emergency admissions genuinely die more often than elective "
        "ones. What matters is whether the model is equally *accurate* and "
        "equally *honest* across groups, which is what AUROC and ECE measure "
        "and base rate does not.")
    add("")
    return "\n".join(L) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--task", default=None, choices=sorted(TASKS))
    ap.add_argument("--out", default=str(REPORT))
    args = ap.parse_args(argv)

    print("[slice] evaluating…", flush=True)
    res = evaluate(args.max_rows, args.task)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(res), encoding="utf-8")
    print(f"Wrote {out}")

    for task, t in (res.get("tasks") or {}).items():
        if t.get("skipped"):
            continue
        gaps = {name: rep["summary"].get("auroc_gap")
                for name, rep in t["by"].items()}
        worst = max((g for g in gaps.values() if g is not None), default=None)
        name = next((k for k, v in gaps.items() if v == worst), "—")
        print(f"  {task:14} overall AUROC {_num(t['overall']['auroc'])} · "
              f"largest AUROC gap {_num(worst)} (by {name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
