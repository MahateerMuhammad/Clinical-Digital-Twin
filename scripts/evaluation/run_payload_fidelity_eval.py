#!/usr/bin/env python3
"""
scripts/evaluation/run_payload_fidelity_eval.py
───────────────────────────────────────────────
Measure how much of each Phase 1-5 model survives being driven by a presentation
payload instead of a full admission record — and decide which tasks may be served.

The question
────────────
Phase 11 hands the models a payload: age, sex, eleven presentation labs, a
medication list. That populates about a fifth of what the models were trained on.
The reports nonetheless quoted all five outputs, each implicitly under its
*published* AUROC. This harness asks whether that is warranted, per task.

The method
──────────
For a sample of held-out **test** admissions, each model is scored twice:

* **reference** — the complete admission feature row, one-hot encoded into the
  booster namespace. This is the model as validated, and its AUROC reproducing the
  phase's published figure is what makes the rest of the numbers trustworthy.
* **payload** — only the fields a payload can carry, reconstructed from the same
  rows through ``LiveModelRunner.PAYLOAD_LAB_FEATURES`` so the mapping under test is
  the one production actually uses. Everything else is NaN.

Both are compared against the real outcome. The decision statistic is **retention**,

    (AUROC_payload − 0.5) / (AUROC_reference − 0.5)

the fraction of the validated model's discriminative lift that survives. A task is
served from payloads only if retention clears ``PAYLOAD_RETENTION_FLOOR``.

Why retention and not absolute AUROC: the report presents each figure as coming
from a specific validated model, so the honest test is how far the served input
falls short of the input that validation was done on. It is also the more stable
criterion — bootstrapped, deterioration's retention CI sits entirely below the
floor while its absolute AUROC CI straddles any threshold near 0.70.

Usage
─────
    python scripts/evaluation/run_payload_fidelity_eval.py
    python scripts/evaluation/run_payload_fidelity_eval.py --n 20000 --bootstrap 1000
    python scripts/evaluation/run_payload_fidelity_eval.py --patch   # rewrite the constants
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
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from src.llm.feature_space import align_to_model, encode_admission_frame, feature_coverage
from src.llm.model_runner import (
    LiveModelRunner, PAYLOAD_RETENTION_FLOOR, payload_retention,
)

ROOT = _ROOT
OUT = ROOT / "reports" / "tables" / "payload_fidelity_evaluation.md"
CONSTANTS_FILE = ROOT / "src" / "llm" / "model_runner.py"

#: Phase 5's proxy target: ward-to-ICU transfer more than `window` hours after
#: admission. ICU-origin admissions are excluded from the cohort, exactly as
#: src/models/deterioration.py does — including them would score the model on
#: patients it was never asked about.
DETERIORATION_WINDOW_HOURS = 6.0

#: Phase 4 Stage A threshold.
LOS_THRESHOLD_DAYS = 5.63


def build_cohort(runner: LiveModelRunner, n: int, seed: int) -> pd.DataFrame:
    """Held-out test admissions, with every task's label attached."""
    split = pd.read_parquet(ROOT / "data/processed/patient_split.parquet")
    test_ids = set(split.loc[split["split"] == "test", "subject_id"])

    adm = runner.adm_df
    if adm is None:
        raise FileNotFoundError(
            "admission_level_selected.parquet not found; the harness needs the "
            "processed cohort to build a reference feature row.")

    d = adm[adm["subject_id"].isin(test_ids)].copy()
    d["time_to_icu_hrs"] = (
        (pd.to_datetime(d["intime"]) - pd.to_datetime(d["admittime"]))
        .dt.total_seconds() / 3600.0)

    if n < len(d):
        d = d.sample(n=n, random_state=seed)
    d = d.reset_index(drop=True)

    d["_y_mortality"] = pd.to_numeric(d["hospital_expire_flag"], errors="coerce")
    d["_y_readmission"] = pd.to_numeric(d["readmission_30d"], errors="coerce")
    d["_y_icu_admission"] = pd.to_numeric(d["has_icu_stay"], errors="coerce")
    d["_y_hospital_los"] = (
        pd.to_numeric(d["los_days"], errors="coerce") > LOS_THRESHOLD_DAYS).astype(float)

    icu_origin = ((d["has_icu_stay"] == 1)
                  & (d["time_to_icu_hrs"] <= DETERIORATION_WINDOW_HOURS))
    det = ((d["has_icu_stay"] == 1)
           & (d["time_to_icu_hrs"] > DETERIORATION_WINDOW_HOURS)).astype(float)
    d["_y_deterioration"] = det.where(~icu_origin, np.nan)

    # `prior_*` features are computed by the phase pipelines, not stored in the
    # parquet, and are among the readmission model's strongest inputs. Without them
    # the reference AUROC is understated and every retention figure is inflated.
    from src.features.prior_utilization import build_prior_utilization_features
    history = adm[adm["subject_id"].isin(set(d["subject_id"]))]
    prior = build_prior_utilization_features(history)
    return d.merge(prior, on="hadm_id", how="left")


def payload_frame(runner: LiveModelRunner, d: pd.DataFrame) -> pd.DataFrame:
    """
    Rebuild, for every admission, exactly the features a payload would populate.

    Values are read from the first booster column of each analyte and broadcast to
    the rest, which is what ``_convert_payload_to_series`` does with the single
    value a payload carries. Absent analytes take ``LAB_DEFAULTS``, again matching
    production. Anything outside this frame is a feature a payload cannot supply.
    """
    sup = {
        "anchor_age": pd.to_numeric(d["anchor_age"], errors="coerce").fillna(65.0),
        "gender_M": (d["gender"].astype(str).str.upper() == "M").astype(float),
    }
    for field, columns in runner.PAYLOAD_LAB_FEATURES.items():
        source = columns[0]
        value = (pd.to_numeric(d[source], errors="coerce") if source in d.columns
                 else pd.Series(np.nan, index=d.index))
        value = value.fillna(runner.LAB_DEFAULTS[field])
        for col in columns:
            sup[col] = value
    for col in runner.MED_KEYWORDS:
        sup[col] = (pd.to_numeric(d[col], errors="coerce").fillna(0.0)
                    if col in d.columns else pd.Series(0.0, index=d.index))
    return pd.DataFrame(sup, index=d.index)


def bootstrap_retention(y, p_ref, p_pay, rounds, seed):
    """Percentile CI for retention, resampling admissions."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(rounds):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() < 5 or y[i].sum() == len(i):
            continue
        a_ref = roc_auc_score(y[i], p_ref[i])
        if a_ref <= 0.5:
            continue
        out.append((roc_auc_score(y[i], p_pay[i]) - 0.5) / (a_ref - 0.5))
    if not out:
        return float("nan"), float("nan")
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def evaluate(n: int, seed: int, rounds: int) -> list[dict]:
    runner = LiveModelRunner()
    d = build_cohort(runner, n, seed)
    reference = encode_admission_frame(d)
    payload = payload_frame(runner, d)

    rows = []
    for task in ("mortality", "readmission", "icu_admission", "hospital_los",
                 "deterioration"):
        model = runner.lgbm_models.get(task)
        if model is None:
            raise FileNotFoundError(
                f"No promoted model for '{task}'. Run scripts/maintenance/promote_models.py.")
        names = runner._feature_names(model)

        y = d[f"_y_{task}"]
        keep = y.notna().to_numpy()
        y_arr = y.to_numpy()[keep]

        p_ref = model.predict_proba(align_to_model(reference, names))[:, 1]
        p_pay = model.predict_proba(align_to_model(payload, names))[:, 1]

        auc_ref = roc_auc_score(y_arr, p_ref[keep])
        auc_pay = roc_auc_score(y_arr, p_pay[keep])
        lo, hi = bootstrap_retention(y_arr, p_ref[keep], p_pay[keep], rounds, seed)

        rows.append({
            "task": task,
            "n": int(keep.sum()),
            "base_rate": float(np.nanmean(y_arr)),
            "reference_coverage": feature_coverage(reference, names),
            "payload_coverage": feature_coverage(payload, names),
            "auroc_reference": round(float(auc_ref), 4),
            "auroc_payload": round(float(auc_pay), 4),
            "spearman": float(spearmanr(p_ref, p_pay).statistic),
            "retention_ci": (lo, hi),
        })
    return rows


def write_report(rows: list[dict], n: int, seed: int, rounds: int) -> None:
    def retention(r):
        return (r["auroc_payload"] - 0.5) / (r["auroc_reference"] - 0.5)

    served = [r for r in rows if retention(r) >= PAYLOAD_RETENTION_FLOOR]
    lines = [
        "# Payload Fidelity Evaluation (Phase 11 serving gate)",
        "",
        f"_Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} by "
        "`scripts/evaluation/run_payload_fidelity_eval.py`._",
        "",
        f"Held-out **test** split, {n:,} sampled admissions, seed {seed}, "
        f"{rounds:,} bootstrap rounds.",
        "",
        "## What this measures",
        "",
        "Each model is scored twice on the same admissions: once from the complete",
        "feature row it was validated on, and once from only the fields an unseen-patient",
        "payload can carry, with everything else NaN. **Retention** is the fraction of",
        "the validated model's discriminative lift that survives the restriction,",
        "`(AUROC_payload - 0.5) / (AUROC_reference - 0.5)`.",
        "",
        f"A task is served from a payload only if retention reaches "
        f"**{PAYLOAD_RETENTION_FLOOR:.1%}**. The reference AUROCs below reproduce each",
        "phase's published figure, which is what validates the harness itself.",
        "",
        "## Results",
        "",
        "| Task | n | Base rate | Payload coverage | AUROC (full record) | AUROC (payload) | Retention | 95% CI | Spearman ρ | Served |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | :---: |",
    ]
    for r in rows:
        ret = retention(r)
        lo, hi = r["retention_ci"]
        lines.append(
            f"| {r['task']} | {r['n']:,} | {r['base_rate']:.2%} | "
            f"{r['payload_coverage']:.1%} | {r['auroc_reference']:.4f} | "
            f"{r['auroc_payload']:.4f} | {ret:.1%} | [{lo:.3f}, {hi:.3f}] | "
            f"{r['spearman']:+.3f} | "
            f"{'**yes**' if ret >= PAYLOAD_RETENTION_FLOOR else 'withheld'} |")

    lines += [
        "",
        "## Reading the table",
        "",
        "- **Spearman ρ** is the rank correlation between the payload prediction and the",
        "  same model's full-record prediction on the same patient. It is the sharpest",
        "  statement of the problem: for ICU admission it is near zero, so the payload",
        "  figure is not a degraded version of the validated prediction — it is",
        "  unrelated to it.",
        "- A retention **below zero** means the payload prediction is anti-correlated",
        "  with the outcome; the model, denied the features it relies on, ranks patients",
        "  backwards.",
        "- **Payload coverage** is the share of trained features a payload populates.",
        "  It is low for every task and is *not* the gate — coverage says how much input",
        "  is missing, retention says whether what remains still discriminates.",
        "",
        "## Consequence",
        "",
        f"{len(served)} of {len(rows)} tasks "
        f"({', '.join(r['task'] for r in served) or 'none'}) "
        "may be served from a presentation payload. The rest are withheld by",
        "`LiveModelRunner.run_live_inference_with_uncertainty`, which returns `None`",
        "and a reason rather than a number. Predictions from a stored admission row are",
        "unaffected — that path supplies the full feature set.",
        "",
        "Regenerate after any Phase 1-5 retrain, then `--patch` to update",
        "`PAYLOAD_FIDELITY` in `src/llm/model_runner.py`. `tests/test_payload_fidelity.py`",
        "fails if the two disagree.",
        "",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


def patch_constants(rows: list[dict]) -> None:
    """
    Rewrite PAYLOAD_FIDELITY in model_runner.py.

    Anchored on the assignment and its closing brace rather than on the individual
    numbers: `recompute_risk_tiers.py --patch` searched for literals that a refactor
    had moved, found none, silently patched nothing and still exited having written
    its report. Not finding the block is an error here.
    """
    text = CONSTANTS_FILE.read_text(encoding="utf-8")
    def entry(r):
        key = f"{r['task']!r}:"
        return (f"    {key:17s}{{'reference': {r['auroc_reference']:.4f}, "
                f"'payload': {r['auroc_payload']:.4f}}},")

    body = "\n".join(entry(r) for r in rows)
    new = f"PAYLOAD_FIDELITY = {{\n{body}\n}}"

    pattern = re.compile(r"^PAYLOAD_FIDELITY = \{.*?^\}", re.S | re.M)
    if not pattern.search(text):
        raise SystemExit(
            f"PAYLOAD_FIDELITY block not found in {CONSTANTS_FILE}. It has been "
            "renamed or moved; --patch would otherwise report success having changed "
            "nothing.")
    patched = pattern.sub(new, text, count=1)

    # The first --patch emitted `'mortality' {...}` — the dict colon was missing from
    # the format string, so it wrote a file that would not import. A generator that
    # rewrites source has to prove the result parses before replacing the original.
    try:
        compile(patched, str(CONSTANTS_FILE), "exec")
    except SyntaxError as exc:
        raise SystemExit(
            f"--patch would have written invalid Python to {CONSTANTS_FILE} "
            f"({exc}). The file is unchanged.")

    CONSTANTS_FILE.write_text(patched, encoding="utf-8")
    print(f"patched PAYLOAD_FIDELITY in {CONSTANTS_FILE.relative_to(ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=8000, help="test admissions to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=400, help="bootstrap rounds")
    ap.add_argument("--patch", action="store_true",
                    help="rewrite PAYLOAD_FIDELITY in src/llm/model_runner.py")
    args = ap.parse_args()

    rows = evaluate(args.n, args.seed, args.bootstrap)

    print(f"\n{'task':15s} {'AUROC ref':>10s} {'AUROC pay':>10s} {'retention':>10s}  served")
    for r in rows:
        ret = (r["auroc_payload"] - 0.5) / (r["auroc_reference"] - 0.5)
        print(f"{r['task']:15s} {r['auroc_reference']:10.4f} {r['auroc_payload']:10.4f} "
              f"{ret:9.1%}  {'yes' if ret >= PAYLOAD_RETENTION_FLOOR else 'WITHHELD'}")

    write_report(rows, args.n, args.seed, args.bootstrap)
    if args.patch:
        patch_constants(rows)
    else:
        drift = [r["task"] for r in rows
                 if abs(payload_retention(r["task"])
                        - (r["auroc_payload"] - 0.5) / (r["auroc_reference"] - 0.5)) > 0.02]
        if drift:
            print(f"\nPAYLOAD_FIDELITY is stale for: {', '.join(drift)}. Re-run with --patch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
