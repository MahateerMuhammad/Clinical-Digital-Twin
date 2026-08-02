#!/usr/bin/env python3
"""
run_phase11_eval.py
───────────────────
Evaluate Phase 11 — the unseen-patient clinical agent (``Engine 2``).

Phase 11 differs from every earlier phase in its input: it is handed a *payload*
describing a patient who is not in the cohort, rather than a `hadm_id` whose full
feature row can be looked up. That difference is the whole point of the phase, and it
is also its central limitation, so this harness measures it rather than assuming it
away.

What is checked
───────────────
1. **Feature coverage.** How much of each model a payload can actually populate.
   Everything else is zero-filled, and a zero-filled feature is indistinguishable
   from a real zero once the matrix is built.
2. **SHAP faithfulness.** Laboratory drivers must carry the payload's own values.
   This is the regression test for the defect that motivated the harness: every lab
   field mapped onto whole-admission column names that Run C removes, so every value
   entered the model as 0.0 and the agent explained a patient made entirely of zeros.
3. **Counterfactual connectivity.** Normalising deranged labs must change the
   prediction. The same broken mapping made every counterfactual return a delta of
   exactly 0.0, which reads as "this intervention has no effect" rather than "this
   input was never wired up" — the more dangerous of the two readings.
4. **Counterfactual directionality.** Normalising deranged labs must not *raise*
   predicted risk. This is an association check on a supervised model, not a causal
   claim; see the disclaimer the simulator itself emits.
5. **Evidence and grounding.** Retrieval must return documents and the composed
   report must pass fail-closed verification.

Usage
─────
    python run_phase11_eval.py
    python run_phase11_eval.py --repeats 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "reports" / "tables" / "phase11_clinical_agent_evaluation.md"

NORMAL = {
    "creatinine_max": 1.0, "bun_max": 15.0, "wbc_max": 8.0, "bicarbonate_min": 24.0,
    "sodium_min": 139.0, "potassium_max": 4.2, "platelets_min": 240.0,
    "hematocrit_min": 41.0, "glucose_max": 100.0, "anion_gap_max": 11.0,
    "chloride_max": 102.0,
}

VITALS_NORMAL = {"sbp_min": 118.0, "hr_max": 78.0, "rr_max": 16.0,
                 "spo2_min": 98.0, "temp_max": 36.9}

#: Clinical phenotypes with their characteristic derangements, and the fields a
#: clinician would expect to normalise under effective treatment.
PHENOTYPES: List[Dict[str, Any]] = [
    {"name": "Septic shock", "diagnosis": "sepsis",
     "labs": {"wbc_max": 24.0, "bicarbonate_min": 14.0, "creatinine_max": 2.6,
              "bun_max": 48.0, "anion_gap_max": 22.0, "platelets_min": 88.0},
     "vitals": {"sbp_min": 78.0, "hr_max": 128.0, "rr_max": 30.0, "temp_max": 39.1},
     "meds": ["vancomycin", "piperacillin"],
     "treat": ["wbc_max", "bicarbonate_min", "anion_gap_max"]},
    {"name": "Stage 3 AKI", "diagnosis": "acute kidney injury",
     "labs": {"creatinine_max": 5.2, "bun_max": 94.0, "potassium_max": 6.1,
              "bicarbonate_min": 15.0, "anion_gap_max": 20.0},
     "vitals": {"sbp_min": 104.0, "hr_max": 92.0},
     "meds": ["furosemide"],
     "treat": ["creatinine_max", "bun_max", "potassium_max", "bicarbonate_min"]},
    {"name": "Diabetic ketoacidosis", "diagnosis": "diabetic ketoacidosis",
     "labs": {"glucose_max": 540.0, "bicarbonate_min": 8.0, "anion_gap_max": 28.0,
              "potassium_max": 5.6, "creatinine_max": 1.8},
     "vitals": {"hr_max": 118.0, "rr_max": 28.0},
     "meds": ["insulin"],
     "treat": ["glucose_max", "bicarbonate_min", "anion_gap_max"]},
    {"name": "Decompensated heart failure", "diagnosis": "heart failure",
     "labs": {"creatinine_max": 2.2, "bun_max": 52.0, "sodium_min": 128.0,
              "hematocrit_min": 32.0},
     "vitals": {"sbp_min": 96.0, "spo2_min": 88.0, "rr_max": 26.0},
     "meds": ["furosemide", "enoxaparin"],
     "treat": ["creatinine_max", "bun_max", "sodium_min"]},
    {"name": "Severe pneumonia", "diagnosis": "pneumonia",
     "labs": {"wbc_max": 19.5, "sodium_min": 130.0, "glucose_max": 180.0,
              "hematocrit_min": 34.0},
     "vitals": {"spo2_min": 85.0, "rr_max": 32.0, "temp_max": 38.9},
     "meds": ["ceftriaxone"],
     "treat": ["wbc_max", "sodium_min"]},
    {"name": "Upper GI haemorrhage", "diagnosis": "gastrointestinal bleeding",
     "labs": {"hematocrit_min": 19.0, "platelets_min": 74.0, "bun_max": 62.0,
              "creatinine_max": 1.9},
     "vitals": {"sbp_min": 82.0, "hr_max": 126.0},
     "meds": ["pantoprazole"],
     "treat": ["hematocrit_min", "platelets_min", "bun_max"]},
]


def build_payload(spec: Dict[str, Any], age: int, sex: str) -> Dict[str, Any]:
    labs = dict(NORMAL)
    labs.update(spec["labs"])
    vitals = dict(VITALS_NORMAL)
    vitals.update(spec.get("vitals", {}))
    return {
        "primary_diagnosis": spec["diagnosis"],
        "demographics": {"age": float(age), "gender": sex},
        "presentation_labs": labs,
        "vital_signs": vitals,
        "active_medications": list(spec.get("meds", [])),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=3,
                    help="age/sex variants per phenotype")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    from src.llm.clinical_assistant import EnterpriseClinicalAgent
    from src.llm.pipeline import ClinicalReportPipeline

    rng = np.random.default_rng(args.seed)
    agent = EnterpriseClinicalAgent()
    pipe = ClinicalReportPipeline()

    rows, coverage, shap_faithful, cf_connected, cf_directional = [], [], [], [], []
    evidence_found, grounded = [], []

    print(f"Evaluating {len(PHENOTYPES)} phenotypes x {args.repeats} variants ...\n")
    for spec in PHENOTYPES:
        deltas, tiers = [], []
        for r in range(args.repeats):
            age = int(rng.integers(45, 88))
            sex = str(rng.choice(["M", "F"]))
            payload = build_payload(spec, age, sex)

            cov = agent.runner.payload_feature_coverage(payload, "mortality")
            coverage.append(cov)

            preds = agent.tool_run_all_models(payload)

            # 2. SHAP drivers that name a lab must carry that lab's supplied value.
            shap_out = agent.tool_explain_shap(payload, top_k=10)
            lab_drivers = [d for d in shap_out["top_shap_features"]
                           if d["feature"].startswith("lab_")
                           and d["feature"].endswith("_24h")]
            supplied = agent.runner._convert_payload_to_series(payload)
            faithful = all(
                abs(d["value"] - float(supplied.get(d["feature"], 0.0))) < 1e-6
                and not (d["value"] == 0.0 and d["feature"] in supplied.index
                         and supplied[d["feature"]] != 0.0)
                for d in lab_drivers) if lab_drivers else False
            shap_faithful.append(bool(faithful))

            # 3/4. Normalise the treatable derangements.
            mods = {f: NORMAL[f] for f in spec["treat"]}
            cf = agent.tool_simulate_counterfactual(payload, mods)
            d = cf["deltas"]
            moved = any(abs(d[k]) > 1e-9 for k in
                        ("delta_p_mortality", "delta_p_icu_admission",
                         "delta_p_deterioration"))
            cf_connected.append(moved)
            cf_directional.append(d["delta_p_mortality"] <= 1e-9)
            deltas.append(d["delta_p_mortality"])
            tiers.append((d["base_tier"], d["mod_tier"]))

            # 5. Evidence + grounding through the verified pipeline.
            res = pipe.generate(payload, case_id=f"p11_{spec['diagnosis']}_{r}",
                                use_llm=False)
            evidence_found.append(len(res.documents) > 0)
            grounded.append(bool(res.grounding.get("ok", False)))

        tier_moves = sum(1 for a, b in tiers if a != b)
        rows.append({
            "name": spec["name"], "dx": spec["diagnosis"],
            "p_mort": preds["p_mortality"], "tier": preds["risk_tier"],
            "d_mort": float(np.mean(deltas)), "tier_moves": tier_moves,
            "n": args.repeats,
        })
        print(f"  {spec['name']:30} p_mort {preds['p_mortality']*100:5.2f}%  "
              f"Δ {np.mean(deltas)*100:+6.2f}pp  tier moved {tier_moves}/{args.repeats}")

    n = len(coverage)
    cov_mean = float(np.mean(coverage))
    r_shap = float(np.mean(shap_faithful))
    r_conn = float(np.mean(cf_connected))
    r_dir = float(np.mean(cf_directional))
    r_evid = float(np.mean(evidence_found))
    r_grnd = float(np.mean(grounded))

    print(f"\n{n} payloads evaluated\n")
    print(f"  mortality-model feature coverage   {cov_mean:.1%}")
    print(f"  SHAP drivers faithful to payload   {r_shap:.1%}")
    print(f"  counterfactual connected           {r_conn:.1%}")
    print(f"  counterfactual directionally sane  {r_dir:.1%}")
    print(f"  evidence retrieved                 {r_evid:.1%}")
    print(f"  reports grounded                   {r_grnd:.1%}")

    checks = [("SHAP faithfulness", r_shap, 1.0),
              ("Counterfactual connectivity", r_conn, 1.0),
              ("Counterfactual direction", r_dir, 0.90),
              ("Evidence retrieval", r_evid, 0.90),
              ("Grounding", r_grnd, 1.0)]
    failed = [f"{k} {v:.1%} < {t:.0%}" for k, v, t in checks if v < t]
    verdict = ("**PASS** — every requirement met." if not failed
               else "**FAIL** — " + "; ".join(failed))
    print(f"\n{verdict}\n")

    table = "\n".join(
        f"| {r['name']} | `{r['dx']}` | {r['p_mort']*100:.2f}% | {r['tier'].split(':')[0]} | "
        f"{r['d_mort']*100:+.2f} pp | {r['tier_moves']}/{r['n']} |" for r in rows)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(f"""# Phase 11 — Unseen-Patient Clinical Agent Evaluation

> [!NOTE]
> Generated by `run_phase11_eval.py`. Regenerate after any Phase 1-5 retrain, since
> the agent calls the promoted models directly.

## 1. What Phase 11 is

Engine 2 of the platform: a clinical agent that accepts a **payload** for a patient
who is not in the cohort, rather than a `hadm_id`. It runs the five models, explains
the mortality prediction with SHAP, retrieves grounded evidence, and simulates
counterfactuals over the supplied physiology.

{len(PHENOTYPES)} clinical phenotypes x {args.repeats} age/sex variants = {n} payloads.

## 2. Results

| Check | Result | Threshold |
| :--- | ---: | ---: |
| SHAP drivers faithful to the payload | **{r_shap:.1%}** | 100% |
| Counterfactual connected to the model | **{r_conn:.1%}** | 100% |
| Counterfactual directionally sane | **{r_dir:.1%}** | 90% |
| Evidence retrieved | **{r_evid:.1%}** | 90% |
| Reports pass fail-closed grounding | **{r_grnd:.1%}** | 100% |

{verdict}

## 3. Per-phenotype behaviour

Counterfactuals normalise the treatable derangements of each phenotype — the fields a
clinician would expect to correct — and report the mean change in predicted mortality.

| Phenotype | Diagnosis | Predicted mortality | Tier | Δ after normalisation | Tier changed |
| :--- | :--- | ---: | :---: | ---: | ---: |
{table}

## 4. The limitation that defines this phase

**A payload populates {cov_mean:.1%} of the mortality model's features.** The
remaining {1 - cov_mean:.1%} are zero-filled: `diagnosis_count`, `procedure_count`,
the admission-type and admission-location dummies, and the per-analyte draw counts
and missing-ratios. None of them can be derived from a payload, because they describe
an admission that has not happened yet.

This is not a defect to be fixed — it is what predicting for an unseen patient means.
But it has two consequences that must travel with any Phase 11 output:

* `diagnosis_count` is consistently the largest SHAP driver, at a zero-filled value.
  The model is partly responding to the *absence* of admission history rather than to
  the physiology supplied.
* Absolute probabilities from a payload are not comparable with those from a cohort
  admission, which populates every feature. The tiering is still meaningful *within*
  payload-based predictions; it should not be read across the two regimes.

## 5. Why the checks in §2 exist

Each corresponds to a defect found in the agent when this harness was first run:

* **SHAP faithfulness** — every laboratory field mapped onto whole-admission column
  names (`lab_creatinine_max`, `lab_bicarbonate_min`, nine more) that Run C removes as
  observation-window leakage. Every lookup missed, so every lab entered the model as
  0.0. The agent produced confident predictions and SHAP explanations for a patient
  made entirely of zeros.
* **Counterfactual connectivity** — the same mapping meant modifications never reached
  the feature vector, so every simulation returned a delta of exactly 0.0. Presented
  to a clinician, that reads as "this intervention would not help", which is the
  opposite of "this input was never connected".

Both are now regression-checked here and in `tests/test_phase11_agent.py`.

## 6. Interpretation

Counterfactuals are associations learned by a supervised model, not treatment effects.
The simulator emits that disclaimer with every result and it is repeated here: a
negative delta means patients with those values were observed to have lower risk, not
that achieving them would lower this patient's risk.
""", encoding="utf-8")
    print(f"Report → {OUT}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
