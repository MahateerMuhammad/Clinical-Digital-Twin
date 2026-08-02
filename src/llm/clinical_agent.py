"""
src/llm/clinical_agent.py
─────────────────────────
Agent facade for a patient **already in the cohort**, addressed by ``hadm_id``.

Relationship to the other agent
───────────────────────────────
``clinical_assistant.EnterpriseClinicalAgent`` serves *unseen* patients supplied as a
payload (Phase 11). This class serves cohort admissions, where the full feature row
can be looked up — so it has access to everything the models were trained on, and to
real Phase 7 twin retrieval, neither of which a payload can provide.

Why this file was rewritten
───────────────────────────
It was a fourth parallel implementation of work already done correctly elsewhere, and
every one of its tools was broken:

* ``tool_rag_search`` called ``rag_store.search_patient_rag`` — a method that exists
  nowhere in the codebase, so the tool raised AttributeError on first use.
* ``tool_retrieve_twins`` returned ``sim_df[sim_df.subject_id != sub_id].head(top_k)``:
  the first rows of the file in storage order, identical for every patient. The 32-d
  embedding was loaded and never used.
* ``tool_check_organ_toxicity`` read ``lab_creatinine_max``, ``lab_bun_max`` and
  ``med_class_antibiotic``. The first two are whole-admission aggregates removed by
  the Run C leakage filter, so both always fell back to their normal defaults (1.0 and
  15.0) and **no alert could ever fire**.
* Its alerts carried ``[NIH DailyMed Package Insert: ENOXAPARIN]`` next to a specific
  dose instruction — a citation for a document never retrieved, attached to a drug
  dose. That is the most dangerous line in the module and it is gone.
* ``execute_agentic_workflow`` assembled a clinical report with no grounding
  verification of any kind.

Each tool now delegates to the tested implementation: twin retrieval and SHAP to
``ClinicalPromptBuilder``, evidence to the RAG engine's real entry point, and the
composed report to ``ClinicalReportPipeline``, which is fail-closed.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath('.'))

import pandas as pd

from src.llm.prompt_builder import ClinicalPromptBuilder

#: Admission-row column -> payload field, for the fields the RAG engine requires.
#: Only leak-free windowed columns appear here; see the module docstring for what
#: reading the whole-admission names silently did.
ROW_TO_PAYLOAD = {
    "creatinine_max":  ["lab_creatinine_first_24h"],
    "bun_max":         ["lab_bun_first_24h"],
    "wbc_max":         ["lab_wbc_last_24h", "lab_wbc_first_24h"],
    "bicarbonate_min": ["lab_bicarbonate_first_24h", "lab_bicarbonate_last_24h"],
    "sodium_min":      ["lab_sodium_first_24h", "lab_sodium_last_24h"],
    "potassium_max":   ["lab_potassium_max_24h", "lab_potassium_last_24h"],
    "platelets_min":   ["lab_platelets_first_24h"],
    "hematocrit_min":  ["lab_hematocrit_first_24h", "lab_hematocrit_last_24h"],
    "glucose_max":     ["lab_glucose_max_24h", "lab_glucose_last_24h"],
}

MED_CLASS_NAMES = {
    "med_class_antibiotic": "antibiotic",
    "med_class_anticoagulant": "anticoagulant",
    "med_class_opioid": "opioid",
    "med_class_insulin": "insulin",
    "med_class_statin": "statin",
    "med_class_beta_blocker": "beta blocker",
    "med_class_ace_inhibitor": "ACE inhibitor",
}


class EnterpriseClinicalAgent:
    """Tool-calling agent over a cohort admission."""

    def __init__(self, data_dir: str = "data/processed",
                 models_dir: str = "models/best_models"):
        self.data_dir = data_dir
        self.builder = ClinicalPromptBuilder(data_dir=data_dir)
        self.runner = self.builder.runner
        self.__icu: Optional[pd.DataFrame] = None
        if models_dir != "models/best_models":
            from src.llm.model_runner import LiveModelRunner
            self.runner = LiveModelRunner(data_dir=data_dir, models_dir=models_dir)
            self.builder.runner = self.runner

    # ── Tool 1: multi-task inference ────────────────────────────────────────
    def tool_run_all_models(self, hadm_id) -> Dict[str, Any]:
        return self.runner.run_live_inference(self.runner.get_patient_row(hadm_id))

    # ── Tool 2: local SHAP ──────────────────────────────────────────────────
    def tool_explain_shap(self, hadm_id, task: str = "mortality",
                          top_k: int = 5) -> Dict[str, Any]:
        """
        Local SHAP drivers, via the prompt builder's implementation.

        The previous version built its own design matrix without sanitising spaces to
        underscores, which is how multi-word categories such as ``EW EMER.`` silently
        became zero columns.
        """
        row = self.runner.get_patient_row(hadm_id)
        drivers = self.builder._shap_drivers(row, task=task, top_k=top_k)
        return {
            "task": task,
            "top_shap_features": [{
                "feature": d["feature"],
                "value": d["value"],
                "shap_impact": d["shap"],
                "direction": ("Associated with increased model-predicted risk"
                              if d["shap"] > 0 else
                              "Associated with decreased model-predicted risk"),
            } for d in drivers],
        }

    # ── Tool 3: Phase 7 digital twins ───────────────────────────────────────
    def tool_retrieve_twins(self, hadm_id, top_k: int = 5) -> List[dict]:
        """True nearest neighbours in the 32-d embedding, excluding the same patient."""
        try:
            return self.builder.get_digital_twins(hadm_id, top_k=top_k)
        except (ValueError, KeyError, FileNotFoundError):
            # Embeddings absent (Phase 7 not run since the last dataset rebuild).
            # Returning [] lets the caller omit the tier; it must not invent twins.
            return []

    # ── Tool 4: evidence retrieval ──────────────────────────────────────────
    @property
    def _icu_vitals(self) -> pd.DataFrame:
        """ICU vital-sign aggregates, indexed by hadm_id. Loaded once, on demand."""
        if self.__icu is None:
            cols = ["hadm_id", "vital_sbp_min", "vital_heart_rate_max",
                    "vital_resp_rate_max", "vital_spo2_min", "vital_temperature_c_max"]
            path = os.path.join(self.data_dir, "icu_level.parquet")
            try:
                icu = pd.read_parquet(path, columns=cols)
            except Exception:
                icu = pd.DataFrame(columns=cols)
            if not icu.empty:
                icu = icu.groupby("hadm_id").agg({
                    "vital_sbp_min": "min", "vital_heart_rate_max": "max",
                    "vital_resp_rate_max": "max", "vital_spo2_min": "min",
                    "vital_temperature_c_max": "max"})
            self.__icu = icu
        return self.__icu

    def build_payload(self, hadm_id, diagnosis: Optional[str] = None,
                      vitals: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Project an admission row into the payload shape the RAG engine validates.

        Vital signs are recorded per ICU stay, so they exist for the ~16% of
        admissions that reach an ICU and for no others. Where they are absent the
        payload is left genuinely incomplete and the pipeline refuses, naming the
        missing fields — the system's first design requirement. Filling them with
        population-normal constants would satisfy the validator while describing a
        patient nobody measured.
        """
        row = self.runner.get_patient_row(hadm_id)
        labs: Dict[str, float] = {}
        for field, candidates in ROW_TO_PAYLOAD.items():
            for col in candidates:
                v = pd.to_numeric(row.get(col), errors="coerce")
                if pd.notna(v):
                    labs[field] = round(float(v), 2)
                    break

        meds = [name for col, name in MED_CLASS_NAMES.items()
                if pd.to_numeric(row.get(col, 0), errors="coerce") == 1]

        vs: Dict[str, float] = {}
        key = int(hadm_id)
        if key in self._icu_vitals.index:
            v = self._icu_vitals.loc[key]
            for field, col in (("sbp_min", "vital_sbp_min"),
                               ("hr_max", "vital_heart_rate_max"),
                               ("rr_max", "vital_resp_rate_max"),
                               ("spo2_min", "vital_spo2_min"),
                               ("temp_max", "vital_temperature_c_max")):
                val = pd.to_numeric(v.get(col), errors="coerce")
                if pd.notna(val):
                    vs[field] = round(float(val), 2)
        if vitals:
            vs.update(vitals)

        payload: Dict[str, Any] = {
            "primary_diagnosis": diagnosis or "inpatient admission",
            "demographics": {
                "age": float(pd.to_numeric(row.get("anchor_age"), errors="coerce") or 65),
                "gender": str(row.get("gender", "M")),
            },
            "presentation_labs": labs,
            "active_medications": meds,
        }
        if vs:
            payload["vital_signs"] = vs
        return payload

    def tool_rag_search(self, hadm_id, diagnosis: Optional[str] = None,
                        top_k: int = 3) -> Dict[str, Any]:
        """
        Retrieve tiered evidence for this admission.

        Routed through ``search_unseen_patient_rag`` — the engine's real entry point.
        The former ``search_patient_rag`` does not exist.
        """
        from src.llm.rag_corpus import rag_store
        return rag_store.search_unseen_patient_rag(
            self.build_payload(hadm_id, diagnosis), top_k=top_k,
            case_id=f"hadm_{int(hadm_id)}", require_complete=False)

    # ── Tool 5: counterfactual ──────────────────────────────────────────────
    def tool_simulate_counterfactual(self, hadm_id, modifications: dict) -> Dict[str, Any]:
        return self.runner.simulate_what_if_unseen_patient(
            self.build_payload(hadm_id), modifications)

    # ── Tool 6: organ-function flags ────────────────────────────────────────
    def tool_check_organ_toxicity(self, hadm_id) -> List[dict]:
        """
        Flag renal impairment alongside nephrotoxic-class exposure.

        These are **observations about this admission's recorded values**, not dosing
        advice. The previous version emitted "reduce Enoxaparin to 30mg daily" under a
        DailyMed citation for a document it had not retrieved — a fabricated reference
        attached to a specific drug dose. Dosing guidance must come from retrieved
        evidence (`tool_rag_search` returns the KDIGO drug-dosing section for exactly
        this presentation), never from a literal in this file.
        """
        row = self.runner.get_patient_row(hadm_id)

        def num(*cols):
            for c in cols:
                v = pd.to_numeric(row.get(c), errors="coerce")
                if pd.notna(v):
                    return float(v)
            return None

        creat = num("lab_creatinine_first_24h")
        bun = num("lab_bun_first_24h")
        on_abx = pd.to_numeric(row.get("med_class_antibiotic", 0), errors="coerce") == 1

        alerts: List[dict] = []
        if creat is None and bun is None:
            return [{
                "type": "RENAL_ASSESSMENT_UNAVAILABLE",
                "severity": "info",
                "message": "No creatinine or BUN recorded within the first 24 hours; "
                           "renal status could not be assessed from this admission.",
            }]

        if (creat is not None and creat > 3.0) or (bun is not None and bun > 80.0):
            parts = []
            if creat is not None:
                parts.append(f"creatinine {creat:.1f} mg/dL")
            if bun is not None:
                parts.append(f"BUN {bun:.1f} mg/dL")
            alerts.append({
                "type": "SEVERE_RENAL_IMPAIRMENT",
                "severity": "critical",
                "message": "Markedly impaired renal function in the first 24 hours ("
                           + ", ".join(parts)
                           + "). Renally-cleared drug doses warrant review against "
                             "current guidance.",
            })

        if on_abx and creat is not None and creat > 2.5:
            alerts.append({
                "type": "NEPHROTOXIC_EXPOSURE_IN_RENAL_IMPAIRMENT",
                "severity": "warning",
                "message": f"Antibiotic class recorded with creatinine {creat:.1f} mg/dL. "
                           "Agents cleared renally may accumulate.",
            })

        return alerts

    # ── composed workflow ───────────────────────────────────────────────────
    def execute_agentic_workflow(self, hadm_id, diagnosis: Optional[str] = None,
                                 use_llm: bool = False) -> str:
        """
        Run every tool and compose a report **through the fail-closed pipeline**.

        The former version concatenated tool output into markdown with no verification,
        so a fabricated citation or an ungrounded number reached the reader unchecked.
        The narrative report is now produced by ``ClinicalReportPipeline``, which
        withholds anything not traceable to the payload, the model outputs or a
        retrieved document; the tool outputs are appended as a structured appendix.
        """
        from src.llm.pipeline import ClinicalReportPipeline

        payload = self.build_payload(hadm_id, diagnosis)

        # Predict from the full admission row, not the payload. The payload exists to
        # drive evidence retrieval and validation; it carries a fraction of the
        # trained features, and the pipeline rightly withholds predictions below that
        # coverage floor. A cohort admission has every feature the models were fitted
        # on, so there is no reason to discard them here.
        preds = self.tool_run_all_models(hadm_id)
        result = ClinicalReportPipeline().generate(
            payload, case_id=f"agent_{int(hadm_id)}", use_llm=use_llm,
            predictions=preds)
        shap_out = self.tool_explain_shap(hadm_id, top_k=5)
        twins = self.tool_retrieve_twins(hadm_id, top_k=5)
        alerts = self.tool_check_organ_toxicity(hadm_id)

        # An empty narrative is what a refusal looks like from the pipeline; say so
        # rather than emitting a report that opens with blank lines.
        narrative = result.report_markdown.rstrip()
        if not narrative:
            missing = ((result.validation or {}).get("missing")
                       if isinstance(result.validation, dict) else None)
            narrative = (
                "# Narrative report withheld\n\n"
                f"The pipeline returned `{result.status}` for this admission, so no "
                "prose summary was composed. The structured tool output below is "
                "still derived entirely from recorded values."
                + (f"\n\nMissing required fields: {', '.join(missing)}."
                   if missing else ""))

        ground = ("passed" if result.grounding.get("ok")
                  else "not assessed" if result.status == "incomplete_input"
                  else "FAILED")
        lines = [narrative, "",
                 "---", "", "## Appendix — tool outputs", "",
                 f"**Admission:** {int(hadm_id)} | **Risk tier:** {preds['risk_tier']} | "
                 f"**Status:** {result.status} | **Grounding:** {ground}",
                 "", "### Local SHAP drivers"]
        for s in shap_out["top_shap_features"]:
            lines.append(f"- `{s['feature']}` = {s['value']:.4g}: "
                         f"{s['shap_impact']:+.4f} ({s['direction']})")

        lines += ["", f"### Phase 7 digital twins (n={len(twins)})"]
        if twins:
            died = sum(t["hospital_expire_flag"] for t in twins)
            icu = sum(t["has_icu_stay"] for t in twins)
            lines.append(f"- observed deaths {died}/{len(twins)}, "
                         f"ICU transfers {icu}/{len(twins)}")
            lines.append("- twin outcomes are observed precedent, not a prediction")
        else:
            lines.append("- unavailable: no Phase 7 embedding for this admission")

        lines += ["", "### Organ-function flags"]
        lines += ([f"- **{a['type']}** ({a['severity']}): {a['message']}" for a in alerts]
                  or ["- none"])

        return "\n".join(lines) + "\n"


if __name__ == "__main__":
    agent = EnterpriseClinicalAgent()
    target = int(sys.argv[1]) if len(sys.argv) > 1 else int(agent.builder.sim_df["hadm_id"].iloc[0])
    print(agent.execute_agentic_workflow(target, diagnosis="acute kidney injury"))
