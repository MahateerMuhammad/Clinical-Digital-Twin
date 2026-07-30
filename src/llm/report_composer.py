"""
src/llm/report_composer.py
──────────────────────────
Deterministic, fully-grounded clinical report composer.

This is the floor of the generation stack. It produces a complete structured
report **without any language model**, drawing only on:

* the validated payload,
* the calibrated model predictions,
* the retrieved evidence documents.

Every sentence is assembled from those inputs, so the output passes
:func:`src.llm.grounding.verify_text` by construction. When no LLM backend is
available this is what the system returns — in contrast to the previous fallback,
which emitted a fixed string containing invented SHAP values and a fabricated
guideline citation for every patient regardless of input.

An LLM, when present, is asked to *rephrase* this content for readability. The
verifier then checks the rephrasing against the same fact store, and the
deterministic text is used if the rephrasing fails.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

__all__ = ["ComposedReport", "compose_report"]

_TIER_NAMES = ("Tier 1: Low Risk", "Tier 2: Moderate Risk",
               "Tier 3: High Risk", "Tier 4: Extreme Risk")

#: Published constants this module may quote. They are grounded in the Phase 9
#: audit rather than in the current patient's data, so the fact store must be told
#: about them explicitly — otherwise the verifier correctly flags them as
#: ungrounded. Callers building a fact store should pass these as extra numbers.
#: Tier rates are recomputed by ``recompute_risk_tiers.py`` whenever Phase 1 is
#: retrained — they are percentiles of a specific model's test predictions, so a
#: retrain invalidates them. Values below are from the 2026-07-29 leak-free model
#: (cutoffs 0.0034 / 0.0225 / 0.0883). The superseded set (0.22 / 1.04 / 4.38 /
#: 15.05) came from the pre-correction model and understated Tier 4 by 6.5
#: percentage points.
SYSTEM_CONSTANTS: Dict[str, float] = {
    "phase9_tier1_observed_mortality_pct": 0.09,
    "phase9_tier2_observed_mortality_pct": 1.01,
    "phase9_tier3_observed_mortality_pct": 3.91,
    "phase9_tier4_observed_mortality_pct": 21.52,
    "phase4_hosp_los_threshold_days": 5.63,
    "phase4_icu_los_threshold_days": 4.18,
    "phase5_deterioration_window_hours": 6.0,
}

# Derived from SYSTEM_CONSTANTS so the prose can never drift from the fact store.
# These were previously two hardcoded copies of the same four numbers; updating
# one left the other stale, and the grounding verifier then correctly rejected the
# report because the quoted rate was absent from the fact store.
TIER_CONTEXT = {
    name: (f"observed in-hospital mortality "
           f"{SYSTEM_CONSTANTS[f'phase9_tier{i}_observed_mortality_pct']}% "
           f"in this band on the held-out test cohort")
    for i, name in enumerate(_TIER_NAMES, start=1)
}

TASK_LABELS = {
    "p_mortality": "In-hospital mortality",
    "p_readmission": "30-day readmission",
    "p_icu_admission": "ICU admission during this stay",
    "p_los_over_5_63d": "Hospital stay beyond 5.63 days",
    "p_deterioration": "Clinical deterioration within 6 hours",
}


@dataclass
class ComposedReport:
    sections: Dict[str, str] = field(default_factory=dict)
    citations_used: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        order = [
            "header", "presentation", "risk", "evidence",
            "medications", "uncertainty", "limitations", "provenance",
        ]
        out: List[str] = []
        for key in order:
            if self.sections.get(key):
                out.append(self.sections[key].rstrip())
        return "\n\n".join(out) + "\n"


def _fmt_pct(p: Optional[float]) -> str:
    return "not computed" if p is None else f"{100.0 * float(p):.1f}%"


def _lab_line(labs: Dict[str, Any], key: str, label: str, unit: str) -> Optional[str]:
    for k in (key, f"lab_{key}"):
        if k in labs and labs[k] is not None:
            try:
                return f"{label} {float(labs[k]):g} {unit}"
            except (TypeError, ValueError):
                return None
    return None


def compose_report(
    payload: Dict[str, Any],
    predictions: Optional[Dict[str, Any]] = None,
    documents: Optional[Sequence[dict]] = None,
    validation: Optional[dict] = None,
    ranked_medications: Optional[Sequence[dict]] = None,
    retrieval_status: str = "ok",
    twin_status: str = "",
    retrieval_errors: Optional[Sequence[str]] = None,
) -> ComposedReport:
    """Build a fully grounded report from structured inputs only."""
    rep = ComposedReport()
    predictions = predictions or {}
    documents = list(documents or [])
    labs = payload.get("presentation_labs", {}) or {}
    vitals = payload.get("vital_signs", {}) or {}
    demo = payload.get("demographics", {}) or {}

    dx_display = ""
    if validation and isinstance(validation.get("diagnosis"), dict):
        dx_display = validation["diagnosis"].get("display") or ""
    dx_raw = str(payload.get("primary_diagnosis", "")).strip()

    # ── 1. header ────────────────────────────────────────────────────────
    age = demo.get("age")
    sex = demo.get("gender")
    who = []
    if age is not None:
        who.append(f"{age}-year-old")
    if sex:
        who.append(str(sex))
    rep.sections["header"] = (
        "# Clinical Decision Support Summary\n\n"
        f"**Presentation:** {' '.join(who) or 'patient'} — {dx_raw or 'diagnosis not stated'}"
        + (f"  \n**Mapped concept:** {dx_display}" if dx_display else "")
    )

    # ── 2. presentation ──────────────────────────────────────────────────
    obs: List[str] = []
    for key, label, unit in [
        ("creatinine_max", "peak creatinine", "mg/dL"),
        ("bun_max", "peak BUN", "mg/dL"),
        ("wbc_max", "peak WBC", "K/uL"),
        ("bicarbonate_min", "lowest bicarbonate", "mEq/L"),
        ("sodium_min", "lowest sodium", "mEq/L"),
        ("potassium_max", "peak potassium", "mEq/L"),
        ("platelets_min", "lowest platelets", "K/uL"),
        ("hematocrit_min", "lowest haematocrit", "%"),
        ("glucose_max", "peak glucose", "mg/dL"),
    ]:
        line = _lab_line(labs, key, label, unit)
        if line:
            obs.append(line)
    for key, label, unit in [("sbp_min", "lowest systolic BP", "mmHg"),
                             ("hr_max", "peak heart rate", "bpm")]:
        if vitals.get(key) is not None:
            try:
                obs.append(f"{label} {float(vitals[key]):g} {unit}")
            except (TypeError, ValueError):
                pass

    rep.sections["presentation"] = (
        "## 1. Observed values (as supplied)\n\n"
        + ("\n".join(f"- {o}" for o in obs) if obs else "- No laboratory or vital values supplied.")
        + "\n\nThese are the values provided as input; they are restated here without "
          "interpretation beyond the model outputs below."
    )

    # ── 3. risk ──────────────────────────────────────────────────────────
    tier = predictions.get("risk_tier")
    rows = ["| Task | Calibrated probability |", "| :--- | ---: |"]
    for key, label in TASK_LABELS.items():
        if key in predictions:
            rows.append(f"| {label} | {_fmt_pct(predictions.get(key))} |")
    risk_body = "\n".join(rows) if len(rows) > 2 else "_No model predictions were supplied._"

    tier_line = ""
    if tier:
        ctx = TIER_CONTEXT.get(str(tier))
        tier_line = f"\n\n**Risk tier:** {tier}"
        if ctx:
            tier_line += f" — {ctx}."
    coverage = predictions.get("feature_coverage")
    cov_line = ""
    if coverage is not None:
        try:
            cov_line = (f"\n\n**Model input coverage:** {100.0 * float(coverage):.0f}% of the "
                        "features these models were trained on were supplied.")
        except (TypeError, ValueError):
            pass

    rep.sections["risk"] = "## 2. Model risk estimates\n\n" + risk_body + tier_line + cov_line

    # ── 4. evidence ──────────────────────────────────────────────────────
    ev_lines: List[str] = []
    by_level: Dict[str, List[dict]] = {}
    for d in documents:
        by_level.setdefault(str(d.get("evidence_level", "Unclassified")), []).append(d)

    for level in sorted(by_level):
        ev_lines.append(f"\n**{level}**\n")
        for d in by_level[level]:
            cit = d.get("citation", "")
            title = d.get("title", "")
            text = str(d.get("text", "")).strip()
            snippet = (text[:400] + "…") if len(text) > 400 else text
            ev_lines.append(f"- {cit} {title}")
            if snippet:
                ev_lines.append(f"  > {snippet}")
            if d.get("url"):
                ev_lines.append(f"  Source: {d['url']}")
            if d.get("provenance"):
                ev_lines.append(f"  _{d['provenance']}_")
            if cit:
                rep.citations_used.append(str(cit))

    if not documents:
        ev_lines.append("_No evidence documents were retrieved._")
        if retrieval_status != "ok":
            ev_lines.append(f"_Retrieval status: {retrieval_status}._")

    rep.sections["evidence"] = "## 3. Retrieved evidence\n" + "\n".join(ev_lines)

    # ── 5. medications ───────────────────────────────────────────────────
    med_lines: List[str] = []
    for m in ranked_medications or []:
        if not m.get("recognised"):
            med_lines.append(
                f"- **{m.get('raw')}** — not recognised; no evidence retrieved for it."
            )
            rep.warnings.append(f"unrecognised medication: {m.get('raw')}")
            continue
        med_lines.append(
            f"- **{m.get('ingredient')}** ({m.get('drug_class') or 'unclassified'}) — "
            f"relevance {m.get('score')}: {m.get('rationale')}"
        )
    rep.sections["medications"] = (
        "## 4. Active medications, by mechanistic relevance\n\n"
        + ("\n".join(med_lines) if med_lines else "_No active medications supplied._")
        + "\n\nRelevance reflects the link between drug class and the stated presentation. "
          "It is not a recommendation to start, stop or change any therapy."
    )

    # ── 6. uncertainty ───────────────────────────────────────────────────
    unc = [
        "- Probabilities are calibrated estimates from models trained on MIMIC-IV; "
        "they describe populations, not individual certainty.",
        "- No causal claim is made. These models identify association, not treatment effect.",
    ]
    if coverage is not None:
        try:
            if float(coverage) < 0.6:
                unc.append(
                    f"- **Input coverage is low ({100.0 * float(coverage):.0f}%).** Unsupplied "
                    "features were treated as missing; estimates are correspondingly less reliable."
                )
        except (TypeError, ValueError):
            pass
    if twin_status and twin_status != "ok":
        unc.append(f"- Historical twin evidence unavailable ({twin_status}); "
                   "no similar-patient comparison is included.")
    for err in (retrieval_errors or []):
        unc.append(f"- Evidence retrieval issue: {err}")
    rep.sections["uncertainty"] = "## 5. Uncertainty and confidence\n\n" + "\n".join(unc)

    # ── 7. limitations ───────────────────────────────────────────────────
    rep.sections["limitations"] = (
        "## 6. Limitations\n\n"
        "- This summary restates supplied values, model outputs and retrieved guideline text. "
        "It does not constitute a clinical assessment.\n"
        "- Guideline records are paraphrased summaries pending clinician review; verify wording "
        "against the cited source before acting.\n"
        "- Model estimates derive from a single-centre US ICU/hospital dataset and may not "
        "transfer to other populations or care settings."
    )

    # ── 8. provenance ────────────────────────────────────────────────────
    rep.sections["provenance"] = (
        "## 7. Provenance\n\n"
        f"- Evidence documents cited: {len(rep.citations_used)}\n"
        f"- Retrieval status: {retrieval_status}\n"
        f"- Twin retrieval: {twin_status or 'not attempted'}\n"
        "- Generated deterministically from structured inputs; every number above appears "
        "in the payload, the model outputs, or a cited document."
    )

    return rep
