"""
src/llm/payload_validation.py
─────────────────────────────
Deterministic completeness and plausibility gate for unseen-patient payloads.

Design rule: **the LLM never decides whether input is sufficient.** This module
does, before any model or retrieval call, and returns a structured list of what
is missing so the caller can ask the user a precise question.

The old behaviour substituted population-normal values for anything absent
(creatinine 1.0, BUN 15, WBC 8.5, SBP 120), which made an empty payload
indistinguishable from a healthy patient. Nothing here defaults silently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.llm.terminology import normalise_diagnosis, normalise_medications

__all__ = [
    "FieldSpec", "ValidationResult", "validate_payload",
    "REQUIRED_FIELDS", "PLAUSIBLE_RANGES",
]


@dataclass(frozen=True)
class FieldSpec:
    path: str                 # dotted path, e.g. "presentation_labs.creatinine_max"
    prompt: str               # what to ask the user
    unit: str = ""
    required: bool = True


# Minimum viable payload. Anything here missing ⇒ refusal, not imputation.
REQUIRED_FIELDS: Tuple[FieldSpec, ...] = (
    FieldSpec("demographics.age", "Patient age", "years"),
    FieldSpec("demographics.gender", "Patient sex (M/F)"),
    FieldSpec("primary_diagnosis", "Primary working diagnosis"),
    FieldSpec("presentation_labs.creatinine_max", "Peak serum creatinine", "mg/dL"),
    FieldSpec("presentation_labs.bun_max", "Peak BUN", "mg/dL"),
    FieldSpec("presentation_labs.wbc_max", "Peak white cell count", "K/uL"),
    FieldSpec("presentation_labs.bicarbonate_min", "Lowest serum bicarbonate", "mEq/L"),
    FieldSpec("presentation_labs.sodium_min", "Lowest serum sodium", "mEq/L"),
    FieldSpec("presentation_labs.potassium_max", "Peak serum potassium", "mEq/L"),
    FieldSpec("presentation_labs.platelets_min", "Lowest platelet count", "K/uL"),
    FieldSpec("presentation_labs.hematocrit_min", "Lowest haematocrit", "%"),
    FieldSpec("presentation_labs.glucose_max", "Peak glucose", "mg/dL"),
    FieldSpec("vital_signs.sbp_min", "Lowest systolic BP", "mmHg"),
    FieldSpec("vital_signs.hr_max", "Peak heart rate", "bpm"),
)

RECOMMENDED_FIELDS: Tuple[FieldSpec, ...] = (
    FieldSpec("presentation_labs.anion_gap_max", "Peak anion gap", "mEq/L", required=False),
    FieldSpec("presentation_labs.lactate_max", "Peak lactate", "mmol/L", required=False),
    FieldSpec("vital_signs.spo2_min", "Lowest SpO2", "%", required=False),
    FieldSpec("vital_signs.rr_max", "Peak respiratory rate", "breaths/min", required=False),
    FieldSpec("vital_signs.temperature_max", "Peak temperature", "degC", required=False),
    FieldSpec("comorbidities", "Known comorbidities", required=False),
    FieldSpec("active_medications", "Active medication list", required=False),

    # Administrative and historical context. Recommended rather than required: these
    # are cheap for a clinician and free from a FHIR pull, but promoting any of them
    # to required would make every payload written before today refuse, and the
    # completeness gate is the guardrail the whole design leans on.
    #
    # They are here because the boosters were fitted on them and the payload path
    # never asked. Of the mortality model's 164 features, 86 are the one-hot
    # expansions of race, language, insurance, marital status and admission
    # type/location alone — supplying six values determines more than half the
    # feature space, which no additional laboratory value comes close to matching.
    FieldSpec("demographics.race", "Recorded race/ethnicity", required=False),
    FieldSpec("demographics.language", "Primary language", required=False),
    FieldSpec("demographics.insurance", "Insurance type", required=False),
    FieldSpec("demographics.marital_status", "Marital status", required=False),
    FieldSpec("admission_context.admission_type",
              "Admission type (e.g. EW EMER., ELECTIVE, URGENT)", required=False),
    FieldSpec("admission_context.admission_location",
              "Admission source (e.g. EMERGENCY ROOM, PHYSICIAN REFERRAL)",
              required=False),

    # Expansion A&D. Phase 2 is built on prior utilisation, so a payload without it
    # asks the readmission model to predict readmission with the patient's own
    # readmission history withheld.
    FieldSpec("prior_utilisation.admissions_30d",
              "Admissions in the past 30 days", "count", required=False),
    FieldSpec("prior_utilisation.admissions_90d",
              "Admissions in the past 90 days", "count", required=False),
    FieldSpec("prior_utilisation.admissions_365d",
              "Admissions in the past 365 days", "count", required=False),
    FieldSpec("prior_utilisation.cumulative_los_days",
              "Total prior inpatient days", "days", required=False),
)

# (low, high) physiologically survivable bounds — outside ⇒ reject as implausible.
PLAUSIBLE_RANGES: Dict[str, Tuple[float, float]] = {
    "demographics.age": (0.0, 120.0),
    "presentation_labs.creatinine_max": (0.1, 30.0),
    "presentation_labs.bun_max": (1.0, 300.0),
    "presentation_labs.wbc_max": (0.0, 200.0),
    "presentation_labs.bicarbonate_min": (2.0, 60.0),
    "presentation_labs.sodium_min": (90.0, 200.0),
    "presentation_labs.potassium_max": (1.0, 12.0),
    "presentation_labs.platelets_min": (0.0, 2000.0),
    "presentation_labs.hematocrit_min": (5.0, 75.0),
    "presentation_labs.glucose_max": (10.0, 2000.0),
    "presentation_labs.anion_gap_max": (0.0, 60.0),
    "presentation_labs.lactate_max": (0.0, 40.0),
    "vital_signs.sbp_min": (30.0, 300.0),
    "vital_signs.hr_max": (10.0, 300.0),
    "vital_signs.spo2_min": (10.0, 100.0),
    "vital_signs.rr_max": (2.0, 80.0),
    "vital_signs.temperature_max": (25.0, 45.0),
}

# alternate spellings accepted for each canonical lab key
_LAB_SYNONYMS = {
    "creatinine_max": ("creatinine_max", "lab_creatinine_max", "creatinine", "cr_max", "scr_max"),
    "bun_max": ("bun_max", "lab_bun_max", "bun", "urea_max"),
    "wbc_max": ("wbc_max", "lab_wbc_max", "wbc", "leukocytes_max", "white_cell_count"),
    "bicarbonate_min": ("bicarbonate_min", "lab_bicarbonate_min", "bicarbonate", "hco3_min", "co2_min"),
    "sodium_min": ("sodium_min", "lab_sodium_min", "sodium", "na_min"),
    "potassium_max": ("potassium_max", "lab_potassium_max", "potassium", "k_max"),
    "platelets_min": ("platelets_min", "lab_platelets_min", "platelets", "plt_min"),
    "hematocrit_min": ("hematocrit_min", "lab_hematocrit_min", "haematocrit_min", "hct_min"),
    "glucose_max": ("glucose_max", "lab_glucose_max", "glucose", "bg_max"),
    "anion_gap_max": ("anion_gap_max", "lab_anion_gap_max", "anion_gap", "ag_max"),
    "lactate_max": ("lactate_max", "lab_lactate_max", "lactate"),
}


@dataclass
class ValidationResult:
    ok: bool
    missing_required: List[Dict[str, str]] = field(default_factory=list)
    missing_recommended: List[Dict[str, str]] = field(default_factory=list)
    implausible: List[Dict[str, Any]] = field(default_factory=list)
    uninterpretable: List[Dict[str, Any]] = field(default_factory=list)
    unmatched_medications: List[str] = field(default_factory=list)
    diagnosis: Dict[str, Any] = field(default_factory=dict)
    normalised: Dict[str, Any] = field(default_factory=dict)
    completeness: float = 0.0

    def question_for_user(self) -> str:
        """Human-readable request for the specific information that is missing."""
        if self.ok and not self.implausible and not self.uninterpretable:
            return ""
        lines: List[str] = []
        if self.missing_required:
            lines.append("I cannot assess this patient yet. Please provide:")
            for m in self.missing_required:
                unit = f" ({m['unit']})" if m.get("unit") else ""
                lines.append(f"  • {m['prompt']}{unit}")
        if self.implausible:
            lines.append("These values are outside physiologically possible ranges "
                         "— please confirm or correct:")
            for m in self.implausible:
                lines.append(f"  • {m['path']} = {m['value']} "
                             f"(expected {m['low']}–{m['high']})")
        if self.uninterpretable:
            lines.append("These values could not be read as numbers:")
            for m in self.uninterpretable:
                lines.append(f"  • {m['path']} = {m['value']!r}")
        if not self.diagnosis.get("concept"):
            raw = self.diagnosis.get("raw") or "(none given)"
            sugg = self.diagnosis.get("suggestions") or []
            if sugg:
                from src.llm.terminology import CONCEPTS
                names = [CONCEPTS[s].display for s in sugg if s in CONCEPTS]
                lines.append(
                    f"  • I could not confidently map the diagnosis {raw!r}. "
                    f"Did you mean one of: {', '.join(names)}? "
                    "Please confirm or restate it."
                )
            else:
                lines.append(
                    f"  • A recognisable primary diagnosis — {raw!r} did not map to any "
                    "clinical concept I hold evidence for."
                )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "completeness": round(self.completeness, 3),
            "missing_required": self.missing_required,
            "missing_recommended": self.missing_recommended,
            "implausible": self.implausible,
            "uninterpretable": self.uninterpretable,
            "unmatched_medications": self.unmatched_medications,
            "diagnosis": self.diagnosis,
        }


def _dig(payload: Any, path: str) -> Tuple[bool, Any]:
    """Fetch a dotted path, tolerating lab-key synonyms. Returns (found, value)."""
    if not isinstance(payload, dict):
        return False, None
    parts = path.split(".")
    node: Any = payload
    for i, p in enumerate(parts):
        if not isinstance(node, dict):
            return False, None
        if p in node:
            node = node[p]
            continue
        # last segment: try synonyms
        if i == len(parts) - 1:
            for canon, alts in _LAB_SYNONYMS.items():
                if p == canon:
                    for a in alts:
                        if a in node:
                            return True, node[a]
            return False, None
        return False, None
    return True, node


def _as_float(value: Any) -> Tuple[bool, Optional[float]]:
    if isinstance(value, bool):
        return False, None
    if isinstance(value, (int, float)):
        f = float(value)
        return (False, None) if math.isnan(f) or math.isinf(f) else (True, f)
    if isinstance(value, str):
        s = value.strip().replace(",", "")
        try:
            f = float(s)
        except ValueError:
            return False, None
        return (False, None) if math.isnan(f) or math.isinf(f) else (True, f)
    return False, None


def validate_payload(payload: Any, strict: bool = True) -> ValidationResult:
    """
    Validate an unseen-patient payload.

    Parameters
    ----------
    payload : dict
        Raw payload as supplied by the caller.
    strict : bool
        When True (default) any missing required field makes ``ok`` False.

    Returns
    -------
    ValidationResult
        ``ok`` is True only when every required field is present, numeric and
        physiologically plausible, and the diagnosis maps to a known concept.
    """
    res = ValidationResult(ok=False)

    if not isinstance(payload, dict):
        res.missing_required = [
            {"path": f.path, "prompt": f.prompt, "unit": f.unit} for f in REQUIRED_FIELDS
        ]
        res.diagnosis = normalise_diagnosis(None).to_dict()
        return res

    normalised: Dict[str, Any] = {}
    n_present = 0

    for spec in REQUIRED_FIELDS + RECOMMENDED_FIELDS:
        found, value = _dig(payload, spec.path)
        empty = (not found) or value is None or (isinstance(value, str) and not value.strip()) \
            or (isinstance(value, (list, tuple, dict)) and len(value) == 0)

        if empty:
            entry = {"path": spec.path, "prompt": spec.prompt, "unit": spec.unit}
            (res.missing_required if spec.required else res.missing_recommended).append(entry)
            continue

        n_present += 1

        if spec.path in PLAUSIBLE_RANGES:
            ok_num, f = _as_float(value)
            if not ok_num:
                res.uninterpretable.append({"path": spec.path, "value": value})
                continue
            low, high = PLAUSIBLE_RANGES[spec.path]
            if f < low or f > high:
                res.implausible.append(
                    {"path": spec.path, "value": f, "low": low, "high": high}
                )
                continue
            normalised[spec.path] = f
        else:
            normalised[spec.path] = value

    dx = normalise_diagnosis(payload.get("primary_diagnosis"))
    res.diagnosis = dx.to_dict()

    drugs = normalise_medications(payload.get("active_medications"))
    res.unmatched_medications = [d.raw for d in drugs if not d.matched]
    normalised["medications"] = [d.to_dict() for d in drugs if d.matched]

    total = len(REQUIRED_FIELDS) + len(RECOMMENDED_FIELDS)
    res.completeness = n_present / total if total else 0.0
    res.normalised = normalised

    res.ok = (
        not res.missing_required
        and not res.implausible
        and not res.uninterpretable
        and dx.matched
    ) if strict else True

    return res
