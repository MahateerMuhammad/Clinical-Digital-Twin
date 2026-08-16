"""
src/llm/drivers.py
──────────────────
Which inputs moved a risk estimate, via SHAP.

Local attributions for one patient against one booster, computed with
``shap.TreeExplainer``. Returns [] rather than inventing values when shap or the
model is unavailable — the predecessor of this code printed four hardcoded
literals (+0.45, +0.38, +0.29, +0.18) under a "LOCAL SHAP" heading for every
patient regardless of their data, and a reader had no way to tell.

The distinction this module exists to preserve
──────────────────────────────────────────────
A gradient-boosted tree routes a *missing* value down a default branch, so
absence carries weight of its own. On a presentation payload, where 82% of the
model's features are unsupplied, several of the largest attributions are the
model reacting to what it was not told.

Reported without that distinction, ``diagnosis_count +0.77`` reads as a finding
about the patient. It is not: it is the model's response to not knowing their
diagnosis count. Every driver therefore carries ``supplied``, and a caller that
shows drivers is expected to show it.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

__all__ = ["Driver", "risk_drivers", "readable_feature", "is_indicator"]


@dataclass(frozen=True)
class Driver:
    feature: str
    label: str
    #: Signed SHAP value in log-odds. Positive raises the estimated risk.
    contribution: float
    #: The value the model saw, or None when the feature was not supplied.
    value: Optional[float]
    supplied: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


#: Booster feature names are pipeline-internal (`lab_wbc_last_24h`). These are
#: the words a clinician uses. Anything unmapped is de-prefixed and spaced
#: rather than dropped: an unlabelled driver is still a true one.
_LABELS: Dict[str, str] = {
    "anchor_age": "Age",
    "gender_F": "Sex: female",
    "gender_M": "Sex: male",
    "diagnosis_count": "Number of coded diagnoses",
    "unique_diagnosis_count": "Distinct coded diagnoses",
    "procedure_count": "Number of coded procedures",
    "has_major_procedure": "Major procedure recorded",
    "charlson_comorbidity_index": "Charlson comorbidity index",
    "n_icu_stays": "Prior ICU stays",
    "los_days": "Length of stay",
}

_STEMS = (
    ("lab_creatinine", "Creatinine"), ("lab_bun", "BUN"),
    ("lab_wbc", "White cells"), ("lab_bicarbonate", "Bicarbonate"),
    ("lab_sodium", "Sodium"), ("lab_potassium", "Potassium"),
    ("lab_platelets", "Platelets"), ("lab_hematocrit", "Haematocrit"),
    ("lab_glucose", "Glucose"), ("lab_lactate", "Lactate"),
    ("lab_anion_gap", "Anion gap"),
    ("vital_sbp", "Systolic BP"), ("vital_hr", "Heart rate"),
    ("vital_spo2", "SpO2"), ("vital_rr", "Respiratory rate"),
    ("vital_temperature", "Temperature"),
)

_WINDOWS = (("_first_24h", ", first 24h"), ("_last_24h", ", last 24h"),
            ("_mean_24h", ", mean"), ("_min_24h", ", lowest"),
            ("_max_24h", ", peak"))


#: One-hot columns. Their value is 1.0 or 0.0, which is meaningless printed
#: beside the label ("Sex: female — 1"), so a caller shows the label alone.
_INDICATOR_PREFIXES = ("admission_type_", "admission_location_", "insurance_",
                       "race_", "language_", "marital_status_", "gender_")


def is_indicator(name: str) -> bool:
    """Is this feature a 0/1 dummy rather than a measured quantity?"""
    return (name.startswith(_INDICATOR_PREFIXES)
            or name in ("has_major_procedure",))


#: MIMIC-IV admission-type codes. Title-casing them produced "Eu Observation"
#: and "Ew Emer." in a clinician's report — abbreviations that mean nothing
#: outside the source database, mangled further by the generic formatter.
_ADMISSION_TYPES = {
    "EU OBSERVATION": "emergency-unit observation",
    "EW EMER.": "emergency admission",
    "DIRECT EMER.": "direct emergency admission",
    "OBSERVATION ADMIT": "observation admission",
    "DIRECT OBSERVATION": "direct observation",
    "AMBULATORY OBSERVATION": "ambulatory observation",
    "ELECTIVE": "elective",
    "URGENT": "urgent",
    "SURGICAL SAME DAY ADMISSION": "same-day surgical admission",
}


def readable_feature(name: str) -> str:
    """A clinician-facing name for a booster feature."""
    if name in _LABELS:
        return _LABELS[name]

    if name.startswith("admission_type_"):
        raw = name[len("admission_type_"):]
        known = _ADMISSION_TYPES.get(raw.upper().replace("_", " "))
        return f"Admitted as: {known or raw.replace('_', ' ').lower()}"

    for prefix, word in (("admission_location_", "Admission from"),
                         ("insurance_", "Insurance"), ("race_", "Race"),
                         ("language_", "Language"),
                         ("marital_status_", "Marital status")):
        if name.startswith(prefix):
            return f"{word}: {name[len(prefix):].replace('_', ' ').title()}"

    for stem, word in _STEMS:
        if name.startswith(stem):
            tail = name[len(stem):]
            for suffix, phrase in _WINDOWS:
                if tail.endswith(suffix):
                    return f"{word}{phrase}"
            return word

    return name.replace("_", " ").capitalize()


def risk_drivers(model: Any, features: pd.DataFrame, top_k: int = 8) -> List[Driver]:
    """
    The ``top_k`` features that moved this prediction furthest, either way.

    ``features`` is one row already aligned to the model's feature order — the
    same frame the prediction was computed from, so the attributions describe
    the number actually served rather than a re-encoding of it.

    Returns [] on any failure. An explanation is a convenience; a prediction
    without one is still valid, and taking the turn down because shap raised
    would trade a working answer for a missing footnote.
    """
    if model is None or features is None or len(features) != 1:
        return []
    try:
        import shap
    except ImportError:
        return []

    try:
        names = list(features.columns)
        values = shap.TreeExplainer(model).shap_values(features)
        # A binary LightGBM classifier returns [negative, positive]; the
        # positive class is the one whose probability is reported.
        if isinstance(values, list):
            values = values[-1]
        values = np.asarray(values).reshape(-1)
        if values.shape[0] != len(names):
            return []
    except Exception:
        return []

    row = features.iloc[0]
    order = np.argsort(-np.abs(values))[:top_k]

    out: List[Driver] = []
    for i in order:
        raw = row.iloc[int(i)]
        supplied = bool(pd.notna(raw))
        out.append(Driver(
            feature=names[int(i)],
            label=readable_feature(names[int(i)]),
            # Rounded to the precision the report prints. The grounding
            # verifier matches digits, so storing 4dp and rendering 3dp
            # withholds the report for quoting its own arithmetic.
            contribution=round(float(values[int(i)]), 3),
            value=round(float(raw), 4) if supplied else None,
            supplied=supplied,
        ))
    return out
