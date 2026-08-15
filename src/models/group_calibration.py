"""
src/models/group_calibration.py
───────────────────────────────
Age-band calibration on top of the global isotonic calibrators.

Why this exists
───────────────
``reports/slice_evaluation.md`` measured what the headline ECE of 0.0036 was
hiding: for patients aged 85 and over the mortality model observes 5.13% and
predicts 3.95%, an ECE of 0.0118 — more than three times the cohort figure, and
in the direction that matters. The oldest, sickest group receives the most
optimistic estimate.

What this is, and is not
────────────────────────
It is **not retraining**. The boosters are untouched. This fits a second
isotonic regression per age band on the *validation* split — exactly what
``calibrate_predictions`` already does globally, and on the same data — then
applies it at serve time when the patient's age is known. Nothing about the
model's ranking changes; only the mapping from score to probability does.

Age was chosen because it is the only strong-disparity dimension that is
**always available at inference**. Race showed a larger gap, but race is an
optional payload field: a calibrator keyed on it would apply to a minority of
real requests and, worse, would silently treat "the clinician did not type it"
as the same state as "the hospital could not record it". Those are different
patients and conflating them is how the disparity got there in the first place.

Fail-open by design
───────────────────
An absent artefact, an unknown age, a band with no fitted calibrator, or any
error during transform all fall back to the global calibrator. A calibration
refinement must never be able to take prediction down — it is an improvement to
a number that was already being served, not a new dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = ["AGE_BANDS", "age_band", "GroupCalibrators", "DEFAULT_PATH",
           "MIN_FIT_ROWS", "MIN_FIT_EVENTS"]

DEFAULT_PATH = Path("models/best_models/group_calibrators.pkl")

#: Bands match ``scripts/evaluation/run_slice_eval.py`` so the fix is measured
#: on the same partition that found the problem. MIMIC caps recorded age at 91.
AGE_BANDS: Tuple[Tuple[int, int], ...] = ((18, 39), (40, 54), (55, 69),
                                          (70, 84), (85, 120))

#: Fitting floors, deliberately stricter than the reporting floors. A poorly
#: supported calibrator is worse than none: it is applied to every future
#: patient in that band, whereas an unmeasured slice merely goes unreported.
MIN_FIT_ROWS = 1000
MIN_FIT_EVENTS = 50


def age_band(age: Any) -> Optional[str]:
    """Band label for an age, or None when it is unknown or out of range."""
    try:
        value = float(age)
    except (TypeError, ValueError):
        return None
    if value != value:                                  # NaN
        return None
    for lo, hi in AGE_BANDS:
        if lo <= value <= hi:
            return f"{lo}-{hi}" if hi < 120 else f"{lo}+"
    return None


@dataclass
class GroupCalibrators:
    """Per-task, per-age-band isotonic calibrators."""

    by_task: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    # ── loading ──
    @classmethod
    def load(cls, path: Optional[Path] = None) -> "GroupCalibrators":
        """Load the artefact, or return an empty instance if it is absent."""
        p = Path(path) if path else DEFAULT_PATH
        if not p.exists():
            return cls()
        try:
            import joblib

            blob = joblib.load(p)
            return cls(by_task=blob.get("by_task", {}),
                       meta=blob.get("meta", {}))
        except Exception:
            # A corrupt artefact must not stop the runner from serving the
            # globally calibrated number it served yesterday.
            return cls()

    def save(self, path: Optional[Path] = None) -> Path:
        import joblib

        p = Path(path) if path else DEFAULT_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"by_task": self.by_task, "meta": self.meta}, p)
        return p

    # ── use ──
    @property
    def available(self) -> bool:
        return bool(self.by_task)

    def bands_for(self, task: str) -> List[str]:
        return sorted(self.by_task.get(task, {}))

    def calibrate(self, task: str, raw_prob: float,
                  *, age: Any = None) -> Optional[float]:
        """
        Band-specific calibrated probability, or None to use the global one.

        None is returned for every reason a band calibrator cannot be trusted —
        no artefact, unknown age, no calibrator fitted for that band, or a
        transform that raised. The caller keeps its existing behaviour in all of
        them.
        """
        band = age_band(age)
        if not band:
            return None
        cal = (self.by_task.get(task) or {}).get(band)
        if cal is None or not hasattr(cal, "predict"):
            return None
        try:
            return float(np.clip(float(cal.predict([raw_prob])[0]),
                                 0.0001, 0.9999))
        except Exception:
            return None

    def describe(self) -> Dict[str, Any]:
        return {"available": self.available,
                "tasks": {t: sorted(b) for t, b in self.by_task.items()},
                "meta": self.meta}
