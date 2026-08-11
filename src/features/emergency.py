"""
src/features/emergency.py
─────────────────────────
Emergency-department feature engineering from the MIMIC-IV-ED module.

Why this module exists
──────────────────────
`chartevents` is ICU-only, so before ED data was added the admission-level matrix
carried *no* vital signs at all — every model was scoring physiology through
proxies (labs, comorbidity, prior utilisation). ED triage and serial vitals are
the only source of ward-grade physiology in MIMIC-IV, and they are recorded
*before* the inpatient admission is registered: 99.7% of ED `intime` values
precede `admittime`, median −4.8 hours. They are therefore legitimately
available to an admission-time model.

Coverage is partial and that is the central design constraint
─────────────────────────────────────────────────────────────
The ED module covers 37.1% of this cohort — not the ~69% implied by
`admission_location == 'EMERGENCY ROOM'`, because MIMIC-IV-ED is a separate
partial capture (only 56.5% of ED-located admissions have an ED record). The
remaining 63% get NaN, never 0.0: a missing heart rate is "not observed", and
filling it with a number would tell the model a patient was bradycardic to a
degree incompatible with life.

Because presence is itself informative, this module emits an explicit
``ed_available`` flag rather than leaving availability implicit in the
missingness pattern. That makes the risk *testable* —
:func:`src.features.leakage_filters.check_availability_leakage` can measure
whether "arrived via ED" predicts the outcome on its own, which is exactly the
failure that forced the Phase 5 rebuild.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger
from src.utils.validation import VITAL_RANGES

log = get_logger(__name__)


#: ED column -> the VITAL_RANGES key governing its physiological bounds.
#:
#: Triage and vitalsign both record temperature in Fahrenheit (cohort mean 98.0),
#: so both map to ``temperature_f``. Clipping is not optional here: the raw table
#: contains dbp up to 661,672 and o2sat up to 9,322. A single unclipped outlier
#: moves a mean far enough to dominate a tree split.
ED_VITAL_RANGES: Dict[str, str] = {
    "temperature": "temperature_f",
    "heartrate": "heart_rate",
    "resprate": "resp_rate",
    "o2sat": "spo2",
    "sbp": "sbp",
    "dbp": "dbp",
}

#: Lower bounds tightened beyond the shared VITAL_RANGES, ED-locally.
#:
#: The shared ranges open at an inclusive 0 (`sbp: (0, 300)`), which lets through
#: a documented MIMIC artifact — rows recording sbp 1 and dbp 0 for living
#: patients, where the cuff failed rather than the circulation. Left in, they
#: become the minimum for that admission and drag `_min` and `_delta` features
#: toward nonsense. These bounds are applied only here so the ICU vitals path and
#: every already-trained phase keep the exact thresholds they were fitted under.
ED_VITAL_FLOORS: Dict[str, float] = {
    "sbp": 20.0,
    "dbp": 10.0,
    "heartrate": 10.0,
    "resprate": 1.0,
}

#: Columns in `edstays` that must never become features.
#:
#: ``disposition`` is the ED *outcome* (ADMITTED / HOME / EXPIRED) — for this
#: cohort it is ADMITTED by construction, so it is simultaneously constant and
#: outcome-derived. ``outtime`` is the departure timestamp; the modelling-safe
#: quantity derived from it is `ed_los_hours`, which is known at admission.
ED_FORBIDDEN_COLUMNS = ("disposition", "outtime")


def _clip_to_physiology(frame: pd.DataFrame, columns=None) -> pd.DataFrame:
    """Blank out values outside the documented physiological range."""
    out = frame.copy()
    for col in (columns or ED_VITAL_RANGES):
        if col not in out.columns:
            continue
        lo, hi = VITAL_RANGES[ED_VITAL_RANGES[col]]
        lo = max(lo, ED_VITAL_FLOORS.get(col, lo))
        values = pd.to_numeric(out[col], errors="coerce")
        out[col] = values.where((values >= lo) & (values <= hi))
    return out


def link_ed_stays_to_admissions(
    edstays: pd.DataFrame,
    admissions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Map each admission to the ED stay that produced it.

    ED visits that ended in discharge home carry a null ``hadm_id`` and are
    dropped: this is an admission-grain pipeline, so joining ED data adds columns
    but never rows. That is what keeps the existing patient split valid — the
    cohort's ``subject_id`` set is unchanged, so the fingerprint guard in
    `src/data/splits.py` will confirm rather than fire.

    573 admissions link to more than one ED stay (max 3). The stay immediately
    preceding admission is the one that caused it, so ties are broken on the
    latest ``intime`` at or before ``admittime``; a stay beginning after
    admission is kept only when it is the sole candidate.
    """
    if edstays.empty or admissions.empty:
        return pd.DataFrame(columns=["hadm_id", "stay_id"])

    eds = edstays[edstays["hadm_id"].notna()].copy()
    if eds.empty:
        return pd.DataFrame(columns=["hadm_id", "stay_id"])

    eds["hadm_id"] = eds["hadm_id"].astype("int64")
    for col in ("intime", "outtime"):
        if col in eds.columns:
            eds[col] = pd.to_datetime(eds[col], errors="coerce")

    adm = admissions[["hadm_id", "admittime"]].dropna(subset=["hadm_id"]).copy()
    adm["hadm_id"] = adm["hadm_id"].astype("int64")
    adm["admittime"] = pd.to_datetime(adm["admittime"], errors="coerce")

    linked = eds.merge(adm, on="hadm_id", how="inner")
    if linked.empty:
        return pd.DataFrame(columns=["hadm_id", "stay_id"])

    linked["ed_offset_hours"] = (
        linked["intime"] - linked["admittime"]
    ).dt.total_seconds() / 3600.0

    # Prefer stays that start before admission, then the latest such stay.
    linked["_precedes"] = (linked["ed_offset_hours"] <= 0).astype(int)
    linked = linked.sort_values(
        ["hadm_id", "_precedes", "ed_offset_hours"],
        ascending=[True, False, False],
    )
    n_stays = linked.groupby("hadm_id")["stay_id"].transform("size")
    chosen = linked.drop_duplicates("hadm_id", keep="first").copy()
    chosen["ed_n_stays"] = n_stays.loc[chosen.index].values

    chosen["ed_los_hours"] = (
        chosen["outtime"] - chosen["intime"]
    ).dt.total_seconds() / 3600.0
    # A negative or implausibly long ED stay is a timestamp error, not a signal.
    chosen.loc[
        (chosen["ed_los_hours"] < 0) | (chosen["ed_los_hours"] > 72), "ed_los_hours"
    ] = np.nan

    keep = ["hadm_id", "stay_id", "intime", "admittime", "ed_los_hours",
            "ed_n_stays", "ed_offset_hours"]
    if "arrival_transport" in chosen.columns:
        chosen["ed_arrival_ambulance"] = (
            chosen["arrival_transport"].astype(str).str.upper().eq("AMBULANCE")
        ).astype(float)
        keep.append("ed_arrival_ambulance")

    result = chosen[keep].reset_index(drop=True)
    log.info(
        "Linked %d ED stays to admissions (%d admissions had >1 candidate)",
        len(result), int((result["ed_n_stays"] > 1).sum()),
    )
    return result


def build_ed_triage_features(
    triage: pd.DataFrame,
    link: pd.DataFrame,
) -> pd.DataFrame:
    """
    First-contact physiology: one measurement per admission, taken at ED arrival.

    ``acuity`` is carried but deliberately named ``ed_triage_acuity`` so it can be
    excluded as its own family. It is a nurse's 1–5 severity judgement (ESI) that
    partly *causes* the ICU decision rather than predicting it — the same shape as
    the testing-volume features Phase 5's rebuild removed. It belongs in a
    sensitivity arm, not the primary model.
    """
    if triage.empty or link.empty:
        return pd.DataFrame(columns=["hadm_id"])

    cols = ["stay_id"] + [c for c in ED_VITAL_RANGES if c in triage.columns]
    for extra in ("pain", "acuity"):
        if extra in triage.columns:
            cols.append(extra)

    tri = _clip_to_physiology(triage[cols].copy())

    if "pain" in tri.columns:
        # Free text in practice ('critical', '13'); only the 0–10 scale is meaningful.
        pain = pd.to_numeric(tri["pain"], errors="coerce")
        tri["pain"] = pain.where((pain >= 0) & (pain <= 10))

    merged = link[["hadm_id", "stay_id"]].merge(tri, on="stay_id", how="inner")
    merged = merged.drop(columns=["stay_id"])

    # Pulse pressure is the derived quantity clinicians actually read for shock.
    if {"sbp", "dbp"}.issubset(merged.columns):
        merged["pulse_pressure"] = merged["sbp"] - merged["dbp"]
        merged.loc[merged["pulse_pressure"] <= 0, "pulse_pressure"] = np.nan
    if {"heartrate", "sbp"}.issubset(merged.columns):
        # Shock index >0.9 is the classic deterioration marker.
        shock = merged["heartrate"] / merged["sbp"].replace(0, np.nan)
        merged["shock_index"] = shock.where(shock.between(0, 5))

    merged = merged.groupby("hadm_id", as_index=False).mean(numeric_only=True)
    merged.columns = [
        c if c == "hadm_id" else f"ed_triage_{c}" for c in merged.columns
    ]
    log.info("ED triage features: %d admissions × %d cols",
             len(merged), merged.shape[1] - 1)
    return merged


def build_ed_vitals_features(
    vitalsign: pd.DataFrame,
    link: pd.DataFrame,
    window_hours: float = 0.0,
) -> pd.DataFrame:
    """
    Serial ED vitals aggregated per admission, cut at ``admittime + window_hours``.

    The cut is applied per ``charttime`` rather than trusting the stay boundary,
    because ``admittime`` falls *inside* the ED stay — the inpatient admission is
    registered while the patient is still physically in the ED, so ED `outtime`
    follows `admittime` for 99.7% of linked stays. At the default of 0.0 that
    discards ~41% of ED vital rows, which is the honest price of an
    admission-time model: those observations did not exist when the prediction is
    made.

    ``window_hours`` exists so the alternative protocol does not require rewriting
    this module. `laboratory.py` keeps draws up to ``admittime + 24h`` for its
    ``lab_*_24h`` family; passing ``window_hours=24.0`` here produces ED vitals on
    that same footing, for models scored under the 24-hour protocol rather than at
    admission. The default stays strict because a feature that is safe under every
    protocol is the one worth defaulting to.
    """
    if vitalsign.empty or link.empty:
        return pd.DataFrame(columns=["hadm_id"])

    value_cols = [c for c in ED_VITAL_RANGES if c in vitalsign.columns]
    if not value_cols:
        return pd.DataFrame(columns=["hadm_id"])

    vs = vitalsign[["stay_id", "charttime"] + value_cols].copy()
    vs["charttime"] = pd.to_datetime(vs["charttime"], errors="coerce")

    vs = link[["hadm_id", "stay_id", "admittime"]].merge(vs, on="stay_id", how="inner")
    cutoff = vs["admittime"] + pd.to_timedelta(window_hours, unit="h")
    before = vs["charttime"].notna() & (vs["charttime"] <= cutoff)
    dropped = int((~before).sum())
    vs = vs.loc[before]
    log.info(
        "ED vitals (<=admittime+%.0fh): dropped %d of %d rows (%.1f%%)",
        window_hours, dropped, dropped + len(vs),
        100.0 * dropped / max(dropped + len(vs), 1),
    )
    if vs.empty:
        return pd.DataFrame(columns=["hadm_id"])

    vs = _clip_to_physiology(vs, value_cols)
    vs = vs.sort_values(["hadm_id", "charttime"])

    stats = vs.groupby("hadm_id")[value_cols].agg(
        ["mean", "min", "max", "std", "first", "last", "count"]
    )
    stats.columns = [f"ed_vital_{col}_{stat}" for col, stat in stats.columns]

    # Direction of travel over the ED stay. A patient whose heart rate rose 30
    # while waiting is a different patient from one who arrived tachycardic and
    # settled, and neither is distinguishable from the mean alone.
    for col in value_cols:
        first, last = stats.get(f"ed_vital_{col}_first"), stats.get(f"ed_vital_{col}_last")
        if first is not None and last is not None:
            stats[f"ed_vital_{col}_delta"] = last - first

    stats = stats.reset_index()
    log.info("ED serial vitals features: %d admissions × %d cols",
             len(stats), stats.shape[1] - 1)
    return stats


def build_ed_medrecon_features(
    medrecon: pd.DataFrame,
    link: pd.DataFrame,
) -> pd.DataFrame:
    """
    Home-medication reconciliation recorded in the ED.

    These are the drugs the patient was already taking on arrival, so unlike
    inpatient `prescriptions` they carry no treatment-response leakage — nothing
    here was chosen in reaction to how the admission unfolded. The count of
    distinct therapeutic classes is a medication-burden proxy that is largely
    independent of the ICD-derived comorbidity already in the matrix.
    """
    if medrecon.empty or link.empty:
        return pd.DataFrame(columns=["hadm_id"])

    cols = ["stay_id"] + [c for c in ("name", "etccode") if c in medrecon.columns]
    rec = link[["hadm_id", "stay_id"]].merge(medrecon[cols], on="stay_id", how="inner")
    if rec.empty:
        return pd.DataFrame(columns=["hadm_id"])

    agg = {"ed_medrecon_n_drugs": ("name", "size")} if "name" in rec.columns else {}
    if "name" in rec.columns:
        agg["ed_medrecon_n_unique_drugs"] = ("name", "nunique")
    if "etccode" in rec.columns:
        agg["ed_medrecon_n_classes"] = ("etccode", "nunique")

    out = rec.groupby("hadm_id").agg(**agg).reset_index()
    if "ed_medrecon_n_drugs" in out.columns:
        out["ed_medrecon_polypharmacy"] = (out["ed_medrecon_n_drugs"] >= 5).astype(float)
    log.info("ED medrecon features: %d admissions × %d cols",
             len(out), out.shape[1] - 1)
    return out


def build_ed_features(
    edstays: pd.DataFrame,
    admissions: pd.DataFrame,
    triage: Optional[pd.DataFrame] = None,
    vitalsign: Optional[pd.DataFrame] = None,
    medrecon: Optional[pd.DataFrame] = None,
    window_hours: float = 0.0,
) -> pd.DataFrame:
    """
    Assemble every ED feature onto one ``hadm_id``-keyed frame.

    Returns only admissions that had an ED stay. The merge back onto the full
    cohort is left to the caller so that absent ED data stays NaN rather than
    being imputed here.
    """
    link = link_ed_stays_to_admissions(edstays, admissions)
    if link.empty:
        return pd.DataFrame(columns=["hadm_id"])

    out = link[["hadm_id", "ed_los_hours", "ed_n_stays"]].copy()
    if "ed_arrival_ambulance" in link.columns:
        out["ed_arrival_ambulance"] = link["ed_arrival_ambulance"].values

    for frame, builder in (
        (triage, build_ed_triage_features),
        (vitalsign, build_ed_vitals_features),
        (medrecon, build_ed_medrecon_features),
    ):
        if frame is None or frame.empty:
            continue
        feats = (builder(frame, link, window_hours=window_hours)
                 if builder is build_ed_vitals_features else builder(frame, link))
        if not feats.empty:
            out = out.merge(feats, on="hadm_id", how="left")

    # Explicit rather than implicit, so availability leakage can be measured.
    out["ed_available"] = 1.0

    log.info("ED features: %d admissions × %d cols", len(out), out.shape[1] - 1)
    return out
