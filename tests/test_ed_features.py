"""
The ED module must add columns without ever adding rows, values, or hindsight.

MIMIC-IV-ED is the first source of ward-grade physiology in this pipeline —
before it, `admission_level_selected` carried zero `vital_*` columns because
`chartevents` is ICU-only. That makes it valuable and dangerous in the same
breath, and three properties have to hold or the whole cohort is compromised:

1. **Rows never change.** ED data joins onto admissions; it does not create them.
   If a single patient entered or left the cohort, the split would repartition
   and every model on disk would be evaluated on patients it trained on. The
   fingerprint guard in `src/data/splits.py` catches that, but only if this layer
   never causes it in the first place.

2. **Nothing is observed after `admittime`.** `admittime` falls *inside* the ED
   stay — registration happens while the patient is still physically in the ED,
   so ED `outtime` follows `admittime` for 99.7% of linked stays. Trusting the
   stay boundary instead of each `charttime` would import ~41% of ED vital rows
   from after the prediction moment.

3. **Missing stays missing.** The ED module covers 37.1% of the cohort. The other
   63% must be NaN, never 0.0 — a filled zero says the patient was measured and
   found to have no pulse.

The unit tests run on synthetic frames so they execute in milliseconds and cannot
be skipped; the cohort-level tests run against the real tables when present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.emergency import (
    ED_FORBIDDEN_COLUMNS,
    ED_VITAL_FLOORS,
    build_ed_features,
    build_ed_triage_features,
    build_ed_vitals_features,
    link_ed_stays_to_admissions,
)

ED_DIR = Path("data/raw/ED")
ADMISSIONS = Path("data/processed/admission_level.parquet")


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def admissions() -> pd.DataFrame:
    return pd.DataFrame({
        "hadm_id": [1, 2, 3],
        "subject_id": [10, 20, 30],
        "admittime": pd.to_datetime(
            ["2180-01-01 12:00", "2180-02-01 12:00", "2180-03-01 12:00"]),
    })


@pytest.fixture
def edstays() -> pd.DataFrame:
    """hadm 1 has two ED stays; hadm 3 has none; one stay never led to admission."""
    return pd.DataFrame({
        "subject_id": [10, 10, 20, 40],
        "hadm_id": [1.0, 1.0, 2.0, np.nan],
        "stay_id": [101, 102, 201, 401],
        "intime": pd.to_datetime([
            "2180-01-01 04:00",   # earlier stay for hadm 1
            "2180-01-01 08:00",   # later, still before admittime -> chosen
            "2180-02-01 06:00",
            "2180-09-01 06:00",   # discharged home, must be dropped
        ]),
        "outtime": pd.to_datetime([
            "2180-01-01 07:00", "2180-01-01 13:00",
            "2180-02-01 13:00", "2180-09-01 09:00",
        ]),
        "arrival_transport": ["WALK IN", "AMBULANCE", "WALK IN", "AMBULANCE"],
        "disposition": ["ADMITTED", "ADMITTED", "ADMITTED", "HOME"],
    })


# ── linkage ───────────────────────────────────────────────────────────────────

def test_link_drops_ed_visits_that_never_became_admissions(edstays, admissions):
    link = link_ed_stays_to_admissions(edstays, admissions)
    assert 401 not in set(link["stay_id"]), "a discharged-home ED visit became a row"
    assert set(link["hadm_id"]) <= set(admissions["hadm_id"])


def test_link_is_one_row_per_admission_choosing_the_stay_before_admission(
        edstays, admissions):
    link = link_ed_stays_to_admissions(edstays, admissions)
    assert link["hadm_id"].is_unique, "an admission linked to >1 ED stay"
    chosen = link.loc[link["hadm_id"] == 1, "stay_id"].item()
    assert chosen == 102, (
        "expected the latest ED stay starting at or before admittime; "
        f"got stay {chosen}")
    assert link.loc[link["hadm_id"] == 1, "ed_n_stays"].item() == 2


def test_link_never_invents_admissions(edstays, admissions):
    """The property the whole split depends on."""
    link = link_ed_stays_to_admissions(edstays, admissions)
    assert set(link["hadm_id"]) <= set(admissions["hadm_id"])
    assert len(link) <= len(admissions)


# ── observation window ────────────────────────────────────────────────────────

def test_vitals_after_admittime_are_excluded(edstays, admissions):
    """
    The measurement that must not survive.

    Stay 102 runs 08:00–13:00 while hadm 1 is admitted at 12:00. The 12:30 reading
    is inside the ED stay but after the prediction moment; keeping it would make
    every `_last`, `_max` and `_delta` feature partly retrospective.
    """
    link = link_ed_stays_to_admissions(edstays, admissions)
    vitals = pd.DataFrame({
        "stay_id": [102, 102, 102],
        "charttime": pd.to_datetime([
            "2180-01-01 09:00", "2180-01-01 11:00", "2180-01-01 12:30"]),
        "heartrate": [80.0, 90.0, 190.0],
    })
    feats = build_ed_vitals_features(vitals, link)
    row = feats.loc[feats["hadm_id"] == 1].iloc[0]
    assert row["ed_vital_heartrate_count"] == 2
    assert row["ed_vital_heartrate_max"] == 90.0, "post-admittime reading leaked in"
    assert row["ed_vital_heartrate_delta"] == 10.0


def test_window_hours_admits_the_24h_protocol(edstays, admissions):
    """`laboratory.py` keeps draws to admittime+24h; ED must be able to match it."""
    link = link_ed_stays_to_admissions(edstays, admissions)
    vitals = pd.DataFrame({
        "stay_id": [102, 102],
        "charttime": pd.to_datetime(["2180-01-01 09:00", "2180-01-01 12:30"]),
        "heartrate": [80.0, 190.0],
    })
    assert build_ed_vitals_features(vitals, link, window_hours=0.0)[
        "ed_vital_heartrate_count"].item() == 1
    assert build_ed_vitals_features(vitals, link, window_hours=24.0)[
        "ed_vital_heartrate_count"].item() == 2


# ── value hygiene ─────────────────────────────────────────────────────────────

def test_impossible_values_are_blanked_not_clamped(edstays, admissions):
    """
    The raw table holds dbp up to 661,672 and o2sat up to 9,322.

    They must become NaN rather than be clipped to the boundary: a cuff failure
    is an absence of measurement, and clamping it to 250 would assert a severe
    hypertension that was never observed.
    """
    link = link_ed_stays_to_admissions(edstays, admissions)
    triage = pd.DataFrame({
        "stay_id": [102, 201],
        "sbp": [661672.0, 120.0],
        "dbp": [0.0, 80.0],
        "heartrate": [88.0, 70.0],
        "o2sat": [9322.0, 98.0],
        "temperature": [98.6, 99.1],
        "resprate": [16.0, 18.0],
        "acuity": [2.0, 3.0],
        "pain": ["critical", "5"],
    })
    feats = build_ed_triage_features(triage, link).set_index("hadm_id")
    assert pd.isna(feats.loc[1, "ed_triage_sbp"])
    assert pd.isna(feats.loc[1, "ed_triage_dbp"]), "dbp 0 passed the inclusive floor"
    assert pd.isna(feats.loc[1, "ed_triage_o2sat"])
    assert feats.loc[2, "ed_triage_sbp"] == 120.0, "a valid value was discarded"
    # Non-numeric pain text must not become a number.
    assert pd.isna(feats.loc[1, "ed_triage_pain"])
    assert feats.loc[2, "ed_triage_pain"] == 5.0


def test_ed_vital_floors_exceed_the_shared_lower_bounds():
    """The tightening must actually tighten, or the artifact rows return."""
    from src.utils.validation import VITAL_RANGES
    from src.features.emergency import ED_VITAL_RANGES

    for col, floor in ED_VITAL_FLOORS.items():
        shared_lo = VITAL_RANGES[ED_VITAL_RANGES[col]][0]
        assert floor > shared_lo, f"{col} floor {floor} does not tighten {shared_lo}"


# ── forbidden columns ─────────────────────────────────────────────────────────

def test_ed_outcome_columns_never_reach_the_feature_frame(edstays, admissions):
    """`disposition` is the ED outcome and is ADMITTED by construction here."""
    feats = build_ed_features(edstays, admissions)
    for forbidden in ED_FORBIDDEN_COLUMNS:
        assert not any(forbidden in c for c in feats.columns), (
            f"'{forbidden}' reached the ED feature frame")


def test_every_exclusion_list_carries_the_ed_guard():
    """
    A new task exclusion list must not silently opt out of the ED guard.

    `ed_available` is a cohort-membership indicator rather than a measurement, and
    `ed_triage_acuity` is a nurse's severity judgement that partly causes the ICU
    decision. Both are cheap for a tree to exploit.
    """
    from src.features import leakage_filters as lf

    names = [n for n in dir(lf)
             if n.endswith("_EXCLUDE") or n.endswith("_EXCLUDE_STRICT")
             or n.endswith("_RUN_B") or n.endswith("_RUN_C")]
    assert names, "no exclusion lists found"
    for name in names:
        lst = getattr(lf, name)
        assert "ed_available" in lst, f"{name} is missing the ed_available guard"
        assert "ed_triage_acuity" in lst, f"{name} is missing the ed_triage_acuity guard"


# ── real-data cohort guarantees ───────────────────────────────────────────────

pytestmark_real = pytest.mark.skipif(
    not (ED_DIR / "edstays.csv").exists() or not ADMISSIONS.exists(),
    reason="ED tables or admission_level.parquet absent",
)


@pytestmark_real
def test_real_ed_join_adds_no_patients_and_no_admissions():
    """
    The cohort must be byte-for-byte the same population after the ED join.

    This is the test that protects every published metric: the split is a
    deterministic function of the sorted subject_id set, so one extra patient
    repartitions 16,361 of them.
    """
    adm = pd.read_parquet(ADMISSIONS, columns=["hadm_id", "subject_id", "admittime"])
    adm["hadm_id"] = adm["hadm_id"].astype("int64")
    eds = pd.read_csv(ED_DIR / "edstays.csv")

    link = link_ed_stays_to_admissions(eds, adm)
    assert link["hadm_id"].is_unique
    assert set(link["hadm_id"]) <= set(adm["hadm_id"])

    merged = adm.merge(link, on="hadm_id", how="left")
    assert len(merged) == len(adm), "the ED join changed the row count"
    assert set(merged["subject_id"]) == set(adm["subject_id"]), \
        "the ED join changed the patient set; the split is now invalid"


@pytestmark_real
def test_real_ed_presence_does_not_predict_the_outcome():
    """
    Availability leakage, measured rather than assumed.

    ED coverage is partial (37.1%), so "has ED data" could in principle carry the
    outcome on its own — the failure that forced the Phase 5 rebuild. Measured, it
    does not: AUROC 0.5097 for mortality. This test pins that, because a future
    change to the linkage rule (say, keeping only stays that precede admission by
    <2h) could quietly turn presence into a severity marker.
    """
    from sklearn.metrics import roc_auc_score

    adm = pd.read_parquet(ADMISSIONS)
    if "hospital_expire_flag" not in adm.columns:
        pytest.skip("hospital_expire_flag absent")
    adm["hadm_id"] = adm["hadm_id"].astype("int64")

    eds = pd.read_csv(ED_DIR / "edstays.csv", usecols=["subject_id", "hadm_id",
                                                       "stay_id", "intime", "outtime"])
    link = link_ed_stays_to_admissions(eds, adm)
    present = adm["hadm_id"].isin(set(link["hadm_id"])).astype(float)
    y = pd.to_numeric(adm["hospital_expire_flag"], errors="coerce")
    ok = y.notna()

    auroc = roc_auc_score(y[ok], present[ok])
    assert abs(auroc - 0.5) < 0.05, (
        f"ED availability alone predicts mortality at AUROC {auroc:.4f}; "
        "presence has become a severity marker and must be handled like the "
        "ICU vital_* family was in the Phase 5 rebuild")


@pytestmark_real
def test_real_missing_ed_data_is_nan_not_zero():
    """The 63% without ED data must be unmeasured, not measured-as-zero."""
    adm = pd.read_parquet(ADMISSIONS, columns=["hadm_id", "subject_id", "admittime"])
    adm["hadm_id"] = adm["hadm_id"].astype("int64")
    eds = pd.read_csv(ED_DIR / "edstays.csv")
    triage = pd.read_csv(ED_DIR / "triage.csv")

    feats = build_ed_features(eds, adm, triage=triage)
    merged = adm[["hadm_id"]].merge(feats, on="hadm_id", how="left")
    absent = merged["ed_available"].isna()
    assert absent.any(), "expected some admissions without ED data"

    for col in ("ed_triage_heartrate", "ed_triage_sbp", "ed_los_hours"):
        assert merged.loc[absent, col].isna().all(), \
            f"{col} was filled for admissions that had no ED stay"
