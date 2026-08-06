"""
NEWS2 must score every value, and must not score values it never saw.

Two defects, both silent, both in the direction of reassurance:

1. **Band gaps.** The bands were written as chained integer comparisons —
   `hr <= 40` then `(hr >= 41) & (hr <= 50)` — which is right for integers and wrong
   for this data. These vitals are means and latest-values, so they are floats. A
   heart rate of 40.5 matched neither test and fell through to the initialised 0: a
   bradycardic patient scored as normal. The same hole sat at every boundary in the
   score.

2. **Missing scored as zero.** An unmeasured parameter contributed 0 points, which is
   the score for *healthy*. A patient nobody had observed came out looking well —
   the exact inversion an early-warning score must not make.

`compute_news2_score` is not currently a model input (`news2_*` is excluded as
availability leakage, since chartevents exists only for ICU stays). These tests pin it
anyway: it is quoted as the clinical rationale for Phase 5, and it becomes live the
moment ward vitals arrive from MIMIC-IV-ED.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.deterioration import _band_score, compute_news2_score

ALL_VITALS = ("vital_heart_rate_latest", "vital_resp_rate_latest", "vital_sbp_latest",
              "vital_spo2_latest", "vital_temperature_c_latest", "vital_gcs_total_latest")


def frame(**kw):
    """One-row frame with every vital normal unless overridden."""
    row = {"vital_heart_rate_latest": 70.0, "vital_resp_rate_latest": 16.0,
           "vital_sbp_latest": 120.0, "vital_spo2_latest": 98.0,
           "vital_temperature_c_latest": 37.0, "vital_gcs_total_latest": 15.0}
    row.update(kw)
    return pd.DataFrame([row])


# ── no value may fall through a band ────────────────────────────────────────

@pytest.mark.parametrize("hr,expected", [
    (35, 3), (40, 3), (40.5, 3),      # 40.5 previously scored 0
    (45, 1), (50, 1), (50.5, 1),      # 50.5 previously scored 0
    (70, 0), (90, 0), (90.5, 0),
    (100, 1), (110, 1), (110.5, 1),
    (120, 2), (130, 2), (130.5, 2),
    (140, 3),
])
def test_heart_rate_bands_have_no_gaps(hr, expected):
    _, sub = compute_news2_score(frame(vital_heart_rate_latest=float(hr)))
    assert sub["news2_hr_score"].iloc[0] == expected, f"HR {hr} scored wrongly"


@pytest.mark.parametrize("field,column,value,expected", [
    ("vital_resp_rate_latest", "news2_resp_score", 8.5, 3),
    ("vital_resp_rate_latest", "news2_resp_score", 11.5, 1),
    ("vital_resp_rate_latest", "news2_resp_score", 24.5, 2),
    ("vital_sbp_latest", "news2_sbp_score", 91.5, 3),
    ("vital_sbp_latest", "news2_sbp_score", 100.5, 2),
    ("vital_sbp_latest", "news2_sbp_score", 110.5, 1),
    ("vital_sbp_latest", "news2_sbp_score", 219.5, 0),
    ("vital_spo2_latest", "news2_spo2_score", 91.5, 3),
    ("vital_spo2_latest", "news2_spo2_score", 93.5, 2),
    ("vital_spo2_latest", "news2_spo2_score", 95.5, 1),
])
def test_fractional_values_at_every_boundary(field, column, value, expected):
    """Each of these previously fell through to 0."""
    _, sub = compute_news2_score(frame(**{field: value}))
    assert sub[column].iloc[0] == expected, f"{field}={value} scored wrongly"


def test_no_gap_anywhere_across_a_dense_sweep():
    """Sweep each parameter in 0.1 steps; nothing may be unscored."""
    for field, column, lo, hi in [
        ("vital_heart_rate_latest", "news2_hr_score", 20, 200),
        ("vital_resp_rate_latest", "news2_resp_score", 4, 40),
        ("vital_sbp_latest", "news2_sbp_score", 60, 250),
        ("vital_spo2_latest", "news2_spo2_score", 70, 100),
        ("vital_temperature_c_latest", "news2_temp_score", 32, 42),
    ]:
        values = np.round(np.arange(lo, hi, 0.1), 1)
        df = pd.concat([frame(**{field: float(v)}) for v in values], ignore_index=True)
        _, sub = compute_news2_score(df)
        assert sub[column].notna().all(), f"{field} left values unscored"


def test_bands_are_monotonic_where_clinically_expected():
    """Falling SpO2 must never reduce the score."""
    values = np.arange(80.0, 100.0, 0.5)
    df = pd.concat([frame(vital_spo2_latest=float(v)) for v in values], ignore_index=True)
    _, sub = compute_news2_score(df)
    scores = sub["news2_spo2_score"].to_numpy()
    assert np.all(np.diff(scores) <= 0), "SpO2 score rose as saturation improved"


# ── missing must not read as normal ─────────────────────────────────────────

def test_missing_vital_scores_nan_not_zero():
    df = frame()
    df["vital_heart_rate_latest"] = np.nan
    _, sub = compute_news2_score(df)
    assert np.isnan(sub["news2_hr_score"].iloc[0]), \
        "an unmeasured heart rate scored 0, which is the score for a normal one"


def test_composite_is_withheld_when_too_few_parameters():
    df = frame()
    for c in ALL_VITALS[2:]:
        df[c] = np.nan
    composite, _ = compute_news2_score(df)
    assert np.isnan(composite.iloc[0]), \
        "a composite built from two parameters was reported as a NEWS2 score"


def test_composite_is_computed_when_enough_parameters_present():
    df = frame()
    df["vital_gcs_total_latest"] = np.nan
    composite, _ = compute_news2_score(df)
    assert composite.iloc[0] == 0, "a fully normal patient should score 0"


def test_absent_columns_are_tolerated():
    """The scorer must not raise when the frame lacks vital columns entirely."""
    composite, sub = compute_news2_score(pd.DataFrame({"hadm_id": [1, 2, 3]}))
    assert len(composite) == 3
    assert composite.isna().all(), "a frame with no vitals produced numeric scores"


# ── the deranged patient must score high ────────────────────────────────────

def test_critically_unwell_patient_scores_high():
    composite, _ = compute_news2_score(frame(
        vital_heart_rate_latest=135.0, vital_resp_rate_latest=30.0,
        vital_sbp_latest=85.0, vital_spo2_latest=88.0,
        vital_temperature_c_latest=39.5, vital_gcs_total_latest=13.0))
    assert composite.iloc[0] >= 15, f"septic-shock physiology scored only {composite.iloc[0]}"


def test_band_score_helper_assigns_exactly_one_band():
    values = pd.Series([0.0, 5.0, 10.0, 15.0])
    out = _band_score(values, [(5, 1), (10, 2), (None, 3)])
    assert list(out) == [1, 2, 3, 3]
