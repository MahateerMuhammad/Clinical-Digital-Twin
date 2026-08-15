"""
The one-hot fill contract in `src/llm/feature_space.py`.

A dummy absent because its category was supplied as something else is a known zero.
A dummy absent because nobody mentioned the category is unknown. Everything here
pins that distinction, because both halves of it were wrong at once: siblings were
filled NaN (understating what the payload said) and coverage credited one feature
per supplied category instead of the whole family.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.llm.feature_space import (
    align_to_model,
    design_matrix,
    encode_admission_frame,
    feature_coverage,
    onehot_known,
)

RACE_FEATURES = ["race_WHITE", "race_BLACK", "race_ASIAN"]


def _payload(**overrides) -> pd.DataFrame:
    row = {"age": 70.0, "race": "WHITE", "admission_type": "URGENT"}
    row.update(overrides)
    return pd.DataFrame([row])


# ── supplied categories license their zeros ──────────────────────────────────

def test_siblings_of_a_supplied_category_are_zero_not_nan():
    X, _ = design_matrix(_payload(), RACE_FEATURES)
    assert X.loc[0, "race_WHITE"] == 1.0
    assert X.loc[0, "race_BLACK"] == 0.0, "a White patient is known not to be Black"
    assert X.loc[0, "race_ASIAN"] == 0.0


def test_a_category_never_mentioned_stays_nan():
    """The frame has no `race` column at all — nothing licenses a zero."""
    X, _ = design_matrix(pd.DataFrame([{"age": 70.0}]), RACE_FEATURES)
    assert X[RACE_FEATURES].isna().all(axis=None)


def test_a_supplied_but_null_category_stays_nan():
    """`race` present and empty is unknown, not a determination of not-White."""
    X, _ = design_matrix(_payload(race=None), RACE_FEATURES)
    assert X[RACE_FEATURES].isna().all(axis=None)


def test_zero_fill_is_per_row_not_per_frame():
    frame = pd.DataFrame([{"race": "WHITE"}, {"race": None}])
    X, _ = design_matrix(frame, RACE_FEATURES)
    assert X.loc[0, "race_BLACK"] == 0.0
    assert np.isnan(X.loc[1, "race_BLACK"]), "row 1 said nothing; it gets no zeros"


def test_reference_level_dropped_at_fit_time_yields_all_zeros():
    """
    Phases 1-4 trained with `drop_first=True`, so the reference level has no column.
    A patient *of* that level must encode as zeros across the family — under the old
    NaN fill they were indistinguishable from a patient whose race was never asked.
    """
    X, _ = design_matrix(_payload(race="OTHER"), RACE_FEATURES)
    assert (X.loc[0, RACE_FEATURES] == 0.0).all()


# ── numeric features keep the opposite rule ──────────────────────────────────

def test_absent_numeric_features_are_still_nan():
    """The inversion applies to dummies only; a missing lab is not a zero lab."""
    X, _ = design_matrix(_payload(), ["lab_creatinine_max_24h", "race_WHITE"])
    assert np.isnan(X.loc[0, "lab_creatinine_max_24h"])
    assert X.loc[0, "race_WHITE"] == 1.0


def test_prefix_collision_resolves_to_the_longest_match():
    """`admission_type` and `admission_location` share a prefix; both are real."""
    frame = pd.DataFrame([{"admission_type": "URGENT", "admission_location": None}])
    X, _ = design_matrix(
        frame, ["admission_type_ELECTIVE", "admission_location_EMERGENCY_ROOM"])
    assert X.loc[0, "admission_type_ELECTIVE"] == 0.0
    assert np.isnan(X.loc[0, "admission_location_EMERGENCY_ROOM"]), \
        "matched on `admission_` it would wrongly inherit admission_type's mask"


# ── coverage counts the whole family ─────────────────────────────────────────

def test_coverage_credits_every_feature_the_category_determines():
    encoded = encode_admission_frame(_payload())
    known = onehot_known(encoded)
    assert feature_coverage(encoded, RACE_FEATURES, known) == 1.0


def test_coverage_does_not_credit_an_unmentioned_category():
    encoded = encode_admission_frame(pd.DataFrame([{"age": 70.0}]))
    assert feature_coverage(encoded, RACE_FEATURES, onehot_known(encoded)) == 0.0


def test_coverage_is_not_credited_when_some_rows_lack_the_category():
    encoded = encode_admission_frame(pd.DataFrame([{"race": "WHITE"}, {"race": None}]))
    cov = feature_coverage(encoded, RACE_FEATURES, onehot_known(encoded))
    assert cov < 1.0, "one number for the frame cannot claim what row 1 never said"


def test_coverage_rises_for_a_realistic_payload():
    """The point of the change: the same payload, measured old way and new."""
    encoded = encode_admission_frame(_payload())
    features = RACE_FEATURES + ["admission_type_ELECTIVE", "age"]
    naive = sum(f in set(encoded.columns) for f in features) / len(features)
    assert feature_coverage(encoded, features, onehot_known(encoded)) > naive


# ── the old behaviour is still reachable ─────────────────────────────────────

def test_align_without_provenance_fills_everything_nan():
    """An unencoded frame carries no map, so no zero is licensed."""
    bare = pd.DataFrame([{"race_WHITE": 1.0}])
    X = align_to_model(bare, RACE_FEATURES, known={})
    assert X.loc[0, "race_WHITE"] == 1.0
    assert X[["race_BLACK", "race_ASIAN"]].isna().all(axis=None)


@pytest.mark.parametrize("n_rows", [1, 5])
def test_shape_and_dtype_are_unchanged(n_rows):
    frame = pd.concat([_payload()] * n_rows, ignore_index=True)
    X, cov = design_matrix(frame, RACE_FEATURES)
    assert X.shape == (n_rows, len(RACE_FEATURES))
    assert (X.dtypes == float).all()
    assert 0.0 <= cov <= 1.0
