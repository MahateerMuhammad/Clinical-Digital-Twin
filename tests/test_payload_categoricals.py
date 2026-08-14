"""
The payload path must reach the boosters through the same encoder they were fitted
with, not through hand-written dummy names.

Writing `gender_M = 1.0` directly looks equivalent and is not: it emits one feature
where the fitted model has a family, so every sibling arrives missing. That is how a
female patient reached Phase 5 as "not male, sex unknown", and how a complete payload
covered 18% of a model trained on 164 features.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.llm.model_runner import (
    PAYLOAD_FIDELITY,
    PAYLOAD_SERVED_TASKS,
    LiveModelRunner,
    _dig_payload,
)
from src.llm.payload_validation import RECOMMENDED_FIELDS, REQUIRED_FIELDS

RICH = {
    "demographics": {"age": 78, "gender": "F", "race": "WHITE", "language": "English",
                     "insurance": "Medicare", "marital_status": "WIDOWED"},
    "primary_diagnosis": "sepsis",
    "presentation_labs": {"creatinine_max": 5.8, "bun_max": 95.0, "wbc_max": 18.2},
    "admission_context": {"admission_type": "EW EMER.",
                          "admission_location": "EMERGENCY ROOM"},
    "prior_utilisation": {"admissions_30d": 1, "admissions_90d": 2,
                          "admissions_365d": 4, "cumulative_los_days": 21},
}


@pytest.fixture(scope="module")
def runner():
    r = LiveModelRunner()
    if r.adm_df is None or not r.lgbm_models.get("mortality"):
        pytest.skip("promoted models or admission parquet not present")
    return r


# ── dotted-path reader tolerates both payload shapes ─────────────────────────

@pytest.mark.parametrize("payload, expected", [
    ({"demographics": {"age": 70}}, 70),
    ({"age": 70}, 70),                      # older flat shape, still in the wild
    ({"demographics": {}}, None),
    ({"demographics": {"age": ""}}, None),  # empty string is not a value
    ({}, None),
])
def test_dig_payload_shapes(payload, expected):
    assert _dig_payload(payload, "demographics.age") == expected


# ── categoricals expand into full families ───────────────────────────────────

def test_supplied_sex_sets_both_sides_of_the_family(runner):
    """Phase 5 carries gender_F and gender_M. A female patient must set both."""
    s = runner._convert_payload_to_series(RICH)
    assert s["gender_F"] == 1.0
    assert s["gender_M"] == 0.0, "hand-written `gender_M` left gender_F absent here"


def test_supplied_race_determines_the_whole_family(runner):
    s = runner._convert_payload_to_series(RICH)
    race = [k for k in s.index if k.startswith("race_")]
    assert len(race) > 20, f"expected the cohort's full race family, got {len(race)}"
    assert s["race_WHITE"] == 1.0
    assert sum(s[k] for k in race) == 1.0, "exactly one level may be hot"


def test_multi_word_categories_are_name_mangled_like_lightgbm(runner):
    """`EW EMER.` must land on `admission_type_EW_EMER.`, spaces rewritten."""
    s = runner._convert_payload_to_series(RICH)
    assert s["admission_type_EW_EMER."] == 1.0
    assert s["admission_type_ELECTIVE"] == 0.0


def test_category_matching_is_case_insensitive(runner):
    payload = {**RICH, "demographics": {**RICH["demographics"], "race": "white"}}
    assert runner._convert_payload_to_series(payload)["race_WHITE"] == 1.0


def test_an_unrecognised_category_never_lands_on_the_wrong_level(runner):
    payload = {**RICH, "demographics": {**RICH["demographics"], "race": "NOT A RACE"}}
    s = runner._convert_payload_to_series(payload)
    hot = [k for k in s.index if k.startswith("race_") and s[k] == 1.0]
    assert hot in ([], ["race_NOT_A_RACE"]), f"matched onto a real level: {hot}"


def test_omitting_categoricals_still_produces_a_scorable_series(runner):
    """The new fields are recommended, not required; a bare payload must survive."""
    bare = {"demographics": {"age": 78, "gender": "F"}, "primary_diagnosis": "sepsis",
            "presentation_labs": {"creatinine_max": 5.8}}
    s = runner._convert_payload_to_series(bare)
    assert len(s) > 0 and s["anchor_age"] == 78.0


# ── prior utilisation reaches the readmission model ──────────────────────────

def test_prior_utilisation_is_carried(runner):
    s = runner._convert_payload_to_series(RICH)
    assert s["prior_admissions_365d"] == 4.0
    assert s["prior_cumulative_los_days"] == 21.0


def test_counts_the_hospital_codes_are_not_claimed(runner):
    """diagnosis_count/procedure_count are EHR-derived; claiming them inflates coverage."""
    s = runner._convert_payload_to_series(RICH)
    assert "diagnosis_count" not in s.index
    assert "procedure_count" not in s.index


# ── coverage and the served set ──────────────────────────────────────────────

def test_a_full_payload_covers_most_of_the_feature_space(runner):
    cov = runner.payload_feature_coverage(RICH, "mortality")
    assert cov > 0.5, f"expected >50% of the mortality feature space, got {cov:.1%}"


def test_schema_and_converter_agree(runner):
    """Every field the converter reads is one the schema actually asks for."""
    asked = {f.path for f in REQUIRED_FIELDS + RECOMMENDED_FIELDS}
    for path in list(runner.PAYLOAD_CATEGORICALS) + list(runner.PAYLOAD_NUMERICS):
        assert path in asked, (
            f"`{path}` is read from payloads but never requested, so a clinician has "
            "no way to supply it and coverage overstates what is reachable")


def test_served_set_still_follows_the_measurement(runner):
    """The floor decides; this only pins that it is not bypassed."""
    from src.llm.model_runner import PAYLOAD_RETENTION_FLOOR, payload_retention

    for task in PAYLOAD_FIDELITY:
        assert (payload_retention(task) >= PAYLOAD_RETENTION_FLOOR) == (
            task in PAYLOAD_SERVED_TASKS)
