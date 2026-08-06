"""
Regression tests for the Phase 11 unseen-patient agent.

Every test here corresponds to a defect the agent shipped with. The two that matter
share a root cause: `_convert_payload_to_series` mapped each payload lab onto a
whole-admission column name (`lab_creatinine_max`, `lab_bicarbonate_min`, nine more)
that the Run C leakage filter removes. None was a booster feature, so every lookup
missed and every laboratory value entered the model as 0.0.

The agent did not fail. It produced calibrated-looking probabilities, a risk tier and
a ranked SHAP explanation for a patient whose physiology was entirely zeros — and the
counterfactual simulator returned a delta of exactly 0.0 for any modification, which
a clinician would read as "this intervention would not help" rather than "this input
was never connected".

Both failure modes are silent by construction, so they are pinned here.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.path.exists("models/best_models/phase1_mortality_winning.pkl"),
    reason="Phase 1-5 models unavailable",
)

DERANGED = {
    "creatinine_max": 4.8, "bun_max": 82.0, "wbc_max": 21.0, "bicarbonate_min": 13.0,
    "sodium_min": 131.0, "potassium_max": 5.9, "platelets_min": 90.0,
    "hematocrit_min": 27.0, "glucose_max": 210.0, "anion_gap_max": 22.0,
    "chloride_max": 99.0,
}

NORMALISED = {"creatinine_max": 1.0, "bun_max": 15.0, "bicarbonate_min": 24.0,
              "anion_gap_max": 11.0}


@pytest.fixture(scope="module")
def agent():
    from src.llm.clinical_assistant import EnterpriseClinicalAgent
    return EnterpriseClinicalAgent()


@pytest.fixture
def payload():
    return {
        "primary_diagnosis": "acute kidney injury",
        "demographics": {"age": 72, "gender": "M"},
        "presentation_labs": dict(DERANGED),
        "vital_signs": {"sbp_min": 88.0, "hr_max": 122.0, "rr_max": 28.0,
                        "spo2_min": 91.0, "temp_max": 38.6},
        "active_medications": ["vancomycin"],
    }


# ── the payload must actually reach the model ───────────────────────────────

def test_payload_labs_reach_booster_features(agent, payload):
    """Each supplied lab must land on a real feature of the mortality booster."""
    import joblib

    series = agent.runner._convert_payload_to_series(payload)
    names = set(joblib.load(
        "models/best_models/phase1_mortality_winning.pkl"
    ).booster_.feature_name())

    landed = {k: v for k, v in series.items() if k in names and k.startswith("lab_")}
    assert landed, "no laboratory value reached a booster feature"
    assert all(v != 0.0 for v in landed.values()), \
        f"laboratory features arrived as zero: {landed}"


def test_no_whole_stay_column_names_in_the_mapping(agent):
    """
    The mapping must not name columns Run C removes.

    `lab_creatinine_max` and its siblings are whole-admission aggregates. Naming them
    is how every value silently became 0.0.
    """
    from src.features.leakage_filters import (
        MORTALITY_EXCLUDE_RUN_C, match_column_patterns,
    )
    mapped = [c for cols in agent.runner.PAYLOAD_LAB_FEATURES.values() for c in cols]
    leaked = match_column_patterns(mapped, MORTALITY_EXCLUDE_RUN_C)
    assert not leaked, f"payload maps onto leak-excluded columns: {leaked}"


def test_coverage_is_reported_and_partial(agent, payload):
    """
    Coverage must be measurable and honestly below 1.0.

    A payload cannot supply admission-derived features, so anything near 100% would
    mean the metric is wrong, not that the payload is complete.
    """
    cov = agent.runner.payload_feature_coverage(payload, "mortality")
    assert 0.05 < cov < 0.60, f"implausible coverage {cov:.1%}"


# ── explanations must describe this patient ─────────────────────────────────

def test_shap_lab_drivers_carry_supplied_values(agent, payload):
    series = agent.runner._convert_payload_to_series(payload)
    drivers = agent.tool_explain_shap(payload, top_k=12)["top_shap_features"]
    labs = [d for d in drivers if d["feature"] in series.index
            and d["feature"].startswith("lab_")]
    assert labs, "no laboratory feature appeared among the SHAP drivers"
    for d in labs:
        assert d["value"] == pytest.approx(float(series[d["feature"]])), \
            f"{d['feature']} explained at {d['value']}, payload says {series[d['feature']]}"


def test_different_payloads_give_different_predictions(agent, payload):
    """Zero-filled inputs made every patient identical."""
    well = {**payload, "presentation_labs": {
        "creatinine_max": 0.9, "bun_max": 12.0, "wbc_max": 6.5,
        "bicarbonate_min": 25.0, "sodium_min": 140.0, "potassium_max": 4.0,
        "platelets_min": 250.0, "hematocrit_min": 42.0, "glucose_max": 95.0,
        "anion_gap_max": 10.0, "chloride_max": 103.0}}
    a = agent.tool_run_all_models(payload)["p_mortality"]
    b = agent.tool_run_all_models(well)["p_mortality"]
    assert a != b, "deranged and normal payloads produced identical risk"
    assert a > b, "the deranged payload was not scored higher"


# ── counterfactuals must be wired to the model ──────────────────────────────

def test_counterfactual_is_connected(agent, payload):
    """A modification must change the prediction; 0.0 meant disconnection."""
    d = agent.tool_simulate_counterfactual(payload, dict(NORMALISED))["deltas"]
    deltas = {k: v for k, v in d.items() if k.startswith("delta_")}
    assert deltas, f"no task produced a delta at all: {d}"
    moved = [k for k, v in deltas.items() if abs(v) > 1e-9]
    assert moved, f"normalising labs changed nothing: {d}"


def test_counterfactual_omits_withheld_tasks(agent, payload):
    """
    A withheld task gets no delta, not a zero one.

    This used to assert deltas for ICU admission and deterioration as well. Both are
    now withheld from payload-based inference — a payload preserves 24% and 59% of
    their validated discrimination respectively — and a withheld task has no
    probability to difference. Emitting 0.0 instead would read as "normalising these
    labs has no effect on ICU risk", which is a stronger and more misleading claim
    than saying nothing.
    """
    from src.llm.model_runner import PAYLOAD_SERVED_TASKS

    res = agent.tool_simulate_counterfactual(payload, dict(NORMALISED))
    d = res["deltas"]
    assert "delta_p_mortality" in d, "mortality is served and must carry a delta"
    for key, reason in d["withheld_tasks"].items():
        assert f"delta_{key}" not in d, f"{key} is withheld but still has a delta"
        assert reason, f"{key} withheld without a stated reason"
    assert "mortality" in PAYLOAD_SERVED_TASKS


def test_counterfactual_direction_is_sane(agent, payload):
    """Normalising deranged values must not raise predicted mortality."""
    d = agent.tool_simulate_counterfactual(payload, dict(NORMALISED))["deltas"]
    assert d["delta_p_mortality"] <= 1e-9, \
        f"normalising labs increased predicted mortality by {d['delta_p_mortality']}"


def test_counterfactual_carries_its_disclaimer(agent, payload):
    """Association, not treatment effect — the caveat must ship with the result."""
    res = agent.tool_simulate_counterfactual(payload, dict(NORMALISED))
    assert res["causal_confidence"] == "Not estimated"
    assert "causal" in res["limitation"].lower() or "causal" in res["disclaimer"].lower()


# ── the report must still pass fail-closed grounding ────────────────────────

def test_report_is_grounded_with_ranked_medications(agent, payload):
    """
    Medication relevance scores are system-computed and must be in the fact store.

    They are rendered as "relevance 9.5"; when absent from the store the verifier
    withheld every report containing a ranked medication.
    """
    from src.llm.pipeline import ClinicalReportPipeline

    res = ClinicalReportPipeline().generate(payload, case_id="t_p11", use_llm=False)
    assert res.status in ("ok", "ok_no_evidence"), f"unexpected status {res.status}"
    assert res.grounding.get("ok"), res.grounding.get("violations")
