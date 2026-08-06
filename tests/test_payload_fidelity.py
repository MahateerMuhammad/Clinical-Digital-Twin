"""
The payload serving gate must match what was actually measured.

`PAYLOAD_FIDELITY` decides which model outputs reach a clinician when the input is a
presentation payload rather than a stored admission record. Its numbers are AUROCs of
one specific set of promoted models on one specific test split, so every Phase 1-5
retrain invalidates them — and nothing at runtime notices, because a stale entry still
produces a confident-looking decision. The same failure mode already occurred twice
with TIER_CUTOFFS; see tests/test_tier_constants_agree.py.

These tests compare the constants against
reports/tables/payload_fidelity_evaluation.md, which is written from the models
themselves, so disagreement means one of the two is out of date.

They also pin the behaviour the gate exists to produce: withheld tasks return None and
a reason, absent features reach the boosters as NaN rather than 0.0, and the stored-row
path is not gated at all.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPORT = Path("reports/tables/payload_fidelity_evaluation.md")

pytestmark = pytest.mark.skipif(
    not REPORT.exists(),
    reason="payload_fidelity_evaluation.md not generated; run "
           "scripts/evaluation/run_payload_fidelity_eval.py",
)


def _published():
    """Parse {task: (auroc_reference, auroc_payload)} out of the report table."""
    rows = {}
    for line in REPORT.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 10 or cells[1] in ("Task", ":---"):
            continue
        if not re.fullmatch(r"[a-z_]+", cells[1]):
            continue
        rows[cells[1]] = (float(cells[5]), float(cells[6]))
    assert rows, "no result rows parsed from payload_fidelity_evaluation.md"
    return rows


# ── the constants must not drift from the measurement ───────────────────────

def test_fidelity_constants_match_the_published_report():
    from src.llm.model_runner import PAYLOAD_FIDELITY

    published = _published()
    assert set(PAYLOAD_FIDELITY) == set(published), (
        f"tasks differ: constants {sorted(PAYLOAD_FIDELITY)} vs report "
        f"{sorted(published)}")
    for task, (ref, pay) in published.items():
        got = PAYLOAD_FIDELITY[task]
        assert (round(got["reference"], 4), round(got["payload"], 4)) == (ref, pay), (
            f"{task}: constants {got} disagree with the report ({ref}, {pay}). "
            "Run: scripts/evaluation/run_payload_fidelity_eval.py --patch")


def test_served_tasks_agree_with_the_published_verdict():
    """`Served` in the report and PAYLOAD_SERVED_TASKS are the same decision."""
    from src.llm.model_runner import PAYLOAD_SERVED_TASKS

    served = set()
    for line in REPORT.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) >= 11 and re.fullmatch(r"[a-z_]+", cells[1]):
            if "yes" in cells[10]:
                served.add(cells[1])
    assert served == set(PAYLOAD_SERVED_TASKS), (
        f"report serves {sorted(served)}, code serves {sorted(PAYLOAD_SERVED_TASKS)}")


def test_served_set_is_derived_not_hand_written():
    """The floor and the served set cannot be edited out of agreement."""
    from src.llm.model_runner import (
        PAYLOAD_FIDELITY, PAYLOAD_RETENTION_FLOOR, PAYLOAD_SERVED_TASKS,
        payload_retention,
    )

    expected = {t for t in PAYLOAD_FIDELITY
                if payload_retention(t) >= PAYLOAD_RETENTION_FLOOR}
    assert expected == set(PAYLOAD_SERVED_TASKS)


def test_every_withheld_task_states_a_reason():
    from src.llm.model_runner import (
        PAYLOAD_FIDELITY, PAYLOAD_SERVED_TASKS, payload_withheld_reason,
    )

    for task in PAYLOAD_FIDELITY:
        reason = payload_withheld_reason(task)
        if task in PAYLOAD_SERVED_TASKS:
            assert reason is None
        else:
            assert reason and "AUROC" in reason, f"{task}: unhelpful reason {reason!r}"


# ── absent features must be NaN, not zero ───────────────────────────────────

def test_align_to_model_leaves_absent_features_missing():
    """
    The defect this replaces: `feat_dict[col] = 0.0` for anything unsupplied.

    A zero is a measurement the boosters split on — creatinine of zero, admission in
    year zero, no bloods sent. NaN is the missing-value representation LightGBM and
    XGBoost were fitted with.
    """
    from src.llm.feature_space import align_to_model

    frame = pd.DataFrame({"present": [3.0], "unrelated": [9.0]})
    X = align_to_model(frame, ["present", "absent"])
    assert list(X.columns) == ["present", "absent"], "column order must be the model's"
    assert X.loc[0, "present"] == 3.0
    assert np.isnan(X.loc[0, "absent"]), "an unsupplied feature was filled, not left NaN"


def test_non_numeric_values_become_missing_not_zero():
    from src.llm.feature_space import align_to_model

    X = align_to_model(pd.DataFrame({"f": ["URGENT"]}), ["f"])
    assert np.isnan(X.loc[0, "f"]), "a string coerced to 0.0 would assert a zero reading"


def test_predict_path_passes_nan_through_to_the_model():
    """Guards the wiring, not just the helper: no fillna may creep back in."""
    from src.llm.model_runner import LiveModelRunner

    captured = {}

    class FakeBooster:
        @staticmethod
        def feature_name():
            return ["a", "b"]

    class FakeModel:
        booster_ = FakeBooster()

        @staticmethod
        def predict_proba(X):
            captured["X"] = X
            return np.array([[0.4, 0.6]])

    runner = LiveModelRunner.__new__(LiveModelRunner)
    runner.lgbm_models = {"mortality": FakeModel()}
    runner.calibrators = {}

    runner._predict_prob("mortality", pd.Series({"a": 1.0}))
    assert np.isnan(captured["X"].loc[0, "b"]), \
        "feature 'b' was never supplied but did not reach the model as NaN"


# ── one-hot expansion of stored rows ────────────────────────────────────────

def test_admission_row_is_one_hot_expanded():
    """
    A stored row carries `admission_type = "URGENT"`, the booster wants
    `admission_type_URGENT`. Without the expansion the row path ran on 78 of 164
    features while claiming to be the full-record reference.
    """
    from src.llm.feature_space import encode_admission_frame

    frame = pd.DataFrame({"admission_type": ["EW EMER."], "anchor_age": [70]})
    out = encode_admission_frame(frame)
    assert "admission_type_EW_EMER." in out.columns, (
        "spaces must be rewritten to underscores; LightGBM does this at fit time and "
        f"skipping it drops every multi-word category. got {list(out.columns)}")
    assert out.loc[0, "admission_type_EW_EMER."] == 1.0
    assert out.loc[0, "anchor_age"] == 70


def test_dummies_are_built_without_drop_first():
    """
    On a single row `drop_first=True` drops that row's *own* category, so an URGENT
    admission would encode identically to the reference level.
    """
    from src.llm.feature_space import encode_admission_frame

    a = encode_admission_frame(pd.DataFrame({"admission_type": ["URGENT"]}))
    b = encode_admission_frame(pd.DataFrame({"admission_type": ["ELECTIVE"]}))
    assert list(a.columns) != list(b.columns) or not a.equals(b), \
        "two different categories encoded identically"
    assert a.loc[0, "admission_type_URGENT"] == 1.0


def test_outcome_columns_never_become_features():
    from src.llm.feature_space import encode_admission_frame

    frame = pd.DataFrame({"anchor_age": [70], "hospital_expire_flag": [1],
                          "los_days": [12.0], "hadm_id": [1]})
    out = encode_admission_frame(frame)
    for leaked in ("hospital_expire_flag", "los_days", "hadm_id"):
        assert leaked not in out.columns


def test_payload_and_row_are_told_apart():
    from src.llm.feature_space import looks_like_admission_row

    assert looks_like_admission_row(pd.Series({"hadm_id": 1, "anchor_age": 70}))
    assert not looks_like_admission_row(pd.Series({"anchor_age": 70, "gender_M": 1.0}))


# ── the report pipeline must actually run the models ────────────────────────

def test_pipeline_loads_a_model_runner_by_default():
    """
    `ClinicalReportPipeline()` must reach the models without being handed one.

    `rag_store` had a lazy loader and `model_runner` did not, so every no-argument
    caller got None, `_predict` returned {} on its first line, and the composed report
    printed "_No model predictions were supplied._" where section 2 should be. Nothing
    raised — a report without a risk section is still well-formed and still passes
    grounding — so only reading the output revealed it.
    """
    from src.llm.pipeline import ClinicalReportPipeline

    pipe = ClinicalReportPipeline()
    assert pipe.model_runner is not None
    assert hasattr(pipe.model_runner, "run_live_inference_with_uncertainty")


def test_injected_model_runner_is_still_honoured():
    """The lazy loader must not override what a caller passed in."""
    from src.llm.pipeline import ClinicalReportPipeline

    sentinel = object()
    assert ClinicalReportPipeline(model_runner=sentinel).model_runner is sentinel


def test_coverage_backstop_does_not_suppress_a_complete_payload():
    """
    The blanket coverage floor must sit below what a valid payload achieves.

    It was 0.30 while a complete payload lands at ~18%, so the backstop fired on every
    payload and discarded all five predictions — including mortality, the one task
    measured to retain most of its validated discrimination. Per-task withholding is
    the decision; this floor only catches input too sparse for that measurement to
    describe.
    """
    from src.llm.model_runner import LiveModelRunner
    from src.llm.pipeline import MIN_FEATURE_COVERAGE

    payload = {
        "demographics": {"age": 61, "gender": "F"},
        "presentation_labs": dict(LiveModelRunner.LAB_DEFAULTS),
        "active_medications": [],
    }
    coverage = LiveModelRunner().payload_feature_coverage(payload, "mortality")
    assert coverage > MIN_FEATURE_COVERAGE, (
        f"a complete payload covers {coverage:.1%} of the mortality model but the "
        f"backstop floor is {MIN_FEATURE_COVERAGE:.0%}, so every prediction is "
        "discarded before the per-task gate is consulted")


def test_withheld_reasons_are_grounded_facts():
    """
    Every number a withheld reason quotes must be in the fact store.

    Fail-closed verification rejects any numeral it cannot source. The withheld reasons
    name AUROCs and a retention percentage, so leaving them unregistered refused every
    report that withheld anything — which, after the gate landed, was all of them.
    """
    import re

    from src.llm.grounding import build_fact_store
    from src.llm.model_runner import PAYLOAD_FIDELITY, payload_withheld_reason
    from src.llm.report_composer import _payload_fidelity_constants

    # Asked through the fact store rather than by comparing floats: the store accepts
    # a rounded rendering of a registered value (0.567 for 0.5673) and a percentage
    # form, and a test that demanded exact equality would fail on text the verifier
    # accepts — testing the assertion rather than the behaviour.
    store = build_fact_store(extra_numbers=_payload_fidelity_constants())
    for task in PAYLOAD_FIDELITY:
        reason = payload_withheld_reason(task)
        if not reason:
            continue
        for literal in re.findall(r"\d+(?:\.\d+)?", reason):
            assert store.knows_number(float(literal)), (
                f"{task}: reason quotes {literal}, which the fact store cannot source "
                f"— the verifier will refuse the report. reason={reason!r}")
