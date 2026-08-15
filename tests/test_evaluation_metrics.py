"""
The evaluation metrics themselves.

An evaluation harness is a measuring instrument, and an unchecked instrument
reports whatever it reports. Every metric below is verified against a worked
example small enough to confirm by hand, because the alternative is a report
full of confident numbers whose only evidence is that the code ran.

The cases that matter most are the ones where two metrics that look alike give
different answers — precision@k versus context_precision@k, and the two
directions of abstention error. Those distinctions are the reason the metrics
exist separately, so they are pinned here.
"""

from __future__ import annotations

import math

import pytest

from src.evaluation import metrics as M


# ── mean ─────────────────────────────────────────────────────────────────────

def test_mean_skips_undefined_rather_than_counting_them_as_zero():
    """A query with no relevant documents has undefined recall, not zero."""
    assert M.mean([1.0, None, 0.0]) == 0.5
    assert M.mean([None, None]) is None


# ── ranking ──────────────────────────────────────────────────────────────────

REL = {"a": 2, "b": 1}


def test_precision_and_recall_at_k():
    ranked = ["a", "x", "b", "y"]
    assert M.precision_at_k(ranked, REL, 2) == 0.5
    assert M.recall_at_k(ranked, REL, 2) == 0.5
    assert M.recall_at_k(ranked, REL, 4) == 1.0


def test_hit_rate_is_binary():
    assert M.hit_rate_at_k(["x", "a"], REL, 2) == 1.0
    assert M.hit_rate_at_k(["x", "y"], REL, 2) == 0.0


def test_reciprocal_rank_uses_the_first_relevant_position():
    assert M.reciprocal_rank(["x", "a", "b"], REL) == 0.5
    assert M.reciprocal_rank(["x", "y"], REL) == 0.0


def test_ndcg_is_one_for_the_ideal_ordering():
    assert M.ndcg_at_k(["a", "b"], REL, 2) == pytest.approx(1.0)


def test_ndcg_penalises_inverted_ordering():
    assert M.ndcg_at_k(["b", "a"], REL, 2) < 1.0


def test_recall_style_metrics_are_undefined_without_a_gold_positive():
    """
    Anything divided by the number of relevant documents is undefined when
    there are none, and must not be recorded as zero — a zero would be averaged
    in and drag the reported mean down for queries that were never measurable.
    """
    for fn in (M.recall_at_k, M.ndcg_at_k, M.hit_rate_at_k,
               M.context_precision_at_k):
        assert fn(["a"], {}, 1) is None, fn.__name__
    assert M.reciprocal_rank(["a"], {}) is None
    assert M.context_recall(["a"], {}) is None


def test_precision_stays_defined_without_a_gold_positive():
    """
    Precision divides by what was *retrieved*, so it is defined whenever
    anything came back. Returning a document when none is relevant is a real
    precision of zero, not an unmeasurable case.
    """
    assert M.precision_at_k(["a"], {}, 1) == 0.0
    assert M.precision_at_k([], {}, 1) is None


# ── the distinction the report leans on ──────────────────────────────────────

def test_context_precision_sees_rank_where_precision_cannot():
    """
    The whole reason both are reported.

    [relevant, irrelevant] and [irrelevant, relevant] are equally precise at
    k=2, and are not equally useful: one puts the right document first.
    """
    good, bad = ["a", "x"], ["x", "a"]
    assert M.precision_at_k(good, REL, 2) == M.precision_at_k(bad, REL, 2) == 0.5
    assert M.context_precision_at_k(good, REL, 2) == 1.0
    assert M.context_precision_at_k(bad, REL, 2) == 0.5


def test_context_precision_is_zero_when_nothing_relevant_is_retrieved():
    assert M.context_precision_at_k(["x", "y"], REL, 2) == 0.0


def test_context_recall_is_unbounded_by_k_by_default():
    """Was the supporting evidence found at all, at any rank?"""
    ranked = ["x", "y", "z", "a", "b"]
    assert M.recall_at_k(ranked, REL, 2) == 0.0
    assert M.context_recall(ranked, REL) == 1.0
    assert M.context_recall(ranked, REL, k=2) == 0.0


# ── classification ───────────────────────────────────────────────────────────

def test_classification_report_on_a_hand_checkable_case():
    pairs = [("a", "a"), ("a", "b"), ("b", "b"), ("c", "c")]
    r = M.classification_report(pairs)
    assert r.n == 4
    assert r.accuracy == 0.75
    assert r.per_class["a"]["recall"] == 0.5
    assert r.per_class["b"]["precision"] == 0.5
    assert r.confusion["a"]["b"] == 1


def test_macro_average_ignores_classes_with_no_support():
    """A label only ever predicted, never expected, must not dilute the average."""
    r = M.classification_report([("a", "a"), ("a", "z")])
    assert "z" in r.per_class
    assert r.macro_recall == pytest.approx(0.5)


def test_errors_carry_the_offending_case():
    r = M.classification_report([("a", "b")], cases=["the message"])
    assert r.errors == [{"case": "the message", "expected": "a", "predicted": "b"}]


# ── extraction ───────────────────────────────────────────────────────────────

def test_extraction_counts_a_wrong_value_as_wrong_not_partially_right():
    """Right field, wrong number is a fabricated observation with a correct label."""
    r = M.extraction_report([{
        "message": "creatinine 3.2",
        "expected": {"creatinine_max": 3.2},
        "predicted": {"creatinine_max": 1.1}}])
    assert r.precision == 0.0
    assert r.recall == 0.0
    assert r.wrong_value


def test_fabrication_requires_the_value_to_be_absent_from_the_message():
    """A value present in the text is a mislabel, not an invention."""
    present = M.extraction_report([{
        "message": "creatinine 3.2 and BUN 48",
        "expected": {"creatinine_max": 3.2},
        "predicted": {"bun_max": 3.2}}])
    assert present.fabrication_rate == 0.0

    absent = M.extraction_report([{
        "message": "no numbers here",
        "expected": {},
        "predicted": {"creatinine_max": 9.9}}])
    assert absent.fabrication_rate == 1.0
    assert absent.fabrications[0]["field"] == "creatinine_max"


def test_a_missed_field_is_not_a_fabrication():
    r = M.extraction_report([{"message": "creatinine 3.2",
                              "expected": {"creatinine_max": 3.2},
                              "predicted": {}}])
    assert r.recall == 0.0
    assert r.fabrication_rate == 0.0
    assert r.missed


def test_numeric_values_compare_across_types():
    r = M.extraction_report([{"message": "creatinine 3.2",
                              "expected": {"creatinine_max": 3.2},
                              "predicted": {"creatinine_max": "3.2"}}])
    assert r.precision == 1.0


def test_a_clean_extraction_scores_perfectly():
    r = M.extraction_report([{"message": "45M",
                              "expected": {"age": 45, "sex": "M"},
                              "predicted": {"age": 45, "sex": "m"}}])
    assert r.f1 == 1.0
    assert r.fabrication_rate == 0.0


# ── abstention ───────────────────────────────────────────────────────────────

def _case(i, should, did):
    return {"id": i, "message": i, "should_answer": should, "answered": did}


def test_the_two_error_directions_are_reported_separately():
    """
    They are not symmetric and must never be averaged.

    Under-refusal is answering without enough information — the unsafe one.
    Over-refusal is friction.
    """
    r = M.abstention_report([
        _case("under", False, True),
        _case("over", True, False),
        _case("ok_answer", True, True),
        _case("ok_refuse", False, False),
    ])
    assert r.accuracy == 0.5
    assert r.under_refusal_rate == 0.5
    assert r.over_refusal_rate == 0.5
    assert [u["id"] for u in r.under_refusals] == ["under"]
    assert [o["id"] for o in r.over_refusals] == ["over"]


def test_a_system_that_refuses_everything_scores_zero_answer_recall():
    r = M.abstention_report([_case("a", True, False), _case("b", False, False)])
    assert r.under_refusal_rate == 0.0      # safe
    assert r.over_refusal_rate == 1.0       # useless


def test_a_system_that_answers_everything_is_maximally_unsafe():
    r = M.abstention_report([_case("a", True, True), _case("b", False, True)])
    assert r.under_refusal_rate == 1.0
    assert r.over_refusal_rate == 0.0


# ── calibration ──────────────────────────────────────────────────────────────

def test_brier_is_zero_for_perfect_forecasts():
    assert M.brier_score([1.0, 0.0], [1, 0]) == 0.0


def test_brier_is_one_for_confidently_wrong_forecasts():
    assert M.brier_score([0.0, 1.0], [1, 0]) == 1.0


def test_a_perfectly_calibrated_forecast_has_zero_ece():
    """Ten cases at p=0.5, five of which occur."""
    probs = [0.5] * 10
    outcomes = [1] * 5 + [0] * 5
    assert M.expected_calibration_error(probs, outcomes) == pytest.approx(0.0)


def test_ece_detects_systematic_overconfidence():
    probs = [0.9] * 10
    outcomes = [1] * 2 + [0] * 8
    assert M.expected_calibration_error(probs, outcomes) == pytest.approx(0.7)


def test_reliability_bins_report_predicted_against_observed():
    bins = M.reliability_bins([0.05, 0.05, 0.95, 0.95], [0, 0, 1, 1], n_bins=10)
    assert len(bins) == 2
    assert bins[0]["observed_rate"] == 0.0
    assert bins[-1]["observed_rate"] == 1.0
    assert sum(b["n"] for b in bins) == 4


def test_calibration_metrics_are_none_without_data():
    assert M.brier_score([], []) is None
    assert M.expected_calibration_error([], []) is None
