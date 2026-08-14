
"""
Subgroup metrics, serve-time telemetry, and shadow replay.

Three instruments, each with a way of being quietly wrong that this file pins:

* a **slice report** that scores a twelve-row subgroup manufactures disparities
  out of sampling noise
* **telemetry** that miscounts refusals as failures makes a working gate look
  broken
* a **replay diff** that grades under-refusal as cosmetic defeats its own point

None of these need models or network.
"""

from __future__ import annotations

import json

import pytest

from src.evaluation import metrics as M


# ══ slice metrics ════════════════════════════════════════════════════════════

def _group(n, positives, prob_pos=0.9, prob_neg=0.1):
    probs = [prob_pos] * positives + [prob_neg] * (n - positives)
    outcomes = [1] * positives + [0] * (n - positives)
    return probs, outcomes


def test_auroc_is_none_when_only_one_class_is_present():
    """Not 0.5. A number would imply a measurement that was not made."""
    assert M.auroc([0.1, 0.9], [1, 1]) is None
    assert M.auroc([0.1, 0.9], [0, 0]) is None
    assert M.auroc([], []) is None


def test_auroc_is_one_for_perfect_separation():
    assert M.auroc([0.9, 0.1], [1, 0]) == pytest.approx(1.0)


def test_a_slice_below_the_support_floor_is_unmeasured_not_scored():
    """
    The guard that stops the report inventing findings.

    Mortality runs near 2%, so a small slice holds a handful of events and its
    AUROC moves on a single case. Reporting that as a disparity would be worse
    than reporting nothing.
    """
    rep = M.slice_report({"tiny": _group(100, 4)})
    s = rep["slices"][0]
    assert s["measured"] is False
    assert "support floor" in s["reason"]
    assert s["auroc"] is None
    # the base rate is still shown: it needs no model to be true
    assert s["base_rate"] == pytest.approx(0.04)


def test_row_count_alone_is_not_enough_support():
    """A large slice with too few events is still unmeasurable."""
    rep = M.slice_report({"big_but_rare": _group(5000, 3)})
    assert rep["slices"][0]["measured"] is False


def test_a_slice_with_enough_support_is_measured():
    rep = M.slice_report({"ok": _group(1000, 100)})
    s = rep["slices"][0]
    assert s["measured"] is True
    assert s["auroc"] == pytest.approx(1.0)
    assert s["events"] == 100


def test_the_summary_reports_the_gap_not_the_average():
    """
    The point of the whole report. An average hides the group it fails.
    """
    # A wider gap between the two probability levels does NOT make AUROC
    # better — it is rank-based, so 0.55 vs 0.45 separates as perfectly as
    # 0.95 vs 0.05. Degrading it requires the distributions to actually
    # overlap, which is what half the positives scoring below the negatives
    # does here.
    good = _group(1000, 100, prob_pos=0.95, prob_neg=0.05)
    poor = ([0.9] * 50 + [0.1] * 50 + [0.5] * 900,
            [1] * 100 + [0] * 900)
    rep = M.slice_report({"good": good, "poor": poor})
    s = rep["summary"]
    assert s["n_measured"] == 2
    assert s["auroc_best"]["slice"] == "good"
    assert s["auroc_worst"]["slice"] == "poor"
    assert s["auroc_gap"] > 0.4


def test_ece_gap_surfaces_a_group_the_average_would_hide():
    """Well calibrated overall, badly calibrated for one group."""
    calibrated = ([0.5] * 1000, [1] * 500 + [0] * 500)
    overconfident = ([0.9] * 1000, [1] * 100 + [0] * 900)
    rep = M.slice_report({"fair": calibrated, "overconfident": overconfident})
    s = rep["summary"]
    assert s["ece_worst"]["slice"] == "overconfident"
    assert s["ece_gap"] > 0.5


def test_unmeasured_slices_are_counted_in_the_summary():
    rep = M.slice_report({"ok": _group(1000, 100), "tiny": _group(50, 2)})
    assert rep["summary"]["n_measured"] == 1
    assert rep["summary"]["n_unmeasured"] == 1


# ══ slice construction ═══════════════════════════════════════════════════════

def test_mimic_race_strings_group_on_their_hierarchy():
    """
    "WHITE - OTHER EUROPEAN" and "ASIAN - CHINESE" share a top level. Grouping
    on the delimiter is a property of the coding scheme, not a judgement about
    which populations to merge.
    """
    from scripts.evaluation.run_slice_eval import _race_group

    assert _race_group("WHITE - OTHER EUROPEAN") == "WHITE"
    assert _race_group("ASIAN - CHINESE") == "ASIAN"
    assert _race_group("HISPANIC/LATINO - PUERTO RICAN") == "HISPANIC"
    assert _race_group("BLACK/AFRICAN AMERICAN") == "BLACK"


def test_unknown_race_keeps_its_own_group():
    """
    On this cohort UNKNOWN carries a mortality rate several times the average —
    it marks patients too unwell for demographics to be collected. Folding it
    into OTHER would hide the group the model handles worst.
    """
    from scripts.evaluation.run_slice_eval import _race_group

    assert _race_group("UNKNOWN") == "UNKNOWN"
    assert _race_group(None) == "UNKNOWN"
    assert _race_group("") == "UNKNOWN"


def test_age_bands_cover_the_cohort_range():
    from scripts.evaluation.run_slice_eval import _age_band

    assert _age_band(18) == "18-39"
    assert _age_band(58) == "55-69"
    assert _age_band(91) == "85+"
    assert _age_band(4) == "unknown"


# ══ telemetry ════════════════════════════════════════════════════════════════

def _rec(**kw):
    base = {"timestamp": "2026-08-14T00:00:00+00:00", "session_id": "s",
            "turn": 1, "status": "answered", "intent": "guideline_lookup",
            "validation": {"ok": True}, "retrieved_sources": ["D1"],
            "gate": {"status": "COMPLETE"}, "missing_information": [],
            "latency_ms": 50.0}
    base.update(kw)
    return base


def test_refusals_are_counted_as_refusals_not_failures():
    """A completeness-gated system refuses most turns. That is it working."""
    from scripts.evaluation.run_telemetry_eval import analyse

    t = analyse([_rec(), _rec(status="declined_incomplete", validation={},
                      retrieved_sources=[])])
    assert t["answer_rate"] == 0.5
    assert t["refusal_rate"] == 0.5


def test_verification_rate_covers_only_composed_answers():
    """A refusal has nothing to verify; counting it would inflate the rate."""
    from scripts.evaluation.run_telemetry_eval import analyse

    t = analyse([_rec(), _rec(status="declined_incomplete", validation={})])
    assert t["verification"]["n"] == 1
    assert t["verification"]["pass_rate"] == 1.0


def test_failed_verifications_are_listed_with_timestamps():
    """Clustering in time distinguishes a regression from a flaky edge case."""
    from scripts.evaluation.run_telemetry_eval import analyse

    bad = _rec(timestamp="2026-08-13T16:05:00+00:00",
               validation={"ok": False,
                           "checks": [{"check": 1, "name": "grounded",
                                       "passed": False, "blocking": True}]})
    t = analyse([_rec(), bad])
    v = t["verification"]
    assert v["pass_rate"] == 0.5
    assert len(v["failures"]) == 1
    assert v["failures"][0]["timestamp"] == "2026-08-13T16:05:00+00:00"
    assert v["failures"][0]["checks"] == ["1:grounded"]


def test_latency_percentiles_are_computed_from_recorded_turns():
    from scripts.evaluation.run_telemetry_eval import analyse

    t = analyse([_rec(latency_ms=x) for x in (10, 20, 30, 40, 1000)])
    assert t["latency_ms"]["p50"] == 30
    assert t["latency_ms"]["max"] == 1000


def test_records_without_latency_do_not_break_the_run():
    """The field was added after traffic already existed."""
    from scripts.evaluation.run_telemetry_eval import analyse

    t = analyse([_rec(latency_ms=None), _rec(latency_ms=None)])
    assert t["latency_ms"]["p50"] is None
    assert t["n_turns"] == 2


def test_a_truncated_log_line_is_skipped_not_fatal(tmp_path):
    from scripts.evaluation.run_telemetry_eval import load

    p = tmp_path / "audit.jsonl"
    p.write_text(json.dumps(_rec()) + "\n" + '{"status": "answ\n',
                 encoding="utf-8")
    assert len(load(p, None)) == 1


def test_telemetry_reads_a_redacted_log():
    """
    Redaction is the default, so telemetry must work without the free text.
    Every field it reads is a decision or an identifier, none of which is
    redacted.
    """
    from src.assistant.audit import AuditRecord
    from scripts.evaluation.run_telemetry_eval import analyse

    rec = AuditRecord(session_id="s", turn=1, user_message="45M septic shock",
                      status="answered", intent="risk_assessment",
                      validation={"ok": True}, latency_ms=12.0)
    t = analyse([rec.to_dict(redact=True)])
    assert t["n_turns"] == 1
    assert t["verification"]["pass_rate"] == 1.0


def test_unknown_intent_rate_is_surfaced():
    """These are the phrasings no rule covers — the next gold cases."""
    from scripts.evaluation.run_telemetry_eval import analyse

    t = analyse([_rec(intent="unknown"), _rec()])
    assert t["unknown_intent_rate"] == 0.5


# ══ shadow replay ════════════════════════════════════════════════════════════

def _run(**kw):
    base = {"status": "answered", "intent": "guideline_lookup",
            "gate_status": "COMPLETE", "blocking_fields": [], "verified": True,
            "n_citations": 3, "sources": ["D1"], "reply_len": 500}
    base.update(kw)
    return base


def _cmp(before, after):
    from scripts.evaluation.run_shadow_replay import compare

    return compare({"captured_at": "t", "runs": {"c": before}},
                   {"captured_at": "t2", "runs": {"c": after}})


def test_identical_behaviour_produces_no_diff():
    assert _cmp(_run(), _run())["changed"] == {}


def test_answering_where_it_previously_refused_is_critical():
    """Under-refusal: the unsafe direction, and the gate's whole purpose."""
    c = _cmp(_run(status="declined_incomplete", verified=None, n_citations=0),
             _run())
    assert c["counts"]["CRITICAL"] >= 1
    assert any(d["severity"] == "CRITICAL" and d["field"] == "status"
               for d in c["changed"]["c"])


def test_losing_verification_is_critical():
    c = _cmp(_run(verified=True), _run(verified=False))
    assert any(d["severity"] == "CRITICAL" and d["field"] == "verified"
               for d in c["changed"]["c"])


def test_refusing_where_it_previously_answered_is_only_a_warning():
    """New friction. Annoying, not dangerous — the asymmetry is deliberate."""
    c = _cmp(_run(), _run(status="declined_incomplete", verified=None))
    assert c["counts"]["CRITICAL"] == 0
    assert c["counts"]["WARNING"] >= 1


def test_an_intent_change_is_a_warning():
    c = _cmp(_run(), _run(intent="drug_dosing"))
    assert any(d["field"] == "intent" and d["severity"] == "WARNING"
               for d in c["changed"]["c"])


def test_losing_citations_is_a_warning():
    c = _cmp(_run(n_citations=3), _run(n_citations=1))
    assert any(d["field"] == "citations" and d["severity"] == "WARNING"
               for d in c["changed"]["c"])


def test_wording_changes_alone_are_only_informational():
    """
    Comparing full prose would flag every config edit and train its owner to
    ignore the report — the failure mode of a check that cries wolf.
    """
    c = _cmp(_run(reply_len=500), _run(reply_len=900))
    assert c["counts"]["CRITICAL"] == 0
    assert c["counts"]["WARNING"] == 0
    assert c["counts"]["INFO"] >= 1


def test_small_wording_drift_is_not_reported_at_all():
    assert _cmp(_run(reply_len=500), _run(reply_len=520))["changed"] == {}


def test_added_and_removed_scenarios_are_reported():
    from scripts.evaluation.run_shadow_replay import compare

    c = compare({"captured_at": "t", "runs": {"a": _run()}},
                {"captured_at": "t2", "runs": {"b": _run()}})
    assert c["added"] == ["b"]
    assert c["removed"] == ["a"]


def test_the_frozen_baseline_is_present_and_populated():
    """
    A missing baseline turns the replay into a no-op that always passes, which
    is worse than not having it: the report would read "no behavioural change".
    """
    from pathlib import Path

    path = Path("tests/gold/shadow_baseline.json")
    assert path.exists(), "run --freeze to create the baseline"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["n_scenarios"] > 50
    sample = next(iter(data["runs"].values()))
    for key in ("status", "intent", "verified", "n_citations"):
        assert key in sample
