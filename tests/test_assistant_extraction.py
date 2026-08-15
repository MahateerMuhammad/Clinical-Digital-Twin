"""
Fact extraction.  Spec 13, 26, 27, 33.1.

This is the one stage where a language model touches patient information, and a
value it invents becomes, to every later stage, something the patient said. The
tests below are adversarial by design: each one hands the extractor a plausible
fabrication and asserts it does not get in.

The load-bearing filter is the quote requirement. A model that decides an
unmentioned patient is 65 has to produce a verbatim span from the message saying
so, and there isn't one.
"""

from __future__ import annotations

import json

import pytest

from src.assistant import extraction as X
from src.assistant.state import PatientContext


class FakeBackend:
    """A backend returning canned JSON. `available` mirrors the real interface."""

    available = True

    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def complete_json(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        return self.payload if isinstance(self.payload, str) else json.dumps(self.payload)


class BrokenBackend:
    available = True

    def complete_json(self, system_prompt, user_prompt):
        raise RuntimeError("rate limited")


@pytest.fixture
def ctx():
    return PatientContext()


# ── the quote filter ─────────────────────────────────────────────────────────

def test_a_fact_with_a_real_quote_is_admitted(ctx):
    msg = "I'm 52 and I take aspirin every morning"
    be = FakeBackend({"facts": [{"field": "age", "value": 52, "quote": "I'm 52"}]})
    res = X.extract(msg, ctx, turn=1, backend=be)
    assert res.accepted and res.accepted[0].field == "age"
    assert ctx.get("age") == 52.0


def test_a_fact_whose_quote_is_absent_is_refused(ctx):
    """
    The fabrication filter. The patient never mentioned an age.

    Asserts on the *age* rather than on an empty ``accepted`` list. The list is
    no longer empty: the deterministic floor also reads "headache" out of this
    message, correctly and with a real quote. Pinning `accepted == []` made the
    absence of a second extractor part of the fabrication contract, which it
    never was — the contract is that an unquoted value does not get through.
    """
    msg = "I have a headache"
    be = FakeBackend({"facts": [
        {"field": "age", "value": 65, "quote": "I am 65 years old"}]})
    res = X.extract(msg, ctx, turn=1, backend=be)
    assert "age" not in [p.field for p in res.accepted]
    assert ctx.get("age") is None
    assert "quote does not appear" in res.rejected[0][1]


def test_a_fact_with_no_quote_at_all_is_refused(ctx):
    be = FakeBackend({"facts": [{"field": "age", "value": 65, "quote": ""}]})
    res = X.extract("I have a headache", ctx, turn=1, backend=be)
    assert "age" not in [p.field for p in res.accepted]
    assert ctx.get("age") is None


def test_a_paraphrased_quote_is_refused(ctx):
    """Paraphrasing the patient is not quoting them."""
    msg = "the pain has been there since Tuesday"
    be = FakeBackend({"facts": [
        {"field": "symptom_onset", "value": "Tuesday",
         "quote": "it started on Tuesday"}]})
    res = X.extract(msg, ctx, turn=1, backend=be)
    assert res.accepted == []


def test_quote_matching_tolerates_whitespace_and_case(ctx):
    msg = "I'm  52   years old"
    be = FakeBackend({"facts": [
        {"field": "age", "value": 52, "quote": "i'm 52 years old"}]})
    assert X.extract(msg, ctx, turn=1, backend=be).accepted


# ── the schema filter ────────────────────────────────────────────────────────

def test_an_invented_field_is_dropped(ctx):
    """Spec 15: a field that does not exist cannot be collected."""
    be = FakeBackend({"facts": [
        {"field": "blood_type", "value": "O+", "quote": "I'm O+"}]})
    res = X.extract("I'm O+", ctx, turn=1, backend=be)
    assert res.accepted == []
    assert "unknown field" in res.rejected[0][1]


def test_a_field_outside_the_allowed_set_is_dropped(ctx):
    """Only what the current intent needs; spec 15's minimum-necessary rule."""
    msg = "I'm 52 and my marital status is single"
    be = FakeBackend({"facts": [
        {"field": "age", "value": 52, "quote": "I'm 52"},
        {"field": "medication_dose", "value": "10mg", "quote": "I'm 52"}]})
    res = X.extract(msg, ctx, turn=1, backend=be, allowed=["age"])
    assert res.fields == ["age"]
    assert any("not needed" in r for _, r in res.rejected)


def test_the_prompt_only_offers_allowed_fields(ctx):
    be = FakeBackend({"facts": []})
    X.extract("hello", ctx, turn=1, backend=be, allowed=["age", "sex"])
    _, user_prompt = be.calls[0]
    assert "- age (" in user_prompt
    assert "- sex (" in user_prompt
    assert "medication_dose" not in user_prompt


# ── the value filter ─────────────────────────────────────────────────────────

def test_an_impossible_value_is_refused_not_clamped(ctx):
    msg = "I am 240 years old"
    be = FakeBackend({"facts": [
        {"field": "age", "value": 240, "quote": "I am 240 years old"}]})
    res = X.extract(msg, ctx, turn=1, backend=be)
    assert res.accepted == []
    assert ctx.get("age") is None
    assert "outside the possible range" in res.rejected[0][1]


def test_an_unparseable_value_is_refused(ctx):
    msg = "I am quite old"
    be = FakeBackend({"facts": [
        {"field": "age", "value": "quite old", "quote": "I am quite old"}]})
    assert X.extract(msg, ctx, turn=1, backend=be).accepted == []


def test_extraction_surfaces_a_contradiction(ctx):
    ctx.record("age", 45, turn=1, source_quote="I am 45")
    msg = "actually I'm 52"
    be = FakeBackend({"facts": [
        {"field": "age", "value": 52, "quote": "actually I'm 52"}]})
    res = X.extract(msg, ctx, turn=2, backend=be)
    assert res.contradictions
    assert ctx.is_contradicted("age")


# ── parsing ──────────────────────────────────────────────────────────────────

def test_malformed_json_is_discarded_not_repaired(ctx):
    be = FakeBackend('{"facts": [{"field": "age", "value": 52')
    res = X.extract("I'm 52", ctx, turn=1, backend=be)
    # falls through to the deterministic floor rather than salvaging
    assert res.used_model is False


def test_json_wrapped_in_prose_is_located():
    facts, failed = X.parse_response(
        'Here you go:\n```json\n{"facts": [{"field": "age", "value": 5, '
        '"quote": "x"}]}\n```\nHope that helps!')
    assert failed is False
    assert facts[0]["field"] == "age"


def test_a_bare_list_is_accepted():
    facts, failed = X.parse_response('[{"field": "age", "value": 5, "quote": "x"}]')
    assert failed is False and len(facts) == 1


def test_empty_response_is_a_parse_failure():
    assert X.parse_response("") == ([], True)


def test_prose_only_response_is_a_parse_failure():
    _, failed = X.parse_response("I could not find any facts.")
    assert failed is True


# ── the deterministic floor ──────────────────────────────────────────────────

def test_no_backend_still_extracts_high_confidence_facts(ctx):
    res = X.extract("I'm 52 and it's a 7 out of 10", ctx, turn=1, backend=None)
    assert res.used_model is False
    assert ctx.get("age") == 52.0
    assert ctx.get("symptom_severity") == 7.0


def test_a_failing_backend_falls_back_rather_than_losing_the_turn(ctx):
    res = X.extract("I am 40 years old", ctx, turn=1, backend=BrokenBackend())
    assert res.used_model is False
    assert ctx.get("age") == 40.0
    assert any("backend call failed" in r for _, r in res.rejected)


def test_the_fallback_extracts_less_never_something_different(ctx):
    """
    Degrading to 'asks more questions' is safe; degrading to 'guesses more' is
    not. The fallback must never produce a fact the message does not state.
    """
    res = X.extract("my chest feels tight", ctx, turn=1, backend=None)
    assert res.accepted == []
    assert ctx.known_fields() == set()


def test_the_fallback_reads_sex_without_inventing_it(ctx):
    X.extract("I'm a woman with a headache", ctx, turn=1, backend=None)
    assert ctx.get("sex") == "female"


def test_severity_pattern_does_not_fire_on_an_age(ctx):
    """'I'm 52' must not become a severity of 52, which is out of bounds anyway."""
    X.extract("I'm 52", ctx, turn=1, backend=None)
    assert ctx.get("symptom_severity") is None


# ── audit ────────────────────────────────────────────────────────────────────

def test_rejections_are_recorded_for_the_audit_trail(ctx):
    """A rejection is the system catching a fabrication; it should be countable."""
    be = FakeBackend({"facts": [
        {"field": "age", "value": 65, "quote": "I am 65"}]})
    res = X.extract("I have a headache", ctx, turn=1, backend=be)
    blob = res.to_dict()
    assert blob["rejected"]
    assert blob["rejected"][0]["proposal"]["field"] == "age"
    assert "age" not in [p["field"] for p in blob["accepted"]]


def test_result_reports_whether_a_model_was_used(ctx):
    be = FakeBackend({"facts": []})
    assert X.extract("hi", ctx, turn=1, backend=be).used_model is True
    assert X.extract("hi", ctx, turn=1, backend=None).used_model is False


def test_an_unavailable_backend_is_not_called(ctx):
    class Unavailable(FakeBackend):
        available = False

    be = Unavailable({"facts": [{"field": "age", "value": 9, "quote": "x"}]})
    X.extract("I am 40 years old", ctx, turn=1, backend=be)
    assert be.calls == []


# ── the model and the floor are additive ─────────────────────────────────────

def test_the_floor_fills_what_the_model_left_alone(ctx):
    """
    A successful model call must not *lose* information.

    It used to: the deterministic floor ran only when the model failed, so a
    model that returned fourteen laboratory values and no diagnosis produced a
    turn where the gate asked for a diagnosis the clinician had already given —
    while the floor reads "septic shock" out of that same sentence every time.
    """
    msg = "72F septic shock, creatinine 3.2"
    be = FakeBackend({"facts": [
        {"field": "creatinine_max", "value": 3.2, "quote": "3.2"}]})
    res = X.extract(msg, ctx, turn=1, backend=be)
    fields = {p.field for p in res.accepted}
    assert "creatinine_max" in fields          # from the model
    assert "primary_diagnosis" in fields       # from the floor
    assert res.used_model is True


def test_the_model_wins_a_field_both_propose(ctx):
    """The floor is a floor, not an override — it never displaces a model value."""
    msg = "I am 52 years old"
    be = FakeBackend({"facts": [{"field": "age", "value": 52, "quote": "52"}]})
    res = X.extract(msg, ctx, turn=1, backend=be)
    ages = [p for p in res.accepted if p.field == "age"]
    assert len(ages) == 1
    assert ctx.get("age") == 52.0


def test_floor_proposals_face_the_same_filters(ctx):
    """
    The floor is not a trusted channel. It proposes; the filters decide.

    Guards the merge above: if floor proposals were appended after the filter
    loop rather than before it, this would be the test that noticed.
    """
    import inspect
    src = inspect.getsource(X.extract)
    merge = src.index("_deterministic(text, allowed)")
    filters = src.index("_quote_is_real")
    assert merge < filters, "floor proposals must be merged before filtering"


# ── a guideline question names its own topic ─────────────────────────────────

def test_a_guideline_question_yields_its_topic(ctx):
    """
    "What are the guidelines for managing psoriasis?" used to be answered with
    "Before I can answer that: Primary working diagnosis" — asking for the thing
    the clinician had just named. An independent judge scored it routing 0.
    """
    X.extract("What are the guidelines for managing psoriasis?",
              ctx, turn=1, backend=None)
    assert ctx.get("primary_diagnosis") == "psoriasis"


def test_the_topic_pattern_needs_a_guidance_word(ctx):
    """
    Narrow by design. A question about risk names no condition, and reading one
    out of it would put an unasserted diagnosis into the payload the models
    score.
    """
    X.extract("what is her risk?", ctx, turn=1, backend=None)
    assert ctx.get("primary_diagnosis") is None


def test_the_topic_pattern_matches_plurals(ctx):
    """`guideline` vs `guidelines` — the defect this codebase keeps shipping."""
    for phrasing in ("What are the guidelines for psoriasis?",
                     "What is the guideline for psoriasis?",
                     "What are the protocols for treating psoriasis?"):
        c = PatientContext()
        X.extract(phrasing, c, turn=1, backend=None)
        assert c.get("primary_diagnosis") == "psoriasis", phrasing


def test_an_uncovered_topic_is_declined_not_answered(ctx):
    """Capturing the topic must not make the corpus look bigger than it is."""
    from src.llm.terminology import normalise_diagnosis
    X.extract("What are the guidelines for managing psoriasis?",
              ctx, turn=1, backend=None)
    assert normalise_diagnosis(ctx.get("primary_diagnosis")).concept is None
