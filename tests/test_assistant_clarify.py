"""
The clarification engine.  Spec 6, 14, 15, 23, 24.

A refusal is only useful if it comes with the right question. These tests pin
the ordering (safety-critical first), the cap (never a questionnaire), the
non-repetition (spec 24), and the loop-breaker — because an assistant that will
not stop asking is a different failure from one that answers too eagerly, and
both strand the patient.
"""

from __future__ import annotations

import pytest

from src.assistant import clarify as C
from src.assistant import gate as G
from src.assistant.intents import Intent
from src.assistant.state import ConversationState, field_spec


def _state(intent=Intent.SYMPTOM_ASSESSMENT, **facts) -> ConversationState:
    st = ConversationState(session_id="s")
    st.intent = intent.value
    st.turn = 1
    for k, v in facts.items():
        st.context.record(k, v, turn=1, source_quote=str(v))
    return st


def _complete_symptom_state() -> ConversationState:
    return _state(
        symptom="headache", symptom_onset="yesterday", symptom_severity=4,
        age=34, sex="male", symptom_duration="a day",
        symptom_location="behind my eyes", symptom_trajectory="the same",
        associated_symptoms=["light sensitivity"])


# ── the basics ───────────────────────────────────────────────────────────────

def test_nothing_is_asked_when_the_gate_permits_an_answer():
    st = _complete_symptom_state()
    c = C.next_questions(st)
    assert c.has_questions is False


def test_a_refusal_produces_specific_questions():
    """Spec 23: never 'please provide additional relevant information'."""
    st = _state(symptom="chest pain")
    c = C.next_questions(st)
    assert c.has_questions
    for q in c.questions:
        assert q.text.strip().endswith("?")
        assert "additional relevant information" not in q.text.lower()


def test_at_most_three_questions_per_turn():
    """Spec 6: do not dump a questionnaire on the patient."""
    st = _state()          # nothing known: eight fields are missing
    c = C.next_questions(st)
    assert len(c.questions) <= C.MAX_QUESTIONS_PER_TURN


def test_safety_critical_questions_come_first():
    """Spec 6: the patient may only answer once, so spend it well."""
    st = _state(symptom="chest pain")
    c = C.next_questions(st)
    assert c.questions[0].level == "safety_critical"
    assert {q.field for q in c.questions} <= {"symptom_onset", "symptom_severity"}


def test_questions_use_the_reviewed_wording_from_the_field_registry():
    st = _state(symptom="chest pain")
    c = C.next_questions(st)
    for q in c.questions:
        assert q.text == field_spec(q.field).prompt


def test_the_worked_example_from_the_spec():
    """Spec 6: chest pain asks about onset and severity before anything else."""
    st = _state(symptom="chest pain")
    fields = set(C.next_questions(st).fields)
    assert fields <= {"symptom_onset", "symptom_severity"}
    assert "medical_history" not in fields
    assert "previous_diagnosis" not in fields


# ── not repeating (spec 24) ──────────────────────────────────────────────────

def test_answered_fields_are_never_asked_again():
    st = _state(symptom="chest pain", symptom_onset="an hour ago",
                symptom_severity=8)
    fields = C.next_questions(st).fields
    assert "symptom_onset" not in fields
    assert "symptom_severity" not in fields


def test_asking_marks_the_field_so_the_next_turn_knows():
    st = _state(symptom="chest pain")
    asked = C.next_questions(st).fields
    assert set(asked) <= st.asked
    for f in asked:
        assert st.times_asked(f) == 1


def test_declined_fields_are_not_asked_again():
    st = _state(symptom="chest pain", symptom_onset="an hour ago")
    st.context.decline("symptom_severity")
    c = C.next_questions(st)
    assert "symptom_severity" not in c.fields


# ── conditional ordering ─────────────────────────────────────────────────────

def test_the_enabling_question_is_asked_before_the_conditional_one():
    """Asking about pregnancy cannot settle a conditional that reads sex."""
    st = _complete_symptom_state()
    st.context = _state(
        symptom="headache", symptom_onset="yesterday", symptom_severity=4,
        age=34, symptom_duration="a day", symptom_location="behind my eyes",
        symptom_trajectory="the same",
        associated_symptoms=["light sensitivity"]).context
    c = C.next_questions(st)
    assert "sex" in c.fields
    assert "pregnancy_status" not in c.fields


# ── contradictions (spec 14) ─────────────────────────────────────────────────

def test_a_contradiction_is_asked_about_before_anything_else():
    st = _complete_symptom_state()
    st.context.record("age", 61, turn=4, source_quote="I'm 61")
    c = C.next_questions(st)
    assert len(c.questions) == 1
    assert c.questions[0].resolves_contradiction is True
    assert "34" in c.questions[0].text and "61" in c.questions[0].text


def test_no_other_question_is_asked_while_a_contradiction_is_open():
    """Every later answer would rest on a fact known to be disputed."""
    st = _state(symptom="chest pain")
    st.context.record("symptom", "back pain", turn=2, source_quote="back pain")
    c = C.next_questions(st)
    assert all(q.resolves_contradiction for q in c.questions)


# ── the loop-breaker ─────────────────────────────────────────────────────────

def test_a_field_is_not_asked_forever():
    """Asking a third time will not produce the answer the first two did not."""
    st = _state(symptom="chest pain", symptom_onset="an hour ago")
    for _ in range(C.MAX_ASK_ATTEMPTS):
        C.next_questions(st)
    c = C.next_questions(st)
    assert "symptom_severity" not in c.fields
    assert "symptom_severity" in c.stalled or c.blocked_by_stalled


def test_a_stalled_safety_critical_field_ends_the_branch_honestly():
    """Spec 33.17: say it cannot answer rather than guessing."""
    st = _state(symptom="chest pain", symptom_onset="an hour ago")
    for _ in range(C.MAX_ASK_ATTEMPTS):
        C.next_questions(st)
    c = C.next_questions(st)
    assert c.blocked_by_stalled is True
    text = c.render().lower()
    assert "cannot answer" in text
    assert "guess" in text


def test_a_stalled_field_is_never_offered_as_skippable():
    """
    The assistant must not offer to proceed without something it cannot proceed
    without.

    It did, and an independent judge caught it: a turn that asked for BUN, WBC
    and bicarbonate while offering, in the same message, to "carry on without"
    age, sex and creatinine. Declining any of those leaves the gate closed —
    `can_answer` stays False — so the offer was simply untrue.

    The old test here was written as `if c.stalled and not c.blocked_by_stalled`
    and so asserted nothing once the offer stopped being generated. A guarded
    assertion is a test that cannot fail.
    """
    st = _state(symptom="headache", symptom_onset="yesterday",
                symptom_severity=3, age=40, sex="male",
                symptom_location="temples", symptom_trajectory="better",
                associated_symptoms=["none"])
    for _ in range(C.MAX_ASK_ATTEMPTS + 1):
        C.next_questions(st)
    c = C.next_questions(st)
    assert "carry on without it" not in c.render()
    if c.stalled:
        assert c.blocked_by_stalled, (
            f"{c.stalled} was stalled without blocking — if that becomes "
            f"reachable the offer to move on needs rewriting, not restoring")


def test_a_stalled_required_field_blocks_rather_than_bargains():
    """The clinician case directly: nothing is safety-critical, all is required."""
    st = _state(intent=Intent.SYMPTOM_ASSESSMENT, symptom="chest pain",
                symptom_onset="an hour ago")
    for _ in range(C.MAX_ASK_ATTEMPTS):
        C.next_questions(st)
    c = C.next_questions(st)
    assert c.blocked_by_stalled is True
    assert "cannot answer" in c.render().lower()


def test_the_referral_line_is_dropped_for_a_clinician():
    """"A doctor can help without it" tells a doctor to consult themselves."""
    st = _state(symptom="chest pain", symptom_onset="an hour ago")
    st.context.decline("symptom_severity")
    patient = C.next_questions(st).render()
    assert "A doctor or pharmacist" in patient

    st2 = _state(symptom="chest pain", symptom_onset="an hour ago")
    st2.context.decline("symptom_severity")
    clinician = C.next_questions(st2, referral="").render()
    assert "A doctor or pharmacist" not in clinician
    assert "cannot answer" in clinician.lower()


def test_a_declined_safety_critical_field_blocks_immediately():
    """No need to stall first — a refusal is already the answer."""
    st = _state(symptom="chest pain", symptom_onset="an hour ago")
    st.context.decline("symptom_severity")
    c = C.next_questions(st)
    assert c.blocked_by_stalled is True
    assert "symptom_severity" in c.stalled
    assert not c.has_questions


# ── rendering ────────────────────────────────────────────────────────────────

def test_a_single_question_still_reads_as_a_question():
    """
    This test previously asserted the opposite — that a lone prompt was rendered
    bare — and that was wrong. Reading the transcripts showed what it produced:
    the entire reply to "Can I give full-dose enoxaparin?" was the two words
    "Peak serum creatinine", which reads as a label, not a request.

    The reviewed prompt text is still used verbatim; only framing is added. A
    test that pins a defect is worse than no test, because it defends it.
    """
    st = _state(symptom="chest pain", symptom_onset="an hour ago")
    text = C.next_questions(st).render()
    prompt = field_spec("symptom_severity").prompt
    assert prompt in text
    assert text != prompt
    assert text.startswith("Before I can answer that:")


def test_a_contradiction_question_is_not_reframed():
    """It is already a full sentence; prefixing it would read as a non-sequitur."""
    st = _state(symptom="chest pain", age=45)
    st.context.record("age", 61, turn=2, source_quote="61")
    text = C.next_questions(st).render()
    assert text.startswith("Earlier you mentioned")
    assert "Before I can answer that:" not in text


def test_several_questions_are_rendered_as_a_short_list():
    st = _state(symptom="chest pain")
    text = C.next_questions(st).render()
    assert "To answer this safely" in text
    assert text.count("•") >= 2


def test_rendering_needs_no_model():
    """A system with no LLM still asks precise questions rather than nothing."""
    import inspect

    from src.assistant import clarify
    source = inspect.getsource(clarify)
    for forbidden in ("backends", "openai", "requests"):
        assert forbidden not in source


# ── limitations carry through ────────────────────────────────────────────────

def test_limitations_are_carried_for_the_eventual_answer():
    st = _complete_symptom_state()
    c = C.next_questions(st)
    assert c.has_questions is False
    assert "medical_history" in c.limitations


def test_clarification_serialises_for_the_audit_trail():
    st = _state(symptom="chest pain")
    blob = C.next_questions(st).to_dict()
    assert blob["questions"]
    assert blob["questions"][0]["level"] == "safety_critical"
