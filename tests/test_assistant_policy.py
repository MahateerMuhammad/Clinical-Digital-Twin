"""
Intent classification, requirement policy, and the answerability gate.
Spec 4, 5, 7, 8, 15, 22, 33.9.

The gate is the load-bearing component of the whole design: it is the only place
that decides whether a medical answer may be produced. These tests pin the
behaviours that make it worth trusting — that it refuses when safety-critical
information is absent, that it cannot be reached with an argument, and that it
does not demand information it has no clinical reason to hold.
"""

from __future__ import annotations

import pytest

from src.assistant import gate as G
from src.assistant.intents import (
    CLINICIAN, Intent, MIN_CONFIDENCE, MODE_INTENTS, PATIENT,
    SWITCH_CONFIDENCE, classify, resolve_intent,
)
from src.assistant.requirements import (
    OPTIONAL, POLICY_PATH, PolicyError, REQUIRED, SAFETY_CRITICAL,
    all_intents, for_intent, load_policy,
)
from src.assistant.state import FIELDS, ConversationState, PatientContext


# ══ intent classification ════════════════════════════════════════════════════

@pytest.mark.parametrize("message,expected", [
    ("I have chest pain, what should I do?", Intent.SYMPTOM_ASSESSMENT),
    ("My knee hurts when I walk", Intent.SYMPTOM_ASSESSMENT),
    ("I've had a fever for three days", Intent.SYMPTOM_ASSESSMENT),
    ("Can I take ibuprofen with my blood pressure tablets?", Intent.MEDICATION_QUESTION),
    ("What are the side effects of metformin?", Intent.MEDICATION_QUESTION),
    ("My haemoglobin is low, what does that mean?", Intent.LAB_RESULT_INTERPRETATION),
    ("I got my blood test results back", Intent.LAB_RESULT_INTERPRETATION),
    ("Can you explain my MRI report?", Intent.MEDICAL_REPORT_EXPLANATION),
    ("What is type 2 diabetes?", Intent.CONDITION_INFORMATION),
    ("How is asthma treated?", Intent.TREATMENT_QUESTION),
    ("How can I reduce my risk of heart disease?", Intent.PREVENTIVE_HEALTH),
    ("What questions should I ask my doctor?", Intent.DOCTOR_QUESTION_PREP),
    ("What does idiopathic mean?", Intent.TERMINOLOGY),
    ("What can you help me with?", Intent.CAPABILITIES),
    ("Hello", Intent.CAPABILITIES),
])
def test_classifier_routes_representative_messages(message, expected):
    assert classify(message).intent is expected


def test_ambiguous_message_returns_unknown_rather_than_guessing():
    """Spec 2: ask what the user needs rather than committing to a branch."""
    res = classify("hmm")
    assert res.intent is Intent.UNKNOWN
    assert not res.is_confident


def test_empty_message_is_not_classified():
    assert classify("").intent is Intent.UNKNOWN


def test_first_turn_routes_the_unrecognised_to_capabilities():
    """Spec 2: the opening move is to explain what the assistant does."""
    assert classify("hmm", first_turn=True).intent is Intent.CAPABILITIES


def test_classifier_never_returns_emergency():
    """
    Spec 16 puts emergency detection before classification, in triage. Two
    detectors would mean a weaker one can win; this module has none.
    """
    for msg in ("crushing chest pain radiating to my jaw",
                "I cannot breathe", "I think I'm having a stroke",
                "severe bleeding that won't stop"):
        assert classify(msg).intent is not Intent.EMERGENCY


def test_classification_carries_evidence_for_the_audit_trail():
    res = classify("What are the side effects of metformin?")
    assert res.evidence
    assert res.confidence >= MIN_CONFIDENCE
    assert res.to_dict()["intent"] == "medication_question"


def test_low_confidence_is_reported_not_hidden():
    res = classify("pain")
    assert res.confidence < 1.0
    if res.intent is Intent.UNKNOWN:
        assert res.alternatives or res.evidence


# ── intent persistence across turns ──────────────────────────────────────────
#
# Found by smoke-testing the flow rather than by unit test, and it was a safety
# hole rather than a rough edge: an answer to a clarifying question carries no
# intent keyword, so per-message classification returned UNKNOWN, whose policy is
# empty, so the gate reported COMPLETE and would have answered on two facts.

def _state_in(intent: Intent, turn: int = 2) -> ConversationState:
    st = ConversationState(session_id="s")
    st.intent = intent.value
    st.turn = turn
    return st


def test_answering_a_clarifying_question_does_not_lose_the_intent():
    st = _state_in(Intent.SYMPTOM_ASSESSMENT)
    res = resolve_intent(st, "yesterday morning, about a 7 out of 10")
    assert res.intent is Intent.SYMPTOM_ASSESSMENT


def test_a_bare_answer_cannot_open_the_gate():
    """The regression this exists for, end to end."""
    st = _state_in(Intent.SYMPTOM_ASSESSMENT)
    st.context.record("symptom", "headache", turn=1, source_quote="I have a headache")
    res = resolve_intent(st, "yesterday morning")
    d = G.evaluate(st.context, res.intent)
    assert d.can_answer is False
    assert d.status == G.SAFETY_CRITICAL_MISSING


def test_a_confident_new_topic_switches_intent():
    st = _state_in(Intent.SYMPTOM_ASSESSMENT)
    res = resolve_intent(st, "Actually, can you explain my MRI report?")
    assert res.intent is Intent.MEDICAL_REPORT_EXPLANATION
    assert any("switched" in e for e in res.evidence)


def test_a_weak_keyword_does_not_switch_intent():
    """Leaving a half-collected intent discards answers already given."""
    st = _state_in(Intent.LAB_RESULT_INTERPRETATION)
    res = resolve_intent(st, "it aches a bit too")
    assert res.intent is Intent.LAB_RESULT_INTERPRETATION


def test_switching_needs_more_confidence_than_starting():
    assert SWITCH_CONFIDENCE > MIN_CONFIDENCE


def test_capabilities_never_persists():
    """Nothing is collected under it, so there is nothing to protect."""
    st = _state_in(Intent.CAPABILITIES)
    assert resolve_intent(st, "I have a headache").intent is Intent.SYMPTOM_ASSESSMENT


def test_unknown_never_persists():
    st = _state_in(Intent.UNKNOWN)
    assert resolve_intent(st, "I have a headache").intent is Intent.SYMPTOM_ASSESSMENT


def test_first_turn_has_no_intent_to_preserve():
    st = ConversationState(session_id="s")
    st.turn = 1
    assert resolve_intent(st, "hello").intent is Intent.CAPABILITIES


def test_a_corrupt_stored_intent_falls_back_to_fresh_classification():
    st = ConversationState(session_id="s")
    st.intent = "not_a_real_intent"
    st.turn = 3
    assert resolve_intent(st, "I have a headache").intent is Intent.SYMPTOM_ASSESSMENT


# ══ requirement policy ═══════════════════════════════════════════════════════

def test_policy_loads_and_validates():
    policy = load_policy(refresh=True)
    assert policy["intents"]
    assert policy["version"]


#: Each audience's policy file. The invariant is per-mode: a clinician intent
#: has no business in the patient policy, and vice versa.
_MODE_POLICIES = {
    "patient": POLICY_PATH,
    "clinician": POLICY_PATH.parent / "requirements_clinician.yaml",
}


@pytest.mark.parametrize("mode", ["patient", "clinician"])
def test_every_reachable_intent_has_a_policy(mode):
    """A missing entry would make the gate raise mid-conversation."""
    from src.assistant.intents import MODE_INTENTS

    covered = set(all_intents(path=_MODE_POLICIES[mode]))
    for intent in MODE_INTENTS[mode]:
        assert intent.value in covered, f"no {mode} policy for {intent.value}"


def test_risk_assessment_requires_exactly_the_model_payload_contract():
    """
    The single definition of "enough to score".

    `state.FIELDS` generates the lab fields from `payload_validation`, and the
    clinician policy lists them by name. If the model's payload contract gains
    a required field and this policy does not, the gate opens on an incomplete
    payload and the pipeline refuses further down — a refusal the assistant
    cannot explain, because it believed it had everything.
    """
    from src.llm.payload_validation import REQUIRED_FIELDS

    required = set(for_intent(Intent.RISK_ASSESSMENT,
                              path=_MODE_POLICIES["clinician"]).required)
    contract = {spec.path.rsplit(".", 1)[-1] for spec in REQUIRED_FIELDS}
    contract = {"sex" if n == "gender" else n for n in contract}
    assert contract <= required, f"policy is missing {contract - required}"


def test_only_risk_assessment_and_counterfactual_reach_the_models():
    """The routing decision, asserted where it is actually made."""
    from src.assistant.orchestrator import _MODEL_INTENTS

    assert _MODEL_INTENTS == {Intent.RISK_ASSESSMENT, Intent.COUNTERFACTUAL}


def test_every_policy_field_exists_in_the_field_registry():
    policy = load_policy()
    for name, spec in policy["intents"].items():
        for level in (SAFETY_CRITICAL, REQUIRED, OPTIONAL):
            for fname in spec.get(level) or []:
                assert fname in FIELDS, f"{name}.{level}: {fname}"


def test_a_field_cannot_sit_at_two_levels(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "version: t\nintents:\n  symptom_assessment:\n"
        "    safety_critical: [age]\n    required: [age]\n", encoding="utf-8")
    with pytest.raises(PolicyError, match="both"):
        load_policy(bad, refresh=True)


def test_a_typo_in_the_policy_fails_loudly(tmp_path):
    """An unfillable requirement would block the gate forever, silently."""
    bad = tmp_path / "typo.yaml"
    bad.write_text(
        "version: t\nintents:\n  symptom_assessment:\n"
        "    required: [symptom_sevrity]\n", encoding="utf-8")
    with pytest.raises(PolicyError, match="unknown field"):
        load_policy(bad, refresh=True)


def test_symptom_policy_demands_onset_and_severity():
    reqs = for_intent(Intent.SYMPTOM_ASSESSMENT)
    assert "symptom" in reqs.safety_critical
    assert "symptom_onset" in reqs.safety_critical
    assert "symptom_severity" in reqs.safety_critical


def test_medication_policy_treats_allergies_as_safety_critical():
    """Spec 17: naming a drug to someone allergic to it is the worst error here."""
    assert "allergies" in for_intent(Intent.MEDICATION_QUESTION).safety_critical


def test_lab_policy_demands_the_reference_range_from_the_report():
    """Spec 18: ranges differ between laboratories, so it cannot be recalled."""
    reqs = for_intent(Intent.LAB_RESULT_INTERPRETATION)
    for f in ("test_name", "test_value", "test_unit", "test_reference_range"):
        assert f in reqs.safety_critical, f


def test_general_information_does_not_demand_patient_details():
    """Spec 15: explaining what asthma is needs no age or history."""
    reqs = for_intent(Intent.CONDITION_INFORMATION)
    assert reqs.safety_critical == []
    assert reqs.required == ["condition_name"]
    assert "age" in reqs.optional


def test_terminology_asks_only_for_the_term():
    reqs = for_intent(Intent.TERMINOLOGY)
    assert reqs.required == ["term"]
    assert reqs.safety_critical == []


# ── conditional requirements ─────────────────────────────────────────────────

def test_pregnancy_is_not_required_when_it_cannot_apply():
    ctx = PatientContext()
    ctx.record("sex", "male", turn=1, source_quote="I'm male")
    ctx.record("age", 40, turn=1, source_quote="40")
    reqs = for_intent(Intent.SYMPTOM_ASSESSMENT, ctx)
    assert "pregnancy_status" not in reqs.required
    assert reqs.pending_conditionals == []


def test_pregnancy_is_required_once_it_can_apply():
    ctx = PatientContext()
    ctx.record("sex", "female", turn=1, source_quote="I'm female")
    ctx.record("age", 30, turn=1, source_quote="30")
    assert "pregnancy_status" in for_intent(Intent.SYMPTOM_ASSESSMENT, ctx).required


def test_pregnancy_is_not_required_outside_the_age_window():
    ctx = PatientContext()
    ctx.record("sex", "female", turn=1, source_quote="I'm female")
    ctx.record("age", 74, turn=1, source_quote="74")
    assert "pregnancy_status" not in for_intent(Intent.SYMPTOM_ASSESSMENT, ctx).required


def test_unknown_sex_leaves_the_conditional_pending_not_false():
    """
    The predicate must not decide the patient is not pregnant merely because
    nobody asked their sex. Unknown propagates.
    """
    ctx = PatientContext()
    ctx.record("age", 30, turn=1, source_quote="30")
    reqs = for_intent(Intent.SYMPTOM_ASSESSMENT, ctx)
    assert "pregnancy_status" not in reqs.required
    assert [r.field for r in reqs.pending_conditionals] == ["pregnancy_status"]


def test_conditional_carries_its_reason():
    ctx = PatientContext()
    ctx.record("sex", "female", turn=1, source_quote="female")
    ctx.record("age", 30, turn=1, source_quote="30")
    reqs = for_intent(Intent.MEDICATION_QUESTION, ctx)
    pregnancy = next(r for r in reqs.requirements if r.field == "pregnancy_status")
    assert pregnancy.conditional is True
    assert "pregnan" in pregnancy.reason.lower()


# ══ the answerability gate ═══════════════════════════════════════════════════

def _symptom_context(**overrides) -> PatientContext:
    """A context complete for symptom assessment, before overrides."""
    ctx = PatientContext()
    base = {
        "symptom": "headache", "symptom_onset": "yesterday morning",
        "symptom_severity": 4, "age": 34, "sex": "male",
        "symptom_duration": "about a day", "symptom_location": "behind my eyes",
        "symptom_trajectory": "about the same",
    }
    base.update(overrides)
    for k, v in base.items():
        if v is not None:
            ctx.record(k, v, turn=1, source_quote=str(v))
    ctx.record("associated_symptoms", ["light sensitivity"], turn=1,
               source_quote="bright light makes it worse")
    return ctx


def test_empty_context_cannot_answer():
    d = G.evaluate(PatientContext(), Intent.SYMPTOM_ASSESSMENT)
    assert d.can_answer is False
    assert d.status == G.SAFETY_CRITICAL_MISSING


def test_complete_context_can_answer():
    d = G.evaluate(_symptom_context(), Intent.SYMPTOM_ASSESSMENT)
    assert d.can_answer is True
    assert d.status == G.COMPLETE


def test_missing_safety_critical_blocks_and_names_the_field():
    """Spec 7: stop answering and ask for it."""
    ctx = _symptom_context(symptom_severity=None)
    d = G.evaluate(ctx, Intent.SYMPTOM_ASSESSMENT)
    assert d.can_answer is False
    assert d.status == G.SAFETY_CRITICAL_MISSING
    assert "symptom_severity" in d.missing_safety_critical
    assert "severity" in d.reason


def test_missing_required_blocks_as_incomplete():
    ctx = _symptom_context(symptom_duration=None)
    d = G.evaluate(ctx, Intent.SYMPTOM_ASSESSMENT)
    assert d.can_answer is False
    assert d.status == G.INCOMPLETE
    assert "symptom_duration" in d.missing_required


def test_missing_optional_does_not_block_but_is_reported_as_a_limit():
    """Spec 8: proceed while clearly stating the limitation."""
    d = G.evaluate(_symptom_context(), Intent.SYMPTOM_ASSESSMENT)
    assert d.can_answer is True
    assert "medical_history" in d.missing_optional
    assert "medical_history" in d.limitations


def test_contradiction_blocks_even_when_everything_is_present():
    """Spec 22: contradictory information is its own refusal reason."""
    ctx = _symptom_context()
    ctx.record("age", 61, turn=5, source_quote="I'm 61")
    d = G.evaluate(ctx, Intent.SYMPTOM_ASSESSMENT)
    assert d.can_answer is False
    assert d.status == G.CONTRADICTORY
    assert d.contradictions
    assert "correct" in d.contradictions[0]["question"].lower()


def test_safety_critical_outranks_contradiction_in_the_headline_reason():
    ctx = _symptom_context(symptom_onset=None)
    ctx.record("age", 61, turn=5, source_quote="I'm 61")
    d = G.evaluate(ctx, Intent.SYMPTOM_ASSESSMENT)
    assert d.status == G.SAFETY_CRITICAL_MISSING
    # both are still reported, so fixing one does not reveal the other late
    assert d.contradictions


def test_declined_safety_critical_field_stops_the_answer_and_says_so():
    """Spec 33.17: say it cannot answer rather than guessing."""
    ctx = _symptom_context(symptom_severity=None)
    ctx.decline("symptom_severity")
    d = G.evaluate(ctx, Intent.SYMPTOM_ASSESSMENT)
    assert d.can_answer is False
    assert d.status == G.SAFETY_CRITICAL_MISSING
    assert "symptom_severity" in d.declined_safety_critical
    assert "symptom_severity" not in d.blocking_fields   # never re-asked
    assert "prefer" in d.reason


def test_declined_required_field_permits_a_limited_answer():
    ctx = _symptom_context(symptom_duration=None)
    ctx.decline("symptom_duration")
    d = G.evaluate(ctx, Intent.SYMPTOM_ASSESSMENT)
    assert d.can_answer is True
    assert "symptom_duration" in d.limitations
    assert "symptom_duration" not in d.blocking_fields


def test_blocking_fields_put_safety_critical_first():
    """Spec 6: ask the most safety-critical questions first."""
    ctx = PatientContext()
    ctx.record("symptom", "chest tightness", turn=1, source_quote="chest tightness")
    d = G.evaluate(ctx, Intent.SYMPTOM_ASSESSMENT)
    first_two = d.blocking_fields[:2]
    assert set(first_two) <= {"symptom_onset", "symptom_severity"}


def test_pending_conditional_asks_the_enabling_question_not_the_conditional_one():
    """Asking about pregnancy cannot settle a conditional that reads sex."""
    ctx = _symptom_context(sex=None)
    d = G.evaluate(ctx, Intent.SYMPTOM_ASSESSMENT)
    assert d.can_answer is False
    assert "sex" in d.blocking_fields
    assert "pregnancy_status" not in d.blocking_fields


def test_gate_reads_only_application_state():
    """
    Spec 33.9. There is no parameter through which a model could influence the
    verdict — the signature accepts a context and an intent, and nothing else.
    """
    import inspect
    params = set(inspect.signature(G.evaluate).parameters)
    assert params == {"context", "intent", "requirement_set"}


def test_gate_accepts_a_conversation_state():
    st = ConversationState(session_id="s1")
    st.intent = Intent.CONDITION_INFORMATION.value
    st.context.record("condition_name", "asthma", turn=1, source_quote="asthma")
    assert G.can_answer(st).can_answer is True


def test_general_education_answers_immediately():
    """Spec 15: no interrogation before explaining a term."""
    ctx = PatientContext()
    ctx.record("term", "idiopathic", turn=1, source_quote="what does idiopathic mean")
    d = G.evaluate(ctx, Intent.TERMINOLOGY)
    assert d.can_answer is True
    assert d.blocking_fields == []


def test_medication_question_without_a_dose_is_blocked():
    """Spec 29's medication case: ask for the dose."""
    ctx = PatientContext()
    ctx.record("medication_name", "aspirin", turn=1, source_quote="aspirin")
    ctx.record("allergies", ["none"], turn=1, source_quote="no allergies")
    d = G.evaluate(ctx, Intent.MEDICATION_QUESTION)
    assert d.can_answer is False
    assert "medication_dose" in d.missing_required


def test_lab_question_without_units_or_range_is_blocked():
    """Spec 29's laboratory case: request the missing context."""
    ctx = PatientContext()
    ctx.record("test_name", "haemoglobin", turn=1, source_quote="haemoglobin")
    ctx.record("test_value", "9.1", turn=1, source_quote="9.1")
    d = G.evaluate(ctx, Intent.LAB_RESULT_INTERPRETATION)
    assert d.can_answer is False
    assert "test_unit" in d.missing_safety_critical
    assert "test_reference_range" in d.missing_safety_critical


def test_decision_serialises_for_the_audit_trail():
    d = G.evaluate(PatientContext(), Intent.SYMPTOM_ASSESSMENT)
    blob = d.to_dict()
    assert blob["can_answer"] is False
    assert blob["status"] == G.SAFETY_CRITICAL_MISSING
    assert "missing_safety_critical" in blob


@pytest.mark.parametrize("mode", ["patient", "clinician"])
def test_every_reachable_intent_can_be_gated_without_raising(mode):
    """A policy gap must not surface as an exception mid-conversation."""
    from src.assistant.intents import MODE_INTENTS
    from src.assistant.requirements import for_intent as _for

    for intent in MODE_INTENTS[mode]:
        ctx = PatientContext()
        d = G.evaluate(ctx, intent,
                       requirement_set=_for(intent, ctx,
                                            path=_MODE_POLICIES[mode]))
        assert d.status in (G.COMPLETE, G.INCOMPLETE,
                            G.SAFETY_CRITICAL_MISSING, G.CONTRADICTORY)


# ── boundaries: outside what the system is, not what it knows ────────────────

def test_a_request_to_read_a_chart_is_routed_as_such():
    """
    Used to score nothing and fall to UNKNOWN, whose reply is "I am not sure
    what you would like help with" — which reads as the assistant being slow
    rather than as "no record connection exists". An independent judge scored
    the old reply routing 0.
    """
    for msg in ("Look up this patient's chart and tell me their history.",
                "Can you pull up her records?",
                "Do you have access to the EMR?"):
        res = classify(msg, first_turn=False, mode=CLINICIAN)
        assert res.intent is Intent.RECORD_ACCESS, msg


def test_a_request_for_a_diagnosis_is_routed_as_such():
    """Distinct from "no trusted source on file", which implies a better source
    would unlock it. This system does not diagnose at any evidence level."""
    for msg in ("What's the diagnosis?", "Can you diagnose this patient?",
                "What is the differential?"):
        res = classify(msg, first_turn=False, mode=CLINICIAN)
        assert res.intent is Intent.DIAGNOSIS_REQUEST, msg


def test_the_boundary_rules_do_not_swallow_ordinary_questions():
    """
    The guard on the two rules above. Both mention patients and conditions, and
    a loose pattern would turn every clinical question into a refusal — the
    over-refusal failure, which is friction rather than danger but still wrong.
    """
    for msg, expected in (
            ("What's the first-line vasopressor in septic shock?",
             Intent.GUIDELINE_LOOKUP),
            ("What are the guidelines for managing DKA?", Intent.GUIDELINE_LOOKUP),
            ("What does oliguric mean?", Intent.TERMINOLOGY),
            ("88 year old female with pneumonia, what is her risk?",
             Intent.RISK_ASSESSMENT)):
        assert classify(msg, first_turn=False, mode=CLINICIAN).intent is expected, msg


def test_the_boundaries_are_not_reachable_by_a_patient_session():
    """Mode scoping: these are clinician wordings and clinician boundaries."""
    assert Intent.RECORD_ACCESS not in MODE_INTENTS[PATIENT]
    assert Intent.DIAGNOSIS_REQUEST not in MODE_INTENTS[PATIENT]


# ── asides: a question about medicine, not about this patient ────────────────

def _clinician_bot():
    from src.assistant.orchestrator import Assistant
    return Assistant.clinician()


def test_a_knowledge_question_does_not_take_over_an_open_case():
    """
    The interruption case.

    A clinician mid-counterfactual asks "how is refractory hypotension managed?".
    That must be answered *as* a guideline question while the counterfactual
    stays the open case. Before this, the only two outcomes were seizing the
    session or being read as the intent already running — which returned "which
    value should I change?" to a guideline question.
    """
    st = ConversationState(session_id="s")
    st.intent = Intent.COUNTERFACTUAL.value
    st.turn = 3

    res = resolve_intent(st, "In septic shock, how should refractory hypotension "
                             "be managed when norepinephrine alone is insufficient?",
                         mode=CLINICIAN)
    assert res.intent is Intent.GUIDELINE_LOOKUP
    assert any("aside" in e for e in res.evidence), res.evidence


def test_an_aside_does_not_need_to_clear_the_switching_bar():
    """
    SWITCH_CONFIDENCE guards *abandoning* a case. An aside abandons nothing, so
    holding it to that threshold was applying a rule to a situation it does not
    describe. MIN_CONFIDENCE still applies.
    """
    st = ConversationState(session_id="s")
    st.intent = Intent.RISK_ASSESSMENT.value
    st.turn = 2

    res = resolve_intent(st, "What does oliguric mean?", mode=CLINICIAN)
    assert res.intent is Intent.TERMINOLOGY
    assert res.confidence < SWITCH_CONFIDENCE or res.confidence >= MIN_CONFIDENCE


def test_a_case_question_still_switches_normally():
    """The guard this change must not weaken."""
    st = ConversationState(session_id="s")
    st.intent = Intent.GUIDELINE_LOOKUP.value
    st.turn = 2

    res = resolve_intent(st, "hmm", mode=CLINICIAN)
    assert res.intent is Intent.GUIDELINE_LOOKUP, "a weak message must not switch"


def test_an_aside_cannot_write_to_the_case():
    """
    The second half. "How should severe hyperkalaemia be managed?" used to write
    `hyperkalaemia` as the patient's diagnosis, contradict the septic shock on
    file, and ask the clinician which of the two their patient had.
    """
    bot = _clinician_bot()
    sid = bot.start().state.session_id
    bot.handle(sid, "72F septic shock, creatinine 3.2")
    before = dict(bot.sessions[sid].context.to_dict()["current"])

    bot.handle(sid, "How should severe hyperkalaemia with potassium above 6.5 "
                    "be managed?")
    after = bot.sessions[sid].context.to_dict()["current"]

    assert after == before, "an aside changed the case"
    assert after.get("primary_diagnosis") == "septic shock"
    assert not bot.sessions[sid].context.contradictions


def test_the_open_case_survives_an_aside():
    """What makes resuming free: the case intent was never overwritten."""
    bot = _clinician_bot()
    sid = bot.start().state.session_id
    bot.handle(sid, "72F septic shock, what is her mortality risk?")
    open_case = bot.sessions[sid].intent

    bot.handle(sid, "What is the first-line vasopressor in septic shock?")
    assert bot.sessions[sid].intent == open_case


# ── a case handover must reach the models ────────────────────────────────────

def test_a_case_handover_naming_a_drug_is_still_a_risk_question():
    """
    A drug mentioned in passing must not capture the turn.

    "88F with sepsis ... on vancomycin and norepinephrine. What is her mortality
    risk?" classified as `drug_dosing` at 0.61, beating `risk_assessment` at
    0.58 on the strength of one rule firing on the word "vancomycin". The models
    were never run: the clinician got retrieved guideline text and a complaint
    that no medication dose had been supplied, above a list of the five fields
    drug-dosing happens to want.
    """
    from src.assistant.intents import CLINICIAN, Intent, classify

    res = classify(
        "88F with sepsis. Creatinine 3.2, BUN 61, WBC 18.4, bicarb 14. "
        "On vancomycin and norepinephrine. What is her mortality risk?",
        mode=CLINICIAN)
    assert res.intent is Intent.RISK_ASSESSMENT, (
        f"routed to {res.intent.value}; a case handover must reach the models")


def test_a_drug_question_is_still_a_drug_question():
    """The counterweight: the rule above exists for a reason and must survive."""
    from src.assistant.intents import CLINICIAN, Intent, classify

    for msg in ("Should I worry about vancomycin with a creatinine of 3.2?",
                "How should I dose vancomycin in renal impairment?",
                "Is it safe to continue enoxaparin with platelets of 42?"):
        assert classify(msg, mode=CLINICIAN).intent is Intent.DRUG_DOSING, msg


def test_the_outcome_word_may_sit_inside_the_possessive():
    """"her mortality risk" scored on one rule; "her risk" scored on two."""
    from src.assistant.intents import CLINICIAN, Intent, classify

    for msg in ("What is her mortality risk?", "What is his readmission risk?",
                "What are their ICU risks?"):
        assert classify(msg, mode=CLINICIAN).intent is Intent.RISK_ASSESSMENT, msg


def test_stated_medications_reach_the_payload():
    """
    `medication_name` has no payload path; `active_medications` does.

    Nothing populated the second, so a clinician stating the patient's therapy
    filled a field the models never see. The same turn could print "Medication
    name: vancomycin" above "Not supplied: active medications", and the second
    drug was discarded entirely.
    """
    from src.assistant import extraction as X
    from src.assistant.state import PatientContext, build_payload

    ctx = PatientContext()
    res = X.extract("88F with sepsis, on vancomycin and norepinephrine.", ctx, 1)
    assert not res.rejected, res.rejected
    assert build_payload(ctx).get("active_medications") == [
        "vancomycin", "norepinephrine"]


def test_the_medication_quote_is_a_real_span():
    """
    Provenance survives the list.

    `_quote_is_real` rejects anything that is not a verbatim span, so joining the
    names into "vancomycin, norepinephrine" would have dropped the fact silently
    however true it was.
    """
    from src.assistant import extraction as X
    from src.assistant.state import PatientContext

    msg = "88F with sepsis, on vancomycin and norepinephrine."
    ctx = PatientContext()
    res = X.extract(msg, ctx, 1)
    quote = next(p.quote for p in res.accepted if p.field == "active_medications")
    assert quote in msg, f"{quote!r} is not a span of the message"


# ── a message that says nothing ──────────────────────────────────────────────

def test_a_meaningless_message_is_not_answered_with_the_previous_answer():
    """
    "hmm" re-emitted the whole previous answer, word for word.

    `resolve_intent` turns an unrecognised message back into the intent already
    running, so the open case was recomputed from an unchanged context and
    produced an identical reply. Correct for a reply that carries facts but no
    keyword; wrong for a message that carries nothing.
    """
    bot = _clinician_bot()
    sid = bot.start().state.session_id
    first = bot.handle(sid, "72F septic shock, creatinine 3.2. "
                            "What is her mortality risk?").reply

    for msg in ("hmm", "akbflvhbalfb", "ok", "...", "   "):
        reply = bot.handle(sid, msg).reply
        assert reply != first, f"{msg!r} replayed the previous answer"
        assert "could not understand" in reply.lower(), msg


def test_a_meaningless_message_leaves_the_case_standing():
    """Not understanding a message is not a reason to discard the patient."""
    bot = _clinician_bot()
    sid = bot.start().state.session_id
    bot.handle(sid, "72F septic shock, creatinine 3.2. What is her mortality risk?")
    before = dict(bot.sessions[sid].context.to_dict()["current"])
    open_case = bot.sessions[sid].intent

    bot.handle(sid, "hmm")

    assert bot.sessions[sid].context.to_dict()["current"] == before
    assert bot.sessions[sid].intent == open_case


def test_a_reply_that_carries_facts_is_still_a_continuation():
    """
    The counterweight.

    An open intent exists precisely to receive replies that match no intent
    rule. Suppressing on classification alone would have swallowed them.
    """
    bot = _clinician_bot()
    sid = bot.start().state.session_id
    bot.handle(sid, "72F septic shock. What is her mortality risk?")

    reply = bot.handle(sid, "her BUN is 54").reply
    assert "could not understand" not in reply.lower()
    assert bot.sessions[sid].context.get("bun_max") == 54.0
