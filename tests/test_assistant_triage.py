"""
Emergency detection.  Spec 16, 29.

These tests are calibration, not coverage. Triage trades false alarms against
misses, and the trade has a correct direction: a warning nobody needed costs a
patient a moment's worry, a missed stroke costs them the treatment window. Every
ambiguous case below asserts that the system errs upward.
"""

from __future__ import annotations

import pytest

from src.assistant import triage as T


# ── unambiguous emergencies (spec 16's list) ─────────────────────────────────

@pytest.mark.parametrize("message,rule", [
    ("I can't breathe", "breathing_difficulty"),
    ("my husband is unresponsive", "loss_of_consciousness"),
    ("I passed out this morning", "loss_of_consciousness"),
    ("her face is drooping and her speech is slurred", "stroke"),
    ("worst headache of my life, came on in seconds", "stroke"),
    ("he just had a seizure", "seizure"),
    ("the bleeding won't stop", "severe_bleeding"),
    ("I'm vomiting blood", "severe_bleeding"),
    ("my throat is closing after the injection", "anaphylaxis"),
    ("my lips are swelling up", "anaphylaxis"),
    ("I took too many paracetamol", "overdose"),
    ("she swallowed bleach", "overdose"),
    ("I want to kill myself", "self_harm"),
    ("there's a rash that doesn't fade", "sepsis"),
])
def test_emergency_is_detected(message, rule):
    res = T.screen(message)
    assert res.severity == T.EMERGENCY, message
    assert rule in [f.rule_id for f in res.flags], message


def test_emergency_bypasses_questioning():
    """Spec 16: no long information-collection workflow."""
    res = T.screen("I can't breathe")
    assert res.bypasses_questioning is True


def test_every_emergency_flag_carries_advice():
    """A rule that fires with nothing to say is worse than no rule."""
    for msg in ("I can't breathe", "I think I'm having a stroke",
                "she swallowed bleach", "I want to end my life"):
        for flag in T.screen(msg).flags:
            assert flag.advice.strip(), (msg, flag.rule_id)


def test_self_harm_response_is_not_an_emergency_department_referral():
    """The default 'go to A&E' framing is the wrong first step for a crisis."""
    flag = next(f for f in T.screen("I want to kill myself").flags
                if f.rule_id == "self_harm")
    assert flag.suppress_default_disclaimer is True
    assert "crisis" in flag.advice.lower() or "line" in flag.advice.lower()


# ── the chest-pain calibration ───────────────────────────────────────────────

def test_bare_chest_pain_is_urgent_not_emergency():
    """
    Spec 6 walks chest pain into a question flow, so bare chest pain must not
    short-circuit to the emergency response — three months of intermittent
    soreness is the same words.
    """
    res = T.screen("I have chest pain, what should I do?")
    assert res.severity == T.URGENT_ASSESS
    assert res.bypasses_questioning is False


def test_bare_chest_pain_still_shows_warning_signs_immediately():
    """Spec 16: do not reassure without evidence."""
    flag = T.screen("I have chest pain").flags[0]
    assert flag.warning_signs
    assert any("arm" in w.lower() or "jaw" in w.lower() for w in flag.warning_signs)


@pytest.mark.parametrize("message", [
    "crushing chest pain",
    "chest pain spreading to my jaw",
    "chest pain and I'm sweating",
    "chest tightness and short of breath",
    "sudden severe chest pain",
    "chest pain and feeling sick",
])
def test_chest_pain_escalates_on_acute_features(message):
    res = T.screen(message)
    assert res.severity == T.EMERGENCY, message
    assert any(f.escalated for f in res.flags), message


def test_escalation_reads_earlier_turns():
    """
    The two facts can arrive in different messages and still describe one
    presentation.
    """
    res = T.screen("now it's spreading to my jaw",
                   history=["I have chest pain"])
    assert res.severity == T.EMERGENCY


def test_a_trigger_does_not_escalate_itself():
    """'severe chest pain' must not match `severe` inside its own span."""
    res = T.screen("I get chest pain when I climb stairs")
    assert res.severity == T.URGENT_ASSESS


def test_escalated_flag_uses_the_emergency_wording():
    flag = T.screen("crushing chest pain radiating to my arm").flags[0]
    assert "heart attack" in flag.advice.lower()


# ── suppression, and its deliberate limits ───────────────────────────────────

def test_explicit_negation_suppresses_an_urgent_rule():
    res = T.screen("I have a cough but no chest pain")
    assert res.severity == T.NONE
    assert res.suppressed


def test_negation_does_not_reach_across_a_clause_boundary():
    """
    The failure that matters: 'no fever but I do have chest pain' must keep the
    chest pain.
    """
    res = T.screen("I have no fever, but I do have chest pain")
    assert res.severity == T.URGENT_ASSESS


def test_historical_mention_is_suppressed():
    res = T.screen("I had chest pain back in 2019")
    assert res.severity == T.NONE


def test_negation_never_suppresses_an_emergency_rule():
    """
    A negation-scope error that hides a stroke is unrecoverable; one that
    produces an extra warning is not. Emergency rules cannot be talked down.
    """
    res = T.screen("it's not like I can't breathe exactly, but")
    assert res.severity == T.EMERGENCY


def test_third_person_still_fires():
    """Someone asking on behalf of another person still needs the answer."""
    res = T.screen("my father is having crushing chest pain")
    assert res.severity == T.EMERGENCY


def test_suppressions_are_recorded_for_review():
    """A decision not to warn has to be auditable."""
    res = T.screen("I have no chest pain")
    assert res.suppressed
    assert "reason" in res.suppressed[0]
    assert res.suppressed[0]["rule_id"] == "chest_pain"


# ── quiet on ordinary messages ───────────────────────────────────────────────

@pytest.mark.parametrize("message", [
    "What is type 2 diabetes?",
    "Can I take ibuprofen with my blood pressure tablets?",
    "My haemoglobin came back at 11.2, is that low?",
    "What questions should I ask my doctor?",
    "I've had a mild headache for two days",
    "Hello",
    "",
])
def test_ordinary_messages_do_not_trigger(message):
    assert T.screen(message).severity == T.NONE


def test_the_word_fit_alone_does_not_fire_seizure():
    """A morphological near-miss is how a rule set becomes untrustworthy."""
    assert T.screen("I don't feel very fit these days").severity == T.NONE


# ── configuration integrity ──────────────────────────────────────────────────

def test_rules_load_and_compile():
    cfg = T.load_rules(refresh=True)
    assert cfg["rules"]
    assert cfg["version"]


def test_every_rule_has_a_valid_severity_and_advice():
    for rule in T.load_rules()["rules"]:
        assert rule["severity"] in (T.EMERGENCY, T.URGENT_ASSESS)
        assert rule["patterns"]
        if rule["severity"] == T.EMERGENCY:
            assert rule["emergency_advice"]
        else:
            assert rule["urgent_advice"]


def test_an_emergency_rule_without_advice_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "version: t\nrules:\n  - id: x\n    severity: emergency\n"
        "    patterns: ['\\\\bboom\\\\b']\n", encoding="utf-8")
    with pytest.raises(T.TriageConfigError, match="no emergency_advice"):
        T.load_rules(bad, refresh=True)


def test_a_bad_regex_is_rejected_at_load(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "version: t\nrules:\n  - id: x\n    severity: emergency\n"
        "    emergency_advice: go\n    patterns: ['[unclosed']\n", encoding="utf-8")
    with pytest.raises(T.TriageConfigError, match="bad pattern"):
        T.load_rules(bad, refresh=True)


def test_duplicate_rule_ids_are_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "version: t\nrules:\n"
        "  - {id: x, severity: emergency, emergency_advice: a, patterns: ['\\\\ba\\\\b']}\n"
        "  - {id: x, severity: emergency, emergency_advice: b, patterns: ['\\\\bb\\\\b']}\n",
        encoding="utf-8")
    with pytest.raises(T.TriageConfigError, match="duplicate"):
        T.load_rules(bad, refresh=True)


def test_triage_needs_no_model_or_network():
    """
    The stage that must work when everything else is broken. If this ever grows
    an import of backends, rag_corpus or requests, that is the regression.
    """
    import inspect

    from src.assistant import triage
    source = inspect.getsource(triage)
    for forbidden in ("backends", "rag_corpus", "requests", "httpx", "openai"):
        assert forbidden not in source, f"triage must not depend on {forbidden}"


def test_result_serialises_for_the_audit_trail():
    blob = T.screen("crushing chest pain").to_dict()
    assert blob["severity"] == T.EMERGENCY
    assert blob["flags"][0]["escalated"] is True
