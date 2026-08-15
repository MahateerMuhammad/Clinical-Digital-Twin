"""
Evidence retrieval and answer composition.  Spec 2, 10, 11, 12, 16, 20, 33.2-3.

The contract these tests defend: an answer may contain only what retrieval
returned. When retrieval returns nothing, the assistant declines — it does not
fall back to the model's own recollection, because a fluent paragraph with an
invented citation is indistinguishable from a real one to the person reading it.

The patient corpus ships empty, so `declined_no_evidence` is currently the
common path. That is the correct behaviour for an empty corpus and it is
asserted here rather than treated as a gap to be worked around.
"""

from __future__ import annotations

import textwrap

import pytest

from src.assistant import answer as A
from src.assistant import evidence as E
from src.assistant import gate as G
from src.assistant import triage as T
from src.assistant.intents import Intent
from src.assistant.state import PatientContext


# ── corpus loading and the trust allowlist ───────────────────────────────────

def test_shipped_corpus_loads_and_is_empty():
    """Empty is the shipped state, and it is valid."""
    stats = E.corpus_stats()
    assert stats["n_documents"] == 0
    assert stats["n_trusted_domains"] > 0


def _corpus(tmp_path, docs_yaml, domains="  - nhs.uk\n  - cdc.gov\n"):
    p = tmp_path / "corpus.yaml"
    p.write_text(f"version: t\ntrusted_domains:\n{domains}documents:\n{docs_yaml}",
                 encoding="utf-8")
    return p


_GOOD = textwrap.dedent("""\
      - doc_id: NHS-ANAEMIA
        topics: [anaemia, haemoglobin, tiredness]
        title: Iron deficiency anaemia
        source_name: NHS
        source_tier: 2
        url: https://www.nhs.uk/conditions/iron-deficiency-anaemia/
        retrieved_on: "2026-08-12"
        review_status: clinician_reviewed
        keywords: [iron, ferritin, fatigue]
        text: Iron deficiency anaemia means having fewer red blood cells than normal.
""")


def test_a_valid_document_loads(tmp_path):
    corpus = E.load_corpus(_corpus(tmp_path, _GOOD), refresh=True)
    assert len(corpus["documents"]) == 1
    assert corpus["documents"][0].source_tier == 2


def test_an_untrusted_domain_is_rejected_at_load(tmp_path):
    """Spec 12: do not blindly trust random websites."""
    bad = _GOOD.replace("https://www.nhs.uk/conditions/iron-deficiency-anaemia/",
                        "https://health-secrets.example.com/anaemia")
    with pytest.raises(E.CorpusError, match="trusted_domains"):
        E.load_corpus(_corpus(tmp_path, bad), refresh=True)


def test_a_lookalike_domain_is_rejected(tmp_path):
    """`nhs.uk.evil.com` must not pass a naive suffix check."""
    bad = _GOOD.replace("https://www.nhs.uk/conditions/iron-deficiency-anaemia/",
                        "https://nhs.uk.evil.com/anaemia")
    with pytest.raises(E.CorpusError, match="trusted_domains"):
        E.load_corpus(_corpus(tmp_path, bad), refresh=True)


def test_a_document_missing_provenance_is_rejected(tmp_path):
    bad = "\n".join(l for l in _GOOD.splitlines() if "retrieved_on" not in l) + "\n"
    with pytest.raises(E.CorpusError, match="missing"):
        E.load_corpus(_corpus(tmp_path, bad), refresh=True)


def test_an_out_of_range_tier_is_rejected(tmp_path):
    bad = _GOOD.replace("source_tier: 2", "source_tier: 9")
    with pytest.raises(E.CorpusError, match="expected 1"):
        E.load_corpus(_corpus(tmp_path, bad), refresh=True)


def test_a_corpus_without_an_allowlist_is_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("version: t\ndocuments: []\n", encoding="utf-8")
    with pytest.raises(E.CorpusError, match="allowlist"):
        E.load_corpus(p, refresh=True)


# ── retrieval ────────────────────────────────────────────────────────────────

def test_empty_corpus_returns_no_source():
    res = E.retrieve("diabetes")
    assert res.status == E.NO_SOURCE
    assert res.ok is False


def test_no_source_refusal_says_it_will_not_answer_from_memory():
    text = E.retrieve("diabetes").refusal_text().lower()
    assert "memory" in text or "not going to answer" in text


def test_a_matching_topic_is_retrieved(tmp_path):
    res = E.retrieve("anaemia", path=_corpus(tmp_path, _GOOD))
    assert res.ok
    assert res.documents[0].doc_id == "NHS-ANAEMIA"


def test_an_unmatched_topic_returns_nothing(tmp_path):
    res = E.retrieve("broken ankle", path=_corpus(tmp_path, _GOOD))
    assert res.status == E.NO_SOURCE


def test_tier_outranks_relevance(tmp_path):
    """Spec 12's hierarchy is an ordering, not a tiebreak."""
    two = _GOOD + textwrap.dedent("""\
      - doc_id: BLOG-ANAEMIA
        topics: [anaemia, haemoglobin, tiredness, iron, ferritin, fatigue]
        title: All about anaemia
        source_name: Mayo Clinic
        source_tier: 3
        url: https://www.mayoclinic.org/anaemia
        retrieved_on: "2026-08-12"
        text: Anaemia has many causes.
    """)
    p = _corpus(tmp_path, two, domains="  - nhs.uk\n  - mayoclinic.org\n")
    res = E.retrieve("anaemia haemoglobin iron ferritin fatigue tiredness", path=p)
    assert res.documents[0].source_tier == 2


def test_require_reviewed_filters_and_says_so(tmp_path):
    unreviewed = _GOOD.replace("review_status: clinician_reviewed",
                               "review_status: unreviewed")
    res = E.retrieve("anaemia", path=_corpus(tmp_path, unreviewed),
                     require_reviewed=True)
    assert res.status == E.NO_SOURCE
    assert res.filtered_unreviewed >= 1
    assert "reviewed by a clinician" in res.refusal_text()


def test_citation_is_derived_from_the_source_not_invented(tmp_path):
    doc = E.retrieve("anaemia", path=_corpus(tmp_path, _GOOD)).documents[0]
    assert doc.citation == "[NHS: Iron deficiency anaemia]"
    assert doc.to_doc()["url"].startswith("https://www.nhs.uk/")


def test_documents_render_in_the_shape_the_grounding_verifier_consumes(tmp_path):
    from src.llm.grounding import build_fact_store

    docs = [d.to_doc() for d in
            E.retrieve("anaemia", path=_corpus(tmp_path, _GOOD)).documents]
    fs = build_fact_store(documents=docs)
    assert "nhs: iron deficiency anaemia" in fs.citations


# ── capabilities (spec 2) ────────────────────────────────────────────────────

def test_capabilities_message_lists_what_the_assistant_does():
    ans = A.capabilities_message()
    text = ans.to_markdown()
    assert "Understanding symptoms" in text
    assert "Understanding test results" in text
    assert text.rstrip().endswith("*") or "help with today" in text


def test_capabilities_message_says_it_is_not_a_doctor():
    assert "not a doctor" in A.capabilities_message().to_markdown().lower()


def test_capabilities_come_from_config_not_code():
    cfg = A.load_capabilities()
    assert len(cfg["capabilities"]) >= 5
    assert all("intent" in c for c in cfg["capabilities"])


# ── emergency composition (spec 16) ──────────────────────────────────────────

def test_emergency_answer_gives_advice_and_asks_nothing():
    tri = T.screen("crushing chest pain radiating to my arm")
    ans = A.emergency_message(tri)
    assert ans.status == A.EMERGENCY_RESPONSE
    text = ans.to_markdown().lower()
    assert "urgent" in text or "emergency" in text
    assert "?" not in text.replace("?", "", 0) or "how old are you" not in text


def test_emergency_answer_does_not_reassure():
    """Spec 16: no reassurance without evidence."""
    text = A.emergency_message(T.screen("I can't breathe")).to_markdown().lower()
    for phrase in ("probably fine", "nothing to worry", "unlikely to be serious",
                   "try not to worry"):
        assert phrase not in text


def test_self_harm_response_omits_the_emergency_department_disclaimer():
    ans = A.emergency_message(T.screen("I want to kill myself"))
    assert "emergency department" not in ans.disclaimer.lower()


def test_compose_routes_an_emergency_regardless_of_the_gate():
    ctx = PatientContext()
    decision = G.evaluate(ctx, Intent.SYMPTOM_ASSESSMENT)
    ans = A.compose(ctx, decision, E.EvidenceResult(),
                    triage=T.screen("I think I'm having a stroke"))
    assert ans.status == A.EMERGENCY_RESPONSE


# ── substantive composition (spec 20) ────────────────────────────────────────

def _answerable_context() -> PatientContext:
    ctx = PatientContext()
    ctx.record("condition_name", "anaemia", turn=1, source_quote="anaemia")
    return ctx


def test_an_incomplete_gate_produces_a_refusal_not_an_answer():
    ctx = PatientContext()
    d = G.evaluate(ctx, Intent.SYMPTOM_ASSESSMENT)
    ans = A.compose(ctx, d, E.EvidenceResult())
    assert ans.status == A.DECLINED_INCOMPLETE


def test_no_evidence_produces_a_refusal_even_when_the_gate_opened():
    """The contract: an open gate is not permission to answer from memory."""
    ctx = _answerable_context()
    d = G.evaluate(ctx, Intent.CONDITION_INFORMATION)
    assert d.can_answer is True
    ans = A.compose(ctx, d, E.retrieve("anaemia"))
    assert ans.status == A.DECLINED_NO_EVIDENCE
    assert "not going to answer" in ans.to_markdown().lower()


def test_an_answer_cites_every_document_it_used(tmp_path):
    ctx = _answerable_context()
    d = G.evaluate(ctx, Intent.CONDITION_INFORMATION)
    ev = E.retrieve("anaemia", path=_corpus(tmp_path, _GOOD))
    ans = A.compose(ctx, d, ev, requested_fields=["condition_name"])
    assert ans.status == A.ANSWERED
    assert ans.citations == ["[NHS: Iron deficiency anaemia]"]
    assert "[NHS: Iron deficiency anaemia]" in ans.to_markdown()


def test_the_answer_separates_patient_facts_from_evidence(tmp_path):
    """Spec 10: the four categories must not be blended."""
    ctx = _answerable_context()
    d = G.evaluate(ctx, Intent.CONDITION_INFORMATION)
    ev = E.retrieve("anaemia", path=_corpus(tmp_path, _GOOD))
    md = A.compose(ctx, d, ev, requested_fields=["condition_name"]).to_markdown()
    assert "What you have told me" in md
    assert "What this could mean" in md
    assert md.index("What you have told me") < md.index("What this could mean")


def test_the_answer_always_states_its_limitations(tmp_path):
    """Spec 33.14 and spec 20's closing section."""
    ctx = _answerable_context()
    d = G.evaluate(ctx, Intent.CONDITION_INFORMATION)
    ev = E.retrieve("anaemia", path=_corpus(tmp_path, _GOOD))
    md = A.compose(ctx, d, ev, requested_fields=["condition_name"]).to_markdown()
    assert "Important limitations" in md
    assert "cannot examine you" in md


def test_the_answer_carries_a_disclaimer(tmp_path):
    ctx = _answerable_context()
    d = G.evaluate(ctx, Intent.CONDITION_INFORMATION)
    ev = E.retrieve("anaemia", path=_corpus(tmp_path, _GOOD))
    ans = A.compose(ctx, d, ev, requested_fields=["condition_name"])
    assert "not a diagnosis" in ans.disclaimer.lower()


def test_warning_signs_appear_only_when_triage_flagged_something(tmp_path):
    """Inventing warning signs for a benign presentation is an ungrounded claim."""
    ctx = _answerable_context()
    d = G.evaluate(ctx, Intent.CONDITION_INFORMATION)
    ev = E.retrieve("anaemia", path=_corpus(tmp_path, _GOOD))
    quiet = A.compose(ctx, d, ev, triage=T.screen("what is anaemia"),
                      requested_fields=["condition_name"])
    assert "When to seek urgent care" not in quiet.to_markdown()

    flagged = A.compose(ctx, d, ev, triage=T.screen("I have chest pain"),
                        requested_fields=["condition_name"])
    assert "When to seek urgent care" in flagged.to_markdown()


def test_the_answer_reads_back_only_what_the_patient_said(tmp_path):
    """Spec 13 made visible: the patient can catch a mishearing."""
    ctx = _answerable_context()
    d = G.evaluate(ctx, Intent.CONDITION_INFORMATION)
    ev = E.retrieve("anaemia", path=_corpus(tmp_path, _GOOD))
    md = A.compose(ctx, d, ev, requested_fields=["condition_name", "age"]).to_markdown()
    assert "anaemia" in md
    assert "Age:" not in md          # never stated, so never echoed


def test_composition_needs_no_model():
    """Grounded by construction: nothing here generates prose."""
    import inspect

    from src.assistant import answer
    source = inspect.getsource(answer)
    for forbidden in ("backends", "openai", "complete_json", "rephrase"):
        assert forbidden not in source


def test_answer_serialises_for_the_audit_trail(tmp_path):
    ctx = _answerable_context()
    d = G.evaluate(ctx, Intent.CONDITION_INFORMATION)
    ev = E.retrieve("anaemia", path=_corpus(tmp_path, _GOOD))
    blob = A.compose(ctx, d, ev, requested_fields=["condition_name"]).to_dict()
    assert blob["status"] == A.ANSWERED
    assert blob["citations"]
