"""
The clinician HTTP surface.

These tests run without the risk models: ``service.start()`` tolerates absent
artefacts, and the routes that need a model return 503 rather than 500. That is
deliberate coverage — a guideline lookup needs no model, and the API must stay
useful when the boosters are not there.

What is asserted here is the *contract*, not the domain logic. Validation, the
coverage floor and the grounding check are tested where they live; duplicating
them at this layer would only prove that FastAPI can call a function.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi", reason="backend extras not installed")

from fastapi.testclient import TestClient   # noqa: E402

from backend.main import app               # noqa: E402
from backend.service import service        # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


INCOMPLETE = {"payload": {
    "demographics": {"age": 45, "gender": "M"},
    "primary_diagnosis": "septic shock",
    "presentation_labs": {"creatinine_max": 3.2}}}

COMPLETE = {"payload": {
    "demographics": {"age": 45, "gender": "M"},
    "primary_diagnosis": "septic shock",
    "presentation_labs": {"creatinine_max": 3.2, "bun_max": 48, "wbc_max": 19.5,
                          "bicarbonate_min": 16, "sodium_min": 132,
                          "potassium_max": 5.1, "platelets_min": 96,
                          "hematocrit_min": 29, "glucose_max": 180},
    "vital_signs": {"sbp_min": 82, "hr_max": 124}}}


# ── meta ─────────────────────────────────────────────────────────────────────

def test_health_reports_what_actually_loaded(client):
    body = client.get("/api/health").json()
    assert body["audience"] == "clinician"
    assert "models_loaded" in body
    assert body["guidelines"]["n_records"] > 0


def test_health_surfaces_the_unreviewed_corpus_count(client):
    """
    The single most important caveat, exposed rather than buried.

    "23 records, 0 clinician-reviewed" is what a clinician needs to know before
    reading anything this service says, so it is in the health payload and not
    only in a README.
    """
    assert "n_clinician_reviewed" in client.get("/api/health").json()["guidelines"]


def test_every_route_is_registered(client):
    paths = client.get("/openapi.json").json()["paths"]
    for expected in ("/api/health", "/api/models/validate", "/api/models/predict",
                     "/api/models/report", "/api/models/whatif",
                     "/api/models/describe", "/api/assistant/sessions",
                     "/api/assistant/sessions/{session_id}",
                     "/api/assistant/sessions/{session_id}/messages"):
        assert expected in paths, expected


# ── validation, which needs no models ────────────────────────────────────────

def test_validate_names_every_missing_field(client):
    body = client.post("/api/models/validate", json=INCOMPLETE).json()
    assert body["ok"] is False
    missing = {m["path"] for m in body["missing_required"]}
    assert "presentation_labs.bun_max" in missing
    assert "vital_signs.sbp_min" in missing


def test_validate_accepts_a_complete_payload(client):
    assert client.post("/api/models/validate", json=COMPLETE).json()["ok"] is True


def test_validate_works_without_the_models(client):
    """A UI form must be able to tell the clinician what is missing regardless."""
    assert client.post("/api/models/validate", json=INCOMPLETE).status_code == 200


def test_a_malformed_body_is_rejected(client):
    assert client.post("/api/models/validate", json={"nope": 1}).status_code == 422


# ── prediction ───────────────────────────────────────────────────────────────

def test_predict_refuses_an_incomplete_payload(client):
    """422 rather than a degraded estimate: there is no partial prediction."""
    r = client.post("/api/models/predict", json=INCOMPLETE)
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "incomplete_payload"
    assert r.json()["detail"]["question_for_user"]


def test_predict_returns_risks_or_a_clean_503(client):
    """
    Both branches, decided at call time.

    `skipif` cannot express this: it is evaluated at collection, before the
    TestClient lifespan has run `service.start()`, so it always saw "no models"
    and then ran against a process where they had since loaded. Whether the
    artefacts are present is a runtime fact and has to be read as one.
    """
    r = client.post("/api/models/predict", json=COMPLETE)
    if not service.models_loaded:
        assert r.status_code == 503, "a missing artefact must not surface as a 500"
        return
    body = r.json()
    assert 0.0 <= body["p_mortality"] <= 1.0
    assert body["risk_tier"]
    # `hospital_los` sits below the payload retention floor by design, and the
    # reason travels with the refusal rather than the task silently vanishing.
    assert "withheld_tasks" in body


# ── the assistant ────────────────────────────────────────────────────────────

def test_a_session_opens_with_the_capability_message(client):
    body = client.post("/api/assistant/sessions").json()
    assert body["status"] == "capabilities_shown"
    assert body["session_id"]
    assert "risk" in body["reply"].lower()


def test_a_refusal_is_a_200_with_the_questions_attached(client):
    """
    Refusals must not be HTTP errors.

    A 4xx invites a caller to retry past the gate; a 200 carrying the missing
    fields invites it to supply them, which is the entire interaction.
    """
    sid = client.post("/api/assistant/sessions").json()["session_id"]
    body = client.post(f"/api/assistant/sessions/{sid}/messages",
                       json={"message": "what is his mortality risk?"}).json()
    assert body["status"] == "declined_incomplete"
    assert body["questions"]
    assert all(q["text"].strip() for q in body["questions"])


def test_the_gate_names_the_payload_fields_a_clinician_would_recognise(client):
    sid = client.post("/api/assistant/sessions").json()["session_id"]
    body = client.post(f"/api/assistant/sessions/{sid}/messages",
                       json={"message": "45M septic shock, mortality risk?"}).json()
    texts = " ".join(q["text"] for q in body["questions"]).lower()
    assert "creatinine" in texts or "bun" in texts or "white cell" in texts
    # patient wording must not leak into a clinician session
    assert "how old are you" not in texts


def test_a_guideline_question_is_answered_and_cited(client):
    sid = client.post("/api/assistant/sessions").json()["session_id"]
    body = client.post(
        f"/api/assistant/sessions/{sid}/messages",
        json={"message": "What is the first-line vasopressor in septic shock?"}).json()
    assert body["status"] == "answered"
    assert body["citations"]
    assert body["verified"] is True
    assert body["intent"] == "guideline_lookup"


def test_an_uncovered_topic_is_declined_rather_than_answered(client):
    """The corpus covers 15 ICU conditions; outside them there is no source."""
    sid = client.post("/api/assistant/sessions").json()["session_id"]
    body = client.post(f"/api/assistant/sessions/{sid}/messages",
                       json={"message": "What are the guidelines for psoriasis?"}).json()
    assert body["status"] != "answered"


def test_triage_is_off_for_clinicians(client):
    """
    The red-flag rules address a person describing their own symptoms. A
    clinician asking about septic shock must not be told to call an ambulance.
    """
    sid = client.post("/api/assistant/sessions").json()["session_id"]
    body = client.post(f"/api/assistant/sessions/{sid}/messages",
                       json={"message": "How do I manage septic shock?"}).json()
    assert body["severity"] == "none"
    assert body["status"] != "emergency_response"
    assert "ambulance" not in body["reply"].lower()


def test_session_state_is_retrievable(client):
    sid = client.post("/api/assistant/sessions").json()["session_id"]
    client.post(f"/api/assistant/sessions/{sid}/messages",
                json={"message": "45 year old male with septic shock, mortality risk?"})
    body = client.get(f"/api/assistant/sessions/{sid}").json()
    assert body["known_facts"].get("age") == 45.0
    assert body["known_facts"].get("primary_diagnosis")
    assert body["missing"]


def test_sessions_are_isolated(client):
    a = client.post("/api/assistant/sessions").json()["session_id"]
    b = client.post("/api/assistant/sessions").json()["session_id"]
    client.post(f"/api/assistant/sessions/{a}/messages",
                json={"message": "45 year old male with septic shock"})
    assert client.get(f"/api/assistant/sessions/{b}").json()["known_facts"] == {}


def test_a_session_can_be_discarded(client):
    sid = client.post("/api/assistant/sessions").json()["session_id"]
    assert client.delete(f"/api/assistant/sessions/{sid}").status_code == 200
    assert client.get(f"/api/assistant/sessions/{sid}").status_code == 404


def test_unknown_session_is_404(client):
    assert client.post("/api/assistant/sessions/nope/messages",
                       json={"message": "hi"}).status_code == 404


def test_empty_message_is_rejected(client):
    sid = client.post("/api/assistant/sessions").json()["session_id"]
    assert client.post(f"/api/assistant/sessions/{sid}/messages",
                       json={"message": ""}).status_code == 422


def test_the_reasoning_trace_is_not_returned_by_default(client):
    """Spec 28: an audit exists for review, not for the reader of the answer."""
    sid = client.post("/api/assistant/sessions").json()["session_id"]
    body = client.post(f"/api/assistant/sessions/{sid}/messages",
                       json={"message": "what is his mortality risk?"}).json()
    assert body["debug"] is None


# ── the contract between the policy and the models ───────────────────────────

def test_the_clinician_policy_matches_the_payload_contract():
    """
    One definition of "enough to score".

    If `payload_validation` gains a required field and the clinician policy does
    not, the gate opens on an incomplete payload and the pipeline refuses
    further down — a refusal the assistant cannot explain, because it believed
    it had everything.
    """
    from src.assistant.intents import Intent
    from src.assistant.orchestrator import CLINICIAN_CONFIG
    from src.assistant.requirements import for_intent
    from src.llm.payload_validation import REQUIRED_FIELDS

    required = set(for_intent(Intent.RISK_ASSESSMENT,
                              path=CLINICIAN_CONFIG["requirements_path"]).required)
    contract = {s.path.rsplit(".", 1)[-1] for s in REQUIRED_FIELDS}
    contract = {"sex" if n == "gender" else n for n in contract}
    assert contract <= required, f"policy is missing {contract - required}"


def test_the_assistant_is_configured_for_clinicians():
    from src.assistant.intents import CLINICIAN

    assert service.assistant is not None
    assert service.assistant.mode == CLINICIAN
    assert service.assistant.triage_enabled is False


# ── what the UI reads ────────────────────────────────────────────────────────
#
# These fields exist so a screen can render the case without re-deriving it from
# markdown. Each assertion below is a promise the UI is built on; breaking one
# silently would leave a panel blank rather than raise.

def test_a_stated_fact_carries_the_words_it_came_from(client):
    """
    Spec 13 made visible. A panel showing "Age: 72" has told the reader a
    number; showing it beside the span it was read from lets them catch a
    misread before it reaches a model.
    """
    sid = client.post("/api/assistant/sessions").json()["session_id"]
    body = client.post(f"/api/assistant/sessions/{sid}/messages",
                       json={"message": "72F with septic shock"}).json()

    assert body["facts"], "no facts extracted from a plain presentation line"
    for fact in body["facts"]:
        assert fact["quote"].strip(), f"{fact['field']} has no source quote"
        assert fact["turn"] >= 1


def test_facts_accumulate_rather_than_reporting_only_this_turn(client):
    """
    The panel shows the case as it stands, not the most recent sentence.

    The opening message asks for risk on purpose. Extraction is scoped to the
    fields the current intent needs (spec 15, minimum-necessary), so after a
    *guideline* question a bare "creatinine 3.2" is correctly refused as not
    needed for that request — which is the system working, and was this test's
    original mistake rather than a defect.
    """
    sid = client.post("/api/assistant/sessions").json()["session_id"]
    client.post(f"/api/assistant/sessions/{sid}/messages",
                json={"message": "72F septic shock — what is her mortality risk?"})
    body = client.post(f"/api/assistant/sessions/{sid}/messages",
                       json={"message": "creatinine 3.2"}).json()

    fields = {f["field"] for f in body["facts"]}
    assert {"age", "creatinine_max"} <= fields, fields


def test_a_guideline_answer_reports_the_documents_behind_it(client):
    """`citations` is rendered text; `sources` is what the UI can link to."""
    sid = client.post("/api/assistant/sessions").json()["session_id"]
    body = client.post(f"/api/assistant/sessions/{sid}/messages",
                       json={"message": "What is the first-line vasopressor "
                                        "in septic shock?"}).json()

    assert body["status"] == "answered"
    assert body["sources"], "answered from evidence but reported no sources"
    for src in body["sources"]:
        assert src["title"] and src["doc_id"]
        assert src["tier"] >= 1
        # The corpus is unreviewed and the UI must be able to say so.
        assert src["review_status"] in {"unreviewed", "clinician_reviewed"}


def test_predictions_are_absent_rather_than_empty_when_no_model_ran(client):
    """
    An empty object would suggest the models ran and found nothing. A guideline
    lookup never touches them.
    """
    sid = client.post("/api/assistant/sessions").json()["session_id"]
    body = client.post(f"/api/assistant/sessions/{sid}/messages",
                       json={"message": "What is the first-line vasopressor "
                                        "in septic shock?"}).json()
    assert body["predictions"] is None


def test_the_evidence_route_publishes_the_corpus_and_its_limits(client):
    """
    Fifteen concepts, twenty-three records, none clinician-reviewed. A reader
    who cannot see that discovers it by being refused, which reads as the system
    being unhelpful rather than as the corpus being small.
    """
    body = client.get("/api/evidence").json()

    assert body["stats"]["n_records"] == len(body["documents"])
    assert body["stats"]["n_clinician_reviewed"] == 0
    assert body["documents"] == sorted(
        body["documents"], key=lambda d: (d["tier"], d["title"])), \
        "documents must arrive ordered by trust, not by file order"
    first = body["documents"][0]
    assert {"doc_id", "title", "tier", "tier_name", "topics"} <= set(first)


def test_the_deterministic_floor_reads_a_full_presentation_line():
    """
    No backend, no key, no quota — the values a clinician actually types must
    still be read. This is the path every demo falls back to, and it broke on
    "bicarb 14" because the shared synonym table listed only "bicarbonate".
    """
    from src.assistant.extraction import extract
    from src.assistant.state import PatientContext

    ctx = PatientContext()
    extract("72F septic shock, Cr 3.2, BUN 48, WBC 19.4, bicarb 14, Na 131, "
            "K 5.2, plt 88, hct 29, glucose 210, SBP 82, HR 118",
            ctx, turn=1, backend=None)

    for field in ("age", "sex", "creatinine_max", "bun_max", "wbc_max",
                  "bicarbonate_min", "sodium_min", "potassium_max",
                  "platelets_min", "hematocrit_min", "glucose_max",
                  "sbp_min", "hr_max"):
        assert ctx.get(field) is not None, f"{field} not read without a model"


# ── SHAP drivers ─────────────────────────────────────────────────────────────

def test_a_scored_case_reports_what_moved_the_estimate(client):
    """
    An explanation the report can print and the verifier will pass.

    Attributions are the system's own arithmetic over its own model, so they
    have to be registered as facts. They were not, and the grounding check
    correctly withheld the entire report for quoting a number it had computed
    itself — the same failure the medication relevance scores caused earlier.
    """
    sid = client.post("/api/assistant/sessions").json()["session_id"]
    body = client.post(f"/api/assistant/sessions/{sid}/messages", json={"message":
        "88F septic shock, Cr 3.2, BUN 48, WBC 19.4, bicarb 14, Na 131, K 5.2, "
        "plt 88, hct 29, glucose 210, SBP 82, HR 118. Mortality risk?"}).json()

    if body["status"] != "answered":
        pytest.skip(f"models unavailable: {body['status']}")

    assert body["verified"] is True, "drivers must not break grounding"
    drivers = body["predictions"]["drivers"]
    assert drivers, "a scored case reported no drivers"
    assert len(drivers) <= 8

    labels = [d["label"] for d in drivers]
    assert any("Age" == l for l in labels), labels
    # Ranked by absolute contribution, so the panel does not reorder per turn.
    magnitudes = [abs(d["contribution"]) for d in drivers]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_a_driver_says_whether_the_value_was_supplied(client):
    """
    The distinction the panel is built on.

    A boosted tree routes a missing value down a default branch, so absence
    carries weight: on this payload several of the largest attributions are the
    model responding to what it was not told. Without `supplied`, a UI renders
    "Number of coded diagnoses +0.77" as a finding about the patient.
    """
    sid = client.post("/api/assistant/sessions").json()["session_id"]
    body = client.post(f"/api/assistant/sessions/{sid}/messages", json={"message":
        "88F septic shock, Cr 3.2, BUN 48, WBC 19.4, bicarb 14, Na 131, K 5.2, "
        "plt 88, hct 29, glucose 210, SBP 82, HR 118. Mortality risk?"}).json()

    if body["status"] != "answered":
        pytest.skip(f"models unavailable: {body['status']}")

    for d in body["predictions"]["drivers"]:
        assert isinstance(d["supplied"], bool)
        # A supplied driver carries the value the model saw; an unsupplied one
        # carries None rather than a zero, which would read as a measurement.
        assert (d["value"] is not None) == d["supplied"], d


def test_the_report_explains_its_own_attributions(client):
    """A signed number under a heading is not an explanation on its own."""
    sid = client.post("/api/assistant/sessions").json()["session_id"]
    body = client.post(f"/api/assistant/sessions/{sid}/messages", json={"message":
        "88F septic shock, Cr 3.2, BUN 48, WBC 19.4, bicarb 14, Na 131, K 5.2, "
        "plt 88, hct 29, glucose 210, SBP 82, HR 118. Mortality risk?"}).json()

    if body["status"] != "answered":
        pytest.skip(f"models unavailable: {body['status']}")

    reply = body["reply"]
    assert "Why the estimate landed where it did" in reply
    assert "not measured in this patient" in reply, (
        "the absence distinction must reach the reader")
    assert "not causes" in reply
