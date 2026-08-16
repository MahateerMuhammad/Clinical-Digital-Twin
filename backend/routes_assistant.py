"""
backend/routes_assistant.py
───────────────────────────
The conversational layer.

``Assistant.handle()`` is the entire turn — triage, intent, extraction, gate,
routing, composition, verification, audit. These routes carry a message in and
select fields out. There is no branching here on status, no re-deciding whether
to answer, and no path by which a request parameter can influence a gate.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

from backend.schemas import (
    Driver, Fact, MessageRequest, Predictions, Question, SessionResponse,
    Source, TaskPrediction, TurnResponse,
)
from backend.service import DEBUG, service

router = APIRouter(prefix="/api/assistant", tags=["assistant"])


def _assistant():
    if service.assistant is None:
        raise HTTPException(status_code=503, detail="assistant not initialised")
    return service.assistant


#: Task keys in report order, with the wording used elsewhere in the product.
#: Ordered rather than derived from dict keys so the panel does not reshuffle
#: between turns as tasks are withheld and restored.
_TASKS = (
    ("p_mortality", "In-hospital mortality"),
    ("p_readmission", "30-day readmission"),
    ("p_icu_admission", "ICU admission"),
    ("p_los_over_5_63d", "Stay beyond 5.63 days"),
    ("p_deterioration", "Deterioration within 48h"),
)


def _predictions(raw: Any) -> Optional[Predictions]:
    """
    Reshape the runner's flat output into the panel the UI renders.

    `run_live_inference_with_uncertainty` returns probabilities as top-level
    keys with a separate `withheld_tasks` map, which is convenient for the
    report writer and awkward for a component. Reshaping happens once, here,
    rather than in every consumer.

    Confidence is deliberately not per task: the runner reports one
    `model_confidence` label for the whole inference, and manufacturing a
    per-task number would show a precision these models do not have.
    """
    if not isinstance(raw, dict) or not any(k in raw for k, _ in _TASKS):
        return None

    withheld = raw.get("withheld_tasks") or {}
    rawp = raw.get("raw_probabilities") or {}
    tasks = [
        TaskPrediction(
            key=key, label=label,
            probability=raw.get(key),
            raw_probability=rawp.get(key),
            withheld=key in withheld,
            reason=str(withheld.get(key, "")),
        )
        for key, label in _TASKS
        if key in raw or key in withheld
    ]
    return Predictions(
        tasks=tasks,
        risk_tier=str(raw.get("risk_tier", "")),
        model_confidence=str(raw.get("model_confidence", "")),
        calibration_statement=str(raw.get("calibration_statement", "")),
        input_kind=str(raw.get("input_kind", "")),
        drivers=[Driver(**d) for d in (raw.get("drivers") or [])
                 if isinstance(d, dict)],
    )


def _sources(evidence: Any, ans: Any) -> List[Source]:
    """
    Retrieved documents, in the order retrieval ranked them.

    Two shapes reach here, because the two paths retrieve differently. The
    guideline path returns `EvidenceDoc` objects on the `EvidenceResult`; the
    risk path lets `ClinicalReportPipeline` retrieve, which leaves plain dicts
    on `answer.documents` and returns an `EvidenceResult` carrying only a
    status. Reading just the first left every scored case citing sources in its
    prose while reporting none to the UI.

    Normalising both here is the narrow fix. The alternative — making the model
    path build EvidenceDocs — means the pipeline's retrieval and the assistant's
    would have to agree on a type, which is a change to two tested components
    for a presentation concern.
    """
    docs = list(getattr(evidence, "documents", None) or [])
    if docs:
        return [
            Source(doc_id=d.doc_id, title=d.title, tier=d.source_tier,
                   url=d.url, citation=d.citation_text,
                   review_status=d.review_status)
            for d in docs
        ]

    # Guideline records are tier 1 wherever they are built (evidence.py:298).
    return [
        Source(doc_id=str(d.get("doc_id", "")), title=str(d.get("title", "")),
               tier=1, url=str(d.get("url", "")),
               citation=str(d.get("citation", "")),
               review_status=str(d.get("review_status", "unreviewed")))
        for d in (getattr(ans, "documents", None) or [])
        if isinstance(d, dict)
    ]


def _turn_response(result: Any) -> TurnResponse:
    ans = result.answer
    clar = result.clarification
    report = result.faithfulness

    debug = None
    if DEBUG:
        debug = {
            "gate": result.gate.to_dict() if result.gate else None,
            "extraction": result.record.extraction if result.record else None,
            "faithfulness": report.to_dict() if report else None,
            "grounding": getattr(ans, "grounding", None) if ans else None,
        }

    return TurnResponse(
        session_id=result.state.session_id,
        turn=result.state.turn,
        status=result.status,
        reply=result.reply,
        questions=[Question(field=q.field, text=q.text, level=q.level)
                   for q in (clar.questions if clar else [])],
        citations=list(ans.citations) if ans else [],
        limitations=list(ans.limitations) if ans else [],
        severity=result.triage.severity if result.triage else "none",
        verified=(report.ok if report else None),
        # The intent of *this turn*, not the session's open case: an aside
        # answered mid-counterfactual is a guideline lookup, and labelling it
        # with the case intent hides exactly the distinction that was just fixed.
        intent=(result.record.intent if result.record and result.record.intent
                else result.state.intent),
        facts=[Fact(field=f.field, value=f.value, quote=f.source_quote,
                    turn=f.turn)
               for f in result.state.context.history],
        predictions=_predictions(getattr(ans, "predictions", None) if ans else None),
        sources=_sources(result.evidence, ans),
        debug=debug,
    )


@router.post("/sessions", response_model=TurnResponse)
def create_session() -> TurnResponse:
    """Open a session and return the capability message (spec 2)."""
    return _turn_response(_assistant().start())


@router.post("/sessions/{session_id}/messages", response_model=TurnResponse)
def post_message(session_id: str, req: MessageRequest) -> TurnResponse:
    """
    One turn.

    A 200 whose ``status`` is ``declined_incomplete`` is the system working:
    ``questions`` carries exactly what it needs next. Treating a refusal as an
    HTTP error would push callers towards retrying past it.
    """
    bot = _assistant()
    if session_id not in bot.sessions:
        raise HTTPException(status_code=404, detail="unknown session")
    return _turn_response(bot.handle(session_id, req.message))


@router.get("/sessions/{session_id}", response_model=SessionResponse)
def get_session(session_id: str) -> SessionResponse:
    """Current state, for a UI restoring a conversation."""
    from src.assistant import gate as G
    from src.assistant.requirements import for_intent

    bot = _assistant()
    if session_id not in bot.sessions:
        raise HTTPException(status_code=404, detail="unknown session")
    state = bot.sessions[session_id]

    missing: List[str] = []
    if state.intent:
        try:
            reqs = for_intent(state.intent, state.context,
                              path=bot.requirements_path)
            missing = G.evaluate(state.context, state.intent,
                                 requirement_set=reqs).blocking_fields
        except Exception:
            missing = []

    return SessionResponse(
        session_id=session_id,
        turn=state.turn,
        intent=state.intent,
        known_facts={n: state.context.get(n)
                     for n in sorted(state.context.known_fields())},
        missing=missing,
        messages=state.messages,
    )


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> Dict[str, Any]:
    """Discard a conversation and everything recorded in it."""
    bot = _assistant()
    if bot.sessions.pop(session_id, None) is None:
        raise HTTPException(status_code=404, detail="unknown session")
    return {"deleted": session_id}
