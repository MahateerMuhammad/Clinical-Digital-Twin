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

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from backend.schemas import (
    MessageRequest, Question, SessionResponse, TurnResponse,
)
from backend.service import DEBUG, service

router = APIRouter(prefix="/api/assistant", tags=["assistant"])


def _assistant():
    if service.assistant is None:
        raise HTTPException(status_code=503, detail="assistant not initialised")
    return service.assistant


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
        intent=result.state.intent,
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
