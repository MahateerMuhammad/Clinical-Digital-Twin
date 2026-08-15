"""
backend/schemas.py
──────────────────
Request and response shapes.

Deliberately few. The domain layer already returns API-shaped dictionaries —
``PipelineResult.to_dict()``, ``ValidationResult.to_dict()``,
``run_live_inference_with_uncertainty()`` — so restating those structures as
pydantic models would create a second schema to keep in step with the first,
for no gain beyond documentation. Where a response is passed through, it is
typed as a dict and passed through.

Models are defined here only where the API genuinely *shapes* something: the
request bodies it must validate, and the assistant turn, whose fields are
selected rather than forwarded.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PayloadRequest(BaseModel):
    """A presentation payload, in the shape ``validate_payload`` expects.

    Not modelled field by field. The payload contract lives in
    ``src/llm/payload_validation.py`` and is enforced there, by the same code
    the pipeline uses — a pydantic copy of it would be a second gate that could
    disagree with the first, and the first is the one with the tests.
    """

    payload: Dict[str, Any] = Field(
        ..., description="Presentation payload: demographics, primary_diagnosis, "
                         "presentation_labs, vital_signs, and optionally "
                         "comorbidities, active_medications, prior_utilisation.")


class WhatIfRequest(PayloadRequest):
    modifications: Dict[str, float] = Field(
        ..., description="Field → new value, e.g. {\"creatinine_max\": 1.5}. "
                         "Keys use the payload's own lab names.")


class ReportRequest(PayloadRequest):
    case_id: str = "api_case"
    use_llm: bool = Field(
        False, description="Run the rephrasing stage. Output is still verified "
                           "against the same facts and withheld if it fails.")


class MessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)


class Question(BaseModel):
    field: str
    text: str
    level: str


class TurnResponse(BaseModel):
    """One assistant turn, as the UI needs it.

    Selected, not forwarded. The audit record holds the gate decision, the
    extraction proposals and the faithfulness checks; spec 28 says an audit is
    not for the person being helped, so it is not returned here unless the
    server is explicitly started in debug mode.
    """

    session_id: str
    turn: int
    status: str
    reply: str
    questions: List[Question] = []
    citations: List[str] = []
    limitations: List[str] = []
    #: none · urgent_assess · emergency. Always "none" for clinician sessions,
    #: where triage is disabled.
    severity: str = "none"
    #: True/False when a verification ran, None when the turn did not compose an
    #: answer (a clarifying question is not something to verify).
    verified: Optional[bool] = None
    intent: Optional[str] = None
    #: Present only when the server runs with CDT_ASSISTANT_DEBUG=1.
    debug: Optional[Dict[str, Any]] = None


class SessionResponse(BaseModel):
    session_id: str
    turn: int
    intent: Optional[str] = None
    known_facts: Dict[str, Any] = {}
    #: What the completeness gate is still waiting for, so a UI can render a
    #: progress indicator without re-deriving the policy.
    missing: List[str] = []
    messages: List[Dict[str, Any]] = []
