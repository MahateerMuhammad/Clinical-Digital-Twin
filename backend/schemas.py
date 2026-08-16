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


class Fact(BaseModel):
    """A value the clinician stated, with the words it was read from.

    The quote is the point (spec 13). A UI that shows "Age: 72" has told the
    reader a number; one that shows it beside "72F septic shock" has told them
    where it came from and lets them catch a misread before it reaches a model.
    """

    field: str
    value: Any
    quote: str
    turn: int


class TaskPrediction(BaseModel):
    """One model output.

    ``withheld`` is not an error. A task whose payload feature coverage falls
    below the retention floor is named with its reason rather than scored badly,
    and the UI is expected to render that as a first-class state — never a blank
    cell and never a zero, both of which read as "no risk".
    """

    key: str
    label: str
    probability: Optional[float] = None
    #: Before calibration. Shown only where the difference is instructive.
    raw_probability: Optional[float] = None
    withheld: bool = False
    reason: str = ""


class Driver(BaseModel):
    """One SHAP attribution.

    ``supplied`` is not decoration. A boosted tree routes a missing value down a
    default branch, so absence carries weight: several of the largest
    attributions on a typed payload are the model responding to what it was not
    told. A UI that hides this reports "Number of coded diagnoses +0.77" as a
    finding about the patient, which it is not.
    """

    feature: str
    label: str
    #: Signed, in log-odds. Positive raises the estimated risk.
    contribution: float
    value: Optional[float] = None
    supplied: bool


class Predictions(BaseModel):
    """The model panel for one turn.

    Confidence is a single label for the run, not a number per task — that is
    what ``run_live_inference_with_uncertainty`` returns, and inventing a
    per-task uncertainty here would be presenting a precision the models do not
    have.
    """

    tasks: List[TaskPrediction] = []
    risk_tier: str = ""
    model_confidence: str = ""
    calibration_statement: str = ""
    input_kind: str = ""
    #: What moved the mortality estimate. Empty when the explainer could not run;
    #: the report says so rather than leaving the reader to notice the absence.
    drivers: List[Driver] = []


class Source(BaseModel):
    """A retrieved document. ``tier`` is trust rank and is not decoration."""

    doc_id: str
    title: str
    tier: int
    url: str = ""
    citation: str = ""
    review_status: str = "unreviewed"


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
    #: Everything the clinician has stated so far, each with its source span.
    #: Cumulative rather than per-turn: a UI panel showing the case as it stands
    #: would otherwise have to accumulate it and could drift from the context
    #: the gate is actually reading.
    facts: List[Fact] = []
    #: Present when the turn ran the risk models; None otherwise. A guideline
    #: lookup does not touch them, and an empty object would suggest it did and
    #: found nothing.
    predictions: Optional[Predictions] = None
    #: The documents behind the answer. `citations` above is the rendered text;
    #: this is the same evidence with its tier and link, so the UI can show what
    #: a claim rests on rather than only that it was cited.
    sources: List[Source] = []
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
