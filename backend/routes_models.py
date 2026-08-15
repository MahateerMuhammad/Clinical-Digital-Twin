"""
backend/routes_models.py
────────────────────────
Direct access to the risk models. Clinician audience.

Every route here is a transport wrapper. The validation, the coverage floor,
the withheld-task logic and the grounding check all live in ``src/llm/`` and
are exercised by the existing suite; these functions call them and return what
they return. Nothing is re-validated, re-shaped or re-decided at this layer —
a second copy of a guard is a guard that can disagree with the first.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from backend.schemas import PayloadRequest, ReportRequest, WhatIfRequest
from backend.service import service
from src.llm.payload_validation import validate_payload

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("/describe")
def describe() -> Dict[str, Any]:
    """Which models loaded, their features and their promoted metrics."""
    return service.require_models().describe_models()


@router.post("/validate")
def validate(req: PayloadRequest) -> Dict[str, Any]:
    """
    Check a payload without scoring it.

    Needs no models, so it answers even when the artefacts are absent — which
    is what a UI form wants: tell the clinician what is missing while they are
    still typing.
    """
    return validate_payload(req.payload).to_dict()


@router.post("/predict")
def predict(req: PayloadRequest) -> Dict[str, Any]:
    """
    Multi-task risk with calibration and uncertainty.

    Refuses an incomplete payload rather than imputing. Tasks whose payload
    feature coverage falls below the retention floor come back under
    ``withheld_tasks`` with the reason, not as a zero.
    """
    report = validate_payload(req.payload)
    if not report.ok:
        raise HTTPException(status_code=422, detail={
            "error": "incomplete_payload",
            "question_for_user": report.question_for_user(),
            "validation": report.to_dict()})
    return service.require_models().run_live_inference_with_uncertainty(req.payload)


@router.post("/report")
def report(req: ReportRequest) -> Dict[str, Any]:
    """
    The full grounded clinical report.

    ``PipelineResult.to_dict()`` already carries status, predictions, the
    validation result, the grounding verdict, timings and the markdown. A 200
    with ``status: "incomplete_input"`` is a successful refusal, not an error —
    the body names the missing fields.
    """
    if service.pipeline is None:
        raise HTTPException(status_code=503,
                            detail=f"pipeline unavailable: {service.load_error}")
    return service.pipeline.generate(req.payload, case_id=req.case_id,
                                     use_llm=req.use_llm).to_dict()


@router.post("/whatif")
def whatif(req: WhatIfRequest) -> Dict[str, Any]:
    """
    Re-score with one or more inputs changed.

    Describes how the model responds to a changed input. It is not evidence
    that changing that input would change the outcome, and the caller is
    expected to present it that way.
    """
    return service.require_models().simulate_what_if_unseen_patient(
        req.payload, req.modifications)
