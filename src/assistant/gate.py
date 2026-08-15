"""
src/assistant/gate.py
─────────────────────
The answerability gate.  Spec sections 8, 22, 33.9.

This is the single place the system decides whether it may answer. Everything it
reads — the patient context, the resolved requirement policy — is application
state. It takes no model output, has no prompt, and cannot be argued with. Spec
33.9 requires that the language model never bypass an application-level safety
gate; the enforcement is that there is no parameter through which it could.

The gate is deliberately dumb. It does not judge whether a missing field matters
— that judgement was made once, by a human, in ``config/requirements.yaml``, by
placing the field at ``required`` rather than ``optional``. Spec 8 asks whether
missing information "materially affects the answer"; this module reads the
answer off the policy instead of re-deriving it per turn, because a per-turn
judgement is a per-turn opportunity to be talked into answering.

Four statuses, mapping onto spec 8's three plus contradiction:

``SAFETY_CRITICAL_MISSING``  a field whose absence makes an answer unsafe
``CONTRADICTORY``            two incompatible statements, unresolved
``INCOMPLETE``               a required field is missing
``COMPLETE``                 answerable; optional gaps become stated limitations

Declined fields
───────────────
A patient may refuse to answer. That is not the same as not having been asked,
and the two must not collapse into each other — re-asking a declined field is a
spec 24 violation, while treating it as answered is a spec 33.4 violation.

So a declined field is *known-unanswerable*: it stops being asked, and the gate
decides what to do with it by level. A declined **safety-critical** field leaves
the system unable to answer safely, and it says so (spec 33.17) rather than
proceeding or nagging. A declined **required** field downgrades the answer to
one with a stated limitation, which is what spec 8 prescribes for information
that is missing but not dangerous to be without.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.assistant.requirements import (
    OPTIONAL, REQUIRED, SAFETY_CRITICAL, RequirementSet, for_intent,
)
from src.assistant.state import ConversationState, PatientContext, field_spec

__all__ = [
    "GateDecision", "can_answer", "evaluate",
    "COMPLETE", "INCOMPLETE", "SAFETY_CRITICAL_MISSING", "CONTRADICTORY",
]

COMPLETE = "COMPLETE"
INCOMPLETE = "INCOMPLETE"
SAFETY_CRITICAL_MISSING = "SAFETY_CRITICAL_MISSING"
CONTRADICTORY = "CONTRADICTORY"


@dataclass
class GateDecision:
    """The gate's verdict, with everything needed to act on it."""

    can_answer: bool
    status: str
    reason: str

    #: Fields to ask about, most important first. Empty when ``can_answer``.
    missing_safety_critical: List[str] = field(default_factory=list)
    missing_required: List[str] = field(default_factory=list)
    #: Absent but not blocking. These become the "what I could not determine"
    #: section of the answer (spec 20), never silent omissions.
    missing_optional: List[str] = field(default_factory=list)

    #: Safety-critical fields the patient declined. Blocking, and not re-askable.
    declined_safety_critical: List[str] = field(default_factory=list)
    #: Required fields the patient declined. Non-blocking, stated as a limit.
    declined_required: List[str] = field(default_factory=list)

    contradictions: List[Dict[str, Any]] = field(default_factory=list)

    #: Conditional requirements that cannot yet be evaluated, with the fields
    #: that would settle them. Asking `sex` unlocks the pregnancy conditional,
    #: so these must be asked before the gate can be trusted as final.
    pending_conditionals: List[str] = field(default_factory=list)

    @property
    def blocking_fields(self) -> List[str]:
        """Everything still worth asking, in priority order (spec 6, spec 23).

        Safety-critical first, then the fields that unlock pending conditionals,
        then the merely required. Declined fields never appear: they have been
        asked and answered, with a refusal.
        """
        return (self.missing_safety_critical
                + [f for f in self.pending_conditionals
                   if f not in self.missing_safety_critical]
                + [f for f in self.missing_required
                   if f not in self.pending_conditionals])

    @property
    def limitations(self) -> List[str]:
        """Fields absent from an answer the gate nonetheless permitted."""
        return self.missing_optional + self.declined_required

    def to_dict(self) -> dict:
        return {
            "can_answer": self.can_answer,
            "status": self.status,
            "reason": self.reason,
            "missing_safety_critical": self.missing_safety_critical,
            "missing_required": self.missing_required,
            "missing_optional": self.missing_optional,
            "declined_safety_critical": self.declined_safety_critical,
            "declined_required": self.declined_required,
            "pending_conditionals": self.pending_conditionals,
            "contradictions": self.contradictions,
        }


def _split(names: List[str], context: PatientContext):
    """Partition a level's fields into (missing, declined)."""
    missing, declined = [], []
    for n in names:
        if n in context.declined:
            declined.append(n)
        elif not context.is_known(n):
            missing.append(n)
    return missing, declined


def evaluate(context: PatientContext, intent: Any,
             *, requirement_set: Optional[RequirementSet] = None) -> GateDecision:
    """
    Decide whether the assistant may answer.

    ``requirement_set`` may be passed to avoid re-resolving the policy; when
    omitted it is resolved from ``intent`` against ``context``, which is what
    makes conditional requirements sensitive to what the patient has said.
    """
    reqs = requirement_set or for_intent(intent, context)

    sc_missing, sc_declined = _split(reqs.at(SAFETY_CRITICAL), context)
    rq_missing, rq_declined = _split(reqs.at(REQUIRED), context)
    op_missing, _ = _split(reqs.at(OPTIONAL), context)

    pending = [r.field for r in reqs.pending_conditionals]
    # The fields a pending conditional reads are what actually need asking. A
    # conditional on `sex` cannot be settled by asking about pregnancy.
    unlockers: List[str] = []
    for r in reqs.pending_conditionals:
        for n in _enabling_fields(intent, r.field, context):
            if n not in unlockers and not context.is_known(n):
                unlockers.append(n)

    contradictions = [c.to_dict() for c in context.contradictions]

    decision = GateDecision(
        can_answer=False, status=INCOMPLETE, reason="",
        missing_safety_critical=sc_missing,
        missing_required=rq_missing,
        missing_optional=op_missing,
        declined_safety_critical=sc_declined,
        declined_required=rq_declined,
        contradictions=contradictions,
        pending_conditionals=unlockers,
    )

    # Order matters only for the headline reason; every blocking condition is
    # reported regardless, so a caller never fixes one and discovers another.
    if sc_declined:
        decision.status = SAFETY_CRITICAL_MISSING
        decision.reason = (
            "cannot answer safely without "
            + _phrase(sc_declined)
            + ", which you preferred not to share")
        return decision

    if sc_missing:
        decision.status = SAFETY_CRITICAL_MISSING
        decision.reason = "safety-critical information missing: " + _phrase(sc_missing)
        return decision

    if contradictions:
        decision.status = CONTRADICTORY
        decision.reason = ("conflicting information about "
                           + _phrase([c["field"] for c in contradictions]))
        return decision

    if unlockers:
        decision.status = INCOMPLETE
        decision.reason = ("cannot yet tell which further information applies "
                           "without " + _phrase(unlockers))
        return decision

    if rq_missing:
        decision.status = INCOMPLETE
        decision.reason = "required information missing: " + _phrase(rq_missing)
        return decision

    decision.can_answer = True
    decision.status = COMPLETE
    decision.reason = (
        "all safety-critical and required information present"
        + (f"; answering with stated limits on {_phrase(decision.limitations)}"
           if decision.limitations else ""))
    return decision


def can_answer(state_or_context: Any, intent: Any = None) -> GateDecision:
    """
    Convenience entry point accepting either a ``ConversationState`` or a bare
    ``PatientContext``.

    Returns the full ``GateDecision`` rather than a bare bool: spec 22's
    pseudocode returns False, but a caller that only learns "no" has to guess
    what to ask, and guessing is what this system is built to avoid.
    """
    if isinstance(state_or_context, ConversationState):
        return evaluate(state_or_context.context,
                        intent if intent is not None else state_or_context.intent)
    return evaluate(state_or_context, intent)


# ── helpers ──────────────────────────────────────────────────────────────────

def _enabling_fields(intent: Any, conditional_field: str,
                     context: PatientContext) -> List[str]:
    """Which unknown fields a pending conditional is waiting on."""
    from src.assistant.requirements import load_policy

    name = str(getattr(intent, "value", intent))
    spec = load_policy()["intents"].get(name) or {}
    out: List[str] = []
    for cond in spec.get("conditional") or []:
        if cond.get("field") != conditional_field:
            continue
        for pred in (cond.get("when") or {}):
            if pred in ("known", "unknown"):
                continue
            base = pred.rsplit("_", 1)[0]
            try:
                field_spec(base)
            except KeyError:
                continue
            out.append(base)
    return out


def _phrase(names: List[str]) -> str:
    words = [n.replace("_", " ") for n in names]
    if len(words) <= 1:
        return words[0] if words else ""
    return ", ".join(words[:-1]) + " and " + words[-1]
