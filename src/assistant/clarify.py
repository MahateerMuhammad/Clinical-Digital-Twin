"""
src/assistant/clarify.py
────────────────────────
Turning a refusal into a question.  Spec sections 6, 14, 15, 23, 24.

The gate decides *whether* to answer. This module decides *what to ask* when the
answer is no, which is the difference between a system that refuses usefully and
one that stonewalls. Spec 23's worked example is exactly this distinction:
"please provide additional relevant information" and "is the chest pain
happening right now, and when did it start?" are both refusals, and only one of
them is any use to a patient.

Four rules, all structural:

**At most three questions per turn.** Spec 6 forbids dumping a questionnaire.
The cap is enforced here rather than requested of a model, so it holds even when
nine fields are missing.

**Safety-critical first.** Spec 6 again: the ordering is not cosmetic, because a
patient may abandon the conversation after one reply and the assistant should
have spent that reply on the question that mattered.

**Never re-ask what is known.** Spec 24. The gate already excludes known fields;
this module additionally excludes declined ones, which are known-unanswerable.

**Stop looping.** A field the patient has been asked about twice without
answering is not going to be answered by asking a third time. It moves to
``stalled``, and the assistant offers to continue without it rather than
trapping the patient in a question it will not stop repeating. Where the field
is safety-critical there is nothing to offer — the assistant says plainly that
it cannot answer safely, which is spec 33.17.

Question *wording* comes from ``state.FIELDS``, written once and reviewable,
rather than generated per turn. A generated question can drift into the vague
form spec 23 names as bad; a fixed one cannot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.assistant import gate as G
from src.assistant.requirements import REQUIRED, SAFETY_CRITICAL
from src.assistant.state import ConversationState, PatientContext, field_spec

__all__ = [
    "Question", "Clarification", "next_questions",
    "MAX_QUESTIONS_PER_TURN", "MAX_ASK_ATTEMPTS",
]

#: Spec 6. Three is enough to make progress and few enough to answer in one
#: message without the patient losing track of which one they are answering.
MAX_QUESTIONS_PER_TURN = 3

#: After this many unanswered asks, stop asking and offer to move on.
MAX_ASK_ATTEMPTS = 2


@dataclass(frozen=True)
class Question:
    """One thing to ask, and why it is being asked."""

    field: str
    text: str
    level: str
    #: Set for a contradiction-resolution question, which has no single field
    #: value to fill — it asks the patient to choose between two they gave.
    resolves_contradiction: bool = False
    reason: str = ""

    def to_dict(self) -> dict:
        return {"field": self.field, "text": self.text, "level": self.level,
                "resolves_contradiction": self.resolves_contradiction,
                "reason": self.reason}


@dataclass
class Clarification:
    """What to say when the gate refuses."""

    questions: List[Question] = field(default_factory=list)
    #: Fields asked too often without an answer. The assistant offers to proceed
    #: without them, or says it cannot, depending on ``blocked_by_stalled``.
    stalled: List[str] = field(default_factory=list)
    #: True when a stalled field is safety-critical, so there is no version of
    #: the answer that can be given without it.
    blocked_by_stalled: bool = False
    #: Non-blocking gaps, carried through so the eventual answer can state them.
    limitations: List[str] = field(default_factory=list)
    #: Where to send someone the assistant cannot help. Addressed to the person
    #: being treated by default; a clinician reading "a doctor can help without
    #: it" is being told to consult themselves, which reads as the system having
    #: misjudged who it is talking to — and undermines the sentence before it,
    #: which is the one that matters.
    referral: str = "A doctor or pharmacist can help without it."

    @property
    def has_questions(self) -> bool:
        return bool(self.questions)

    @property
    def fields(self) -> List[str]:
        return [q.field for q in self.questions if not q.resolves_contradiction]

    def render(self) -> str:
        """
        Plain text for the patient.

        Deterministic. A language model may later rephrase this for tone, but
        it is generated here so that a system with no model available still
        asks precise questions rather than nothing.
        """
        lines: List[str] = []
        if self.blocked_by_stalled:
            names = ", ".join(s.replace("_", " ") for s in self.stalled)
            lines.append(
                f"I cannot answer this safely without knowing {names}, and I do "
                f"not want to guess. {self.referral}".rstrip())
            return "\n".join(lines)

        if self.questions:
            if len(self.questions) == 1:
                # A lone prompt rendered bare reads as a fragment — the reply to
                # "Can I give full-dose enoxaparin?" was the two words "Peak
                # serum creatinine" and nothing else. One line of framing costs
                # nothing and makes it a question rather than a label.
                q = self.questions[0]
                lines.append(q.text if q.resolves_contradiction
                             else f"Before I can answer that: {q.text}")
            else:
                lines.append("To answer this safely I need a little more:")
                lines += [f"  • {q.text}" for q in self.questions]

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"questions": [q.to_dict() for q in self.questions],
                "stalled": self.stalled,
                "blocked_by_stalled": self.blocked_by_stalled,
                "limitations": self.limitations}


def next_questions(state: ConversationState,
                   decision: Optional[G.GateDecision] = None,
                   *, intent: Any = None,
                   max_questions: int = MAX_QUESTIONS_PER_TURN,
                   prompts: Optional[Dict[str, str]] = None,
                   referral: Optional[str] = None) -> Clarification:
    """
    Choose what to ask next.

    ``state`` is mutated: fields returned here are marked as asked, so the next
    turn knows not to repeat them (spec 24) and the stall counter advances. The
    caller is expected to actually put these questions to the patient.
    """
    ctx = state.context
    if decision is None:
        decision = G.evaluate(ctx, intent if intent is not None else state.intent)

    out = Clarification(limitations=list(decision.limitations))
    if referral is not None:
        out.referral = referral

    if decision.can_answer:
        return out

    # 1. Contradictions come first. Every later answer would be built on a fact
    #    the system knows is disputed, so there is nothing useful to ask until
    #    this is settled (spec 14).
    for c in decision.contradictions[:max_questions]:
        out.questions.append(Question(
            field=c["field"], text=c["question"], level="contradiction",
            resolves_contradiction=True,
            reason="two different values were given for this"))
    if out.questions:
        state.mark_asked([q.field for q in out.questions])
        return out

    # 2. A safety-critical field the patient declined ends the conversation
    #    branch rather than starting a new question (spec 33.17).
    if decision.declined_safety_critical:
        out.stalled = list(decision.declined_safety_critical)
        out.blocked_by_stalled = True
        return out

    # 3. Everything still worth asking, already in priority order:
    #    safety-critical, then conditional unlockers, then required.
    candidates = decision.blocking_fields

    askable: List[str] = []
    stalled: List[str] = []
    for name in candidates:
        (stalled if state.times_asked(name) >= MAX_ASK_ATTEMPTS
         else askable).append(name)

    sc = set(decision.missing_safety_critical)

    # A stalled field that *blocks* cannot be worked around, and collecting the
    # remaining fields first would not change that — the assistant would gather
    # everything else and still refuse. Say so now.
    #
    # This used to test only for safety-critical, and the consequence was an
    # offer the system could not honour. In the clinician policy nothing is
    # safety-critical and everything blocking is `required`, so a stalled field
    # fell through to "tell me and I will carry on without it" — while the gate
    # went on refusing, because a required field is required. Declining
    # `creatinine_max` and re-running the gate still returns `can_answer:
    # False`; the offer was simply untrue.
    #
    # An independent judge found it in a transcript where the assistant asked
    # for three fields and, in the same breath, offered to proceed without three
    # different ones. Being truthful about safety and untruthful about itself is
    # still being untruthful.
    blocking = set(decision.blocking_fields)
    blocking_stalled = [n for n in stalled if n in sc or n in blocking]
    if blocking_stalled:
        out.stalled = blocking_stalled
        out.blocked_by_stalled = True
        return out

    # Levels are not mixed within a turn. Spec 6 asks for the safety-critical
    # questions first and *then* the rest; a turn containing one of each reads
    # as a single flat questionnaire and spends the patient's attention — which
    # they may give only once — on whichever question they happen to answer.
    critical_first = [n for n in askable if n in sc]
    batch = critical_first or askable

    overrides = prompts or {}
    for name in batch[:max_questions]:
        spec = field_spec(name)
        level = SAFETY_CRITICAL if name in sc else REQUIRED
        out.questions.append(Question(
            field=name, text=overrides.get(name) or spec.prompt, level=level,
            reason=("needed to answer safely" if level == SAFETY_CRITICAL
                    else "needed for a reliable answer")))

    # Nothing is left to record as stalled-but-workable. Candidates come from
    # `decision.blocking_fields`, so every one of them blocks by construction
    # and the branch above has already returned. The "carry on without it"
    # offer that used to live here could therefore never be honoured — see the
    # comment above — and an offer that cannot be kept is worse than none.
    state.mark_asked(out.fields)
    return out
