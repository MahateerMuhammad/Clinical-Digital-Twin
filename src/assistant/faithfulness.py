"""
src/assistant/faithfulness.py
─────────────────────────────
The pre-return verification.  Spec 21, 9, 10, 33.

Spec 21 lists ten questions to ask before returning any medical response, and
requires that a YES to any of them stops the response. This module asks them in
code, so the answer does not depend on a model's willingness to incriminate its
own output.

Relationship to ``src.llm.grounding``
────────────────────────────────────
Checks 6 and 7 — unsupported claims, fabricated citations — are already solved
there, thoroughly, against a closed world of permissible facts. They are reused
rather than reimplemented. This module adds the checks that are specific to a
patient conversation: that no patient fact was invented, that a possibility was
not stated as a diagnosis, that uncertainty survived into the output.

What this can and cannot establish
──────────────────────────────────
It establishes *traceability*: every number and citation in the response came
from the patient or from a retrieved document. It does not establish clinical
correctness. A correctly-quoted guideline that does not apply to this patient
passes every check here. That gap is closed by clinician review of the corpus,
not by code — it is the project's one open safety item and this module does not
close it.

Because composition in ``answer.py`` is deterministic, these checks currently
pass by construction. They are written now because the point of the check is to
still be there when an LLM rephrasing pass is added and construction stops being
a guarantee.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from src.assistant import answer as A
from src.assistant import gate as G
from src.assistant import triage as T
from src.assistant.state import FIELDS, PatientContext
from src.llm.grounding import FactStore, build_fact_store, verify_text

__all__ = ["Check", "FaithfulnessReport", "verify"]


@dataclass(frozen=True)
class Check:
    """One of spec 21's questions, and how it came out."""

    number: int
    name: str
    passed: bool
    detail: str = ""
    #: Blocking checks stop the response. Non-blocking ones are recorded and
    #: surfaced, because "the answer is thin" is not "the answer is unsafe".
    blocking: bool = True

    def to_dict(self) -> dict:
        return {"check": self.number, "name": self.name, "passed": self.passed,
                "detail": self.detail, "blocking": self.blocking}


@dataclass
class FaithfulnessReport:
    checks: List[Check] = field(default_factory=list)
    grounding: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(c.blocking and not c.passed for c in self.checks)

    @property
    def failures(self) -> List[Check]:
        return [c for c in self.checks if not c.passed]

    def render(self) -> str:
        if self.ok:
            return "FAITHFULNESS OK — all blocking checks passed."
        return "FAITHFULNESS FAILED:\n" + "\n".join(
            f"  [{c.number:2}] {c.name}: {c.detail}" for c in self.failures
            if c.blocking)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "checks": [c.to_dict() for c in self.checks],
                "grounding": self.grounding}


# ── patterns ─────────────────────────────────────────────────────────────────

#: Language that asserts a diagnosis rather than a possibility (spec 33.8).
#: Narrow on purpose: "this could be" and "one cause of this is" must survive,
#: because hedged explanation is the entire product.
#:
#: The lookahead is load-bearing. Without it the pattern matched "What you have
#: told me" — the answer's own section heading, mandated by spec 20 — so every
#: correctly-composed answer failed check 5 and would have been withheld. A
#: verifier that rejects valid output is not a strict verifier, it is a broken
#: one, and it trains its owner to disable it.
_DIAGNOSTIC = re.compile(
    r"\byou (?:have|are suffering from|are diagnosed with)\b"
    r"(?!\s+(?:told|said|mentioned|given|provided|described|shared)\b)"
    r"|\bthis is (?:definitely|certainly|clearly)\b"
    r"|\byour diagnosis is\b"
    r"|\bit is (?:definitely|certainly) \w+\b",
    re.I)

#: Directive medication language, which needs more than a general source
#: behind it (spec 17, spec 33).
_MED_DIRECTIVE = re.compile(
    r"\b(?:take|start|stop|increase|decrease|double|halve|skip)\b[^.]{0,40}"
    r"\b(?:dose|tablet|capsule|pill|mg\b|medication|medicine)\b"
    r"|\byou should (?:take|stop|start)\b",
    re.I)

#: Reassurance the system has no basis for (spec 16, spec 9).
_UNFOUNDED_REASSURANCE = re.compile(
    r"\b(?:nothing to worry about|probably fine|no cause for concern|"
    r"you'?ll be fine|it'?s nothing|definitely not serious)\b", re.I)

#: Phrases that mark uncertainty as preserved (spec 33.14).
_UNCERTAINTY = re.compile(
    r"\b(?:could|may|might|possible|possibilit\w+|cannot|can'?t|not able to|"
    r"uncertain|unclear|do not know|don'?t know|general information)\b", re.I)


def _patient_facts(context: PatientContext) -> Dict[str, Any]:
    """
    Stated facts with their types intact, for the fact store.

    Types matter here and it took a live failure to notice. This previously
    returned ``List[str]``: ``build_fact_store`` registers ints and floats as
    permissible *numbers* and strings as *entities*, so stringifying every value
    meant no number the clinician supplied was ever registered as one. Check 1
    then rejected the patient's own creatinine as ungrounded, and every report
    carrying a numeric value would have been withheld.

    It stayed hidden because the patient path answers with prose and citations
    and almost never quotes a figure back.
    """
    out: Dict[str, Any] = {}
    for name in FIELDS:
        value = context.get(name)
        if value in (None, [], ""):
            continue
        out[name] = value
    return out


def _patient_value_strings(context: PatientContext) -> List[str]:
    """Flattened string forms, for the echo check (check 2)."""
    out: List[str] = []
    for value in _patient_facts(context).values():
        for v in (value if isinstance(value, list) else [value]):
            out.append(str(v))
    return out


# ── the ten questions ────────────────────────────────────────────────────────

def verify(response: A.Answer,
           context: PatientContext,
           decision: G.GateDecision,
           *, documents: Optional[Sequence[dict]] = None,
           triage: Optional[T.TriageResult] = None,
           intent: Any = None,
           predictions: Optional[Dict[str, Any]] = None,
           upstream_grounding: Optional[Dict[str, Any]] = None,
           ) -> FaithfulnessReport:
    """
    Run spec 21's checklist over a composed response.

    ``report.ok`` is False when any blocking check failed, and the caller must
    then withhold the response rather than repair it — spec 21 says correct it
    first, and the only correction available without a model is to refuse.
    """
    report = FaithfulnessReport()
    text = response.to_markdown()
    docs = list(documents if documents is not None else response.documents)

    # An emergency response is composed entirely from the red-flag rules and
    # cites no documents; grounding it against a document store would fail on
    # its own advice text. It is verified by construction — the wording comes
    # from a reviewed config file — so the numeric checks are skipped and the
    # safety checks are not.
    is_emergency = response.status == A.EMERGENCY_RESPONSE

    # Model outputs are part of the permissible world when the answer came from
    # the models: a 1.9% mortality figure traces to a prediction, not to
    # anything the clinician typed, and without them every risk report fails
    # check 1 on its own headline number.
    fact_store: FactStore = build_fact_store(
        payload=_patient_facts(context),
        predictions=predictions,
        documents=[d for d in docs],
    )

    # 1 / 6 / 7 — only provided information, no unsupported claims, no
    # fabricated citations. Delegated to the existing verifier.
    if upstream_grounding:
        # The text was already verified by the stage that composed it, against a
        # closed world this function does not have: the full payload, the model
        # predictions and the system constants — the length-of-stay threshold
        # among them, which appears in a withheld-task label and is neither a
        # patient value nor a prediction.
        #
        # Re-deriving the verdict here from the rendered markdown with a poorer
        # fact set does not make it stricter, it makes it wrong: it rejected a
        # correctly grounded report over the "5.63" in its own withheld-task
        # notice. The upstream verdict is carried, not second-guessed.
        ok = bool(upstream_grounding.get("ok"))
        report.grounding = dict(upstream_grounding)
        report.checks.append(Check(
            1, "only provided information", ok,
            "verified upstream against payload, predictions and evidence"))
        report.checks.append(Check(
            7, "no fabricated citations", ok, "verified upstream"))
    elif is_emergency or not text.strip():
        report.checks.append(Check(1, "only provided information", True,
                                   "emergency response: composed from reviewed "
                                   "rules, not from generated text"))
        report.checks.append(Check(7, "no fabricated citations", True,
                                   "no citations claimed"))
    else:
        gr = verify_text(text, fact_store, require_citation_for_directives=False)
        report.grounding = gr.to_dict()
        numeric = [v for v in gr.violations if v.kind == "ungrounded_number"]
        citations = [v for v in gr.violations if v.kind == "unknown_citation"]
        report.checks.append(Check(
            1, "only provided information", not numeric,
            "; ".join(v.detail for v in numeric[:3])))
        report.checks.append(Check(
            7, "no fabricated citations", not citations,
            "; ".join(v.detail for v in citations[:3])))

    # 2 — no invented patient information. Every value echoed back under "what
    # you have told me" must exist in the context.
    told = next((b for t, b in response.sections
                 if t.lower().startswith("what you have told")), "")
    known = {v.lower() for v in _patient_value_strings(context)}
    invented = []
    for line in told.splitlines():
        if ":" not in line:
            continue
        value = line.split(":", 1)[1].strip().lower()
        for part in [p.strip() for p in value.split(",") if p.strip()]:
            if part and part not in known:
                invented.append(part)
    report.checks.append(Check(
        2, "no invented patient information", not invented,
        f"echoed values absent from the context: {invented[:3]}"))

    # 3 — nothing clinically important assumed. `PatientContext` cannot hold an
    # unquoted fact, so this is structural; it is asserted rather than measured.
    report.checks.append(Check(
        3, "nothing clinically important assumed", True,
        "patient facts require a verbatim quote to be recorded"))

    # 4 — did not answer past a gate that was closed.
    answered = response.status == A.ANSWERED
    report.checks.append(Check(
        4, "did not answer with required information missing",
        not (answered and not decision.can_answer),
        f"gate status was {decision.status}"))

    # 5 — possibility was not presented as diagnosis.
    diagnostic = _DIAGNOSTIC.findall(text)
    report.checks.append(Check(
        5, "possibility not stated as diagnosis", not diagnostic,
        f"diagnostic assertions: {diagnostic[:3]}"))

    # 8 — no emergency sign was passed over. If triage flagged something, the
    # response must actually carry it.
    missed = []
    if triage is not None and triage.flags and not is_emergency:
        lowered = text.lower()
        for f in triage.flags:
            if not any(w.lower()[:18] in lowered for w in f.warning_signs) \
                    and f.advice.strip()[:25].lower() not in lowered:
                missed.append(f.rule_id)
    unfounded = _UNFOUNDED_REASSURANCE.findall(text)
    report.checks.append(Check(
        8, "no emergency sign overlooked", not missed and not unfounded,
        f"missed flags {missed}; unfounded reassurance {unfounded[:2]}"))

    # 9 — no medication-specific direction without the information spec 17 needs.
    directives = _MED_DIRECTIVE.findall(text)
    med_ok = True
    if directives:
        needed = ("medication_name", "medication_dose", "allergies")
        med_ok = all(context.is_known(n) for n in needed)
    report.checks.append(Check(
        9, "no medication direction without sufficient information", med_ok,
        f"directive language present: {directives[:2]}"))

    # 10 — uncertainty communicated. Non-blocking: a thin answer is not a
    # dangerous one, and blocking here would reject a correct refusal.
    report.checks.append(Check(
        10, "uncertainty communicated",
        bool(_UNCERTAINTY.search(text)) or not answered,
        "no hedging or limitation language found", blocking=False))

    return report
