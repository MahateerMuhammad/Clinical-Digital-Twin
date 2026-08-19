"""
src/assistant/intents.py
────────────────────────
Intent classification.  Spec section 4.

Chunk 1 is deterministic: scored keyword and phrase patterns, no model call.
That is not a placeholder for a classifier — it is the floor beneath one. A
language model added later (chunk 3) may *refine* the intent, but spec 27 puts
classification confidence in application code, and a deterministic prior means a
cold, rate-limited or malfunctioning model degrades the assistant to "ask what
you need help with" rather than to a confident wrong branch.

What this module does **not** do
────────────────────────────────
It does not detect emergencies. Spec 16 requires emergency screening to run
*before* intent classification and to bypass the information-collection
workflow entirely; that belongs to ``src.assistant.triage`` and it is
authoritative. ``Intent.EMERGENCY`` exists in this enum so that downstream code
has a name for the state triage produces, and this classifier never returns it —
if a keyword rule here could return EMERGENCY, there would be two competing
emergency detectors and the weaker one would eventually win an argument.

Ambiguity resolves to ``UNKNOWN``, which is a real answer. Spec 2 has the
assistant ask what the patient needs rather than guess, so a low-confidence
classification is routed to that question instead of to a branch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["Intent", "IntentResult", "classify", "resolve_intent",
           "MIN_CONFIDENCE", "SWITCH_CONFIDENCE",
           "PATIENT", "CLINICIAN", "MODES", "MODE_INTENTS",
           "KNOWLEDGE_INTENTS"]


#: Audience. The same message means different things depending on who typed it:
#: "how do I manage septic shock" is a guideline lookup from a clinician and a
#: frightening symptom report from a patient. Mode selects which rules apply, so
#: a patient session can never be routed to a clinician branch and vice versa.
PATIENT = "patient"
CLINICIAN = "clinician"
MODES = (PATIENT, CLINICIAN)


class Intent(str, Enum):
    # ── patient-facing ──
    SYMPTOM_ASSESSMENT = "symptom_assessment"
    MEDICATION_QUESTION = "medication_question"
    MEDICAL_REPORT_EXPLANATION = "medical_report_explanation"
    CONDITION_INFORMATION = "condition_information"
    TREATMENT_QUESTION = "treatment_question"
    PREVENTIVE_HEALTH = "preventive_health"
    DOCTOR_QUESTION_PREP = "doctor_question_prep"
    GENERAL_EDUCATION = "general_education"

    # ── clinician-facing ──
    #: The only intent that reaches the risk models. Its requirement policy
    #: lists the fourteen fields `payload_validation.REQUIRED_FIELDS` names, so
    #: "enough to score" has one definition rather than two.
    RISK_ASSESSMENT = "risk_assessment"
    GUIDELINE_LOOKUP = "guideline_lookup"
    DRUG_DOSING = "drug_dosing"
    #: Contraindications, warnings and interactions — properties of the drug,
    #: not of the patient. Split from `DRUG_DOSING` because they share a
    #: vocabulary and need opposite things: a dose depends on this patient's
    #: renal function, a contraindication list is the same sentence whoever is
    #: being treated. Merged, "what are the contraindications for vancomycin?"
    #: inherited the dosing policy and was answered by demanding a creatinine.
    DRUG_SAFETY = "drug_safety"
    COUNTERFACTUAL = "counterfactual"
    #: Asks the assistant to read a patient record it has no access to.
    #: Routed so the refusal can name the real boundary. Without it the message
    #: scored nothing and fell to UNKNOWN, whose reply — "I am not sure what you
    #: would like help with" — reads as the assistant being slow rather than as
    #: a statement that no record connection exists. Both are refusals; only one
    #: is true.
    RECORD_ACCESS = "record_access"
    #: Asks for a diagnosis. The system does not diagnose, and saying so is a
    #: different refusal from "no trusted source on file", which was the reply
    #: this used to get and which implies the right source would unlock it.
    DIAGNOSIS_REQUEST = "diagnosis_request"

    # ── both ──
    LAB_RESULT_INTERPRETATION = "lab_result_interpretation"
    TERMINOLOGY = "terminology"
    #: Produced by ``src.assistant.triage``, never by this module.
    EMERGENCY = "emergency"
    #: The opening turn, or a request to know what the assistant can do.
    CAPABILITIES = "capabilities"
    UNKNOWN = "unknown"


#: Questions about medicine rather than about the patient in front of you.
#:
#: A clinician working a case interrupts it constantly — "what's the first-line
#: vasopressor again?" in the middle of collecting labs. Those turns must be
#: answerable *without* disturbing the case, and the two failures that motivated
#: this both came from treating them as case turns:
#:
#:   * a guideline question asked mid-counterfactual could not take the session
#:     over (rightly — `SWITCH_CONFIDENCE` protects a half-collected case), so
#:     it was answered as "which value should I change?"
#:   * "how should severe hyperkalaemia be managed?" wrote `hyperkalaemia` as
#:     the patient's diagnosis, contradicting the septic shock already on file,
#:     and the reply asked the clinician which of the two their patient had.
#:
#: Both are correct behaviour applied to the wrong kind of turn. Marking the
#: intent is what lets the orchestrator tell them apart.
#:
#: `drug_dosing` is deliberately absent. "Can I give full-dose enoxaparin?" is a
#: question about this patient and needs their creatinine; treating it as
#: general knowledge would cut it off from the case it depends on.
#: `drug_safety` is present for the same reason `drug_dosing` is not: a package
#: insert describes the drug, not the patient, so answering one mid-case must
#: not write the drug into their record or disturb a half-collected assessment.
KNOWLEDGE_INTENTS: frozenset = frozenset({
    Intent.GUIDELINE_LOOKUP,
    Intent.DRUG_SAFETY,
    Intent.TERMINOLOGY,
    Intent.CONDITION_INFORMATION,
    Intent.GENERAL_EDUCATION,
})


#: Which intents are reachable in each mode. `EMERGENCY` appears in neither
#: because this module never returns it; triage owns it.
MODE_INTENTS: Dict[str, frozenset] = {
    PATIENT: frozenset({
        Intent.SYMPTOM_ASSESSMENT, Intent.MEDICATION_QUESTION,
        Intent.MEDICAL_REPORT_EXPLANATION, Intent.CONDITION_INFORMATION,
        Intent.TREATMENT_QUESTION, Intent.PREVENTIVE_HEALTH,
        Intent.DOCTOR_QUESTION_PREP, Intent.GENERAL_EDUCATION,
        Intent.LAB_RESULT_INTERPRETATION, Intent.TERMINOLOGY,
        Intent.CAPABILITIES, Intent.UNKNOWN,
    }),
    CLINICIAN: frozenset({
        Intent.RISK_ASSESSMENT, Intent.GUIDELINE_LOOKUP, Intent.DRUG_DOSING,
        Intent.DRUG_SAFETY,
        Intent.COUNTERFACTUAL, Intent.LAB_RESULT_INTERPRETATION,
        Intent.TERMINOLOGY, Intent.CAPABILITIES, Intent.UNKNOWN,
        Intent.RECORD_ACCESS, Intent.DIAGNOSIS_REQUEST,
    }),
}


#: Score below which the classification is not trusted and UNKNOWN is returned.
#: Set so that a single weak keyword ("pain" alone) is not enough to commit to a
#: branch, but a keyword plus a question frame is.
MIN_CONFIDENCE = 0.30

#: Score required to *abandon* an intent already under way, deliberately higher
#: than ``MIN_CONFIDENCE``. Starting a topic is cheap; leaving a half-collected
#: one throws away answers the patient has already given.
SWITCH_CONFIDENCE = 0.55


#: Drugs whose name alone signals the message is about the drug. Shared by
#: `DRUG_DOSING` and `DRUG_SAFETY` so the two cannot drift: a drug added for one
#: reading and not the other would route "contraindications" and "dose" to
#: different intents for the same agent.
_DRUG_NAMES = (r"vancomycin|vanc|cefepime|gentamicin|enoxaparin|heparin|"
               r"piperacillin|meropenem|amikacin|tobramycin|colistin")

#: The drug is being listed as current therapy rather than asked about.
#: "On vancomycin and norepinephrine" is part of a case description.
_NOT_PRESCRIBED = (r"(?<!\bon )(?<!\breceiving )(?<!\btaking )(?<!\bgiven )"
                   r"(?<!\bgetting )(?<!\bstarted on )")


@dataclass(frozen=True)
class _Rule:
    pattern: re.Pattern
    weight: float
    label: str


def _r(intent: Intent, expr: str, weight: float, label: str) -> Tuple[Intent, _Rule]:
    return intent, _Rule(re.compile(expr, re.I), weight, label)


# Patterns are word-bounded. Substring matching produced the kind of error this
# project keeps rediscovering: "cold" inside "could", "ache" inside "headaches"
# being fine but "ed" inside "red" not.
_RULES: List[Tuple[Intent, _Rule]] = [
    # ── symptoms ──
    _r(Intent.SYMPTOM_ASSESSMENT, r"\bi (?:have|feel|felt|get|am having|'ve got)\b", 0.45, "first-person symptom report"),
    _r(Intent.SYMPTOM_ASSESSMENT, r"\bmy \w+ (?:hurts?|aches?|is sore|feels)\b", 0.45, "body part complaint"),
    _r(Intent.SYMPTOM_ASSESSMENT, r"\b(?:pain|ache|aching|sore|hurts?|hurting)\b", 0.25, "pain word"),
    _r(Intent.SYMPTOM_ASSESSMENT, r"\b(?:fever|nausea|vomiting|dizzy|dizziness|rash|cough|swelling|fatigue|diarrh(?:o)?ea|itching|cramps?)\b", 0.30, "symptom word"),
    _r(Intent.SYMPTOM_ASSESSMENT, r"\bsymptoms?\b", 0.20, "symptom noun"),
    _r(Intent.SYMPTOM_ASSESSMENT, r"\bshould i (?:be )?(?:worry|worried|be concerned|see a doctor)\b", 0.30, "concern question"),

    # ── medication ──
    _r(Intent.MEDICATION_QUESTION, r"\b(?:medication|medicine|drug|tablet|capsule|pill|dose|dosage|mg|prescription|prescribed)\b", 0.35, "medication noun"),
    _r(Intent.MEDICATION_QUESTION, r"\b(?:i take|i'm taking|i am taking|am on|i'm on)\b", 0.35, "current medication"),
    _r(Intent.MEDICATION_QUESTION, r"\b(?:side ?effects?|interactions?|contraindicat\w+)\b", 0.40, "medication safety"),
    _r(Intent.MEDICATION_QUESTION, r"\b(?:can i (?:take|stop|skip)|should i stop|safe to take)\b", 0.40, "medication decision"),

    # ── laboratory ──
    _r(Intent.LAB_RESULT_INTERPRETATION, r"\b(?:blood test|lab(?:oratory)? (?:result|test|work)|test results?)\b", 0.45, "lab result phrase"),
    _r(Intent.LAB_RESULT_INTERPRETATION, r"\b(?:h(?:a)?emoglobin|hba1c|cholesterol|creatinine|glucose|white cell|platelets?|tsh|ferritin|vitamin d)\b", 0.40, "analyte name"),
    _r(Intent.LAB_RESULT_INTERPRETATION, r"\b(?:reference range|normal range|out of range|is (?:high|low|normal|elevated))\b", 0.35, "range language"),
    _r(Intent.LAB_RESULT_INTERPRETATION, r"\bmy \w+ (?:is|was|came back) (?:high|low|elevated|abnormal|\d)", 0.40, "value report"),
    # Clinician phrasing. The rules above are first-person ("my haemoglobin"),
    # which a doctor presenting someone else's result never writes.
    _r(Intent.LAB_RESULT_INTERPRETATION, r"\b(?:came back (?:at|as)|interpret|what do you make of)\b", 0.45, "clinician value report"),
    _r(Intent.LAB_RESULT_INTERPRETATION, r"\b\d+(?:\.\d+)?\s*(?:mg/dL|mmol/L|mEq/L|g/dL|K/uL|ng/mL|mmHg|bpm)\b", 0.45, "value with unit"),
    _r(Intent.LAB_RESULT_INTERPRETATION, r"\b(?:reference|range)\s*(?:of\s*)?\d+\s*[-–]\s*\d+\b", 0.40, "reference range given"),

    # ── report explanation ──
    _r(Intent.MEDICAL_REPORT_EXPLANATION, r"\b(?:my )?(?:report|scan|x-?ray|mri|ct scan|ultrasound|biopsy|discharge summary|pathology)\b", 0.45, "report noun"),
    _r(Intent.MEDICAL_REPORT_EXPLANATION, r"\b(?:what does (?:this|my) (?:report|scan|result)|explain (?:this|my) report)\b", 0.50, "explain report"),

    # ── condition information ──
    _r(Intent.CONDITION_INFORMATION, r"\bwhat (?:is|are|causes)\b", 0.30, "definition question"),
    _r(Intent.CONDITION_INFORMATION, r"\b(?:tell me about|information about|learn about)\b", 0.35, "information request"),
    _r(Intent.CONDITION_INFORMATION, r"\b(?:diabetes|hypertension|asthma|migraine|anaemia|anemia|arthritis|copd|eczema|thyroid)\b", 0.30, "condition name"),

    # ── treatment ──
    _r(Intent.TREATMENT_QUESTION, r"\b(?:treat(?:s|ed|ing|ment|ments)?|therapy|therapies|cure[ds]?|managed?)\b", 0.35, "treatment noun"),
    _r(Intent.TREATMENT_QUESTION, r"\b(?:surgery|operation|procedure|physiotherapy)\b", 0.30, "procedure noun"),

    # ── prevention ──
    _r(Intent.PREVENTIVE_HEALTH, r"\b(?:prevent|prevention|reduce (?:my )?risk|screening|vaccin\w+|immunis\w+|immuniz\w+)\b", 0.45, "prevention noun"),
    _r(Intent.PREVENTIVE_HEALTH, r"\b(?:healthy|diet|exercise|lifestyle)\b", 0.20, "lifestyle word"),

    # ── doctor prep ──
    # Word order varies more than a fixed phrase allows: "questions to ask",
    # "questions I should ask", "questions should I ask" are the same request.
    _r(Intent.DOCTOR_QUESTION_PREP, r"\bquestions?\b[\w\s]{0,20}?\bask\b", 0.60, "appointment prep"),
    _r(Intent.DOCTOR_QUESTION_PREP, r"\b(?:what should i ask|prepare for (?:my )?(?:appointment|visit))\b", 0.60, "appointment prep"),
    _r(Intent.DOCTOR_QUESTION_PREP, r"\b(?:appointment|seeing (?:my|the) doctor|consultation)\b", 0.20, "appointment noun"),

    # ── terminology ──
    _r(Intent.TERMINOLOGY, r"\bwhat does \w+ mean\b", 0.55, "term definition"),
    # Outweighs the topical rules deliberately. "What is the meaning of
    # nephrotoxic?" contains a dosing keyword and routed to drug_dosing, which
    # then demanded a creatinine value to define a word. An explicit definition
    # frame is the strongest signal in the sentence.
    _r(Intent.TERMINOLOGY, r"\b(?:meaning of|definition of|what is the term)\b", 0.60, "definition request"),
    # Imperative forms. "define oliguria" and "explain the term anion gap" are
    # how the request is actually phrased and matched nothing.
    _r(Intent.TERMINOLOGY, r"^\s*(?:define|explain)\b", 0.55, "imperative definition request"),
    _r(Intent.TERMINOLOGY, r"\bexplain the term\b", 0.60, "explicit term request"),
    _r(Intent.TERMINOLOGY, r"\bin (?:plain|simple) (?:english|terms|language)\b", 0.35, "plain-language request"),

    # ── capabilities ──
    _r(Intent.CAPABILITIES, r"\b(?:what can you (?:do|help)|how can you help|what do you do|who are you)\b", 0.75, "capability question"),
    _r(Intent.CAPABILITIES, r"^\s*(?:hi|hello|hey|good (?:morning|afternoon|evening))\b[\s!.,]*$", 0.75, "greeting only"),

    # ══ clinician-only ══════════════════════════════════════════════════════
    # ── risk assessment: the only path to the models ──
    _r(Intent.RISK_ASSESSMENT, r"\b(?:mortality|deterioration|readmission)\b", 0.50, "outcome name"),
    _r(Intent.RISK_ASSESSMENT, r"\b(?:risk (?:score|of|assessment)|predicted risk|prognosis|acuity)\b", 0.45, "risk language"),
    # A bare "risk" belonging to a patient. "What is her risk?" and "his risk"
    # are how the question is actually asked; the phrasings above all require a
    # qualifier, so the models were never consulted and the turn was answered
    # from the guideline corpus instead.
    # The outcome may sit between the possessive and the noun: "her mortality
    # risk", "his readmission risk". Requiring them adjacent meant the most
    # natural phrasing of the commonest question this system answers scored on
    # the outcome word alone, 0.50, and lost to a drug mentioned in passing.
    _r(Intent.RISK_ASSESSMENT, r"\b(?:his|her|their|the patient'?s?|this patient'?s?)\s+(?:\w+\s+)?risks?\b", 0.50, "possessive risk"),
    _r(Intent.RISK_ASSESSMENT, r"\bwhat(?:'s| is| are)\s+(?:the\s+)?risks?\b", 0.45, "bare risk question"),
    _r(Intent.RISK_ASSESSMENT, r"\b(?:icu (?:admission|transfer)|length of stay|\blos\b)\b", 0.45, "outcome name"),
    _r(Intent.RISK_ASSESSMENT, r"\b(?:score (?:this|him|her|the patient)|run the model|what are (?:his|her|their) odds)\b", 0.55, "explicit scoring request"),
    _r(Intent.RISK_ASSESSMENT, r"\bhow likely is (?:he|she|they|this patient)\b", 0.50, "likelihood question"),

    # ── guideline lookup ──
    # Plurals. `\bguideline\b` cannot match "guidelines" — the boundary fails on
    # the trailing s. "What are the guidelines for managing psoriasis?" scored
    # zero and was answered with the capability menu. This is the same defect
    # class as the `fluids_*` exclusion pattern that never matched `fluid_*`.
    _r(Intent.GUIDELINE_LOOKUP, r"\b(?:guidelines?|guidance|recommendations?|consensus|bundles?|protocols?)\b", 0.50, "guideline noun"),
    _r(Intent.GUIDELINE_LOOKUP, r"\b(?:first[- ]line|second[- ]line|standard of care|indicated|contraindicated)\b", 0.45, "management language"),
    _r(Intent.GUIDELINE_LOOKUP, r"\b(?:what does (?:the )?(?:kdigo|ssc|surviving sepsis|aha|acc|ada|gold|nice)\b)", 0.60, "society named"),
    _r(Intent.GUIDELINE_LOOKUP, r"\b(?:how (?:should|do) (?:i|we) manage|management of|approach to)\b", 0.45, "management question"),
    _r(Intent.GUIDELINE_LOOKUP, r"\b(?:target|threshold) (?:map|mean arterial|blood pressure|sats?|saturation)\b", 0.40, "target question"),

    # Weighted above GUIDELINE_LOOKUP's nouns on purpose: "look up this
    # patient's chart and tell me the guidelines they're on" is a records
    # request first, and answering the guideline half would imply the chart had
    # been read.
    _r(Intent.RECORD_ACCESS, r"\b(?:look up|pull up|open|access|retrieve|fetch|check)\b[^.?!]{0,30}\b(?:chart|record|records|notes?|emr|ehr|file)\b", 0.75, "record lookup"),
    _r(Intent.RECORD_ACCESS, r"\b(?:their|his|her|the patient'?s?)\s+(?:chart|records?|notes?|history|labs?)\b[^.?!]{0,20}\b(?:tell me|show me|what|summar\w+)", 0.60, "asks the system to read a record"),
    _r(Intent.RECORD_ACCESS, r"\bdo you have access to\b", 0.70, "asks about access"),

    _r(Intent.DIAGNOSIS_REQUEST, r"\bwhat(?:'s| is| are)? (?:the |your |his |her |their )?(?:diagnosis|diagnoses|differential)\b", 0.75, "asks for a diagnosis"),
    _r(Intent.DIAGNOSIS_REQUEST, r"\b(?:diagnose|what(?:'s| is) wrong with)\b", 0.65, "asks the system to diagnose"),
    _r(Intent.DIAGNOSIS_REQUEST, r"\bwhat does (?:this|he|she|the patient) have\b", 0.65, "asks what the patient has"),

    # ── drug dosing ──
    _r(Intent.DRUG_DOSING, r"\b(?:dos(?:e|ing|age)|renal(?:ly)? (?:adjust|dose|clear)|nephrotoxic|trough|level)\b", 0.45, "dosing noun"),
    # Outweighs the analyte rules on purpose. "Should I worry about vancomycin
    # with a creatinine of 3.2" is a drug question that mentions a lab value;
    # scored on the analyte alone it routed to lab interpretation and asked for
    # units and a reference range nobody was asking about.
    #
    # Not when the drug is being *listed as current therapy*. "On vancomycin and
    # norepinephrine" is part of a case description, not a dosing question, and
    # this rule was strong enough to carry a whole case handover — labs, vitals,
    # and an explicit "what is her mortality risk?" — into the drug-dosing path,
    # where it was answered with retrieved guideline text and a complaint that no
    # medication dose had been supplied. The models were never run.
    _r(Intent.DRUG_DOSING, _NOT_PRESCRIBED + r"\b(?:" + _DRUG_NAMES + r")\b", 0.55, "drug named"),
    _r(Intent.DRUG_DOSING, r"\b(?:worried|concerned) about\b", 0.20, "safety concern"),
    _r(Intent.DRUG_DOSING, r"\b(?:safe to (?:give|continue)|should i (?:hold|stop|continue))\b", 0.45, "drug decision"),

    # ── drug safety ──
    # A property of the drug, answerable from its package insert without knowing
    # whose chart it is. Deliberately excludes "safe to give", "should I hold"
    # and "nephrotoxic", which are `DRUG_DOSING`: those ask whether to give it to
    # *this* patient, and that does depend on their renal function.
    _r(Intent.DRUG_SAFETY,
       r"\b(?:contraindicat\w+|side ?effects?|adverse (?:effects?|reactions?|events?)|"
       r"drug interactions?|interacts? with|boxed warning|black.?box|"
       r"warnings? and (?:cautions?|precautions?))\b", 0.55, "drug safety question"),
    # Lower than the dosing rule's 0.55 for the same names: on its own a drug
    # name is not a safety question, and only the pairing should outrank dosing.
    _r(Intent.DRUG_SAFETY, _NOT_PRESCRIBED + r"\b(?:" + _DRUG_NAMES + r")\b", 0.30, "drug named"),

    # ── counterfactual ──
    _r(Intent.COUNTERFACTUAL, r"\bwhat if\b", 0.55, "counterfactual frame"),
    _r(Intent.COUNTERFACTUAL, r"\b(?:if (?:the |his |her |their )?\w+ (?:were|was|came down|dropped|improved|fell|rose))\b", 0.50, "hypothetical change"),
    _r(Intent.COUNTERFACTUAL, r"\b(?:how much would|would it change|counterfactual|sensitivity)\b", 0.45, "change question"),
]

#: Where a rule fires for two intents, these break the tie in favour of the more
#: specific reading. A message naming a laboratory analyte *and* a symptom is a
#: laboratory question with context, not a symptom report with a stray word.
_PRIORITY: Dict[Intent, float] = {
    Intent.CAPABILITIES: 0.30,
    # A counterfactual names an outcome ("would his mortality risk change?") and
    # would otherwise lose to RISK_ASSESSMENT on that word alone. "What if" is
    # the more specific reading and wins.
    Intent.COUNTERFACTUAL: 0.16,
    Intent.MEDICAL_REPORT_EXPLANATION: 0.12,
    Intent.LAB_RESULT_INTERPRETATION: 0.10,
    Intent.DOCTOR_QUESTION_PREP: 0.10,
    Intent.RISK_ASSESSMENT: 0.08,
    Intent.MEDICATION_QUESTION: 0.06,
    # Above `DRUG_DOSING`: both fire on the drug's name, and where the message
    # also carries explicit safety language that is the more specific reading.
    Intent.DRUG_SAFETY: 0.10,
    Intent.DRUG_DOSING: 0.06,
    Intent.TERMINOLOGY: 0.04,
}


@dataclass
class IntentResult:
    """A classification, with the evidence that produced it."""

    intent: Intent
    confidence: float
    #: Human-readable rule labels that fired, for the audit trail (spec 28).
    evidence: List[str] = field(default_factory=list)
    #: Runner-up intents above the floor, so a later LLM pass has somewhere to
    #: disagree from and the audit shows what was nearly chosen.
    alternatives: List[Tuple[str, float]] = field(default_factory=list)

    @property
    def is_confident(self) -> bool:
        return self.intent is not Intent.UNKNOWN

    def to_dict(self) -> dict:
        return {"intent": self.intent.value,
                "confidence": round(self.confidence, 3),
                "evidence": self.evidence,
                "alternatives": [(i, round(s, 3)) for i, s in self.alternatives]}


def _concept_fallback(text: str, mode: str) -> Optional["IntentResult"]:
    """
    A bare clinical concept, with no question frame, is a guideline lookup.

    Clinicians search as well as ask. "antibiotic timing in septic shock" and
    "when to start dialysis in AKI" carry no interrogative and matched no rule,
    so they scored zero and — on a first turn — were answered with the
    capability menu. Forty-five of the forty-five gold retrieval queries are
    phrased this way, which is what made the gap visible.

    Deliberately clinician-only. The equivalent patient message is far more
    likely to be a symptom report than a literature search, and routing "chest
    pain" to a guideline lookup would skip the questions that matter.
    """
    if mode != CLINICIAN:
        return None
    try:
        from src.llm.terminology import normalise_diagnosis
    except Exception:                                   # pragma: no cover
        return None

    dx = normalise_diagnosis(text)
    if not dx.matched:
        return None
    return IntentResult(
        Intent.GUIDELINE_LOOKUP, MIN_CONFIDENCE,
        [f"no interrogative, but names the concept {dx.concept!r}"])


def classify(message: str, *, first_turn: bool = False,
             mode: str = PATIENT) -> IntentResult:
    """
    Classify a message.

    ``first_turn`` biases an unrecognised opening message towards CAPABILITIES,
    which is what spec 2 asks for: the assistant introduces what it can help
    with rather than guessing at an unclear request.

    ``mode`` restricts the reachable intents. Rules for the other audience are
    not merely deprioritised, they are not evaluated — so a patient describing
    chest pain cannot land on ``drug_dosing`` because they mentioned a tablet,
    and a clinician cannot land on ``symptom_assessment``, whose questions
    ("How bad is it, 1 to 10?") are addressed to the wrong person.
    """
    if mode not in MODE_INTENTS:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")

    text = (message or "").strip()
    if not text:
        return IntentResult(Intent.CAPABILITIES if first_turn else Intent.UNKNOWN,
                            0.0, ["empty message"])

    reachable = MODE_INTENTS[mode]
    scores: Dict[Intent, float] = {}
    evidence: Dict[Intent, List[str]] = {}
    for intent, rule in _RULES:
        if intent not in reachable:
            continue
        if rule.pattern.search(text):
            scores[intent] = scores.get(intent, 0.0) + rule.weight
            evidence.setdefault(intent, []).append(rule.label)

    if not scores:
        fallback = _concept_fallback(text, mode)
        if fallback is not None:
            return fallback
        return IntentResult(Intent.CAPABILITIES if first_turn else Intent.UNKNOWN,
                            0.0, ["no pattern matched"])

    for intent in scores:
        scores[intent] += _PRIORITY.get(intent, 0.0)

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0].value))
    best, best_score = ranked[0]
    # Cap rather than normalise: several independent rules firing is genuine
    # evidence, but the number is a confidence, not a probability over intents.
    confidence = min(best_score, 1.0)

    if confidence < MIN_CONFIDENCE:
        fallback = _concept_fallback(text, mode)
        if fallback is not None:
            return fallback
        return IntentResult(
            Intent.CAPABILITIES if first_turn else Intent.UNKNOWN,
            confidence,
            [f"best match {best.value} below confidence floor {MIN_CONFIDENCE}"],
            [(i.value, s) for i, s in ranked[:3]],
        )

    return IntentResult(
        best, confidence, evidence.get(best, []),
        [(i.value, s) for i, s in ranked[1:4] if s >= MIN_CONFIDENCE],
    )


def resolve_intent(state: Any, message: str, mode: str = PATIENT) -> IntentResult:
    """
    Classify a message *in the context of the conversation so far*.

    Per-message classification is not enough, and the gap is a safety hole
    rather than a rough edge. When the assistant asks "when did it start?" and
    the patient replies "yesterday morning, about a 7 out of 10", that reply
    contains no intent keyword at all. Classified on its own it scores nothing
    and falls to ``UNKNOWN`` — whose requirement policy is empty, so the
    completeness gate finds nothing missing and reports that it may answer. The
    system would then produce a symptom assessment having collected two facts,
    with every guard reporting green.

    So an intent under way persists. It is abandoned only when a new message
    argues for a different one at ``SWITCH_CONFIDENCE``, which is set above the
    floor for *starting* an intent: opening a topic is cheap, whereas discarding
    a half-collected one throws away answers the patient already gave and, by
    spec 24, must not be asked for again.

    ``state`` is a ``ConversationState``; it is typed loosely to keep this
    module free of an import cycle with ``state.py``.
    """
    current_name = getattr(state, "intent", None)
    turn = getattr(state, "turn", 0) or 0
    fresh = classify(message, first_turn=(turn <= 1 and not current_name),
                     mode=mode)

    if not current_name:
        return fresh

    try:
        current = Intent(current_name)
    except ValueError:
        return fresh

    # An intent stored under the other audience must not be continued here.
    if current not in MODE_INTENTS[mode]:
        return fresh

    # These hold nothing worth preserving — no facts are collected under them.
    if current in (Intent.UNKNOWN, Intent.CAPABILITIES):
        return fresh

    if fresh.intent is current:
        return fresh

    if fresh.intent is not Intent.UNKNOWN and fresh.confidence >= SWITCH_CONFIDENCE:
        fresh.evidence.append(
            f"switched from {current.value} at confidence {fresh.confidence:.2f}")
        return fresh

    # An aside is not a switch, so it does not have to clear the switching bar.
    #
    # SWITCH_CONFIDENCE exists to stop a half-collected case being abandoned.
    # A knowledge question abandons nothing: the orchestrator answers it against
    # a scratch context and leaves `state.intent` where it was. Holding it to the
    # same threshold made "how should severe hyperkalaemia be managed?" — asked
    # during a counterfactual — come back as "which value should I change?",
    # because the only two options were *take the session over* or *be read as
    # the intent already running*. There was no third option for "answer this
    # and carry on", which is most of what a clinician does mid-case.
    #
    # MIN_CONFIDENCE still applies: a weak signal falls through to the case, as
    # before. What changes is only that a *confident* aside no longer needs to
    # win an argument it was never having.
    if fresh.intent in KNOWLEDGE_INTENTS and fresh.confidence >= MIN_CONFIDENCE:
        fresh.evidence.append(
            f"aside: answered as {fresh.intent.value} at confidence "
            f"{fresh.confidence:.2f}; {current.value} remains the open case")
        return fresh

    return IntentResult(
        current, fresh.confidence,
        [f"continuing {current.value}: no competing intent reached "
         f"{SWITCH_CONFIDENCE}"],
        fresh.alternatives,
    )
