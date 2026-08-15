"""
src/assistant/state.py
──────────────────────
Conversation and patient state.  Spec sections 3, 13, 14, 24.

Two properties do the safety work here, and both are structural rather than
prompt-level:

**Facts are append-only.**  ``PatientContext`` never overwrites a value the
patient stated. A patient who says "I am 45" and later "I am 52" has not had
their age updated — they have produced a contradiction, and the assistant must
ask which is correct (spec 14). Silently keeping the newer value is the failure
this design forbids, because there is no way to tell a correction from a typo
from a different person using the same session.

**Every fact carries provenance.**  A ``Fact`` records the turn it came from and
the patient's own words. Nothing enters the context without a quote behind it,
so "did the model invent this?" is answerable by inspection rather than by
argument. Derived or assumed values have no way to be recorded at all: there is
no code path that constructs a ``Fact`` without a source quote.

Absence is represented by absence. There are no default values, no population
means, no zero-fills. ``context.get("age")`` returns ``None`` when age is
unknown, and ``None`` never compares equal to a plausible number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from src.llm.payload_validation import (
    PLAUSIBLE_RANGES as _PAYLOAD_RANGES,
    RECOMMENDED_FIELDS as _RECOMMENDED_PAYLOAD,
    REQUIRED_FIELDS as _REQUIRED_PAYLOAD,
)

__all__ = [
    "Fact", "Contradiction", "PatientContext", "ConversationState",
    "FieldSpec", "FIELDS", "field_spec", "build_payload", "SCALAR", "LIST",
]

SCALAR = "scalar"
LIST = "list"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── field registry ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FieldSpec:
    """
    One piece of patient information the system can hold.

    ``prompt`` is the question shown to the patient when the field is missing.
    Spec 23 requires these to be specific, short and answerable by a layperson,
    so they live here next to the field rather than being generated per turn —
    a generated question can drift into "please provide additional relevant
    information", which is the example the spec gives of what not to do.
    """

    name: str
    kind: str                       # SCALAR or LIST
    prompt: str
    unit: str = ""
    #: Physiologically possible bounds for numeric fields. A value outside these
    #: is rejected as uninterpretable rather than recorded — recording it would
    #: put an impossible number into the closed world of "patient-stated facts",
    #: where every later stage is entitled to trust it.
    bounds: Optional[Tuple[float, float]] = None
    #: Accepted values for enumerated fields, lowercased.
    choices: Optional[Tuple[str, ...]] = None
    #: Dotted path in the model payload this field supplies, when it supplies
    #: one. Present so ``build_payload`` is mechanical rather than a second
    #: hand-maintained mapping that can drift from the first.
    payload_path: Optional[str] = None


def _f(name, kind, prompt, unit="", bounds=None, choices=None,
       payload_path=None) -> FieldSpec:
    return FieldSpec(name=name, kind=kind, prompt=prompt, unit=unit,
                     bounds=bounds, choices=choices, payload_path=payload_path)


#: Every field the assistant can hold, by name.
#:
#: Deliberately small. Spec 15 forbids collecting information merely because it
#: exists, so there is no field for occupation, address, education or marital
#: status: nothing in the requirement policy can ask for what is not defined
#: here, which makes over-collection a compile-time impossibility rather than a
#: prompt-time instruction.
FIELDS: Dict[str, FieldSpec] = {s.name: s for s in (
    # ── demographics ──
    _f("age", SCALAR, "How old are you?", "years", bounds=(0.0, 120.0),
       payload_path="demographics.age"),
    _f("sex", SCALAR, "What is your sex at birth?",
       choices=("f", "female", "m", "male", "intersex", "prefer not to say"),
       payload_path="demographics.gender"),
    _f("pregnancy_status", SCALAR, "Are you currently pregnant, or could you be?",
       choices=("yes", "no", "unsure", "not applicable")),

    # ── the presenting problem ──
    _f("symptom", SCALAR, "What is the main symptom that is bothering you?"),
    _f("symptom_onset", SCALAR, "When did it start?"),
    _f("symptom_duration", SCALAR, "How long has it lasted?"),
    _f("symptom_severity", SCALAR,
       "How bad is it right now, on a scale of 1 to 10?", bounds=(0.0, 10.0)),
    _f("symptom_location", SCALAR, "Where in your body do you feel it?"),
    _f("symptom_character", SCALAR,
       "How would you describe it — sharp, dull, burning, pressure, or something else?"),
    _f("symptom_frequency", SCALAR, "Does it come and go, or is it constant?"),
    _f("symptom_trajectory", SCALAR,
       "Is it getting better, getting worse, or staying about the same?"),
    _f("associated_symptoms", LIST, "Are you having any other symptoms alongside it?"),
    _f("recent_events", LIST,
       "Has anything happened recently that might be related — an injury, illness, "
       "travel, or a change in medication?"),

    # ── background ──
    _f("medical_history", LIST,
       "Do you have any ongoing medical conditions?"),
    _f("current_medications", LIST,
       "What medications are you currently taking?"),
    _f("allergies", LIST,
       "Do you have any allergies to medicines?"),
    _f("previous_diagnosis", SCALAR,
       "Has a doctor already given you a diagnosis for this?"),

    # ── medication questions (spec 17) ──
    _f("medication_name", SCALAR, "Which medication are you asking about?"),
    _f("medication_dose", SCALAR,
       "What is the strength of each dose — for example 81 mg or 500 mg?"),
    _f("medication_form", SCALAR,
       "What form is it in — tablet, capsule, liquid, injection, or something else?"),
    _f("medication_frequency", SCALAR,
       "How often do you take it?"),
    _f("medication_reason", SCALAR, "What were you prescribed it for?"),

    # ── laboratory questions (spec 18) ──
    _f("test_name", SCALAR, "Which test result are you asking about?"),
    _f("test_value", SCALAR, "What is the result value?"),
    _f("test_unit", SCALAR,
       "What unit is it reported in? It is usually printed next to the number."),
    _f("test_reference_range", SCALAR,
       "What reference range does the report give? Ranges differ between "
       "laboratories, so I need the one printed on your report."),
    _f("test_date", SCALAR, "When was the test taken?"),

    # ── other intents ──
    _f("condition_name", SCALAR, "Which condition would you like to know about?"),
    _f("term", SCALAR, "Which term would you like explained?"),
    _f("report_text", SCALAR, "Please paste or upload the report text."),
    _f("topic", SCALAR, "What topic would you like information about?"),
)}


# ── clinical fields, derived from the model's own payload contract ───────────
#
# Generated, not typed out. `payload_validation` already defines what the risk
# models need — prompt, unit, and physiologically possible bounds — and
# restating that list here would create a second definition of "enough to
# score" which could drift from the one the pipeline actually enforces. This
# project has shipped that defect before; it is the recurring one.
#
# The assistant's field name is the last segment of the payload path, so
# `presentation_labs.creatinine_max` becomes `creatinine_max`. Fields already
# defined above (age, sex) keep their patient-facing prompt.
_LIST_PAYLOAD_FIELDS = frozenset({"comorbidities", "active_medications"})

for _spec in _REQUIRED_PAYLOAD + _RECOMMENDED_PAYLOAD:
    _name = _spec.path.rsplit(".", 1)[-1]
    if _name in FIELDS:
        continue
    FIELDS[_name] = _f(
        _name,
        LIST if _name in _LIST_PAYLOAD_FIELDS else SCALAR,
        _spec.prompt,
        _spec.unit,
        bounds=_PAYLOAD_RANGES.get(_spec.path),
        payload_path=_spec.path,
    )

FIELDS["primary_diagnosis"] = _f(
    "primary_diagnosis", SCALAR, "Primary working diagnosis",
    payload_path="primary_diagnosis")


def field_spec(name: str) -> FieldSpec:
    """Look up a field, failing loudly on an unknown one.

    A typo in the requirement policy would otherwise become a field that can
    never be filled, so the completeness gate would block forever with a
    question the patient cannot answer.
    """
    try:
        return FIELDS[name]
    except KeyError:
        raise KeyError(
            f"unknown patient field {name!r}; add it to src.assistant.state.FIELDS "
            f"or correct the requirement policy"
        ) from None


# ── facts ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Fact:
    """One patient-stated value, with the words it came from."""

    field: str
    value: Any
    turn: int
    source_quote: str
    recorded_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Contradiction:
    """Two incompatible statements about the same scalar field."""

    field: str
    previous: Fact
    current: Fact

    def question(self) -> str:
        """The clarifying question, quoting both statements (spec 14)."""
        return (
            f"Earlier you mentioned {self.field.replace('_', ' ')} was "
            f"{self.previous.value}, but now you have said {self.current.value}. "
            f"Which is correct?"
        )

    def to_dict(self) -> dict:
        return {"field": self.field,
                "previous": self.previous.to_dict(),
                "current": self.current.to_dict(),
                "question": self.question()}


class ValueRejected(ValueError):
    """A value could not be recorded: unparseable, or outside possible bounds."""


# ── patient context ──────────────────────────────────────────────────────────

def _normalise(spec: FieldSpec, value: Any) -> Any:
    """Coerce and bounds-check a value, or raise ``ValueRejected``.

    Normalisation is limited to whitespace, case for enumerated fields, and
    numeric parsing. It never rounds, never snaps to a nearby value and never
    substitutes a default — spec 13 requires that a stated 52 stays 52.
    """
    if value is None:
        raise ValueRejected(f"{spec.name}: no value")

    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise ValueRejected(f"{spec.name}: empty string")

    if spec.choices is not None:
        low = str(value).strip().lower()
        if low not in spec.choices:
            raise ValueRejected(
                f"{spec.name}: {value!r} is not one of {', '.join(spec.choices)}")
        return low

    if spec.bounds is not None:
        try:
            num = float(str(value).strip())
        except (TypeError, ValueError):
            raise ValueRejected(f"{spec.name}: {value!r} is not a number") from None
        if num != num:                                   # NaN
            raise ValueRejected(f"{spec.name}: not a number")
        low, high = spec.bounds
        if not (low <= num <= high):
            raise ValueRejected(
                f"{spec.name}: {num} is outside the possible range {low}–{high}")
        return num

    return value


def _same(a: Any, b: Any) -> bool:
    """Whether two recorded values are the same statement.

    Case- and whitespace-insensitive for text; exact for numbers. "52" and 52.0
    are the same age because both went through ``_normalise``; "Aspirin" and
    "aspirin" are the same drug.
    """
    if isinstance(a, float) and isinstance(b, float):
        return a == b
    return str(a).strip().lower() == str(b).strip().lower()


@dataclass
class PatientContext:
    """
    Append-only store of what the patient has actually said.

    ``history`` holds every statement ever recorded, in order. ``get`` returns
    the current value of a scalar field, which is the most recent *resolved*
    statement — and while a contradiction is open there is no current value at
    all, because choosing one would be the silent resolution spec 14 forbids.
    """

    history: List[Fact] = field(default_factory=list)
    contradictions: List[Contradiction] = field(default_factory=list)
    #: Fields the patient explicitly said they do not know or will not give.
    #: Distinct from missing: re-asking these is a spec 24 violation.
    declined: Set[str] = field(default_factory=set)

    # ── reading ──
    def statements(self, name: str) -> List[Fact]:
        return [f for f in self.history if f.field == name]

    def get(self, name: str) -> Any:
        """Current value: the value for a scalar, the accumulated list for a list.

        Returns ``None`` (or ``[]``) when unknown. Never a default.
        """
        spec = field_spec(name)
        facts = self.statements(name)
        if spec.kind == LIST:
            out: List[Any] = []
            for f in facts:
                for v in (f.value if isinstance(f.value, (list, tuple)) else [f.value]):
                    if not any(_same(v, o) for o in out):
                        out.append(v)
            return out
        if self.is_contradicted(name):
            return None
        return facts[-1].value if facts else None

    def is_known(self, name: str) -> bool:
        """Whether the field has a usable value.

        A declined field counts as known-unanswerable: the assistant must stop
        asking, and the gate must decide whether it can proceed without it.
        """
        if name in self.declined:
            return True
        value = self.get(name)
        return bool(value) if field_spec(name).kind == LIST else value is not None

    def is_contradicted(self, name: str) -> bool:
        return any(c.field == name for c in self.contradictions)

    def known_fields(self) -> Set[str]:
        return {n for n in FIELDS if self.is_known(n)}

    # ── writing ──
    def record(self, name: str, value: Any, turn: int,
               source_quote: str) -> Optional[Contradiction]:
        """
        Record a patient statement.

        Returns a ``Contradiction`` when this conflicts with an earlier scalar
        statement, in which case the new fact is still appended — the history is
        a record of what was said, not of what is true — but ``get`` reports the
        field as unknown until ``resolve`` is called.

        Raises ``ValueRejected`` when the value is unparseable or impossible.
        Callers must not swallow this and substitute something plausible.
        """
        spec = field_spec(name)
        if not str(source_quote).strip():
            raise ValueRejected(
                f"{name}: refusing to record a fact with no source quote; "
                f"every value must be traceable to the patient's own words")

        if spec.kind == LIST:
            values = value if isinstance(value, (list, tuple)) else [value]
            cleaned = [_normalise(spec, v) for v in values]
            self.history.append(Fact(name, cleaned, turn, source_quote))
            self.declined.discard(name)
            return None

        clean = _normalise(spec, value)
        prior = self.statements(name)
        self.history.append(Fact(name, clean, turn, source_quote))
        self.declined.discard(name)

        if prior:
            last = prior[-1]
            if not _same(last.value, clean):
                c = Contradiction(name, last, self.history[-1])
                self.contradictions.append(c)
                return c
        return None

    def decline(self, name: str) -> None:
        """Mark a field as one the patient cannot or will not answer."""
        field_spec(name)
        self.declined.add(name)

    def resolve(self, name: str, value: Any, turn: int,
                source_quote: str) -> None:
        """
        Settle an open contradiction with the patient's explicit choice.

        Only the patient can call this outcome, which is why it takes a quote:
        there is no automatic resolution rule, and no "most recent wins".
        """
        spec = field_spec(name)
        clean = _normalise(spec, value)
        self.contradictions = [c for c in self.contradictions if c.field != name]
        self.history.append(Fact(name, clean, turn, source_quote))

    # ── serialisation ──
    def to_dict(self) -> dict:
        return {
            "history": [f.to_dict() for f in self.history],
            "contradictions": [c.to_dict() for c in self.contradictions],
            "declined": sorted(self.declined),
            "current": {n: self.get(n) for n in sorted(self.known_fields())},
        }


#: Values the assistant records for `sex` mapped to what the models were fitted
#: on. The cohort column is M/F; the assistant accepts the words people use.
_GENDER_TO_PAYLOAD = {"f": "F", "female": "F", "m": "M", "male": "M"}


def build_payload(context: "PatientContext") -> Dict[str, Any]:
    """
    Assemble a model payload from what the clinician has actually stated.

    Walks ``FIELDS`` and places each known value at its ``payload_path``. Only
    known values are placed: an absent field is left out entirely rather than
    filled, so ``validate_payload`` sees a genuinely incomplete payload and
    names what is missing. Filling gaps here would defeat the completeness gate
    by making an empty payload indistinguishable from a healthy patient — the
    exact behaviour ``payload_validation`` was written to end.
    """
    payload: Dict[str, Any] = {}
    for name, spec in FIELDS.items():
        if not spec.payload_path or not context.is_known(name):
            continue
        value = context.get(name)
        if value in (None, [], ""):
            continue
        if name == "sex":
            value = _GENDER_TO_PAYLOAD.get(str(value).lower())
            if value is None:
                continue

        node = payload
        parts = spec.payload_path.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return payload


# ── conversation ─────────────────────────────────────────────────────────────

@dataclass
class ConversationState:
    """
    Everything the assistant knows about one conversation.

    Held in memory and serialisable to JSON. There is no database in this
    project; a server layer can persist ``to_dict()`` wherever it likes without
    this module changing.
    """

    session_id: str
    context: PatientContext = field(default_factory=PatientContext)
    turn: int = 0
    #: Intent name resolved for the current request, set by ``intents``.
    intent: Optional[str] = None
    #: Fields already asked about, so the assistant does not repeat itself
    #: (spec 24) even when the patient's reply did not answer the question.
    asked: Set[str] = field(default_factory=set)
    #: How many times each field has been asked. A field the patient keeps not
    #: answering must not be asked forever: the clarification engine reads this
    #: to stop looping and offer to move on instead, which is the difference
    #: between a careful assistant and one that has trapped the patient.
    ask_counts: Dict[str, int] = field(default_factory=dict)
    messages: List[Dict[str, str]] = field(default_factory=list)
    #: Whether the capability menu has been shown. Repeating it at someone who
    #: has already read it is a loop, not help.
    capabilities_shown: bool = False
    started_at: str = field(default_factory=_utcnow)

    def add_message(self, role: str, content: str) -> None:
        if role not in ("user", "assistant"):
            raise ValueError(f"unknown role {role!r}")
        self.messages.append({"role": role, "content": content,
                              "turn": self.turn, "at": _utcnow()})

    def begin_turn(self) -> int:
        self.turn += 1
        return self.turn

    def mark_asked(self, names: Iterable[str]) -> None:
        for n in names:
            field_spec(n)
            self.asked.add(n)
            self.ask_counts[n] = self.ask_counts.get(n, 0) + 1

    def times_asked(self, name: str) -> int:
        return self.ask_counts.get(name, 0)

    def unanswered_asks(self) -> Set[str]:
        """Fields that were asked about and are still unknown."""
        return {n for n in self.asked if not self.context.is_known(n)}

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "turn": self.turn,
            "intent": self.intent,
            "asked": sorted(self.asked),
            "ask_counts": dict(self.ask_counts),
            "context": self.context.to_dict(),
            "messages": self.messages,
        }

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str, **kw)
