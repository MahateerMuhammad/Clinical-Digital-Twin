"""
src/assistant/extraction.py
───────────────────────────
Turning what a patient typed into structured facts.  Spec 13, 26, 27, 33.1.

This is the only stage where a language model touches patient information, and
it is the stage where hallucinated facts would do the most damage: a value that
enters ``PatientContext`` is treated as something the patient said by every
later stage, including the completeness gate and the grounding verifier. So the
model's output is treated as a *proposal*, and application code decides what is
admitted.

Four filters, in order, and a proposal must survive all of them:

1. **Parse.** Output that is not JSON is discarded. It is never repaired by
   guessing what the model meant.
2. **Schema.** Field names not in ``state.FIELDS`` are dropped. The model cannot
   invent a field, so it cannot smuggle in an observation the policy never
   authorised anyone to collect (spec 15).
3. **Quote.** Every proposed fact must carry a verbatim span from the patient's
   message, and that span must actually appear in the message. This is the
   filter that does the real work: a model that decides an unmentioned patient
   is 65 has to produce a quote saying so, and there isn't one.
4. **Normalise.** The value goes through ``PatientContext.record``, which bounds-
   checks it and detects contradictions. Impossible values are rejected rather
   than clamped.

The deterministic fallback
──────────────────────────
With no model available, ``extract`` still runs a small set of high-precision
patterns — age, sex, a numeric severity. It extracts less, which surfaces as the
completeness gate asking more questions. It never extracts something different.
That ordering matters: degrading to "asks more" is safe, degrading to "guesses
more" is not, and a system whose safety depends on an API being reachable is not
safe.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.assistant.state import (
    FIELDS, LIST, PatientContext, ValueRejected, field_spec,
)

__all__ = ["Proposal", "ExtractionResult", "extract", "build_prompt",
           "parse_response", "EXTRACTION_SYSTEM_PROMPT"]


@dataclass(frozen=True)
class Proposal:
    """A fact the extractor believes the patient stated."""

    field: str
    value: Any
    quote: str

    def to_dict(self) -> dict:
        return {"field": self.field, "value": self.value, "quote": self.quote}


@dataclass
class ExtractionResult:
    """What was admitted, what was refused, and why."""

    accepted: List[Proposal] = field(default_factory=list)
    #: (proposal-ish, reason). Kept for the audit trail: a rejected proposal is
    #: the system catching a fabrication, which is worth being able to count.
    rejected: List[Tuple[Dict[str, Any], str]] = field(default_factory=list)
    contradictions: List[Any] = field(default_factory=list)
    used_model: bool = False
    parse_failed: bool = False

    @property
    def fields(self) -> List[str]:
        return [p.field for p in self.accepted]

    def to_dict(self) -> dict:
        return {"accepted": [p.to_dict() for p in self.accepted],
                "rejected": [{"proposal": p, "reason": r} for p, r in self.rejected],
                "contradictions": [getattr(c, "field", str(c))
                                   for c in self.contradictions],
                "used_model": self.used_model,
                "parse_failed": self.parse_failed}


# ── prompting ────────────────────────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """\
You extract structured facts from a patient's message. You do not diagnose, \
advise, or interpret.

Return a JSON object with one key, "facts", whose value is a list. Each item is:
  {"field": "<field name>", "value": <value>, "quote": "<exact span from the message>"}

Rules, all absolute:
- Use ONLY field names from the list you are given. Ignore anything else the \
patient mentions.
- "quote" must be copied character-for-character from the patient's message. If \
you cannot copy an exact span that states the fact, do not emit the fact.
- Do not infer. If the patient did not state their age, there is no age fact. \
If they named a medicine but not its dose, there is no dose fact.
- Do not normalise units, round numbers, or expand abbreviations in "value" \
beyond what the patient wrote.
- If the message states nothing extractable, return {"facts": []}.
"""


def _field_catalogue(allowed: Optional[List[str]] = None) -> str:
    names = allowed if allowed is not None else sorted(FIELDS)
    lines = []
    for n in names:
        spec = FIELDS.get(n)
        if spec is None:
            continue
        kind = "list of strings" if spec.kind == LIST else "single value"
        extra = ""
        if spec.choices:
            extra = f"; one of: {', '.join(spec.choices)}"
        elif spec.bounds:
            extra = f"; numeric {spec.bounds[0]}–{spec.bounds[1]}"
        lines.append(f"- {n} ({kind}{extra}): {spec.prompt}")
    return "\n".join(lines)


def build_prompt(message: str, allowed: Optional[List[str]] = None) -> str:
    """
    The user-side prompt.

    ``allowed`` narrows the catalogue to the fields the current intent actually
    needs. Offering the model every field invites it to fill in ones nobody
    asked for, which is the over-collection spec 15 forbids — and a field the
    policy never requested has no reviewed question behind it either.
    """
    return (f"Patient message:\n\"\"\"\n{message}\n\"\"\"\n\n"
            f"Fields you may extract:\n{_field_catalogue(allowed)}\n\n"
            f"Return the JSON object now.")


# ── parsing ──────────────────────────────────────────────────────────────────

_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def parse_response(raw: str) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Pull the proposal list out of a model response.

    Returns ``(proposals, parse_failed)``. A response that cannot be parsed
    yields an empty list and ``parse_failed=True`` — never a partial salvage.
    Guessing at malformed JSON is how an invented value gets in through the one
    door that was supposed to be checking.
    """
    text = (raw or "").strip()
    if not text:
        return [], True

    # Models wrap JSON in prose or fences often enough to be worth one attempt
    # at locating the object — but only locating it, not reconstructing it.
    for candidate in (text, (_JSON_BLOCK.search(text) or _Empty()).group(0)
                      if _JSON_BLOCK.search(text) else None):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, list):
            return [f for f in obj if isinstance(f, dict)], False
        if isinstance(obj, dict):
            facts = obj.get("facts")
            if isinstance(facts, list):
                return [f for f in facts if isinstance(f, dict)], False
            # A bare single fact is a common shape; accept it, still validated.
            if "field" in obj:
                return [obj], False
            return [], False
    return [], True


class _Empty:
    @staticmethod
    def group(_):
        return None


# ── the deterministic floor ──────────────────────────────────────────────────

#: Common presenting complaints, matched literally.
#:
#: A closed lexicon rather than a "the noun after 'I have'" rule, because that
#: rule extracts "a question", "an appointment" and "no idea" as symptoms. Being
#: unable to name a symptom is a good failure — the assistant asks. Naming the
#: wrong one is not, because nothing downstream re-examines it.
_SYMPTOM_LEXICON: Tuple[str, ...] = (
    "chest pain", "chest tightness", "back pain", "stomach pain", "abdominal pain",
    "sore throat", "shortness of breath", "difficulty breathing",
    "headache", "migraine", "fever", "cough", "nausea", "vomiting", "diarrhoea",
    "diarrhea", "constipation", "dizziness", "fatigue", "rash", "itching",
    "swelling", "numbness", "tingling", "palpitations", "heartburn",
    "toothache", "earache", "insomnia", "bloating", "cramps", "bleeding",
)

_SYMPTOM_RE = re.compile(
    r"\b(" + "|".join(sorted((re.escape(s) for s in _SYMPTOM_LEXICON),
                             key=len, reverse=True)) + r")\b", re.I)

#: Onset expressions that name a time without needing to be parsed into one.
#: The value is stored as the patient wrote it — spec 13 forbids normalising it
#: into a date the patient never gave.
_ONSET_RE = re.compile(
    r"\b(?:started|began|came on|since|from)\s+"
    r"((?:about\s+|around\s+)?(?:a\s+|an\s+)?"
    r"(?:yesterday|today|this (?:morning|afternoon|evening)|last (?:night|week)|"
    r"\d+\s*(?:min(?:ute)?s?|hours?|hrs?|days?|weeks?|months?|years?)"
    r"(?:\s+ago)?))\b", re.I)

_ONSET_BARE_RE = re.compile(
    r"\b(yesterday|this morning|this afternoon|last night|"
    r"\d+\s*(?:min(?:ute)?s?|hours?|hrs?|days?|weeks?|months?)\s+ago)\b", re.I)

_DURATION_RE = re.compile(
    r"\bfor\s+((?:about\s+|around\s+)?(?:a\s+|an\s+)?"
    r"\d*\s*(?:min(?:ute)?s?|hours?|hrs?|days?|weeks?|months?|years?))\b", re.I)

_DET_PATTERNS: Tuple[Tuple[str, re.Pattern], ...] = (
    # Clinician shorthand. "45M", "72 yo F", "45-year-old male" are how a case
    # is actually presented, and none of them are first-person — the patterns
    # below them only fire on "I am 45", so without these the risk path could
    # not read an age or sex out of a normal handover sentence.
    ("age", re.compile(r"\b(\d{1,3})\s*(?:yo|y/o|yrs?|years?[- ]old)\b", re.I)),
    ("age", re.compile(r"\b(\d{1,3})\s*(?:yo|y/o)?\s*[MF]\b")),
    ("sex", re.compile(r"\b\d{1,3}\s*(?:yo|y/o|yrs?|years?[- ]old)?\s*(male|female)\b", re.I)),
    # "72yo F" and "72 y/o M" are as common as "72M"; requiring the digits to
    # sit against the letter missed every one of them.
    ("sex", re.compile(r"\b\d{1,3}\s*(?:yo|y/o|yrs?|years?[- ]old)?[,\s]*([MF])\b")),
    # Third-person and correction phrasing. A clinician writes "he's 74" or
    # "sorry, she is 74" — never "I am 74". Without these a correction extracted
    # nothing, so no contradiction was raised and the assistant restated the
    # stale age as fact on the next turn. That is the failure spec 13 and 14
    # exist to prevent, and it was invisible to every automated suite because
    # the gold sets only ever stated an age once.
    ("age", re.compile(r"\b(?:he|she|they|patient)(?:'s|'re| is| was| are)\s+(\d{1,3})\b(?!\s*(?:/|out of|%))", re.I)),
    ("age", re.compile(r"\bage[d]?\s+(?:is\s+)?(\d{1,3})\b", re.I)),
    ("age", re.compile(r"\bi(?:'m| am)\s+(\d{1,3})\b(?!\s*(?:/|out of))", re.I)),
    ("age", re.compile(r"\b(\d{1,3})\s+years?\s+old\b", re.I)),
    ("sex", re.compile(r"\bi(?:'m| am)\s+(?:a\s+)?(male|female|man|woman)\b", re.I)),
    ("symptom_severity", re.compile(
        r"\b(\d{1,2})\s*(?:/|out of)\s*10\b", re.I)),
    ("symptom", _SYMPTOM_RE),
    ("symptom_onset", _ONSET_RE),
    ("symptom_onset", _ONSET_BARE_RE),
    ("symptom_duration", _DURATION_RE),
    # Terminology requests name their own term. Asking "which term would you
    # like explained?" of someone who just wrote "what does oliguric mean?" is
    # the kind of friction that makes a tool feel broken even when every guard
    # is behaving correctly.
    ("term", re.compile(r"\bwhat does\s+(?:the\s+(?:term|word)\s+)?([a-z][a-z\- ]{2,30}?)\s+mean\b", re.I)),
    ("term", re.compile(r"^\s*(?:define|explain)\s+(?:the\s+(?:term|word)\s+)?([a-z][a-z\- ]{2,30})\s*\??\s*$", re.I)),
    ("term", re.compile(r"\b(?:meaning|definition) of\s+(?:the\s+(?:term|word)\s+)?([a-z][a-z\- ]{2,30}?)\s*\??$", re.I)),
)

_SEX_MAP = {"man": "male", "woman": "female", "male": "male", "female": "female"}


_DIAGNOSIS_ALIASES: Optional[List[Tuple[str, re.Pattern]]] = None


def _diagnosis_patterns() -> List[Tuple[str, re.Pattern]]:
    """
    Alias → matcher for every diagnosis the guideline corpus is keyed on.

    Built from ``terminology.CONCEPTS``, which the clinician pipeline already
    uses to map free text onto guideline concepts. Without this the
    deterministic floor cannot read "septic shock" out of "what is the
    first-line vasopressor in septic shock?", so the commonest clinician
    question — a guideline lookup naming its own condition — stalls at the gate
    asking for a diagnosis the message already contains.

    Longest alias first, so "septic shock" is preferred over "sepsis" and the
    quote shown back to the clinician is what they actually wrote.
    """
    global _DIAGNOSIS_ALIASES
    if _DIAGNOSIS_ALIASES is None:
        from src.llm.terminology import CONCEPTS

        pairs: List[Tuple[str, re.Pattern]] = []
        for concept in CONCEPTS.values():
            terms = {concept.display, *getattr(concept, "aliases", ())}
            for term in terms:
                term = str(term).strip()
                # Display names like "Sepsis / Septic Shock" are labels, not
                # things anyone types; split them into their halves.
                for part in (p.strip() for p in term.split("/")):
                    if len(part) >= 3:
                        pairs.append((part, re.compile(
                            r"\b" + re.escape(part) + r"\b", re.I)))
        pairs.sort(key=lambda p: len(p[0]), reverse=True)
        _DIAGNOSIS_ALIASES = pairs
    return _DIAGNOSIS_ALIASES


_DRUG_PATTERN: Optional[re.Pattern] = None


def _drug_pattern() -> re.Pattern:
    """
    One alternation over the known drug lexicon.

    The same names ``grounding.py`` checks generated text against, so a drug the
    verifier would recognise is a drug extraction can read. Four characters
    minimum — shorter tokens in the lexicon collide with ordinary words.
    """
    global _DRUG_PATTERN
    if _DRUG_PATTERN is None:
        from src.llm.terminology import BRAND_TO_INGREDIENT, INGREDIENT_CLASS

        names = sorted(
            (n for n in set(INGREDIENT_CLASS) | set(BRAND_TO_INGREDIENT)
             if len(n) >= 4),
            key=len, reverse=True)
        _DRUG_PATTERN = re.compile(
            r"\b(?:" + "|".join(re.escape(n) for n in names) + r")\b", re.I)
    return _DRUG_PATTERN


_LAB_PATTERNS: Optional[List[Tuple[str, re.Pattern]]] = None

#: Vital-sign shorthand. Not in the lab synonym table, and each needs its own
#: reading: "BP 82/40" gives the systolic from the first number only.
_VITAL_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("sbp_min", r"(?:sbp|systolic|bp)\s*(?:of|is|was|=|:)?\s*(\d{2,3})(?:\s*/\s*\d{1,3})?"),
    ("hr_max", r"(?:hr|heart rate|pulse)\s*(?:of|is|was|=|:)?\s*(\d{2,3})"),
    ("spo2_min", r"(?:spo2|sats?|saturation[s]?|o2 sat[s]?)\s*(?:of|is|was|=|:)?\s*(\d{2,3})"),
    ("rr_max", r"(?:rr|resp(?:iratory)? rate)\s*(?:of|is|was|=|:)?\s*(\d{1,2})"),
    ("temperature_max", r"(?:temp(?:erature)?)\s*(?:of|is|was|=|:)?\s*(\d{2}(?:\.\d)?)"),
)


def _lab_patterns() -> List[Tuple[str, re.Pattern]]:
    """
    "<analyte> <number>" matchers, built from the payload contract's own table.

    ``payload_validation._LAB_SYNONYMS`` already lists what a clinician types
    for each analyte — cr, scr, hco3, plt, hct. Reusing it means the words the
    validator accepts are exactly the words extraction reads, rather than two
    lists that agree today.

    Without these the risk path is unusable without an API key: "creatinine
    3.2, BUN 48, WBC 19.5" is how the values arrive and nothing else reads it.
    """
    global _LAB_PATTERNS
    if _LAB_PATTERNS is None:
        from src.llm.payload_validation import _LAB_SYNONYMS

        pairs: List[Tuple[str, re.Pattern]] = []
        for canon, alts in _LAB_SYNONYMS.items():
            words = {a.replace("_max", "").replace("_min", "").replace("_", " ")
                     for a in alts} | {canon.replace("_max", "").replace("_min", "")}
            for word in sorted(words, key=len, reverse=True):
                word = word.strip()
                # Single-character aliases are kept — "K 5.1" is how potassium
                # is written, and excluding it left the gate asking for a value
                # the clinician had already given, twice, until the loop-breaker
                # fired and the conversation dead-ended. The pattern still
                # requires a number immediately after, so a bare "K" matches
                # nothing.
                if not word:
                    continue
                pairs.append((canon, re.compile(
                    r"\b" + re.escape(word) +
                    r"\b\s*(?:of|is|was|at|=|:)?\s*(\d+(?:\.\d+)?)", re.I)))
        for canon, expr in _VITAL_PATTERNS:
            pairs.append((canon, re.compile(r"\b" + expr, re.I)))
        _LAB_PATTERNS = pairs
    return _LAB_PATTERNS


#: The topic of a guideline-shaped question. Deliberately narrow: it fires only
#: after an explicit guidance word, so "what is her risk?" cannot be read as
#: naming a condition. Plurals are spelled out — `guidelines?`, `protocols?` —
#: because this codebase has now shipped the singular-only word-boundary bug
#: three times (`fluids_*`, `vitals_*`, `\bguideline\b`).
_TOPIC_PATTERN = re.compile(
    r"\b(?:guidelines?|protocols?|recommendations?|management|treatment)\b"
    r"\s+(?:for|of|in)\s+(?:managing\s+|treating\s+)?"
    r"(?P<topic>[a-z][a-z'\- ]{2,40}?)\s*[?.,;]?\s*$",
    re.IGNORECASE)


def _deterministic(message: str, allowed: Optional[List[str]]) -> List[Proposal]:
    """High-precision patterns only. Silence is the correct output when unsure."""
    out: List[Proposal] = []
    seen = set()

    for canon, pat in _lab_patterns():
        if canon in seen or (allowed is not None and canon not in allowed):
            continue
        m = pat.search(message)
        if m:
            out.append(Proposal(canon, m.group(1), m.group(0)))
            seen.add(canon)

    if allowed is None or "primary_diagnosis" in allowed:
        for _, pat in _diagnosis_patterns():
            m = pat.search(message)
            if m:
                out.append(Proposal("primary_diagnosis", m.group(0), m.group(0)))
                seen.add("primary_diagnosis")
                break
        # A guideline question names its own topic, and the lexicon above only
        # knows the fifteen concepts the corpus covers. "What are the guidelines
        # for managing psoriasis?" matched nothing, so the assistant asked for a
        # primary working diagnosis — which the clinician had just given in the
        # question. Capturing the topic lets the reply be the true one: that
        # psoriasis is outside the corpus. Naming a topic is not diagnosing it;
        # the value still faces every filter, and an unmapped concept retrieves
        # nothing and is declined.
        if "primary_diagnosis" not in seen:
            m = _TOPIC_PATTERN.search(message)
            if m:
                out.append(Proposal("primary_diagnosis",
                                    m.group("topic").strip(), m.group(0)))
                seen.add("primary_diagnosis")

    if allowed is None or "medication_name" in allowed:
        m = _drug_pattern().search(message)
        if m:
            out.append(Proposal("medication_name", m.group(0), m.group(0)))
            seen.add("medication_name")

    for name, pat in _DET_PATTERNS:
        if allowed is not None and name not in allowed:
            continue
        if name in seen:
            continue
        m = pat.search(message)
        if not m:
            continue
        value = m.group(1)
        if name == "sex":
            value = _SEX_MAP.get(value.lower(), value.lower())
        out.append(Proposal(name, value, m.group(0)))
        seen.add(name)
    return out


# ── admission ────────────────────────────────────────────────────────────────

def _quote_is_real(quote: str, message: str) -> bool:
    """
    Whether the quote actually appears in what the patient wrote.

    Whitespace-insensitive, case-insensitive; nothing else. A model that
    paraphrases the patient has not quoted them, and the whole point of the
    check is that a fabricated fact cannot produce a real span to sit behind.
    """
    if not quote or not str(quote).strip():
        return False
    norm = lambda s: re.sub(r"\s+", " ", str(s)).strip().lower()
    return norm(quote) in norm(message)


def extract(message: str, context: PatientContext, turn: int,
            *, backend: Any = None, allowed: Optional[List[str]] = None,
            ) -> ExtractionResult:
    """
    Extract facts from ``message`` and record the admissible ones in ``context``.

    ``backend`` is any object with ``complete_json(system, user)`` — the
    ``OpenRouterBackend`` shape. Passing ``None``, or a backend that fails, falls
    through to the deterministic patterns. A model failure must never be able to
    turn into a silent gap in the safety chain, so the fallback is not optional.
    """
    result = ExtractionResult()
    text = (message or "").strip()
    if not text:
        return result

    proposals: List[Dict[str, Any]] = []

    if backend is not None and getattr(backend, "available", False):
        try:
            raw = backend.complete_json(EXTRACTION_SYSTEM_PROMPT,
                                        build_prompt(text, allowed))
            proposals, result.parse_failed = parse_response(raw)
            result.used_model = not result.parse_failed
        except Exception as exc:                       # network, auth, rate limit
            result.rejected.append(({}, f"backend call failed: {exc}"))
            proposals, result.parse_failed = [], True

    # The floor runs on every turn, not only when the model fails, and fills
    # only the fields the model left alone.
    #
    # It used to be an either/or, which made a successful model call *lose*
    # information: given all 61 fields in scope the model returned fourteen
    # laboratory values and no diagnosis, while the floor's lexicon reads
    # "septic shock" out of the same sentence every time. The gate then asked
    # for a diagnosis the clinician had already given — a worse turn than the
    # one with no model at all.
    #
    # This is not a weakening: floor proposals go through the identical four
    # filters below, so anything unquoted, out of range or off-schema is
    # rejected exactly as a model's would be. The model wins on any field both
    # propose, being the better reader of unusual phrasing.
    claimed = {p.get("field") for p in proposals}
    proposals += [p.to_dict() for p in _deterministic(text, allowed)
                  if p.field not in claimed]

    for raw_prop in proposals:
        name = raw_prop.get("field")
        value = raw_prop.get("value")
        quote = raw_prop.get("quote", "")

        if name not in FIELDS:
            result.rejected.append((raw_prop, f"unknown field {name!r}"))
            continue
        if allowed is not None and name not in allowed:
            result.rejected.append(
                (raw_prop, f"{name} is not needed for this request"))
            continue
        if value is None or (isinstance(value, str) and not value.strip()):
            result.rejected.append((raw_prop, "no value"))
            continue
        if not _quote_is_real(quote, text):
            result.rejected.append(
                (raw_prop, "quote does not appear in the patient's message"))
            continue

        try:
            contradiction = context.record(name, value, turn, str(quote))
        except ValueRejected as exc:
            result.rejected.append((raw_prop, str(exc)))
            continue

        result.accepted.append(Proposal(name, value, str(quote)))
        if contradiction is not None:
            result.contradictions.append(contradiction)

    return result
