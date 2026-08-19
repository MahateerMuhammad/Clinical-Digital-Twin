"""
src/llm/grounding.py
────────────────────
Hallucination control for generated clinical text.

The premise: an LLM in this system is a *verbaliser*, not a reasoner. Every number
and every clinical claim it emits must already exist in structured inputs — the
validated payload, the model predictions, or the retrieved evidence. Anything else
is fabrication, regardless of how plausible it reads.

The failure this prevents is concrete. The previous fallback generator emitted
"Serum Creatinine elevations (+0.45 SHAP) and WBC leukocytosis (+0.38 SHAP)" for
every patient — SHAP values that were never computed, for a model that was not
consulted, under a docstring claiming "100% factual".

Two mechanisms:

``FactStore``   the closed world of permissible facts, built from the inputs.
``verify_text`` scans generated text and returns every violation, so the caller
                can regenerate, redact, or refuse.

This is a *containment* mechanism, not a correctness proof: it can show that every
number traces to an input, but it cannot tell you the clinical reasoning is sound.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = [
    "Violation", "FactStore", "VerificationReport", "verify_text",
    "verify_rephrase", "build_fact_store", "SEVERITY_ORDER",
]

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}

# Numbers that carry no clinical claim and need no provenance.
_TRIVIAL_NUMBERS = {
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
    12.0, 24.0, 48.0, 72.0, 100.0,
}

_NUMBER_RE = re.compile(r"(?<![\w/])(\d+(?:\.\d+)?)\s*(%|mg/dL|mmol/L|mEq/L|mmHg|bpm|K/uL|days?|hours?|h\b)?",
                        re.I)
_CITATION_RE = re.compile(r"\[([^\[\]]{3,200})\]")

#: Bare URLs and DOIs. Numbers inside them are bibliographic identifiers, never
#: clinical values, and must be excluded from numeric grounding.
_URL_RE = re.compile(r"(?:https?://\S+|www\.\S+|\b10\.\d{4,9}/[^\s\)\]]+)", re.I)

#: Numerals belonging to fixed outcome labels rather than to patient values.
#: A hyphenated interval ("30-day", "6-hour", "90-day") names the outcome window;
#: "within N hours/days" is the same thing spelled out. Deliberately narrow — it
#: matches an integer bound to a time unit, never a bare number.
_LABEL_RE = re.compile(
    r"\b\d{1,3}-(?:hour|hr|day|days|week|month|year)\b"
    r"|\bwithin\s+\d{1,3}\s+(?:hour|hours|day|days|week|weeks|month|months)\b",
    re.I,
)

# Phrases that assert a computation the system may not have performed.
_CLAIM_PATTERNS = [
    (re.compile(r"\bSHAP\b", re.I), "shap_claim",
     "mentions SHAP attribution"),
    (re.compile(r"\b(?:p\s*[<=>]\s*0?\.\d+|95%\s*CI|confidence interval)\b", re.I),
     "statistical_claim", "asserts a statistical result"),
    (re.compile(r"\b(?:proven|guaranteed|will (?:prevent|cure|ensure)|definitely)\b", re.I),
     "causal_overclaim", "asserts certainty or causation"),
    # SHAP is attribution, not causation: it says how far a feature moved this
    # model's output, not what moved the patient. "Creatinine contributed most"
    # and "creatinine drove the mortality risk" quote the same number, and only
    # the second makes a claim about the illness. Nothing else in this file
    # separates them — no digit changes and no citation is invented — so a
    # rephrase could restate every attribution causally and verify clean.
    #
    # Deliberately verbs of production, not of association: "associated with",
    # "seen alongside", "accompanied by" are the correct language and must pass.
    # The negative lookbehinds are not a refinement, they are the difference
    # between this rule working and being switched off. "causes" is a noun as
    # readily as a verb, and the composer's own driver disclaimer — "these are
    # associations learned from past admissions, not causes" — is the single
    # sentence in the report most responsible for preventing this overclaim.
    # Without the guard the rule withheld every scored report for containing its
    # own warning against the thing the rule looks for.
    (re.compile(r"(?<!not )(?<!never )(?<!no )(?<!rather than )"
                r"\b(?:drove|drives|driving|caused|causes|causing|led to|leads to|"
                r"resulted in|results in|responsible for|due to the patient'?s)\b",
                re.I),
     "causal_overclaim", "states an attribution as a cause"),
    (re.compile(r"\b(?:I recommend|you should administer|start the patient on)\b", re.I),
     "directive_without_citation", "issues a treatment directive"),
]


@dataclass
class Violation:
    kind: str
    severity: str
    detail: str
    span: str = ""

    def to_dict(self) -> dict:
        return {"kind": self.kind, "severity": self.severity,
                "detail": self.detail, "span": self.span}


@dataclass
class FactStore:
    """The closed world of facts a generation is permitted to state."""

    numbers: Dict[float, str] = field(default_factory=dict)   # value -> provenance
    entities: Set[str] = field(default_factory=set)           # drugs, concepts
    citations: Set[str] = field(default_factory=set)          # admissible citation strings
    doc_ids: Set[str] = field(default_factory=set)
    text_corpus: str = ""                                     # retrieved text, for claim support

    def add_number(self, value: Any, provenance: str,
                   *, percent_variants: bool = False) -> None:
        try:
            f = float(value)
        except (TypeError, ValueError):
            return
        if f != f:            # NaN
            return
        for variant in self._variants(f, percent=percent_variants):
            self.numbers.setdefault(round(variant, 6), provenance)

    @staticmethod
    def _variants(f: float, *, percent: bool = False) -> Iterable[float]:
        """
        The renderings of ``f`` a report may legitimately print.

        ``percent`` widens the set with the ×100 forms, so a probability stored
        as 0.0286 also grounds "2.86%" and "2.9%". It is opt-in, and that is the
        whole point of the flag: applied to every value in [0, 1] regardless of
        what the value *means*, it turned any small decimal in a retrieved
        document into a licence for its hundredfold.

        Concretely, KDIGO defines stage 1 AKI by a creatinine rise of
        0.3 mg/dL. With the expansion unconditional, that threshold registered
        30.0 as a permissible number, so a generated sentence claiming "30%" of
        anything — a risk, a mortality rate, a cohort proportion — passed
        grounding with provenance pointing at a renal guideline. Evidence
        numbers are also stored un-scoped to their source document, so the
        licence was not even confined to reports citing that guideline.

        Only quantities the system knows to be probabilities are expanded:
        model predictions, and the published constants it may quote. Payload
        values are labs, counts and ages in real units, none of which is ever
        correctly restated as a percentage of itself.
        """
        out = {f, round(f, 1), round(f, 2), round(f, 3)}
        if percent and 0.0 <= f <= 1.0:
            pct = f * 100.0
            out |= {pct, round(pct, 0), round(pct, 1), round(pct, 2)}
        return out

    def knows_number(self, value: float, tolerance: float = 0.011) -> Optional[str]:
        key = round(value, 6)
        if key in self.numbers:
            return self.numbers[key]
        for known, prov in self.numbers.items():
            if abs(known - value) <= tolerance:
                return prov
        return None

    def summary(self) -> dict:
        return {"n_numbers": len(self.numbers), "n_entities": len(self.entities),
                "n_citations": len(self.citations), "n_docs": len(self.doc_ids)}


def build_fact_store(
    payload: Optional[dict] = None,
    predictions: Optional[dict] = None,
    documents: Optional[Sequence[dict]] = None,
    extra_numbers: Optional[Dict[str, Any]] = None,
) -> FactStore:
    """Assemble the permissible-fact world from the structured inputs."""
    fs = FactStore()

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
        elif isinstance(node, bool):
            return
        elif isinstance(node, (int, float)):
            # No payload field is a probability: every one is a lab in its own
            # units, a count, or an age. See `_variants`.
            fs.add_number(node, f"payload:{path}")
        elif isinstance(node, str):
            s = node.strip()
            if s and len(s) < 80:
                fs.entities.add(s.lower())

    if payload:
        walk(payload, "")
        # register normalised drug identities too: a patient charted as "vanc" is
        # legitimately described as being on vancomycin
        try:
            from src.llm.terminology import normalise_diagnosis, normalise_medications
            for d in normalise_medications(payload.get("active_medications")):
                if d.matched:
                    fs.entities.add(d.ingredient.lower())
                    if d.drug_class:
                        fs.entities.add(d.drug_class.lower().replace("_", " "))
            dx = normalise_diagnosis(payload.get("primary_diagnosis"))
            for c in dx.all_concepts:
                fs.entities.add(c.replace("_", " "))
            if dx.display:
                fs.entities.add(dx.display.lower())
        except Exception:      # pragma: no cover - terminology is optional here
            pass
    if predictions:
        for k, v in predictions.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                # A calibrated probability is the one quantity the report really
                # does restate as a percentage.
                fs.add_number(v, f"prediction:{k}", percent_variants=True)
            elif isinstance(v, str):
                fs.entities.add(v.lower())
    if extra_numbers:
        # System-computed quantities: published constants, payload-fidelity
        # AUROCs, medication relevance scores, SHAP attributions. The rates and
        # retention figures among them are genuinely printed both ways.
        for k, v in extra_numbers.items():
            fs.add_number(v, f"context:{k}", percent_variants=True)

    corpus_parts: List[str] = []
    for d in documents or []:
        cit = str(d.get("citation", "")).strip()
        if cit:
            fs.citations.add(cit.strip("[]").strip().lower())
        did = str(d.get("doc_id", "")).strip()
        if did:
            fs.doc_ids.add(did.lower())
            fs.citations.add(did.lower())
        for key in ("title", "text", "strength", "section"):
            val = d.get(key)
            if isinstance(val, str) and val:
                corpus_parts.append(val)
                # bracketed spans inside document text are themselves admissible
                # citations — e.g. a DailyMed label title "[WG CRITICAL CARE, LLC]"
                for br in _CITATION_RE.findall(val):
                    fs.citations.add(br.strip().lower())
        # Numbers appearing inside retrieved evidence are quotable — as they are
        # written there, and not multiplied by a hundred. A guideline threshold
        # is a threshold, not a proportion of anything.
        for key in ("text", "title", "strength"):
            val = d.get(key)
            if isinstance(val, str):
                for m in _NUMBER_RE.finditer(val):
                    fs.add_number(m.group(1), f"evidence:{did or cit}")

    fs.text_corpus = "\n".join(corpus_parts).lower()
    return fs


@dataclass
class VerificationReport:
    ok: bool
    violations: List[Violation] = field(default_factory=list)
    n_numbers_checked: int = 0
    n_citations_checked: int = 0
    fact_store: Optional[dict] = None

    @property
    def critical(self) -> List[Violation]:
        return [v for v in self.violations if v.severity == "critical"]

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "n_numbers_checked": self.n_numbers_checked,
            "n_citations_checked": self.n_citations_checked,
            "violations": [v.to_dict() for v in self.violations],
            "fact_store": self.fact_store,
        }

    def render(self) -> str:
        if self.ok:
            return "GROUNDING OK — every number and citation traces to an input."
        lines = [f"GROUNDING FAILED — {len(self.violations)} violation(s):"]
        for v in sorted(self.violations, key=lambda x: SEVERITY_ORDER.get(x.severity, 9)):
            lines.append(f"  [{v.severity.upper():8}] {v.kind}: {v.detail}"
                         + (f"  …{v.span}…" if v.span else ""))
        return "\n".join(lines)


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def _normalise(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _verbatim_quote_spans(text: str, fact_store: FactStore) -> List[Tuple[int, int]]:
    """
    Spans holding retrieved text quoted verbatim, which make no claim of their own.

    The report renders each guideline record as a Markdown blockquote of the
    source passage. Those passages are written by guideline authors, who say
    "AKI caused by sepsis" and quote confidence intervals, and the claim
    patterns cannot tell that language apart from the same words asserted by the
    system in its own voice. Flagging it would withhold a correct report for
    faithfully reproducing its own evidence — the failure mode that teaches an
    owner to switch the verifier off.

    The exemption is earned, not granted by punctuation: a ``>`` line counts
    only when its content really occurs in the retrieved corpus. A rephrase
    cannot smuggle an assertion out of scope by formatting it as a quote.
    """
    corpus = _normalise(fact_store.text_corpus)
    if not corpus:
        return []

    spans: List[Tuple[int, int]] = []
    pos = 0
    for line in text.splitlines(keepends=True):
        start, pos = pos, pos + len(line)
        body = line.strip()
        if not body.startswith(">"):
            continue
        # The composer truncates a long passage at 400 characters and marks the
        # cut with an ellipsis; the prefix is still verbatim.
        body = body.lstrip(">").strip().rstrip("…").strip()
        if len(body) >= 20 and _normalise(body) in corpus:
            spans.append((start, pos))
    return spans


_DRUG_LEXICON_CACHE: Optional[Set[str]] = None


def _drug_lexicon() -> Set[str]:
    """Known ingredient and brand names, from the terminology module."""
    global _DRUG_LEXICON_CACHE
    if _DRUG_LEXICON_CACHE is None:
        try:
            from src.llm.terminology import BRAND_TO_INGREDIENT, INGREDIENT_CLASS
            _DRUG_LEXICON_CACHE = {
                n for n in set(INGREDIENT_CLASS) | set(BRAND_TO_INGREDIENT)
                if len(n) >= 4
            }
        except Exception:      # pragma: no cover
            _DRUG_LEXICON_CACHE = set()
    return _DRUG_LEXICON_CACHE


# Standard clinical time windows carry no patient-specific claim.
_TRIVIAL_TIME_UNITS = {"h", "hour", "hours", "day", "days"}


def verify_text(
    text: str,
    fact_store: FactStore,
    *,
    require_citation_for_directives: bool = True,
    allow_trivial_numbers: bool = True,
) -> VerificationReport:
    """
    Check generated text against the closed world of permissible facts.

    Returns every violation found. ``ok`` is True only when there are none of
    severity ``critical`` or ``high``.
    """
    report = VerificationReport(ok=True, fact_store=fact_store.summary())
    if not text or not str(text).strip():
        report.ok = False
        report.violations.append(Violation("empty_generation", "critical",
                                           "generator produced no text"))
        return report

    text = str(text)

    # Identifiers inside citation brackets (PMIDs, SPL ids, guideline section
    # numbers) are bibliographic, not clinical claims — exclude those spans from
    # numeric grounding.
    citation_spans = [(m.start(), m.end()) for m in _CITATION_RE.finditer(text)]

    # URLs and DOIs likewise. These sit in the provenance section, *outside* citation
    # brackets, so the bracket rule never reached them: a reference such as
    # ``https://www.ahajournals.org/doi/10.1161/CIR.0000000000001038`` yielded "1161"
    # and "0000000000001038" as ungrounded clinical values and the report was withheld.
    # On real admissions that rejected 64% of otherwise valid reports — a fail-closed
    # verifier is only useful if it closes on genuine fabrication.
    citation_spans += [(m.start(), m.end()) for m in _URL_RE.finditer(text)]

    # Fixed label text. "30-day readmission" and "deterioration within 6 hours" name
    # a task; their numerals define *which outcome is being reported*, not a value
    # claimed about the patient. They are constants of the system, identical in every
    # report, and cannot be fabricated.
    #
    # This was latent rather than harmless: the risk table is the only place these
    # labels appear, and predictions had been withheld from every report by the
    # payload coverage floor. The moment a report actually displayed its predictions,
    # the verifier rejected it for the "30" in its own column heading.
    citation_spans += [(m.start(), m.end()) for m in _LABEL_RE.finditer(text)]

    def _in_citation(pos: int) -> bool:
        return any(a <= pos < b for a, b in citation_spans)

    # 1. every number must trace to an input
    for m in _NUMBER_RE.finditer(text):
        if _in_citation(m.start()):
            continue
        raw = m.group(1)
        try:
            val = float(raw)
        except ValueError:
            continue
        report.n_numbers_checked += 1
        unit = (m.group(2) or "").strip().lower().rstrip(".")
        if allow_trivial_numbers and val in _TRIVIAL_NUMBERS and (
            not unit or unit in _TRIVIAL_TIME_UNITS
        ):
            continue
        # a year-like integer inside a citation is bibliographic, not clinical
        if 1900 <= val <= 2100 and float(int(val)) == val:
            continue
        if fact_store.knows_number(val) is None:
            start = max(0, m.start() - 35)
            report.violations.append(Violation(
                kind="ungrounded_number", severity="critical",
                detail=f"value {raw} does not appear in the payload, predictions or evidence",
                span=text[start:m.end() + 25].replace("\n", " "),
            ))

    # 2. every citation must correspond to a retrieved document
    for m in _CITATION_RE.finditer(text):
        cit = m.group(1).strip().lower()
        report.n_citations_checked += 1
        if cit in ("no evidence", "none", "system safety refusal"):
            continue
        known = (cit in fact_store.citations
                 or any(cit in c or c in cit for c in fact_store.citations))
        if not known:
            report.violations.append(Violation(
                kind="unknown_citation", severity="critical",
                detail=f"citation [{m.group(1)}] was not among the retrieved documents",
                span=m.group(0),
            ))

    # 3. computation claims the system may not have performed.
    # Verbatim quotations of the retrieved evidence are the source speaking, not
    # the system, and are exempt — see `_verbatim_quote_spans`.
    quoted = _verbatim_quote_spans(text, fact_store)

    for pattern, kind, desc in _CLAIM_PATTERNS:
        for m in pattern.finditer(text):
            if any(a <= m.start() < b for a, b in quoted):
                continue
            if kind == "shap_claim" and any("shap" in e for e in fact_store.entities):
                continue
            severity = "critical" if kind in ("shap_claim", "statistical_claim") else "high"
            if kind == "directive_without_citation":
                if not require_citation_for_directives:
                    continue
                sent = next((s for s in _sentences(text) if m.group(0) in s), "")
                if _CITATION_RE.search(sent):
                    continue
                severity = "high"
            report.violations.append(Violation(
                kind=kind, severity=severity,
                detail=f"{desc} that is not supported by a retrieved document",
                span=m.group(0),
            ))

    # 4. drug names must come from the patient's list or the evidence.
    # Matched against the real drug lexicon rather than a morphological pattern —
    # a suffix rule flagged "Creatinine" and "Guideline" as medications.
    low_text = text.lower()
    for drug in _drug_lexicon():
        if not re.search(r"\b" + re.escape(drug) + r"\b", low_text):
            continue
        if drug in fact_store.entities or drug in fact_store.text_corpus:
            continue
        if any(drug in e for e in fact_store.entities):
            continue
        report.violations.append(Violation(
            kind="ungrounded_medication", severity="high",
            detail=f"'{drug}' is named as a medication but is neither on the patient's "
                   "active list nor mentioned in any retrieved document",
            span=drug,
        ))

    report.ok = not any(v.severity in ("critical", "high") for v in report.violations)
    return report


def verify_rephrase(candidate: str, source: str,
                    fact_store: FactStore, **kwargs: Any) -> VerificationReport:
    """
    Verify a rephrasing against both the fact store and the text it rephrased.

    ``verify_text`` only ever inspects tokens it finds. Everything it can catch
    is therefore something the generator *added*; a generator that removes
    something is invisible to it, and reports success on a document it did not
    check. The two are not equally likely failures — an instructed rephraser
    drops far more readily than it invents.

    Two ways that matters here. A figure written out in words is not a mismatch,
    it is not a number: "0.34" rephrased as "roughly a third" leaves nothing for
    ``_NUMBER_RE`` to match, and the report passes with ``n_numbers_checked``
    silently lower. And a dropped citation detaches a recommendation from the
    guideline that carries it while every remaining citation still verifies.

    So the counts are compared rather than the strings. Any rephrasing that
    arrives carrying fewer checkable numbers or fewer citations than it was
    given has changed the document's substance, whatever it did to the prose,
    and is rejected. This is coarse on purpose: it needs no lexicon of number
    words and no notion of equivalent phrasing, and it closes the whole class
    including the members not yet thought of.
    """
    report = verify_text(candidate, fact_store, **kwargs)
    baseline = verify_text(source, fact_store, **kwargs)

    for label, got, had in (("number", report.n_numbers_checked,
                             baseline.n_numbers_checked),
                            ("citation", report.n_citations_checked,
                             baseline.n_citations_checked)):
        if got < had:
            report.violations.append(Violation(
                kind=f"{label}s_dropped", severity="critical",
                detail=f"the rephrasing states {had - got} fewer {label}(s) than "
                       f"the report it was given ({had} → {got}); a value written "
                       "out in words or a citation removed is not verifiable",
            ))
            report.ok = False
    return report
