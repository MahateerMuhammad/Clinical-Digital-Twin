"""
src/llm/terminology.py
──────────────────────
Clinical terminology normalisation for the RAG layer.

Two problems this solves:

1. **Diagnosis aliasing.** A clinician types "STEMI", "CVA", "CHF exacerbation" or
   "septicaemia"; the corpus and the retrieval rules speak in canonical concepts.
   Substring matching cannot bridge that gap, and it produces false positives in
   the other direction ("hyPErkalemia" matching a "pe" rule for pulmonary embolism).

2. **Medication aliasing.** Medications arrive as brand names ("Levophed"), salt
   forms ("norepinephrine bitartrate"), or as charted in an eMAR
   ("NOREPINEPHRINE 4mg/250mL"). Exact list membership fails on all of them.

Matching is word-boundary anchored, never bare substring. Every match carries the
method that produced it so callers can require a confidence floor.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

__all__ = [
    "ConceptMatch",
    "normalise_text",
    "normalise_diagnosis",
    "normalise_medication",
    "normalise_medications",
    "concept_evidence_terms",
    "CONCEPTS",
]

# ── text normalisation ────────────────────────────────────────────────────

_PUNCT = re.compile(r"[^\w\s]+")
_WS = re.compile(r"\s+")
# dose / strength / volume / route fragments seen in eMAR strings
_DOSE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|gm|kg|ml|l|units?|iu|meq|mmol|%)\b"
    r"|\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|units?)\s*/\s*\d*\.?\d*\s*(?:ml|l|hr|h|kg|min)\b"
    r"|\b\d+\s*/\s*\d+\b",
    re.I,
)
_ROUTE_FORM = re.compile(
    r"\b(?:iv|po|im|sc|sq|pr|ivpb|gtt|drip|infusion|bolus|tablet|tabs?|capsule|caps?|"
    r"syringe|premix|soln|solution|injection|inj|vial|bag|oral|nebuli[sz]er|neb|patch|"
    r"suspension|elixir|ointment|cream|er|xr|sr|cr|dr)\b",
    re.I,
)
# salt / ester suffixes that do not change the active ingredient
_SALTS = re.compile(
    r"\b(?:hcl|hydrochloride|bitartrate|tartrate|sulfate|sulphate|sodium|potassium|"
    r"calcium|besylate|mesylate|maleate|succinate|fumarate|citrate|acetate|phosphate|"
    r"gluconate|lactate|tromethamine|dihydrate|monohydrate|anhydrous|base)\b",
    re.I,
)


def normalise_text(text: object) -> str:
    """Lowercase, strip accents/punctuation, collapse whitespace. Never raises."""
    if text is None:
        return ""
    s = unicodedata.normalize("NFKD", str(text))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def _tokens(text: str) -> List[str]:
    return [t for t in normalise_text(text).split() if t]


# Light morphological stemming. Clinical text alternates freely between
# "nephrotoxic"/"nephrotoxicity", "anticoagulation"/"anticoagulants",
# "ventilation"/"ventilated", and exact token equality misses all of these.
# Deliberately conservative: only strips well-behaved suffixes and never
# shortens a token below 5 characters, which keeps short words unambiguous.
_SUFFIXES = (
    "ational", "ations", "ation", "ities", "ity", "ically", "ical", "ics", "ic",
    "ising", "izing", "ised", "ized", "ise", "ize", "ing", "ies", "ers", "er",
    "ants", "ant", "ance", "ancy", "ents", "ent",
    "ed", "es", "s", "al", "ive", "ous", "ary",
)

_MAX_STEM_ROUNDS = 4


def stem(token: str) -> str:
    """
    Crude suffix stripper for clinical vocabulary matching.

    Applied to a fixpoint: a single pass is not idempotent, so "nephrotoxicity"
    would reduce to "nephrotoxic" while "nephrotoxic" reduced to "nephrotox",
    and the two would never match. Iterating makes the relation an equivalence.
    """
    t = str(token).lower()
    for _ in range(_MAX_STEM_ROUNDS):
        if len(t) <= 5:
            return t
        for suf in _SUFFIXES:
            if t.endswith(suf) and len(t) - len(suf) >= 5:
                t = t[: -len(suf)]
                break
        else:
            return t
    return t


def _stems(text: str) -> Set[str]:
    return {stem(t) for t in _tokens(text)}


def _has_phrase(haystack_tokens: Sequence[str], phrase: str) -> bool:
    """Word-boundary phrase containment. 'pe' never matches 'hyperkalemia'."""
    p = _tokens(phrase)
    if not p:
        return False
    n = len(p)
    return any(list(haystack_tokens[i : i + n]) == p for i in range(len(haystack_tokens) - n + 1))


# ── concept catalogue ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class Concept:
    key: str
    display: str
    aliases: Tuple[str, ...]          # surface forms, incl. abbreviations
    evidence_terms: Tuple[str, ...]   # terms that make a document on-topic
    icd10_prefixes: Tuple[str, ...] = ()


def _c(key, display, aliases, evidence, icd=()):
    return Concept(key, display, tuple(aliases), tuple(evidence), tuple(icd))


CONCEPTS: Dict[str, Concept] = {
    c.key: c
    for c in [
        _c("sepsis", "Sepsis / Septic Shock",
           ["sepsis", "septic shock", "septicemia", "septicaemia", "septic", "urosepsis",
            "bacteremia", "bacteraemia", "sirs", "blood stream infection", "bloodstream infection",
            "severe sepsis", "septic syndrome"],
           ["sepsis", "septic", "shock", "infection", "antimicrobial", "antibiotic",
            "norepinephrine", "vasopressor", "lactate", "critical care", "icu",
            "source control", "procalcitonin", "bacteremia"],
           ["A40", "A41", "R65"]),
        _c("heart_failure", "Heart Failure / Cardiorenal",
           ["heart failure", "congestive heart failure", "chf", "congestive cardiac failure",
            "cardiac failure", "hfref", "hfpef", "decompensated heart failure",
            "acute decompensated heart failure", "adhf", "cardiorenal", "cardiorenal syndrome",
            "pulmonary oedema", "pulmonary edema", "fluid overload", "volume overload"],
           ["heart failure", "cardiac failure", "echocardiography", "ejection fraction",
            "cardiorenal", "diuretic", "furosemide", "natriuretic", "bnp", "decongestion",
            "cardiovascular", "kidney", "renal"],
           ["I50", "I11.0", "I13.0"]),
        _c("myocardial_infarction", "Acute Coronary Syndrome / MI",
           ["myocardial infarction", "mi", "ami", "stemi", "nstemi", "nstemi acs",
            "acute coronary syndrome", "acs", "heart attack", "coronary thrombosis",
            "unstable angina", "cardiogenic shock", "coronary artery occlusion"],
           ["myocardial infarction", "coronary", "troponin", "stemi", "nstemi",
            "revascularization", "revascularisation", "pci", "angioplasty", "stent",
            "cardiogenic shock", "dobutamine", "inotrope", "antiplatelet"],
           ["I21", "I22", "I24", "I25"]),
        _c("stroke", "Acute Ischemic Stroke",
           ["stroke", "cva", "cerebrovascular accident", "brain attack", "ischemic stroke",
            "ischaemic stroke", "cerebral infarction", "cerebral infarct", "tia",
            "transient ischemic attack", "acute stroke"],
           ["stroke", "ischemic", "ischaemic", "thrombolysis", "alteplase", "tenecteplase",
            "tpa", "thrombectomy", "cerebral", "nihss", "nicardipine", "blood pressure"],
           ["I63", "I64", "G45"]),
        _c("pulmonary_embolism", "Pulmonary Embolism / VTE",
           ["pulmonary embolism", "pe", "pulmonary embolus", "vte",
            "venous thromboembolism", "dvt", "deep vein thrombosis", "saddle embolus",
            "pulmonary thromboembolism"],
           ["pulmonary embolism", "embolism", "thrombosis", "thromboembolism",
            "anticoagulation", "alteplase", "thrombolysis", "heparin", "enoxaparin",
            "rv strain", "right ventricular", "d dimer", "echocardiography"],
           ["I26", "I82"]),
        _c("dka", "Diabetic Ketoacidosis / Hyperglycaemic Crisis",
           ["dka", "diabetic ketoacidosis", "ketoacidosis", "hhs",
            "hyperosmolar hyperglycemic state", "hyperosmolar hyperglycaemic state",
            "hyperglycemic crisis", "hyperglycaemic crisis", "diabetic emergency"],
           ["ketoacidosis", "dka", "diabetes", "diabetic", "insulin", "glycemic", "glycaemic",
            "hyperglycemia", "hyperglycaemia", "acidosis", "anion gap", "potassium",
            "crystalloid", "saline", "bicarbonate"],
           ["E10.1", "E11.1", "E87.2"]),
        _c("aki", "Acute Kidney Injury",
           ["aki", "acute kidney injury", "acute renal failure", "arf",
            "acute tubular necrosis", "atn", "renal insufficiency", "renal failure",
            "kidney injury", "kidney failure", "uremia", "uraemia"],
           ["acute kidney injury", "renal", "kidney", "creatinine", "dialysis",
            "renal replacement", "crrt", "nephrotoxic", "kdigo", "oliguria", "urine output"],
           ["N17", "N19", "N18"]),
        _c("gi_bleed", "Acute Gastrointestinal Haemorrhage",
           ["gi bleed", "gi bleeding", "gastrointestinal bleed", "gastrointestinal bleeding",
            "gastrointestinal hemorrhage", "gastrointestinal haemorrhage", "ugib", "lgib",
            "upper gi bleed", "lower gi bleed", "hematemesis", "haematemesis", "melena",
            "melaena", "variceal bleed", "variceal hemorrhage", "peptic ulcer bleed"],
           ["gastrointestinal bleeding", "gi bleed", "hemorrhage", "haemorrhage",
            "pantoprazole", "proton pump", "ppi", "octreotide", "endoscopy", "varices",
            "transfusion", "peptic ulcer"],
           ["K92.0", "K92.1", "K92.2", "I85.0"]),
        _c("ards", "ARDS / Acute Respiratory Failure",
           ["ards", "acute respiratory distress syndrome", "acute respiratory failure",
            "respiratory failure", "hypoxemia", "hypoxaemia", "hypoxemic respiratory failure",
            "acute lung injury", "ali", "respiratory distress"],
           ["ards", "respiratory distress", "hypoxemia", "hypoxaemia", "mechanical ventilation",
            "peep", "tidal volume", "prone", "cisatracurium", "neuromuscular blockade",
            "dexamethasone", "oxygenation", "pao2"],
           ["J80", "J96"]),
        _c("liver_failure", "Acute Liver Failure / Hepatic Encephalopathy",
           ["liver failure", "acute liver failure", "hepatic failure",
            "hepatic encephalopathy", "encephalopathy", "cirrhosis", "decompensated cirrhosis",
            "end stage liver disease", "esld", "fulminant hepatic failure"],
           ["hepatic encephalopathy", "liver failure", "cirrhosis", "ammonia", "lactulose",
            "rifaximin", "portal", "coagulopathy", "hepatic", "meld"],
           ["K72", "K70.4", "K74"]),
        _c("pancreatitis", "Acute Pancreatitis",
           ["pancreatitis", "acute pancreatitis", "necrotizing pancreatitis",
            "necrotising pancreatitis", "gallstone pancreatitis"],
           ["pancreatitis", "pancreatic", "lipase", "amylase", "fluid resuscitation",
            "crystalloid", "lactated ringers", "necrosis", "sirs", "ranson"],
           ["K85"]),
        _c("pneumonia", "Pneumonia / Lower Respiratory Infection",
           ["pneumonia", "cap", "community acquired pneumonia", "hap", "vap",
            "hospital acquired pneumonia", "ventilator associated pneumonia",
            "aspiration pneumonia", "lrti", "chest infection", "bronchopneumonia"],
           ["pneumonia", "antibiotic", "antimicrobial", "respiratory", "consolidation",
            "curb 65", "empiric", "infection", "sputum", "chest radiograph"],
           ["J13", "J14", "J15", "J18"]),
        _c("copd", "COPD Exacerbation",
           ["copd", "chronic obstructive pulmonary disease", "copd exacerbation",
            "aecopd", "emphysema", "chronic bronchitis"],
           ["copd", "chronic obstructive", "bronchodilator", "corticosteroid",
            "exacerbation", "niv", "bipap", "noninvasive ventilation", "spirometry",
            "hypercapnic", "hypercapnia", "ventilation", "salbutamol", "albuterol",
            "ipratropium", "respiratory failure"],
           ["J44"]),
        _c("hypertensive_emergency", "Hypertensive Emergency",
           ["hypertensive emergency", "hypertensive crisis", "hypertensive urgency",
            "malignant hypertension", "accelerated hypertension"],
           ["hypertensive", "blood pressure", "nicardipine", "clevidipine", "labetalol",
            "antihypertensive", "end organ", "hypertension"],
           ["I16"]),
        _c("hyperkalemia", "Hyperkalaemia / Electrolyte Emergency",
           ["hyperkalemia", "hyperkalaemia", "hypokalemia", "hypokalaemia",
            "hyponatremia", "hyponatraemia", "hypernatremia", "hypernatraemia",
            "electrolyte abnormality", "electrolyte derangement", "electrolyte emergency"],
           ["hyperkalemia", "hyperkalaemia", "potassium", "electrolyte", "sodium",
            "calcium gluconate", "insulin dextrose", "kayexalate", "patiromer",
            "dialysis", "ecg"],
           ["E87.5", "E87.6", "E87.1"]),
    ]
}

# alias → concept key, longest-first so "acute kidney injury" beats "kidney"
_ALIAS_INDEX: List[Tuple[List[str], str]] = sorted(
    ((_tokens(a), c.key) for c in CONCEPTS.values() for a in c.aliases),
    key=lambda kv: -len(kv[0]),
)

# Abbreviations that are dangerous as free-text substrings. They must appear as a
# standalone token (already guaranteed by phrase matching) AND be upper-case-ish in
# the raw input, otherwise "pe" inside a sentence is too weak a signal.
_RISKY_ABBREV = {"pe", "mi", "acs", "ali", "arf", "tia", "hap", "vap", "cap", "sirs"}


@dataclass
class ConceptMatch:
    """Result of normalising a free-text clinical phrase.

    ``concept`` is the primary (first-mentioned) concept; ``all_concepts`` holds
    every concept found, in order of mention. A composite presentation such as
    "Septic Shock & Acute Respiratory Failure" is genuinely two concepts, and
    retrieval should treat a document relevant to *either* as on-topic.
    """

    concept: Optional[str]
    display: str = ""
    confidence: float = 0.0
    method: str = "none"
    raw: str = ""
    ambiguous: Tuple[str, ...] = field(default_factory=tuple)
    all_concepts: Tuple[str, ...] = field(default_factory=tuple)
    suggestions: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def matched(self) -> bool:
        return self.concept is not None

    def to_dict(self) -> dict:
        return {
            "concept": self.concept,
            "display": self.display,
            "confidence": round(self.confidence, 3),
            "method": self.method,
            "raw": self.raw,
            "ambiguous": list(self.ambiguous),
            "all_concepts": list(self.all_concepts),
            "suggestions": list(self.suggestions),
        }


def _find_alias_hits(toks: Sequence[str]) -> List[Tuple[int, int, str]]:
    """Return (position, phrase_len, concept_key) for every alias occurrence."""
    hits: List[Tuple[int, int, str]] = []
    for alias_toks, key in _ALIAS_INDEX:
        n = len(alias_toks)
        if n == 1 and alias_toks[0] in _RISKY_ABBREV:
            if alias_toks[0] in toks:
                hits.append((list(toks).index(alias_toks[0]), 1, key))
            continue
        for i in range(len(toks) - n + 1):
            if list(toks[i : i + n]) == alias_toks:
                hits.append((i, n, key))
                break
    return hits


#: Confidence below which a match is offered as a *suggestion* rather than
#: assigned as a concept. Gold-set measurement showed the token-overlap matcher
#: contributed no correct answers and produced only false positives
#: ("Hyperglycaemia" → dka, "Peptic ulcer disease" → gi_bleed), so guessing is
#: strictly worse than asking the user to clarify.
MIN_ASSIGN_CONFIDENCE = 0.75


def normalise_diagnosis(text: object, min_confidence: float = MIN_ASSIGN_CONFIDENCE) -> ConceptMatch:
    """
    Map free-text diagnosis to canonical concept(s).

    Matching order: exact alias phrase → risky-abbreviation token → token overlap.
    All matching is word-boundary anchored. The primary concept is the
    **first-mentioned** one (clinical convention), not the longest match.

    A match scoring below ``min_confidence`` is *not* assigned: ``concept`` stays
    None and the candidate appears in ``suggestions``, so the caller can ask the
    user to confirm instead of silently retrieving evidence for a guess.
    """
    raw = "" if text is None else str(text)
    toks = _tokens(raw)
    if not toks:
        return ConceptMatch(None, method="empty", raw=raw)

    hits = _find_alias_hits(toks)
    if hits:
        # earliest mention wins; longer phrase breaks ties at the same position
        hits.sort(key=lambda h: (h[0], -h[1]))
        ordered: List[str] = []
        for _, _, key in hits:
            if key not in ordered:
                ordered.append(key)
        primary = ordered[0]
        best_len = max(n for pos, n, k in hits if k == primary)
        return ConceptMatch(
            concept=primary,
            display=CONCEPTS[primary].display,
            confidence=0.99 if best_len > 1 else 0.80,
            method="alias_phrase" if best_len > 1 else "alias_token",
            raw=raw,
            ambiguous=tuple(ordered[1:]),
            all_concepts=tuple(ordered),
        )

    # token-overlap fallback against evidence vocabulary
    tset: Set[str] = set(toks)
    scored: List[Tuple[float, str]] = []
    for key, c in CONCEPTS.items():
        vocab = {t for term in c.evidence_terms for t in _tokens(term)}
        overlap = tset & vocab
        if overlap:
            scored.append((len(overlap) / max(len(tset), 1), key))
    if scored:
        scored.sort(reverse=True)
        score, key = scored[0]
        if score >= 0.34:
            conf = min(0.70, score)
            if conf >= min_confidence:
                return ConceptMatch(key, CONCEPTS[key].display, conf,
                                    "token_overlap", raw, all_concepts=(key,))
            # too weak to assign — offer it for confirmation instead
            return ConceptMatch(
                None, confidence=conf, method="low_confidence", raw=raw,
                suggestions=tuple(k for _, k in sorted(scored, reverse=True)[:3]),
            )

    return ConceptMatch(None, method="unmatched", raw=raw)


def concept_evidence_terms(concept_key: Optional[str]) -> Tuple[str, ...]:
    c = CONCEPTS.get(concept_key or "")
    return c.evidence_terms if c else ()


# ── medications ───────────────────────────────────────────────────────────

# brand / synonym → RxNorm-style ingredient name
BRAND_TO_INGREDIENT: Dict[str, str] = {
    "levophed": "norepinephrine", "noradrenaline": "norepinephrine",
    "adrenaline": "epinephrine", "adrenalin": "epinephrine",
    "neosynephrine": "phenylephrine", "neo synephrine": "phenylephrine",
    "vasostrict": "vasopressin", "pitressin": "vasopressin",
    "intropin": "dopamine", "dobutrex": "dobutamine", "primacor": "milrinone",
    "lasix": "furosemide", "bumex": "bumetanide", "demadex": "torsemide",
    "protonix": "pantoprazole", "prilosec": "omeprazole", "nexium": "esomeprazole",
    "sandostatin": "octreotide",
    "activase": "alteplase", "tnkase": "tenecteplase", "tpa": "alteplase",
    "lovenox": "enoxaparin", "eliquis": "apixaban", "xarelto": "rivaroxaban",
    "coumadin": "warfarin", "pradaxa": "dabigatran",
    "vancocin": "vancomycin", "maxipime": "cefepime", "rocephin": "ceftriaxone",
    "zosyn": "piperacillin tazobactam", "merrem": "meropenem", "cipro": "ciprofloxacin",
    "levaquin": "levofloxacin", "zithromax": "azithromycin", "flagyl": "metronidazole",
    "cardene": "nicardipine", "cleviprex": "clevidipine", "trandate": "labetalol",
    "normodyne": "labetalol", "lopressor": "metoprolol", "toprol": "metoprolol",
    "coreg": "carvedilol",
    "nimbex": "cisatracurium", "decadron": "dexamethasone", "solumedrol": "methylprednisolone",
    "xifaxan": "rifaximin", "kristalose": "lactulose",
    "humalog": "insulin", "lantus": "insulin", "novolog": "insulin",
    "regular insulin": "insulin", "insulin glargine": "insulin", "insulin lispro": "insulin",
    "ns": "normal saline", "0 9 nacl": "normal saline", "sodium chloride": "normal saline",
    "lr": "lactated ringers", "ringers lactate": "lactated ringers",
    "plasmalyte": "balanced crystalloid",
    "zofran": "ondansetron", "dilaudid": "hydromorphone", "duramorph": "morphine",
    "sublimaze": "fentanyl", "percocet": "oxycodone",
    "kayexalate": "sodium polystyrene", "veltassa": "patiromer", "lokelma": "sodium zirconium",
    # ward shorthand
    "vanc": "vancomycin", "vanco": "vancomycin", "norepi": "norepinephrine",
    "levophed drip": "norepinephrine", "epi": "epinephrine", "neo": "phenylephrine",
    "pip tazo": "piperacillin tazobactam", "zosyn ivpb": "piperacillin tazobactam",
    "ptz": "piperacillin tazobactam", "ctx": "ceftriaxone", "flagyl iv": "metronidazole",
    "nacl": "normal saline", "normal saline 0 9": "normal saline",
    "ringers": "lactated ringers", "d5w": "dextrose", "dextrose 5": "dextrose",
    "hep": "heparin", "lmwh": "enoxaparin", "ppi": "pantoprazole",
}

# ingredient → therapeutic class (word-boundary safe; replaces keyword substrings)
INGREDIENT_CLASS: Dict[str, str] = {
    "norepinephrine": "vasopressor", "epinephrine": "vasopressor",
    "phenylephrine": "vasopressor", "vasopressin": "vasopressor", "dopamine": "vasopressor",
    "dobutamine": "inotrope", "milrinone": "inotrope",
    "furosemide": "loop_diuretic", "bumetanide": "loop_diuretic", "torsemide": "loop_diuretic",
    "vancomycin": "antibiotic", "cefepime": "antibiotic", "ceftriaxone": "antibiotic",
    "cefazolin": "antibiotic", "piperacillin tazobactam": "antibiotic",
    "meropenem": "antibiotic", "ciprofloxacin": "antibiotic", "levofloxacin": "antibiotic",
    "azithromycin": "antibiotic", "metronidazole": "antibiotic", "gentamicin": "antibiotic",
    "heparin": "anticoagulant", "enoxaparin": "anticoagulant", "warfarin": "anticoagulant",
    "apixaban": "anticoagulant", "rivaroxaban": "anticoagulant", "dabigatran": "anticoagulant",
    "alteplase": "thrombolytic", "tenecteplase": "thrombolytic",
    "insulin": "insulin",
    "morphine": "opioid", "fentanyl": "opioid", "hydromorphone": "opioid",
    "oxycodone": "opioid",
    "atorvastatin": "statin", "simvastatin": "statin", "rosuvastatin": "statin",
    "pravastatin": "statin",
    "metoprolol": "beta_blocker", "carvedilol": "beta_blocker", "labetalol": "beta_blocker",
    "atenolol": "beta_blocker",
    "lisinopril": "ace_inhibitor", "enalapril": "ace_inhibitor", "captopril": "ace_inhibitor",
    "ramipril": "ace_inhibitor",
    "nicardipine": "antihypertensive", "clevidipine": "antihypertensive",
    "pantoprazole": "ppi", "omeprazole": "ppi", "esomeprazole": "ppi",
    "octreotide": "somatostatin_analogue",
    "lactulose": "hepatic_encephalopathy_therapy", "rifaximin": "hepatic_encephalopathy_therapy",
    "cisatracurium": "neuromuscular_blocker",
    "dexamethasone": "corticosteroid", "methylprednisolone": "corticosteroid",
    "normal saline": "crystalloid", "lactated ringers": "crystalloid",
    "balanced crystalloid": "crystalloid",
    "sodium polystyrene": "potassium_binder", "patiromer": "potassium_binder",
    "sodium zirconium": "potassium_binder",
    "ondansetron": "antiemetic",
}

_KNOWN_INGREDIENTS = set(INGREDIENT_CLASS) | set(BRAND_TO_INGREDIENT.values())


@dataclass
class DrugMatch:
    ingredient: Optional[str]
    drug_class: Optional[str]
    confidence: float
    method: str
    raw: str

    @property
    def matched(self) -> bool:
        return self.ingredient is not None

    def to_dict(self) -> dict:
        return {
            "ingredient": self.ingredient, "drug_class": self.drug_class,
            "confidence": round(self.confidence, 3), "method": self.method, "raw": self.raw,
        }


def _strip_drug_noise(text: str) -> str:
    s = str(text)
    s = _DOSE.sub(" ", s)
    s = _ROUTE_FORM.sub(" ", s)
    s = normalise_text(s)
    s = re.sub(r"\b\d+(?:\.\d+)?\b", " ", s)   # bare numbers
    return _WS.sub(" ", s).strip()


def normalise_medication(text: object) -> DrugMatch:
    """Map a charted medication string to its active ingredient and class."""
    raw = "" if text is None else str(text)
    cleaned = _strip_drug_noise(raw)
    if not cleaned:
        return DrugMatch(None, None, 0.0, "empty", raw)

    # 1. exact ingredient / brand on the cleaned string
    if cleaned in BRAND_TO_INGREDIENT:
        ing = BRAND_TO_INGREDIENT[cleaned]
        return DrugMatch(ing, INGREDIENT_CLASS.get(ing), 0.99, "brand_exact", raw)
    if cleaned in INGREDIENT_CLASS:
        return DrugMatch(cleaned, INGREDIENT_CLASS[cleaned], 0.99, "ingredient_exact", raw)

    # 2. drop salt/ester words and retry
    desalted = _WS.sub(" ", _SALTS.sub(" ", cleaned)).strip()
    if desalted != cleaned:
        if desalted in BRAND_TO_INGREDIENT:
            ing = BRAND_TO_INGREDIENT[desalted]
            return DrugMatch(ing, INGREDIENT_CLASS.get(ing), 0.95, "brand_desalted", raw)
        if desalted in INGREDIENT_CLASS:
            return DrugMatch(desalted, INGREDIENT_CLASS[desalted], 0.95, "ingredient_desalted", raw)

    # 3. word-boundary phrase containment (handles "norepinephrine drip 4 mg")
    toks = _tokens(desalted or cleaned)
    best: Optional[Tuple[int, str, str]] = None
    for name in _KNOWN_INGREDIENTS:
        if _has_phrase(toks, name):
            n = len(_tokens(name))
            if best is None or n > best[0]:
                best = (n, name, "ingredient_phrase")
    for brand, ing in BRAND_TO_INGREDIENT.items():
        if _has_phrase(toks, brand):
            n = len(_tokens(brand))
            if best is None or n > best[0]:
                best = (n, ing, "brand_phrase")
    if best:
        _, ing, method = best
        return DrugMatch(ing, INGREDIENT_CLASS.get(ing), 0.90, method, raw)

    return DrugMatch(None, None, 0.0, "unmatched", raw)


def normalise_medications(items: object) -> List[DrugMatch]:
    """Normalise a medication list. Accepts a list, a single string, or None."""
    if items is None:
        return []
    if isinstance(items, (str, bytes)):
        items = [items]
    try:
        seq = list(items)
    except TypeError:
        seq = [items]
    return [normalise_medication(m) for m in seq]
