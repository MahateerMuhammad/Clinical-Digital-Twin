"""
src/assistant/evidence.py
─────────────────────────
Retrieval of trusted, citable medical information.  Spec 11, 12, 33.2, 33.3.

The contract is narrow and it is the point of the module: **an answer may only
contain what came back from here.** If retrieval returns nothing, the answer
stage declines. It does not fall back to the language model's own knowledge,
because a model's recollection of a guideline carries no citation, no version,
and no way to check it — and a fluent paragraph with an invented reference is
indistinguishable from a real one to the person reading it.

So ``retrieve`` has an explicit ``no_source`` status, and it is a normal outcome
rather than an error. The corpus starting empty is what makes that path real
instead of theoretical.

Source hierarchy (spec 12)
──────────────────────────
Every document carries a ``source_tier``, 1–6, and results are ranked by tier
before relevance. A patient-information page from a national health service
outranks a journal abstract for this audience — not because the abstract is
worse evidence, but because it is not written for the person asking.

Trust is enforced at load, not at answer time: a document whose URL is outside
``trusted_domains`` is rejected when the corpus is read, so it can never reach a
ranking function.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote, urlparse

import yaml

__all__ = [
    "EvidenceDoc", "EvidenceResult", "retrieve", "load_corpus",
    "OK", "NO_SOURCE", "TIER_NAMES", "PATIENT_CORPUS_PATH",
    "PATIENT_CORPUS", "GUIDELINES", "DRUG_LABELS", "DEFAULT_SOURCES",
]

#: Selectable corpora. `patient` is the plain-language file that ships empty;
#: `guidelines` is the curated society corpus in `src/llm/guidelines.py`, which
#: is written for clinicians and is the right source for that audience;
#: `drug_labels` is the manufacturer's package insert, fetched from OpenFDA.
#:
#: The first two are indexed by *condition*. A question about a drug rather than
#: about a disease — "what are the contraindications for vancomycin?" — matched
#: nothing in either, and with no drug-evidence path available it fell through
#: to `drug_dosing`, the only intent that knows what a drug name is, which
#: demanded a creatinine before it would discuss a package insert.
PATIENT_CORPUS = "patient"
GUIDELINES = "guidelines"
DRUG_LABELS = "drug_labels"
DEFAULT_SOURCES = (PATIENT_CORPUS,)

OK = "ok"
NO_SOURCE = "no_source"

#: Spec 12's hierarchy, most authoritative first.
TIER_NAMES: Dict[int, str] = {
    1: "Official clinical guideline",
    2: "Government or public-health organisation",
    3: "Major medical institution",
    4: "Peer-reviewed medical literature",
    5: "Approved medical database",
    6: "General medical reference",
}

PATIENT_CORPUS_PATH = Path(__file__).resolve().parent / "config" / "patient_corpus.yaml"

_CACHE: Dict[str, dict] = {}


class CorpusError(ValueError):
    """The evidence corpus is malformed, or cites an untrusted source."""


@dataclass(frozen=True)
class EvidenceDoc:
    """One retrievable document, in the shape the answer stage cites."""

    doc_id: str
    title: str
    text: str
    source_name: str
    source_tier: int
    url: str
    topics: Tuple[str, ...] = ()
    retrieved_on: str = ""
    review_status: str = "unreviewed"
    verbatim: bool = False
    keywords: Tuple[str, ...] = ()
    #: A citation supplied by the source rather than constructed here. The
    #: society corpus already formats its own — "[KDIGO Clinical Practice
    #: Guideline for AKI 2012, Section 3.9: Drug Dosing]" — and rebuilding one
    #: from source_name plus title printed the document name twice.
    citation_text: str = ""

    @property
    def citation(self) -> str:
        return self.citation_text or f"[{self.source_name}: {self.title}]"

    @property
    def tier_name(self) -> str:
        return TIER_NAMES.get(self.source_tier, "Unclassified source")

    def to_doc(self) -> dict:
        """The shape ``src.llm.grounding.build_fact_store`` consumes.

        Reusing that verifier is the reason this mapping exists: the citation
        and text registered here are exactly what a generated answer is later
        checked against, so a citation the answer invents cannot match.
        """
        return {"doc_id": self.doc_id, "citation": self.citation,
                "title": self.title, "text": self.text,
                "category": self.tier_name, "url": self.url,
                "evidence_level": f"Tier {self.source_tier}: {self.tier_name}",
                "review_status": self.review_status}

    def to_dict(self) -> dict:
        d = self.to_doc()
        d["source_name"] = self.source_name
        d["source_tier"] = self.source_tier
        d["retrieved_on"] = self.retrieved_on
        return d


@dataclass
class EvidenceResult:
    """What retrieval found, and — when it found nothing — what it looked for."""

    status: str = NO_SOURCE
    documents: List[EvidenceDoc] = field(default_factory=list)
    searched_terms: List[str] = field(default_factory=list)
    #: Set when documents exist but all were filtered out by a review
    #: requirement, which is a different state from "nothing on file".
    filtered_unreviewed: int = 0

    @property
    def ok(self) -> bool:
        return self.status == OK and bool(self.documents)

    def refusal_text(self) -> str:
        """What to tell the patient when there is no source to answer from."""
        if self.filtered_unreviewed:
            return ("I have information on this, but none of it has been "
                    "reviewed by a clinician yet, so I am not going to give it "
                    "to you as guidance. Please ask a doctor or pharmacist.")
        return ("I do not have a trusted source on file that covers this, and I "
                "am not going to answer from memory. I could sound confident "
                "and be wrong. A doctor, pharmacist or your national health "
                "service website would be a better place to ask.")

    def to_dict(self) -> dict:
        return {"status": self.status,
                "documents": [d.to_dict() for d in self.documents],
                "searched_terms": self.searched_terms,
                "filtered_unreviewed": self.filtered_unreviewed}


# ── loading ──────────────────────────────────────────────────────────────────

def _domain_ok(url: str, trusted: Sequence[str]) -> bool:
    host = (urlparse(str(url)).hostname or "").lower()
    if not host:
        return False
    return any(host == d or host.endswith("." + d) for d in trusted)


def load_corpus(path: Optional[Path] = None, *, refresh: bool = False) -> dict:
    """
    Load and validate the patient corpus.

    An empty ``documents:`` list is valid and is the shipped state. A malformed
    or untrusted entry is not: it raises, because a corpus that silently drops
    what it cannot validate is a corpus whose coverage nobody can state.
    """
    p = Path(path) if path else PATIENT_CORPUS_PATH
    key = str(p)
    if refresh:
        _CACHE.pop(key, None)
    if key in _CACHE:
        return _CACHE[key]

    if not p.exists():
        raise CorpusError(f"patient corpus not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    trusted = tuple(raw.get("trusted_domains") or ())
    if not trusted:
        raise CorpusError(f"{p}: no `trusted_domains`; spec 12 requires an allowlist")

    docs: List[EvidenceDoc] = []
    seen = set()
    for i, d in enumerate(raw.get("documents") or []):
        if not isinstance(d, dict):
            raise CorpusError(f"{p}: document {i} is not a mapping")
        missing = [k for k in ("doc_id", "title", "text", "source_name",
                               "source_tier", "url", "retrieved_on")
                   if not str(d.get(k) or "").strip()]
        if missing:
            raise CorpusError(
                f"{p}: document {d.get('doc_id', i)!r} is missing {missing}")
        if d["doc_id"] in seen:
            raise CorpusError(f"{p}: duplicate doc_id {d['doc_id']!r}")
        seen.add(d["doc_id"])
        try:
            tier = int(d["source_tier"])
        except (TypeError, ValueError):
            raise CorpusError(
                f"{p}: {d['doc_id']!r} has non-numeric source_tier") from None
        if tier not in TIER_NAMES:
            raise CorpusError(
                f"{p}: {d['doc_id']!r} has source_tier {tier}, expected 1–6")
        if not _domain_ok(d["url"], trusted):
            raise CorpusError(
                f"{p}: {d['doc_id']!r} cites {d['url']!r}, which is not in "
                f"trusted_domains; spec 12 forbids untrusted sources")

        docs.append(EvidenceDoc(
            doc_id=str(d["doc_id"]), title=str(d["title"]), text=str(d["text"]),
            source_name=str(d["source_name"]), source_tier=tier,
            url=str(d["url"]),
            topics=tuple(str(t).lower() for t in (d.get("topics") or ())),
            retrieved_on=str(d["retrieved_on"]),
            review_status=str(d.get("review_status") or "unreviewed"),
            verbatim=bool(d.get("verbatim")),
            keywords=tuple(str(k).lower() for k in (d.get("keywords") or ())),
        ))

    out = {"version": raw.get("version", "unknown"),
           "trusted_domains": trusted, "documents": docs}
    _CACHE[key] = out
    return out


# ── retrieval ────────────────────────────────────────────────────────────────

_WORD = re.compile(r"[a-z0-9]+")


def _terms(*parts: Any) -> List[str]:
    out: List[str] = []
    for p in parts:
        if p is None:
            continue
        for chunk in (p if isinstance(p, (list, tuple)) else [p]):
            out += [w for w in _WORD.findall(str(chunk).lower()) if len(w) > 2]
    seen, uniq = set(), []
    for w in out:
        if w not in seen:
            seen.add(w)
            uniq.append(w)
    return uniq


def _score(doc: EvidenceDoc, terms: Sequence[str]) -> int:
    hay = " ".join([doc.title, doc.text, " ".join(doc.topics),
                    " ".join(doc.keywords)]).lower()
    topical = sum(3 for t in terms if t in doc.topics)
    keyworded = sum(2 for t in terms if t in doc.keywords)
    textual = sum(1 for t in terms if t in hay)
    return topical + keyworded + textual


def _guideline_docs(subjects: Sequence[Any], terms: Sequence[str],
                    top_k: int) -> List[EvidenceDoc]:
    """
    Retrieve from the curated society guideline corpus.

    Free text is mapped to canonical concepts by ``terminology.normalise_diagnosis``
    — the same normaliser the clinician pipeline uses, so "septic shock", "sepsis"
    and "severe sepsis" reach the same records here as they do there. Matching on
    raw words instead would answer "septic shock" and miss "urosepsis".

    Society guidelines are tier 1 by definition (spec 12), so they outrank
    everything in the patient corpus when both are searched.
    """
    from src.llm.guidelines import GUIDELINE_CORPUS, retrieve_guidelines
    from src.llm.terminology import normalise_diagnosis

    society = {r.doc_id: r.society for r in GUIDELINE_CORPUS}

    concepts: List[str] = []
    for subject in subjects:
        for chunk in (subject if isinstance(subject, (list, tuple)) else [subject]):
            if not chunk:
                continue
            dx = normalise_diagnosis(str(chunk))
            for c in dx.all_concepts:
                if c not in concepts:
                    concepts.append(c)
    if not concepts:
        return []

    out: List[EvidenceDoc] = []
    for d in retrieve_guidelines(concepts, query_terms=terms, top_k=top_k):
        out.append(EvidenceDoc(
            doc_id=d["doc_id"], title=d["title"], text=d["text"],
            source_name=society.get(d["doc_id"], "Clinical guideline"),
            source_tier=1, url=d.get("url", ""),
            topics=tuple(concepts), retrieved_on="",
            review_status=d.get("review_status", "unreviewed"),
            citation_text=d["citation"],
        ))
    return out


#: OpenFDA's label endpoint, searched on generic name so brand queries have
#: already been resolved to an ingredient before we get here.
_FDA_LABEL_URL = ('https://api.fda.gov/drug/label.json'
                  '?search=openfda.generic_name:"{drug}"&limit=1')

#: Label sections worth quoting to a clinician asking about drug safety, in the
#: order a package insert presents them. A label carrying none of these has
#: nothing to say on the question and is treated as no source at all.
_FDA_SECTIONS: Tuple[Tuple[str, str], ...] = (
    ("boxed_warning", "BOXED WARNING"),
    ("contraindications", "CONTRAINDICATIONS"),
    ("warnings_and_cautions", "WARNINGS AND CAUTIONS"),
    ("warnings", "WARNINGS"),
    ("drug_interactions", "DRUG INTERACTIONS"),
)

#: Per section, so one enormous warnings block cannot crowd out the others.
_FDA_SECTION_CHARS = 700


def _drug_label_docs(subjects: Sequence[Any], top_k: int) -> List[EvidenceDoc]:
    """
    Retrieve manufacturers' package inserts from OpenFDA.

    Only *recognised* medications are queried. ``normalise_medication`` resolves
    brand names, salt forms and eMAR strings to an ingredient, and anything it
    does not recognise is dropped rather than sent — which keeps the endpoint
    from being asked about "septic shock", and keeps an unrecognised word from
    silently becoming a drug the answer then cites.

    ``src.llm.rag_corpus.fetch_live_fda_dailymed`` covers the same endpoint and
    is deliberately not reused. It swallows every exception and then returns a
    placeholder document naming the drug with no label content, which is right
    for a report that lists what it consulted and wrong here: this module's
    contract is that an answer may only contain what retrieval returned, so a
    hollow document is worse than none. Retrieval failing and the label having
    nothing to say must both end as ``NO_SOURCE`` and a decline. Instantiating
    ``ClinicalRAG`` would also read three parquet files, including the notes
    table, on a conversational turn.
    """
    from src.llm.evidence_cache import RetrievalUnavailable, get_default_cache
    from src.llm.terminology import normalise_medication

    ingredients: List[str] = []
    for subject in subjects:
        for chunk in (subject if isinstance(subject, (list, tuple)) else [subject]):
            if not chunk:
                continue
            m = normalise_medication(str(chunk))
            if m.matched and m.ingredient not in ingredients:
                ingredients.append(m.ingredient)
    if not ingredients:
        return []

    cache = get_default_cache()
    out: List[EvidenceDoc] = []
    for drug in ingredients[:top_k]:
        try:
            data = cache.get_json(_FDA_LABEL_URL.format(drug=quote(drug)))
        except RetrievalUnavailable:
            # Transport failure is not evidence of absence, but this module has
            # no way to say "ask me again later" — and inventing a document to
            # carry that message would put unsourced text in front of a
            # clinician. Declining is the honest floor.
            continue
        except Exception:
            continue

        results = data.get("results") or []
        if not results:
            continue
        label = results[0]

        parts: List[str] = []
        for key, heading in _FDA_SECTIONS:
            body = label.get(key) or []
            if not body:
                continue
            text = str(body[0]).strip()
            if len(text) > _FDA_SECTION_CHARS:
                text = text[:_FDA_SECTION_CHARS - 1].rstrip() + "…"
            parts.append(f"{heading}: {text}")
        if not parts:
            continue

        openfda = label.get("openfda") or {}
        brands = [b for b in (openfda.get("brand_name") or []) if str(b).strip()]
        title = (f"FDA package insert: {drug}"
                 + (f" ({brands[0]})" if brands else ""))

        out.append(EvidenceDoc(
            doc_id=f"fda_label_{drug.replace(' ', '_')}",
            title=title,
            text="\n\n".join(parts),
            # The FDA is a government body, which is what tier 2 names. It is
            # not tier 1: a package insert is a regulatory document about a
            # product, not a society's recommendation for managing a patient.
            source_name="U.S. Food and Drug Administration",
            source_tier=2,
            url="https://open.fda.gov/apis/drug/label/",
            topics=(drug,),
            retrieved_on="",
            # Nobody has reviewed these either, and saying so keeps
            # `require_reviewed` meaning the same thing across every source.
            review_status="unreviewed",
            verbatim=True,
            keywords=tuple(str(b).lower() for b in brands[:4]),
            citation_text=f"[FDA package insert: {drug}]",
        ))
    return out


def retrieve(*subjects: Any, top_k: int = 4, require_reviewed: bool = False,
             path: Optional[Path] = None, min_score: int = 1,
             sources: Sequence[str] = DEFAULT_SOURCES) -> EvidenceResult:
    """
    Retrieve documents covering ``subjects`` — any mix of strings and lists.

    ``sources`` selects which corpora are searched. Returns
    ``status == NO_SOURCE`` when nothing matches, which is not an error to be
    worked around: it is the signal that the assistant must decline. Callers
    must not substitute model knowledge for an empty result.
    """
    terms = _terms(*subjects)
    result = EvidenceResult(searched_terms=terms)
    if not terms:
        return result

    candidates: List[EvidenceDoc] = []
    if PATIENT_CORPUS in sources:
        candidates += load_corpus(path)["documents"]
    guideline_hits: List[EvidenceDoc] = []
    if GUIDELINES in sources:
        guideline_hits = _guideline_docs(subjects, terms, top_k)
    if DRUG_LABELS in sources:
        # Concept-matched upstream by the drug normaliser, exactly as the
        # guideline hits are by the diagnosis normaliser, so they join the same
        # pool that is scored but never dropped for word overlap.
        guideline_hits += _drug_label_docs(subjects, top_k)

    # Guideline hits are already concept-matched by `retrieve_guidelines`, so
    # they are scored on relevance but never dropped for it. Word overlap would
    # discard them: "septic shock" yields the terms {septic, shock} while the
    # record is filed under the concept `sepsis`, which shares no token with
    # either — the match is semantic and was made upstream.
    pool: List[tuple] = [(d, _score(d, terms)) for d in candidates]
    pool = [(d, s) for d, s in pool if s >= min_score]
    pool += [(d, max(_score(d, terms), min_score)) for d in guideline_hits]

    if require_reviewed:
        before = len(pool)
        pool = [(d, s) for d, s in pool
                if d.review_status == "clinician_reviewed"]
        result.filtered_unreviewed = before - len(pool)

    if not pool:
        return result

    # Tier first, then relevance: spec 12's hierarchy is not a tiebreak.
    pool.sort(key=lambda ds: (ds[0].source_tier, -ds[1], ds[0].doc_id))
    result.documents = [d for d, _ in pool[:top_k]]
    result.status = OK
    return result


def corpus_stats(path: Optional[Path] = None) -> dict:
    corpus = load_corpus(path)
    docs = corpus["documents"]
    tiers: Dict[int, int] = {}
    for d in docs:
        tiers[d.source_tier] = tiers.get(d.source_tier, 0) + 1
    return {
        "version": corpus["version"],
        "n_documents": len(docs),
        "n_clinician_reviewed": sum(1 for d in docs
                                    if d.review_status == "clinician_reviewed"),
        "by_tier": dict(sorted(tiers.items())),
        "topics": sorted({t for d in docs for t in d.topics}),
        "n_trusted_domains": len(corpus["trusted_domains"]),
    }
