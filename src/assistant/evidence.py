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
from urllib.parse import urlparse

import yaml

__all__ = [
    "EvidenceDoc", "EvidenceResult", "retrieve", "load_corpus",
    "OK", "NO_SOURCE", "TIER_NAMES", "PATIENT_CORPUS_PATH",
    "PATIENT_CORPUS", "GUIDELINES", "DEFAULT_SOURCES",
]

#: Selectable corpora. `patient` is the plain-language file that ships empty;
#: `guidelines` is the curated society corpus in `src/llm/guidelines.py`, which
#: is written for clinicians and is the right source for that audience.
PATIENT_CORPUS = "patient"
GUIDELINES = "guidelines"
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
                "am not going to answer from memory — I could sound confident "
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
