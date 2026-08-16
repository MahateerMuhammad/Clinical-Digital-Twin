"""
backend/routes_evidence.py
──────────────────────────
What the assistant can cite, listed.

The corpus is fifteen concepts and twenty-three records, none of them
clinician-reviewed. Those are real limits, and a reader who cannot see them
discovers them by asking a question and being refused — which reads as the
system being unhelpful rather than as the corpus being small. Publishing the
list turns an apparent failure into a stated boundary.
"""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter

router = APIRouter(prefix="/api/evidence", tags=["evidence"])


@router.get("")
def list_evidence() -> Dict[str, Any]:
    """Corpus statistics and every document that can be retrieved."""
    from src.assistant.evidence import TIER_NAMES
    from src.llm.guidelines import GUIDELINE_CORPUS, corpus_stats

    # Every record in this corpus is a society guideline, and `evidence.py`
    # assigns tier 1 when it builds an EvidenceDoc from one. The tier is read
    # from that fact rather than from a second table here, which could disagree
    # with the retrieval layer about how far a document is to be trusted.
    TIER = 1

    docs: List[Dict[str, Any]] = [
        {
            "doc_id": rec.doc_id,
            "title": f"{rec.society} {rec.document} ({rec.year}) — {rec.section}",
            "tier": TIER,
            "tier_name": TIER_NAMES[TIER],
            "source_name": rec.society,
            "url": rec.url,
            "topics": sorted(rec.concepts),
            "review_status": rec.review_status,
            "strength": rec.strength,
        }
        for rec in GUIDELINE_CORPUS
    ]
    docs.sort(key=lambda d: (d["tier"], d["title"]))
    return {"stats": corpus_stats(), "documents": docs}
