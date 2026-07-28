"""
src/llm/retrieval_eval.py
─────────────────────────
Retrieval quality measurement for the RAG layer.

Before this module there were no retrieval metrics anywhere in the project — the
Phase 11 benchmark reported "100.0%" on quantities that were never computed. This
supplies the standard IR metrics against version-controlled gold sets so retrieval
claims become reproducible numbers.

Three evaluations:

``terminology``  free-text surface form → canonical concept (accuracy, plus a
                 separate false-positive rate on terms that must stay unmapped)
``guidelines``   query → ranked Level 1 doc_ids (recall@k, precision@k, nDCG@k, MRR)
``relevance``    (presentation, document) → admit/reject (precision, recall, F1)

All gold sets live in ``tests/gold/*.json`` and are editable without touching code,
so a clinician can revise judgements directly.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

__all__ = [
    "recall_at_k", "precision_at_k", "dcg", "ndcg_at_k", "reciprocal_rank",
    "EvalResult", "evaluate_terminology", "evaluate_guideline_retrieval",
    "evaluate_topical_relevance", "run_full_evaluation", "render_markdown_report",
]

GOLD_DIR = Path(__file__).resolve().parents[2] / "tests" / "gold"


# ── metric primitives ─────────────────────────────────────────────────────

def recall_at_k(retrieved: Sequence[str], relevant: Dict[str, int], k: int) -> Optional[float]:
    """Fraction of relevant documents present in the top k. None if nothing is relevant."""
    rel = {d for d, g in relevant.items() if g > 0}
    if not rel:
        return None
    return len(rel & set(retrieved[:k])) / len(rel)


def precision_at_k(retrieved: Sequence[str], relevant: Dict[str, int], k: int) -> float:
    """Fraction of the top k that is relevant."""
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    hits = sum(1 for d in top if relevant.get(d, 0) > 0)
    return hits / len(top)


def dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(retrieved: Sequence[str], relevant: Dict[str, int], k: int) -> Optional[float]:
    """Graded nDCG. None when the query has no relevant documents."""
    if not any(g > 0 for g in relevant.values()):
        return None
    gains = [float(relevant.get(d, 0)) for d in retrieved[:k]]
    ideal = sorted((float(g) for g in relevant.values()), reverse=True)[:k]
    idcg = dcg(ideal)
    return (dcg(gains) / idcg) if idcg > 0 else None


def reciprocal_rank(retrieved: Sequence[str], relevant: Dict[str, int]) -> Optional[float]:
    """1/rank of the first relevant document; None if the query has no relevant docs."""
    if not any(g > 0 for g in relevant.values()):
        return None
    for i, d in enumerate(retrieved):
        if relevant.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def _mean(xs: Sequence[Optional[float]]) -> Optional[float]:
    vals = [x for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None


# ── result container ──────────────────────────────────────────────────────

@dataclass
class EvalResult:
    name: str
    n: int
    metrics: Dict[str, Optional[float]] = field(default_factory=dict)
    failures: List[dict] = field(default_factory=list)
    breakdown: Dict[str, Dict[str, float]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "n": self.n,
            "metrics": {k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in self.metrics.items()},
            "n_failures": len(self.failures),
            "failures": self.failures,
            "breakdown": self.breakdown,
            "notes": self.notes,
        }


def _load(path_or_name) -> dict:
    p = Path(path_or_name)
    if not p.is_absolute():
        p = GOLD_DIR / p
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ── 1. terminology ────────────────────────────────────────────────────────

def evaluate_terminology(gold_file="terminology_gold.json") -> EvalResult:
    from src.llm.terminology import normalise_diagnosis

    gold = _load(gold_file)
    cases = gold["cases"]
    res = EvalResult(name="Terminology normalisation", n=len(cases))

    correct = 0
    positives = negatives = 0
    fp = fn = 0
    composite_total = composite_ok = 0
    by_kind: Dict[str, List[int]] = {}

    for c in cases:
        expected = c.get("concept")
        m = normalise_diagnosis(c["surface"])
        got = m.concept
        ok = (got == expected)

        # composite cases additionally require every co-mentioned concept
        if c.get("also_expected"):
            composite_total += 1
            all_ok = ok and all(x in m.all_concepts for x in c["also_expected"])
            if all_ok:
                composite_ok += 1
            ok = all_ok

        if expected is None:
            negatives += 1
            if got is not None:
                fp += 1
        else:
            positives += 1
            if got is None:
                fn += 1

        correct += int(ok)
        by_kind.setdefault(c.get("kind", "unspecified"), []).append(int(ok))

        if not ok:
            res.failures.append({
                "surface": c["surface"], "expected": expected, "got": got,
                "also_expected": c.get("also_expected"),
                "got_all": list(m.all_concepts), "kind": c.get("kind"),
                "method": m.method,
            })

    res.metrics = {
        "accuracy": correct / len(cases) if cases else None,
        "false_positive_rate_on_null_terms": (fp / negatives) if negatives else None,
        "false_negative_rate_on_real_terms": (fn / positives) if positives else None,
        "composite_all_concepts_accuracy": (composite_ok / composite_total) if composite_total else None,
    }
    res.breakdown = {
        k: {"n": len(v), "accuracy": sum(v) / len(v)} for k, v in sorted(by_kind.items())
    }
    res.notes.append(gold["_meta"]["review_note"])
    return res


# ── 2. guideline retrieval ────────────────────────────────────────────────

def evaluate_guideline_retrieval(gold_file="guideline_retrieval_gold.json",
                                 ks=(1, 3, 5)) -> EvalResult:
    from src.llm.guidelines import retrieve_guidelines
    from src.llm.terminology import normalise_diagnosis

    gold = _load(gold_file)
    queries = gold["queries"]
    res = EvalResult(name="Level 1 guideline retrieval", n=len(queries))

    per_k: Dict[int, Dict[str, List[Optional[float]]]] = {
        k: {"recall": [], "precision": [], "ndcg": []} for k in ks
    }
    rrs: List[Optional[float]] = []
    empty_correct = empty_total = 0

    for q in queries:
        concepts = q.get("concepts")
        if not concepts:
            # derive concepts the way production does, to test the whole path
            concepts = list(normalise_diagnosis(q["query"]).all_concepts)
        terms = q["query"].split()
        docs = retrieve_guidelines(concepts, query_terms=terms, top_k=max(ks))
        retrieved = [d["doc_id"] for d in docs]
        relevant = q.get("relevant", {}) or {}

        if not relevant:
            empty_total += 1
            if not retrieved:
                empty_correct += 1
            else:
                res.failures.append({
                    "qid": q["qid"], "query": q["query"],
                    "issue": "returned documents for an out-of-scope query",
                    "retrieved": retrieved,
                })
            continue

        for k in ks:
            per_k[k]["recall"].append(recall_at_k(retrieved, relevant, k))
            per_k[k]["precision"].append(precision_at_k(retrieved, relevant, k))
            per_k[k]["ndcg"].append(ndcg_at_k(retrieved, relevant, k))
        rr = reciprocal_rank(retrieved, relevant)
        rrs.append(rr)

        if rr is not None and rr < 1.0:
            top_grade = relevant.get(retrieved[0], 0) if retrieved else 0
            if top_grade < 2:
                res.failures.append({
                    "qid": q["qid"], "query": q["query"],
                    "issue": "top-ranked document is not a grade-2 match",
                    "retrieved": retrieved[:3],
                    "expected_grade2": [d for d, g in relevant.items() if g == 2],
                })

    for k in ks:
        res.metrics[f"recall@{k}"] = _mean(per_k[k]["recall"])
        res.metrics[f"precision@{k}"] = _mean(per_k[k]["precision"])
        res.metrics[f"ndcg@{k}"] = _mean(per_k[k]["ndcg"])
    res.metrics["mrr"] = _mean(rrs)
    res.metrics["out_of_scope_correctly_empty"] = (
        empty_correct / empty_total if empty_total else None
    )
    res.notes.append(gold["_meta"]["review_note"])
    return res


# ── 3. topical relevance ──────────────────────────────────────────────────

def evaluate_topical_relevance(gold_file="topical_relevance_gold.json") -> EvalResult:
    import tempfile

    from src.llm import rag_corpus as rc
    from src.llm.evidence_cache import EvidenceCache

    gold = _load(gold_file)
    pairs = gold["pairs"]
    res = EvalResult(name="Topical relevance judgement", n=len(pairs))

    tmp = tempfile.mkdtemp(prefix="rageval_")
    e = rc.LiveRealtimeMedicalRAGEngine.__new__(rc.LiveRealtimeMedicalRAGEngine)
    e.data_dir = tmp
    e.models_dir = tmp
    e.cache_dir = tmp
    e.adm_df = e.sim_df = e.df_notes = None
    e.notes_path = ""
    e.citation_log_file = str(Path(tmp) / "c.json"); e.citation_registry = {}
    e.abstract_log_file = str(Path(tmp) / "a.json"); e.abstract_registry = {}
    e.audit_log_file = str(Path(tmp) / "u.json"); e.audit_log = []
    e.evidence_cache = EvidenceCache(tmp, offline=True)
    e.last_twin_status = ""; e.last_retrieval_errors = []
    e.w0_numpy = e.b0_numpy = None

    tp = fp = tn = fn = 0
    trap_total = trap_ok = 0

    for p in pairs:
        payload = {"primary_diagnosis": p["diagnosis"]}
        if p.get("medications"):
            payload["active_medications"] = p["medications"]
        got = bool(e.verify_topical_relevance(p["title"], p["title"], payload))
        want = bool(p["label"])

        if want and got:
            tp += 1
        elif want and not got:
            fn += 1
        elif not want and got:
            fp += 1
        else:
            tn += 1

        if p.get("trap"):
            trap_total += 1
            trap_ok += int(got == want)

        if got != want:
            res.failures.append({
                "id": p["id"], "diagnosis": p["diagnosis"], "title": p["title"],
                "expected": "admit" if want else "reject",
                "got": "admit" if got else "reject",
                "trap": p.get("trap"),
            })

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and (precision + recall) else None)

    res.metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / len(pairs) if pairs else None,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else None,
        "pe_substring_trap_accuracy": (trap_ok / trap_total) if trap_total else None,
    }
    res.breakdown = {"confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn}}
    res.notes.append(gold["_meta"]["review_note"])
    return res


# ── runner ────────────────────────────────────────────────────────────────

def run_full_evaluation() -> Dict[str, EvalResult]:
    return {
        "terminology": evaluate_terminology(),
        "guidelines": evaluate_guideline_retrieval(),
        "relevance": evaluate_topical_relevance(),
    }


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def render_markdown_report(results: Dict[str, EvalResult]) -> str:
    from datetime import datetime, timezone

    from src.llm.guidelines import corpus_stats

    lines: List[str] = []
    lines.append("# RAG Retrieval Evaluation\n")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n")
    lines.append("Gold sets: `tests/gold/*.json` · Harness: `src/llm/retrieval_eval.py`\n")

    lines.append("\n> [!WARNING]")
    lines.append("> **All gold sets are `review_status: unreviewed`.** They were authored")
    lines.append("> alongside the code they measure, so these numbers demonstrate internal")
    lines.append("> consistency and guard against regression. They are **not** independent")
    lines.append("> clinical validation, and must not be cited as retrieval accuracy against")
    lines.append("> clinician ground truth until each row has been reviewed.\n")

    st = corpus_stats()
    lines.append(f"\nGuideline corpus: **{st['n_records']} records** across "
                 f"**{st['n_concepts_covered']} concepts** "
                 f"({st['n_clinician_reviewed']} clinician-reviewed).\n")

    lines.append("\n## 1. Summary\n")
    lines.append("| Evaluation | N | Headline metric | Value |")
    lines.append("| :--- | ---: | :--- | ---: |")
    head = {
        "terminology": ("accuracy", "Concept accuracy"),
        "guidelines": ("ndcg@3", "nDCG@3"),
        "relevance": ("f1", "F1"),
    }
    for key, r in results.items():
        mk, label = head.get(key, (None, ""))
        lines.append(f"| {r.name} | {r.n} | {label} | {_fmt(r.metrics.get(mk))} |")

    for key, r in results.items():
        lines.append(f"\n## {r.name}\n")
        lines.append(f"Cases: **{r.n}** · Failures: **{len(r.failures)}**\n")
        lines.append("| Metric | Value |")
        lines.append("| :--- | ---: |")
        for k, v in r.metrics.items():
            lines.append(f"| {k} | {_fmt(v)} |")

        if r.breakdown:
            lines.append("\n<details><summary>Breakdown</summary>\n")
            lines.append("\n| Group | N | Accuracy |")
            lines.append("| :--- | ---: | ---: |")
            for g, d in r.breakdown.items():
                if "accuracy" in d:
                    lines.append(f"| {g} | {d.get('n', '')} | {_fmt(d['accuracy'])} |")
                else:
                    lines.append(f"| {g} | | {d} |")
            lines.append("\n</details>\n")

        if r.failures:
            lines.append("\n<details><summary>Failing cases</summary>\n")
            lines.append("\n```json")
            lines.append(json.dumps(r.failures, indent=2)[:6000])
            lines.append("```\n")
            lines.append("</details>\n")

    return "\n".join(lines) + "\n"
