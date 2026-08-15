"""
src/evaluation/metrics.py
─────────────────────────
Metric implementations for the evaluation pipeline.

Pure functions over plain data. Nothing here loads a model, reads a file or
knows what a clinician is — which is what makes the numbers checkable: every
metric can be verified by hand on a three-row example, and several are, in
``tests/test_evaluation_metrics.py``.

Two families that are easy to conflate
──────────────────────────────────────
``precision_at_k`` asks *how much of what I returned was relevant*.
``context_precision_at_k`` asks *did the relevant items rank near the top* —
it averages precision measured at each position that holds a relevant item, so
returning the right document last scores far worse than returning it first.
A system can score 1.0 on the first and 0.4 on the second, and the difference
is exactly the thing a reranker would fix.

``recall_at_k`` and ``context_recall`` differ the same way: the first is
position-bounded, the second asks whether the evidence needed to support the
answer was retrieved at all, at any rank.

Missing values are ``None``, never 0.0
──────────────────────────────────────
A query with no relevant documents in the gold set has undefined recall. This
project has already shipped one bug where an unmeasurable quantity was recorded
as zero and then averaged; ``mean()`` skips ``None`` rather than counting it.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "mean", "hit_rate_at_k", "precision_at_k", "recall_at_k", "f1",
    "reciprocal_rank", "dcg", "ndcg_at_k",
    "context_precision_at_k", "context_recall",
    "ClassificationReport", "classification_report",
    "ExtractionReport", "extraction_report",
    "AbstentionReport", "abstention_report",
    "brier_score", "expected_calibration_error", "reliability_bins",
    "auroc", "SliceResult", "slice_report", "MIN_SLICE_ROWS", "MIN_SLICE_EVENTS",
]

#: A slice smaller than this is reported as unmeasured rather than as a finding.
#:
#: Both thresholds are needed and the second is the one that bites. Mortality
#: runs at ~2%, so a 500-row slice can hold four deaths — an AUROC computed on
#: four events swings wildly on one case and would be presented as a disparity.
#: Requiring events as well as rows is what stops the report manufacturing
#: alarming differences out of sampling noise.
MIN_SLICE_ROWS = 500
MIN_SLICE_EVENTS = 25


def mean(xs: Iterable[Optional[float]]) -> Optional[float]:
    """Mean over defined values. Returns None when nothing is defined."""
    vals = [float(x) for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None


# ── ranking ──────────────────────────────────────────────────────────────────

def _relevant_ids(relevant: Dict[str, int]) -> set:
    return {k for k, v in relevant.items() if v > 0}


def hit_rate_at_k(retrieved: Sequence[str], relevant: Dict[str, int],
                  k: int) -> Optional[float]:
    """1.0 if any relevant document appears in the top k."""
    rel = _relevant_ids(relevant)
    if not rel:
        return None
    return 1.0 if rel & set(retrieved[:k]) else 0.0


def precision_at_k(retrieved: Sequence[str], relevant: Dict[str, int],
                   k: int) -> Optional[float]:
    """Fraction of the top k that is relevant. Undefined if nothing retrieved."""
    top = retrieved[:k]
    if not top:
        return None
    rel = _relevant_ids(relevant)
    return sum(1 for d in top if d in rel) / len(top)


def recall_at_k(retrieved: Sequence[str], relevant: Dict[str, int],
                k: int) -> Optional[float]:
    rel = _relevant_ids(relevant)
    if not rel:
        return None
    return len(rel & set(retrieved[:k])) / len(rel)


def f1(precision: Optional[float], recall: Optional[float]) -> Optional[float]:
    if precision is None or recall is None or (precision + recall) == 0:
        return 0.0 if (precision is not None and recall is not None) else None
    return 2 * precision * recall / (precision + recall)


def reciprocal_rank(retrieved: Sequence[str],
                    relevant: Dict[str, int]) -> Optional[float]:
    rel = _relevant_ids(relevant)
    if not rel:
        return None
    for i, doc in enumerate(retrieved, start=1):
        if doc in rel:
            return 1.0 / i
    return 0.0


def dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(retrieved: Sequence[str], relevant: Dict[str, int],
              k: int) -> Optional[float]:
    if not relevant:
        return None
    gains = [float(relevant.get(d, 0)) for d in retrieved[:k]]
    ideal = sorted((float(v) for v in relevant.values()), reverse=True)[:k]
    denom = dcg(ideal)
    return (dcg(gains) / denom) if denom > 0 else None


# ── RAG context metrics ──────────────────────────────────────────────────────

def context_precision_at_k(retrieved: Sequence[str], relevant: Dict[str, int],
                           k: int) -> Optional[float]:
    """
    Rank-sensitive precision: mean of precision@i over positions holding a
    relevant document.

    This is the metric that exposes a missing reranker. ``precision_at_k``
    cannot distinguish [relevant, irrelevant] from [irrelevant, relevant] —
    both score 0.5 — while this scores them 1.0 and 0.25. Where the answer is
    composed from the top documents, that ordering is what the reader sees.
    """
    top = retrieved[:k]
    rel = _relevant_ids(relevant)
    if not rel or not top:
        return None
    hits = [i for i, d in enumerate(top, start=1) if d in rel]
    if not hits:
        return 0.0
    return mean([len([h for h in hits if h <= i]) / i for i in hits])


def context_recall(retrieved: Sequence[str],
                   relevant: Dict[str, int],
                   k: Optional[int] = None) -> Optional[float]:
    """
    Fraction of the documents needed to support the answer that were retrieved.

    Unbounded by default: the question is whether the supporting evidence was
    found at all. A low value means the answer was composed without material the
    gold set says it needed — the failure mode that produces a confident,
    correctly-cited, incomplete answer.
    """
    rel = _relevant_ids(relevant)
    if not rel:
        return None
    pool = set(retrieved if k is None else retrieved[:k])
    return len(rel & pool) / len(rel)


# ── classification ───────────────────────────────────────────────────────────

@dataclass
class ClassificationReport:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confusion: Dict[str, Dict[str, int]] = field(default_factory=dict)
    n: int = 0
    errors: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"n": self.n, "accuracy": self.accuracy,
                "macro_precision": self.macro_precision,
                "macro_recall": self.macro_recall, "macro_f1": self.macro_f1,
                "per_class": self.per_class, "confusion": self.confusion,
                "errors": self.errors}


def classification_report(pairs: Sequence[Tuple[str, str]],
                          *, cases: Optional[Sequence[str]] = None,
                          ) -> ClassificationReport:
    """
    ``pairs`` is (expected, predicted). Macro averages weight every class
    equally, so a rare-but-important class cannot be hidden by a common one.
    """
    if not pairs:
        return ClassificationReport(0.0, 0.0, 0.0, 0.0, n=0)

    labels = sorted({e for e, _ in pairs} | {p for _, p in pairs})
    tp = Counter()
    fp = Counter()
    fn = Counter()
    confusion: Dict[str, Dict[str, int]] = {e: {p: 0 for p in labels} for e in labels}

    errors: List[Dict[str, str]] = []
    for i, (exp, pred) in enumerate(pairs):
        confusion[exp][pred] += 1
        if exp == pred:
            tp[exp] += 1
        else:
            fp[pred] += 1
            fn[exp] += 1
            errors.append({"case": (cases[i] if cases and i < len(cases) else ""),
                           "expected": exp, "predicted": pred})

    per_class: Dict[str, Dict[str, float]] = {}
    for label in labels:
        p = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) else 0.0
        r = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) else 0.0
        per_class[label] = {"precision": p, "recall": r,
                            "f1": f1(p, r) or 0.0,
                            "support": tp[label] + fn[label]}

    support_labels = [l for l in labels if per_class[l]["support"] > 0]
    return ClassificationReport(
        accuracy=sum(tp.values()) / len(pairs),
        macro_precision=mean([per_class[l]["precision"] for l in support_labels]) or 0.0,
        macro_recall=mean([per_class[l]["recall"] for l in support_labels]) or 0.0,
        macro_f1=mean([per_class[l]["f1"] for l in support_labels]) or 0.0,
        per_class=per_class, confusion=confusion, n=len(pairs), errors=errors)


# ── extraction ───────────────────────────────────────────────────────────────

@dataclass
class ExtractionReport:
    precision: float
    recall: float
    f1: float
    #: Fields extracted with a value the source text does not contain. This is
    #: the hallucination rate for the one stage that writes into patient state,
    #: and the number that must be zero.
    fabrication_rate: float
    n_expected: int = 0
    n_predicted: int = 0
    fabrications: List[Dict[str, Any]] = field(default_factory=list)
    missed: List[Dict[str, Any]] = field(default_factory=list)
    wrong_value: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"precision": self.precision, "recall": self.recall, "f1": self.f1,
                "fabrication_rate": self.fabrication_rate,
                "n_expected": self.n_expected, "n_predicted": self.n_predicted,
                "fabrications": self.fabrications, "missed": self.missed,
                "wrong_value": self.wrong_value}


def _same_value(a: Any, b: Any) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return str(a).strip().lower() == str(b).strip().lower()


def extraction_report(cases: Sequence[Dict[str, Any]]) -> ExtractionReport:
    """
    ``cases``: {"message", "expected": {field: value}, "predicted": {field: value}}.

    A predicted field counts as correct only when its value matches. Extracting
    the right field with the wrong number is not a partial success here — it is
    a fabricated observation wearing a correct label.
    """
    tp = fp = fn = 0
    fabrications: List[Dict[str, Any]] = []
    missed: List[Dict[str, Any]] = []
    wrong: List[Dict[str, Any]] = []
    n_expected = n_predicted = 0

    for case in cases:
        msg = str(case.get("message", ""))
        low = msg.lower()
        expected = case.get("expected") or {}
        predicted = case.get("predicted") or {}
        n_expected += len(expected)
        n_predicted += len(predicted)

        for fname, value in predicted.items():
            if fname in expected and _same_value(expected[fname], value):
                tp += 1
                continue
            fp += 1
            entry = {"message": msg, "field": fname, "value": value,
                     "expected": expected.get(fname)}
            wrong.append(entry)
            # A value whose text does not occur in the message was not read out
            # of it. That is the fabrication case — distinct from extracting a
            # field the gold set did not ask for, which is merely noisy.
            if str(value).strip().lower() not in low:
                fabrications.append(entry)

        for fname, value in expected.items():
            if fname not in predicted:
                fn += 1
                missed.append({"message": msg, "field": fname, "expected": value})

    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return ExtractionReport(
        precision=p, recall=r, f1=f1(p, r) or 0.0,
        fabrication_rate=(len(fabrications) / n_predicted) if n_predicted else 0.0,
        n_expected=n_expected, n_predicted=n_predicted,
        fabrications=fabrications, missed=missed, wrong_value=wrong)


# ── abstention ───────────────────────────────────────────────────────────────

@dataclass
class AbstentionReport:
    """
    How well the system knows when not to answer.

    The two error types are not symmetric and must never be averaged into one
    number. **Under-refusal** — answering when the gate should have closed — is
    the unsafe direction and is what spec 22 exists to prevent.
    **Over-refusal** is friction: annoying, not dangerous.
    """

    accuracy: float
    #: Of the cases that should have been refused, the fraction that were.
    refusal_recall: float
    #: Of the cases that should have been answered, the fraction that were.
    answer_recall: float
    under_refusals: List[Dict[str, str]] = field(default_factory=list)
    over_refusals: List[Dict[str, str]] = field(default_factory=list)
    n: int = 0

    @property
    def under_refusal_rate(self) -> float:
        return 1.0 - self.refusal_recall

    @property
    def over_refusal_rate(self) -> float:
        return 1.0 - self.answer_recall

    def to_dict(self) -> dict:
        return {"n": self.n, "accuracy": self.accuracy,
                "refusal_recall": self.refusal_recall,
                "answer_recall": self.answer_recall,
                "under_refusal_rate": self.under_refusal_rate,
                "over_refusal_rate": self.over_refusal_rate,
                "under_refusals": self.under_refusals,
                "over_refusals": self.over_refusals}


def abstention_report(cases: Sequence[Dict[str, Any]]) -> AbstentionReport:
    """``cases``: {"id", "message", "should_answer": bool, "answered": bool}."""
    if not cases:
        return AbstentionReport(0.0, 0.0, 0.0, n=0)

    correct = 0
    should_refuse = [c for c in cases if not c.get("should_answer")]
    should_answer = [c for c in cases if c.get("should_answer")]
    under: List[Dict[str, str]] = []
    over: List[Dict[str, str]] = []

    for c in cases:
        answered = bool(c.get("answered"))
        if answered == bool(c.get("should_answer")):
            correct += 1
        elif answered:
            under.append({"id": str(c.get("id", "")),
                          "message": str(c.get("message", "")),
                          "why": str(c.get("why", ""))})
        else:
            over.append({"id": str(c.get("id", "")),
                         "message": str(c.get("message", "")),
                         "why": str(c.get("why", ""))})

    return AbstentionReport(
        accuracy=correct / len(cases),
        refusal_recall=((len(should_refuse) - len(under)) / len(should_refuse))
        if should_refuse else 1.0,
        answer_recall=((len(should_answer) - len(over)) / len(should_answer))
        if should_answer else 1.0,
        under_refusals=under, over_refusals=over, n=len(cases))


# ── calibration ──────────────────────────────────────────────────────────────

def brier_score(probs: Sequence[float], outcomes: Sequence[int]) -> Optional[float]:
    """Mean squared error of a probabilistic forecast. Lower is better."""
    pairs = [(p, o) for p, o in zip(probs, outcomes) if p is not None]
    if not pairs:
        return None
    return sum((p - o) ** 2 for p, o in pairs) / len(pairs)


def reliability_bins(probs: Sequence[float], outcomes: Sequence[int],
                     n_bins: int = 10) -> List[Dict[str, float]]:
    """Predicted vs observed frequency per probability bin."""
    bins: List[Dict[str, float]] = []
    pairs = [(float(p), int(o)) for p, o in zip(probs, outcomes) if p is not None]
    if not pairs:
        return bins
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        chunk = [(p, o) for p, o in pairs
                 if (p >= lo and (p < hi or (i == n_bins - 1 and p <= hi)))]
        if not chunk:
            continue
        bins.append({
            "bin_low": lo, "bin_high": hi, "n": len(chunk),
            "mean_predicted": sum(p for p, _ in chunk) / len(chunk),
            "observed_rate": sum(o for _, o in chunk) / len(chunk),
        })
    return bins


def expected_calibration_error(probs: Sequence[float], outcomes: Sequence[int],
                               n_bins: int = 10) -> Optional[float]:
    """
    Support-weighted mean gap between predicted probability and observed rate.

    Reported alongside AUROC because they answer different questions: AUROC says
    the ranking is right, ECE says the numbers mean what they say. A clinician
    reading "2% mortality" is relying on the second, and the existing evaluation
    only measured the first.
    """
    bins = reliability_bins(probs, outcomes, n_bins)
    total = sum(b["n"] for b in bins)
    if not total:
        return None
    return sum(b["n"] * abs(b["mean_predicted"] - b["observed_rate"])
               for b in bins) / total


# ── discrimination and subgroup analysis ─────────────────────────────────────

def auroc(probs: Sequence[float], outcomes: Sequence[int]) -> Optional[float]:
    """
    Area under the ROC curve, or None when it is undefined.

    Undefined means one class only — every case in the slice had the same
    outcome. That is common in small subgroups and is not a score of 0.5; a
    number would imply a measurement that was not made.
    """
    pairs = [(float(p), int(o)) for p, o in zip(probs, outcomes) if p is not None]
    if not pairs:
        return None
    ys = {o for _, o in pairs}
    if len(ys) < 2:
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score([o for _, o in pairs], [p for p, _ in pairs]))


@dataclass
class SliceResult:
    """One subgroup's performance, or the reason it was not measured."""

    name: str
    n: int
    events: int
    measured: bool
    reason: str = ""
    base_rate: Optional[float] = None
    mean_predicted: Optional[float] = None
    auroc: Optional[float] = None
    brier: Optional[float] = None
    ece: Optional[float] = None

    def to_dict(self) -> dict:
        return {"name": self.name, "n": self.n, "events": self.events,
                "measured": self.measured, "reason": self.reason,
                "base_rate": self.base_rate, "mean_predicted": self.mean_predicted,
                "auroc": self.auroc, "brier": self.brier, "ece": self.ece}


def slice_report(groups: Dict[str, Tuple[Sequence[float], Sequence[int]]],
                 *, min_rows: int = MIN_SLICE_ROWS,
                 min_events: int = MIN_SLICE_EVENTS) -> Dict[str, Any]:
    """
    Per-subgroup metrics plus the disparity summary.

    ``groups`` maps a slice name to (predicted probabilities, observed outcomes).

    The headline is the gap, not the average. A model can post an excellent
    overall ECE while being badly calibrated for one group, and the overall
    number is what hides it — so the summary reports the worst slice and the
    spread across slices, which is what someone reviewing a clinical model
    actually needs to see.
    """
    results: List[SliceResult] = []
    for name, (probs, outcomes) in groups.items():
        n = len(outcomes)
        events = int(sum(outcomes))
        if n < min_rows or events < min_events:
            results.append(SliceResult(
                name=name, n=n, events=events, measured=False,
                reason=(f"below support floor (needs n>={min_rows} and "
                        f"events>={min_events})"),
                base_rate=(events / n) if n else None))
            continue
        results.append(SliceResult(
            name=name, n=n, events=events, measured=True,
            base_rate=events / n,
            mean_predicted=mean(list(probs)),
            auroc=auroc(probs, outcomes),
            brier=brier_score(probs, outcomes),
            ece=expected_calibration_error(probs, outcomes)))

    measured = [r for r in results if r.measured]
    aurocs = [(r.name, r.auroc) for r in measured if r.auroc is not None]
    eces = [(r.name, r.ece) for r in measured if r.ece is not None]

    summary: Dict[str, Any] = {
        "n_slices": len(results),
        "n_measured": len(measured),
        "n_unmeasured": len(results) - len(measured),
    }
    if aurocs:
        best = max(aurocs, key=lambda kv: kv[1])
        worst = min(aurocs, key=lambda kv: kv[1])
        summary["auroc_best"] = {"slice": best[0], "value": best[1]}
        summary["auroc_worst"] = {"slice": worst[0], "value": worst[1]}
        summary["auroc_gap"] = best[1] - worst[1]
    if eces:
        worst_ece = max(eces, key=lambda kv: kv[1])
        best_ece = min(eces, key=lambda kv: kv[1])
        summary["ece_worst"] = {"slice": worst_ece[0], "value": worst_ece[1]}
        summary["ece_best"] = {"slice": best_ece[0], "value": best_ece[1]}
        summary["ece_gap"] = worst_ece[1] - best_ece[1]

    return {"summary": summary, "slices": [r.to_dict() for r in results]}
