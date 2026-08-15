#!/usr/bin/env python
"""
scripts/evaluation/run_assistant_eval.py
────────────────────────────────────────
End-to-end evaluation of the clinician assistant.

    PYTHONPATH=. .venv/bin/python scripts/evaluation/run_assistant_eval.py
    … --no-models      skip anything that needs the boosters
    … --quick          skip the live-retrieval suites (network, cold ~1 min/query)

Writes ``reports/assistant_evaluation.md``.

What this measures, and what it cannot
──────────────────────────────────────
Six suites, each answering a question the others cannot:

**Retrieval**       does the right document come back, and does it come back near
                    the top? Ranking metrics plus the two RAG context metrics.
**Intent**          is the routing decision right? This decides whether a model
                    is consulted at all, so an error here is not a wording
                    problem, it is the wrong pipeline.
**Extraction**      are the values written into patient state the ones the
                    clinician actually typed? Fabrication rate must be zero.
**Abstention**      does it refuse when it should? Under-refusal is the unsafe
                    direction and is reported separately from over-refusal.
**Faithfulness**    does every number and citation in an answer trace to an
                    input or a retrieved document?
**Calibration**     do the probabilities mean what they say? AUROC says the
                    ranking is right; Brier and ECE say the numbers are usable.

None of it establishes clinical correctness. Every gold set here was authored
by the same engineering effort that wrote the code, so these numbers measure
internal consistency and regression safety. A correctly-retrieved guideline that
does not apply to the patient passes every check in this file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath("."))

from src.evaluation import metrics as M

GOLD = Path("tests/gold")
REPORT = Path("reports/assistant_evaluation.md")
K_VALUES = (1, 3, 5)


def _load(name: str) -> dict:
    with (GOLD / name).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _pct(v: Optional[float], places: int = 1) -> str:
    return "n/a" if v is None else f"{v * 100:.{places}f}%"


def _num(v: Optional[float], places: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{places}f}"


# ══ 1. retrieval ═════════════════════════════════════════════════════════════

def eval_retrieval() -> Dict[str, Any]:
    """
    Ranking quality over the curated guideline corpus.

    Retrieval here is concept-anchored and lexical: `normalise_diagnosis` maps
    the query onto canonical concepts, the concept index returns candidates, and
    they are re-scored by query-term overlap. No embedding model and no
    reranker are involved — `context_precision` is the metric that would move if
    one were added, because it is the only one that penalises a relevant
    document returned late.
    """
    from src.llm.guidelines import retrieve_guidelines
    from src.llm.terminology import normalise_diagnosis

    gold = _load("guideline_retrieval_gold.json")
    rows: List[Dict[str, Any]] = []
    per_k: Dict[int, Dict[str, List]] = {k: {"p": [], "r": [], "ndcg": [],
                                             "hit": [], "cp": []} for k in K_VALUES}
    rr: List[Optional[float]] = []
    crecall: List[Optional[float]] = []
    latencies: List[float] = []

    for q in gold["queries"]:
        query = q.get("query", "")
        relevant = {str(k): int(v) for k, v in (q.get("relevant") or {}).items()}
        concepts = q.get("concepts")
        if not concepts:
            concepts = list(normalise_diagnosis(query).all_concepts)

        t0 = time.perf_counter()
        docs = retrieve_guidelines(concepts, query_terms=query.lower().split(),
                                   top_k=max(K_VALUES))
        latencies.append((time.perf_counter() - t0) * 1000)
        ids = [d["doc_id"] for d in docs]

        for k in K_VALUES:
            per_k[k]["p"].append(M.precision_at_k(ids, relevant, k))
            per_k[k]["r"].append(M.recall_at_k(ids, relevant, k))
            per_k[k]["ndcg"].append(M.ndcg_at_k(ids, relevant, k))
            per_k[k]["hit"].append(M.hit_rate_at_k(ids, relevant, k))
            per_k[k]["cp"].append(M.context_precision_at_k(ids, relevant, k))
        rr.append(M.reciprocal_rank(ids, relevant))
        crecall.append(M.context_recall(ids, relevant))
        rows.append({"query": query, "retrieved": ids[:5],
                     "relevant": sorted(relevant)})

    return {
        "n_queries": len(gold["queries"]),
        "mrr": M.mean(rr),
        "context_recall": M.mean(crecall),
        "median_latency_ms": sorted(latencies)[len(latencies) // 2] if latencies else None,
        "at_k": {k: {"precision": M.mean(v["p"]), "recall": M.mean(v["r"]),
                     "ndcg": M.mean(v["ndcg"]), "hit_rate": M.mean(v["hit"]),
                     "context_precision": M.mean(v["cp"])}
                 for k, v in per_k.items()},
        "rows": rows,
    }


# ══ 2. intent routing ════════════════════════════════════════════════════════

def eval_intent() -> Dict[str, Any]:
    from src.assistant.intents import CLINICIAN, classify

    gold = _load("clinician_intent_gold.json")
    pairs, cases = [], []
    for c in gold["cases"]:
        pairs.append((c["intent"], classify(c["message"], mode=CLINICIAN).intent.value))
        cases.append(c["message"])
    report = M.classification_report(pairs, cases=cases)

    # The routing decision in isolation: does a model get consulted or not?
    model_pairs = [("model" if e == "risk_assessment" else "evidence",
                    "model" if p == "risk_assessment" else "evidence")
                   for e, p in pairs]
    routing = M.classification_report(model_pairs)
    out = report.to_dict()
    out["routing"] = routing.to_dict()
    return out


# ══ 3. extraction ════════════════════════════════════════════════════════════

def eval_extraction(backend=None) -> Dict[str, Any]:
    from src.assistant.extraction import extract
    from src.assistant.state import PatientContext

    gold = _load("clinician_extraction_gold.json")
    cases = []
    for c in gold["cases"]:
        ctx = PatientContext()
        res = extract(c["message"], ctx, turn=1, backend=backend)
        predicted = {p.field: p.value for p in res.accepted}
        cases.append({"message": c["message"], "expected": c["expected"],
                      "predicted": predicted})
    out = M.extraction_report(cases).to_dict()
    out["backend"] = "openrouter" if backend else "deterministic"
    return out


# ══ 4. abstention ════════════════════════════════════════════════════════════

def eval_abstention(assistant_factory) -> Dict[str, Any]:
    from src.assistant import answer as A

    gold = _load("abstention_gold.json")
    cases = []
    for c in gold["cases"]:
        bot = assistant_factory()
        sid = bot.start().state.session_id
        result = None
        for turn in c["turns"]:
            result = bot.handle(sid, turn)
        answered = bool(result and result.status == A.ANSWERED)
        cases.append({"id": c["id"], "message": " | ".join(c["turns"]),
                      "should_answer": c["should_answer"], "answered": answered,
                      "why": c.get("why", ""), "status": result.status if result else ""})
    out = M.abstention_report(cases).to_dict()
    out["cases"] = cases
    return out


# ══ 5. faithfulness ══════════════════════════════════════════════════════════

def eval_faithfulness(assistant_factory) -> Dict[str, Any]:
    """
    Of the answers actually produced, how many survive verification?

    Only answered turns count. A refusal has nothing to be faithful about, and
    including refusals would let a system that never answers score 100% — the
    exact failure the rephrase harness caught earlier in this project.
    """
    from src.assistant import answer as A

    # The abstention set alone yields only a handful of answers — most of its
    # cases are designed to be refused. A verified-rate over four samples is not
    # a measurement, so the guideline queries are added: they are answerable by
    # construction and exercise the citation path that faithfulness is about.
    cases = [{"id": c["id"], "turns": c["turns"]}
             for c in _load("abstention_gold.json")["cases"]]
    cases += [{"id": q["qid"], "turns": [q["query"]]}
              for q in _load("guideline_retrieval_gold.json")["queries"]]

    verified = unverified = 0
    citation_bearing = 0
    failures: List[Dict[str, Any]] = []

    for c in cases:
        bot = assistant_factory()
        sid = bot.start().state.session_id
        result = None
        for turn in c["turns"]:
            result = bot.handle(sid, turn)
        if not result or result.status != A.ANSWERED:
            continue
        ok = bool(result.faithfulness and result.faithfulness.ok)
        verified += int(ok)
        unverified += int(not ok)
        if result.answer and result.answer.citations:
            citation_bearing += 1
        if not ok and result.faithfulness:
            failures.append({
                "id": c["id"],
                "failed": [f"{ch.number}:{ch.name}"
                           for ch in result.faithfulness.checks
                           if not ch.passed and ch.blocking]})

    total = verified + unverified
    return {"n_answers": total,
            "verified_rate": (verified / total) if total else None,
            "cited_rate": (citation_bearing / total) if total else None,
            "failures": failures}


# ══ 6. calibration ═══════════════════════════════════════════════════════════

def eval_calibration(max_rows: int = 4000) -> Dict[str, Any]:
    """
    Brier and ECE for the promoted models on the held-out test split.

    Reported because AUROC and calibration answer different questions and only
    the first was measured. A model can rank perfectly and still be badly
    calibrated, and a clinician reading "2% mortality" is relying on the second.
    """
    import numpy as np
    import pandas as pd

    from src.llm.model_runner import LiveModelRunner

    path = Path("data/processed/admission_level_selected.parquet")
    if not path.exists():
        return {"skipped": f"{path} not found"}

    runner = LiveModelRunner()
    df = pd.read_parquet(path)

    # The feature table carries no split column, so the split has to be joined
    # from `patient_split.parquet`. Without this the calibration numbers are
    # computed over rows the isotonic calibrators were fitted on, which does not
    # measure calibration — it measures memorisation, and would have been
    # reported as an excellent ECE.
    split_path = Path("data/processed/patient_split.parquet")
    if "split" in df.columns:
        df = df[df["split"].astype(str).str.lower() == "test"]
    elif split_path.exists() and "subject_id" in df.columns:
        split = pd.read_parquet(split_path)[["subject_id", "split"]]
        df = df.merge(split, on="subject_id", how="inner")
        df = df[df["split"].astype(str).str.lower() == "test"]
    else:
        return {"skipped": "no patient split available; refusing to score on "
                           "rows the calibrators may have been fitted on"}
    if df.empty:
        return {"skipped": "no test rows"}
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=0)

    targets = {"mortality": "hospital_expire_flag",
               "icu_admission": "has_icu_stay",
               "readmission": "readmission_30d"}
    out: Dict[str, Any] = {"n_rows": int(len(df)), "tasks": {}}

    for task, target in targets.items():
        model = runner.lgbm_models.get(task)
        if model is None or target not in df.columns:
            out["tasks"][task] = {"skipped": "model or target absent"}
            continue
        try:
            feats = model.booster_.feature_name()
            X = df.reindex(columns=feats)
            raw = model.predict_proba(X)[:, 1]
            y = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int).tolist()

            # The boosters are wrapped in isotonic calibrators and the runner
            # never serves a raw probability. Scoring `predict_proba` directly
            # measures a number no caller ever sees: on this split it reported
            # mortality at a 14.5% mean against a 1.9% base rate, ECE 0.126 —
            # an alarming result about a model that is not the one deployed.
            # Both are reported so the calibrator's contribution is visible.
            calibrator = runner.calibrators.get(task)
            cal = (np.asarray(calibrator.predict(raw), dtype=float)
                   if calibrator is not None and hasattr(calibrator, "predict")
                   else None)

            entry = {
                "n": int(len(y)),
                "base_rate": float(np.mean(y)),
                "calibrated": cal is not None,
                "raw": {"mean_predicted": float(np.mean(raw)),
                        "brier": M.brier_score(list(raw), y),
                        "ece": M.expected_calibration_error(list(raw), y)},
            }
            served = cal if cal is not None else raw
            entry["served"] = {
                "mean_predicted": float(np.mean(served)),
                "brier": M.brier_score(list(served), y),
                "ece": M.expected_calibration_error(list(served), y),
                "bins": M.reliability_bins(list(served), y),
            }
            out["tasks"][task] = entry
        except Exception as exc:
            out["tasks"][task] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


# ══ report ═══════════════════════════════════════════════════════════════════

def render(results: Dict[str, Any]) -> str:
    L: List[str] = []
    add = L.append

    add("# Clinician Assistant — Evaluation Report")
    add("")
    add(f"*Generated {date.today().isoformat()} by "
        "`scripts/evaluation/run_assistant_eval.py`.*")
    add("")
    add("Every gold set below was authored by the same engineering effort that "
        "wrote the code. These numbers measure **internal consistency and "
        "regression safety**, not clinical correctness: a correctly-retrieved "
        "guideline that does not apply to the patient passes every check here.")
    add("")

    # ── retrieval ──
    r = results.get("retrieval")
    if r:
        add("## 1. Retrieval")
        add("")
        add("Concept-anchored lexical retrieval over the curated guideline "
            "corpus. `normalise_diagnosis` maps the query to canonical "
            "concepts, the concept index returns candidates, and query-term "
            "overlap re-scores them. **No embedding retrieval and no reranker** "
            "are involved at this tier.")
        add("")
        add(f"- Queries: **{r['n_queries']}**")
        add(f"- MRR: **{_num(r['mrr'])}**")
        add(f"- Context recall (unbounded): **{_pct(r['context_recall'])}**")
        add(f"- Median retrieval latency: **{_num(r['median_latency_ms'], 2)} ms**")
        add("")
        add("| k | Precision@k | Recall@k | nDCG@k | Hit rate@k | Context precision@k |")
        add("| ---: | ---: | ---: | ---: | ---: | ---: |")
        for k, v in sorted(r["at_k"].items()):
            add(f"| {k} | {_pct(v['precision'])} | {_pct(v['recall'])} | "
                f"{_num(v['ndcg'])} | {_pct(v['hit_rate'])} | "
                f"{_pct(v['context_precision'])} |")
        add("")
        add("*Context precision is the rank-sensitive metric: it penalises a "
            "relevant document returned late, which plain precision@k cannot "
            "see. It is the number a reranker would move.*")
        add("")

    # ── intent ──
    i = results.get("intent")
    if i:
        add("## 2. Intent routing")
        add("")
        add(f"- Cases: **{i['n']}** · Accuracy: **{_pct(i['accuracy'])}** · "
            f"Macro F1: **{_num(i['macro_f1'])}**")
        rt = i.get("routing") or {}
        add(f"- Model-vs-evidence routing accuracy: **{_pct(rt.get('accuracy'))}**")
        add("")
        add("| Intent | Precision | Recall | F1 | Support |")
        add("| :--- | ---: | ---: | ---: | ---: |")
        for label, v in sorted(i["per_class"].items()):
            if not v["support"]:
                continue
            add(f"| `{label}` | {_pct(v['precision'])} | {_pct(v['recall'])} | "
                f"{_num(v['f1'])} | {int(v['support'])} |")
        add("")
        if i["errors"]:
            add("**Misroutes**")
            add("")
            for e in i["errors"]:
                add(f"- `{e['expected']}` → `{e['predicted']}` — \"{e['case']}\"")
            add("")

    # ── extraction ──
    x = results.get("extraction")
    if x:
        add("## 3. Fact extraction")
        add("")
        add(f"Backend: **{x['backend']}**")
        add("")
        add(f"- Precision **{_pct(x['precision'])}** · Recall "
            f"**{_pct(x['recall'])}** · F1 **{_num(x['f1'])}**")
        add(f"- **Fabrication rate: {_pct(x['fabrication_rate'])}** "
            f"({len(x['fabrications'])} of {x['n_predicted']} extracted values)")
        add("")
        add("*Fabrication rate is the one that must be zero: it counts values "
            "written into patient state that do not appear in the message. "
            "Recall below 100% means the gate asks more questions — friction, "
            "not danger.*")
        add("")
        if x["fabrications"]:
            add("**Fabrications**")
            add("")
            for fb in x["fabrications"]:
                add(f"- `{fb['field']}` = {fb['value']!r} from \"{fb['message']}\"")
            add("")
        if x["missed"]:
            add(f"<details><summary>Missed fields ({len(x['missed'])})</summary>")
            add("")
            for m in x["missed"]:
                add(f"- `{m['field']}` = {m['expected']!r} — \"{m['message']}\"")
            add("")
            add("</details>")
            add("")

    # ── abstention ──
    a = results.get("abstention")
    if a:
        add("## 4. Abstention")
        add("")
        add(f"- Cases: **{a['n']}** · Accuracy: **{_pct(a['accuracy'])}**")
        add(f"- **Under-refusal rate: {_pct(a['under_refusal_rate'])}** "
            "(answered when it should have refused — the unsafe direction)")
        add(f"- Over-refusal rate: {_pct(a['over_refusal_rate'])} "
            "(refused when it could have answered — friction)")
        add("")
        if a["under_refusals"]:
            add("**Under-refusals — every one of these is a defect**")
            add("")
            for u in a["under_refusals"]:
                add(f"- `{u['id']}` — {u['why']}")
            add("")
        if a["over_refusals"]:
            add("**Over-refusals**")
            add("")
            for o in a["over_refusals"]:
                add(f"- `{o['id']}` — {o['why']}")
            add("")

    # ── faithfulness ──
    f = results.get("faithfulness")
    if f:
        add("## 5. Faithfulness")
        add("")
        add(f"- Answers produced: **{f['n_answers']}**")
        add(f"- Verified: **{_pct(f['verified_rate'])}**")
        add(f"- Carrying at least one citation: **{_pct(f['cited_rate'])}**")
        add("")
        add("*Computed over answered turns only. Including refusals would let a "
            "system that never answers score 100%.*")
        add("")
        for fail in f["failures"]:
            add(f"- `{fail['id']}` failed: {', '.join(fail['failed'])}")
        if f["failures"]:
            add("")

    # ── calibration ──
    c = results.get("calibration")
    if c:
        add("## 6. Model calibration")
        add("")
        if c.get("skipped"):
            add(f"*Skipped: {c['skipped']}*")
            add("")
        else:
            add(f"Held-out test split, {c['n_rows']} rows. **Served** is the "
                "isotonic-calibrated probability the runner actually returns; "
                "**raw** is the bare booster output, shown so the calibrator's "
                "contribution is visible.")
            add("")
            add("| Task | n | Base rate | Served mean | Served Brier | Served ECE "
                "| Raw Brier | Raw ECE |")
            add("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            for task, v in c["tasks"].items():
                if "skipped" in v or "error" in v:
                    add(f"| `{task}` | — | — | — | — | — | — | "
                        f"{v.get('skipped') or v.get('error')} |")
                    continue
                s_, r_ = v["served"], v["raw"]
                add(f"| `{task}` | {v['n']} | {_pct(v['base_rate'], 2)} | "
                    f"{_pct(s_['mean_predicted'], 2)} | {_num(s_['brier'], 4)} | "
                    f"{_num(s_['ece'], 4)} | {_num(r_['brier'], 4)} | "
                    f"{_num(r_['ece'], 4)} |")
            add("")
            add("*AUROC says the ranking is right; ECE says the number means "
                "what it says. Both matter, and only the first was measured "
                "before this report.*")
            add("")

    return "\n".join(L) + "\n"


# ══ main ═════════════════════════════════════════════════════════════════════

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-models", action="store_true",
                    help="skip suites that need the boosters")
    ap.add_argument("--quick", action="store_true",
                    help="skip calibration (loads the full test split)")
    ap.add_argument("--out", default=str(REPORT))
    args = ap.parse_args(argv)

    from src.assistant.orchestrator import Assistant

    pipeline = runner = None
    if not args.no_models:
        try:
            from src.llm.model_runner import LiveModelRunner
            from src.llm.pipeline import ClinicalReportPipeline
            runner = LiveModelRunner()
            pipeline = ClinicalReportPipeline(model_runner=runner)
        except Exception as exc:
            print(f"[warn] models unavailable: {exc}")

    def factory():
        return Assistant.clinician(model_runner=runner, pipeline=pipeline)

    results: Dict[str, Any] = {}
    for name, fn in (("retrieval", eval_retrieval),
                     ("intent", eval_intent),
                     ("extraction", eval_extraction),
                     ("abstention", lambda: eval_abstention(factory)),
                     ("faithfulness", lambda: eval_faithfulness(factory))):
        t0 = time.perf_counter()
        print(f"[{name}] running…", flush=True)
        results[name] = fn()
        print(f"[{name}] done in {time.perf_counter() - t0:.1f}s", flush=True)

    if not args.quick and not args.no_models:
        print("[calibration] running…", flush=True)
        results["calibration"] = eval_calibration()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(results), encoding="utf-8")
    print(f"\nWrote {out}")

    # A compact console summary, so a failure is visible without opening the file.
    a = results.get("abstention") or {}
    x = results.get("extraction") or {}
    print("\n── headline ──")
    print(f"  retrieval MRR          {_num((results.get('retrieval') or {}).get('mrr'))}")
    print(f"  intent accuracy        {_pct((results.get('intent') or {}).get('accuracy'))}")
    print(f"  fabrication rate       {_pct(x.get('fabrication_rate'))}")
    print(f"  under-refusal rate     {_pct(a.get('under_refusal_rate'))}")
    print(f"  faithfulness verified  "
          f"{_pct((results.get('faithfulness') or {}).get('verified_rate'))}")

    # Non-zero exit on the two conditions that are defects rather than scores.
    bad = []
    if x.get("fabrication_rate"):
        bad.append("fabrication rate is non-zero")
    if a.get("under_refusal_rate"):
        bad.append("under-refusal rate is non-zero")
    if bad:
        print("\nFAILED: " + "; ".join(bad))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
