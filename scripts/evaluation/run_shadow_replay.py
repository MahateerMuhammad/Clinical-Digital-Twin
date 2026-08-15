#!/usr/bin/env python
"""
scripts/evaluation/run_shadow_replay.py
───────────────────────────────────────
Behavioural regression detection by replay and diff.

    # once, when behaviour is known good
    PYTHONPATH=. .venv/bin/python scripts/evaluation/run_shadow_replay.py --freeze

    # after any change
    PYTHONPATH=. .venv/bin/python scripts/evaluation/run_shadow_replay.py

Baseline: ``tests/gold/shadow_baseline.json``.
Report:   ``reports/shadow_replay.md``.  Exit code 1 on a critical diff.

What this catches that the gold sets do not
───────────────────────────────────────────
A gold set asserts what you thought to assert. This asserts **everything else**:
if a turn that answered yesterday refuses today, or loses a citation, or routes
to a different intent, the diff shows it — even where no test covers that
behaviour and nobody predicted it would change.

That matters most for changes that are correct in isolation and surprising in
aggregate. Raising an intent rule's weight to fix one misroute is a two-line
edit with no test in its way; whether it silently moved eleven other messages
is not visible from the diff of the source.

Severity, and why the ordering is not symmetric
───────────────────────────────────────────────
``CRITICAL``  verification passed and now fails, or a turn that refused now
              answers. The second is under-refusal — the unsafe direction, and
              the one the completeness gate exists to prevent.
``WARNING``   a turn that answered now refuses (new friction), the intent
              changed, or citations were lost.
``INFO``      the wording moved but every structural outcome held.

Deliberately *not* compared: the exact prose. Comparing full text would flag
every config edit and train its owner to ignore the report — the failure mode of
a check that cries wolf. A length delta is recorded so a large rewrite is still
visible as INFO.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath("."))

GOLD = Path("tests/gold")
BASELINE = GOLD / "shadow_baseline.json"
REPORT = Path("reports/shadow_replay.md")

CRITICAL, WARNING, INFO = "CRITICAL", "WARNING", "INFO"


def scenarios() -> List[Dict[str, Any]]:
    """
    The replay set: every abstention case plus every guideline query.

    Reusing the gold sets rather than inventing a third corpus keeps one
    definition of "the traffic we care about". They are replayed here for a
    different purpose — the gold sets ask *is this right*, the replay asks *is
    this the same*, and a change can be a regression without being wrong.
    """
    out: List[Dict[str, Any]] = []
    with (GOLD / "abstention_gold.json").open(encoding="utf-8") as fh:
        for c in json.load(fh)["cases"]:
            out.append({"id": f"abst:{c['id']}", "turns": c["turns"]})
    with (GOLD / "guideline_retrieval_gold.json").open(encoding="utf-8") as fh:
        for q in json.load(fh)["queries"]:
            out.append({"id": f"guide:{q['qid']}", "turns": [q["query"]]})
    with (GOLD / "clinician_intent_gold.json").open(encoding="utf-8") as fh:
        for i, c in enumerate(json.load(fh)["cases"]):
            out.append({"id": f"intent:{i:02d}", "turns": [c["message"]]})
    return out


def capture(assistant_factory) -> Dict[str, Any]:
    """Run every scenario and record its structural outcome."""
    runs: Dict[str, Any] = {}
    for sc in scenarios():
        bot = assistant_factory()
        sid = bot.start().state.session_id
        result = None
        for turn in sc["turns"]:
            result = bot.handle(sid, turn)
        if result is None:
            continue
        ans = result.answer
        runs[sc["id"]] = {
            "status": result.status,
            "intent": result.state.intent,
            "gate_status": result.gate.status if result.gate else None,
            "blocking_fields": sorted(result.gate.blocking_fields)
            if result.gate else [],
            "verified": (result.faithfulness.ok
                         if result.faithfulness else None),
            "n_citations": len(ans.citations) if ans else 0,
            "sources": sorted(str(d.get("doc_id", "")) for d in ans.documents)
            if ans else [],
            "reply_len": len(result.reply or ""),
        }
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_scenarios": len(runs),
        "runs": runs,
    }


def _severity(before: Dict[str, Any], after: Dict[str, Any]) -> List[Dict[str, str]]:
    diffs: List[Dict[str, str]] = []

    b_answered = before["status"] == "answered"
    a_answered = after["status"] == "answered"

    if before.get("verified") is True and after.get("verified") is False:
        diffs.append({"severity": CRITICAL, "field": "verified",
                      "detail": "verification passed before and fails now"})
    if not b_answered and a_answered:
        diffs.append({"severity": CRITICAL, "field": "status",
                      "detail": f"now answers where it refused "
                                f"({before['status']} → {after['status']})"})
    if b_answered and not a_answered:
        diffs.append({"severity": WARNING, "field": "status",
                      "detail": f"now refuses where it answered "
                                f"({before['status']} → {after['status']})"})
    if before["intent"] != after["intent"]:
        diffs.append({"severity": WARNING, "field": "intent",
                      "detail": f"{before['intent']} → {after['intent']}"})
    if before["n_citations"] and after["n_citations"] < before["n_citations"]:
        diffs.append({"severity": WARNING, "field": "citations",
                      "detail": f"{before['n_citations']} → {after['n_citations']}"})
    if before["sources"] != after["sources"]:
        diffs.append({"severity": INFO, "field": "sources",
                      "detail": f"{before['sources']} → {after['sources']}"})
    if before["blocking_fields"] != after["blocking_fields"]:
        diffs.append({"severity": INFO, "field": "blocking_fields",
                      "detail": f"{before['blocking_fields']} → "
                                f"{after['blocking_fields']}"})
    if abs(before["reply_len"] - after["reply_len"]) > 40:
        diffs.append({"severity": INFO, "field": "reply_len",
                      "detail": f"{before['reply_len']} → {after['reply_len']} chars"})
    return diffs


def compare(base: Dict[str, Any], now: Dict[str, Any]) -> Dict[str, Any]:
    b_runs, n_runs = base["runs"], now["runs"]
    changed: Dict[str, List[Dict[str, str]]] = {}
    for key in sorted(set(b_runs) & set(n_runs)):
        diffs = _severity(b_runs[key], n_runs[key])
        if diffs:
            changed[key] = diffs

    counts = {CRITICAL: 0, WARNING: 0, INFO: 0}
    for diffs in changed.values():
        for d in diffs:
            counts[d["severity"]] += 1

    return {
        "baseline_captured_at": base.get("captured_at"),
        "n_compared": len(set(b_runs) & set(n_runs)),
        "added": sorted(set(n_runs) - set(b_runs)),
        "removed": sorted(set(b_runs) - set(n_runs)),
        "changed": changed,
        "counts": counts,
        "n_unchanged": len(set(b_runs) & set(n_runs)) - len(changed),
    }


def render(cmp: Dict[str, Any]) -> str:
    L: List[str] = []
    add = L.append
    c = cmp["counts"]

    add("# Shadow Replay")
    add("")
    add(f"*Generated {date.today().isoformat()} by "
        "`scripts/evaluation/run_shadow_replay.py`.*")
    add("")
    add(f"Baseline captured `{cmp['baseline_captured_at']}` · "
        f"{cmp['n_compared']} scenarios replayed · "
        f"**{cmp['n_unchanged']} unchanged**")
    add("")
    add(f"- **{c[CRITICAL]} critical** · {c[WARNING]} warning · {c[INFO]} info")
    add("")

    if not cmp["changed"] and not cmp["added"] and not cmp["removed"]:
        add("No behavioural change. Every scenario produced the same status, "
            "intent, gate outcome, verification verdict and citation set as the "
            "baseline.")
        add("")
        return "\n".join(L) + "\n"

    if cmp["added"]:
        add(f"**New scenarios** (no baseline): {', '.join(cmp['added'][:10])}")
        add("")
    if cmp["removed"]:
        add(f"**Missing scenarios**: {', '.join(cmp['removed'][:10])}")
        add("")

    for sev in (CRITICAL, WARNING, INFO):
        rows = [(k, d) for k, diffs in cmp["changed"].items()
                for d in diffs if d["severity"] == sev]
        if not rows:
            continue
        add(f"## {sev} ({len(rows)})")
        add("")
        add("| Scenario | Field | Change |")
        add("| :--- | :--- | :--- |")
        for key, d in rows:
            add(f"| `{key}` | {d['field']} | {d['detail']} |")
        add("")

    if c[CRITICAL]:
        add("---")
        add("")
        add("**A critical diff means one of two things happened:** verification "
            "stopped passing on output that previously passed, or a turn that "
            "refused now answers. The second is under-refusal — the system "
            "producing a substantive answer without the information it "
            "previously judged necessary — and it is the failure the "
            "completeness gate exists to prevent. Neither should be accepted "
            "without a deliberate reason, and if the new behaviour is correct "
            "the baseline should be re-frozen so the change is recorded rather "
            "than merely tolerated.")
        add("")
    return "\n".join(L) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--freeze", action="store_true",
                    help="capture the current behaviour as the new baseline")
    ap.add_argument("--baseline", default=str(BASELINE))
    ap.add_argument("--out", default=str(REPORT))
    ap.add_argument("--no-models", action="store_true")
    args = ap.parse_args(argv)

    from src.assistant.orchestrator import Assistant

    runner = pipeline = None
    if not args.no_models:
        try:
            from src.llm.model_runner import LiveModelRunner
            from src.llm.pipeline import ClinicalReportPipeline
            runner = LiveModelRunner()
            pipeline = ClinicalReportPipeline(model_runner=runner)
        except Exception as exc:
            print(f"[warn] models unavailable: {exc}")

    def factory():
        # A fresh in-memory audit per run: the replay must not pollute the
        # serve-time telemetry log with synthetic traffic, which would make
        # the two reports describe different populations.
        from src.assistant.audit import AuditLog
        return Assistant.clinician(model_runner=runner, pipeline=pipeline,
                                   audit_log=AuditLog(path=None))

    print("[shadow] replaying…", flush=True)
    now = capture(factory)
    base_path = Path(args.baseline)

    if args.freeze:
        base_path.parent.mkdir(parents=True, exist_ok=True)
        base_path.write_text(json.dumps(now, indent=2) + "\n", encoding="utf-8")
        print(f"Froze {now['n_scenarios']} scenarios → {base_path}")
        return 0

    if not base_path.exists():
        print(f"No baseline at {base_path}. Run with --freeze first.")
        return 2

    with base_path.open(encoding="utf-8") as fh:
        base = json.load(fh)
    cmp = compare(base, now)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(cmp), encoding="utf-8")
    print(f"Wrote {out}")

    c = cmp["counts"]
    print(f"  {cmp['n_unchanged']}/{cmp['n_compared']} unchanged · "
          f"{c[CRITICAL]} critical · {c[WARNING]} warning · {c[INFO]} info")
    return 1 if c[CRITICAL] else 0


if __name__ == "__main__":
    raise SystemExit(main())
