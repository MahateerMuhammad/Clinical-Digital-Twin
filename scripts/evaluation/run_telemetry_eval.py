#!/usr/bin/env python
"""
scripts/evaluation/run_telemetry_eval.py
────────────────────────────────────────
Serve-time telemetry from the audit log.

    PYTHONPATH=. .venv/bin/python scripts/evaluation/run_telemetry_eval.py
    … --log logs/assistant_audit.jsonl
    … --since 2026-08-13        only records on or after this date

Writes ``reports/serve_time_telemetry.md``.

This is not online evaluation, and calling it that would be wrong
─────────────────────────────────────────────────────────────────
Online evaluation measures a system against **real users**: their traffic
distribution, their behaviour, and outcome signals — did the clinician act on
the answer, did they abandon the conversation, did the advice change a decision.
None of that is available here, and no amount of instrumentation substitutes for
it. Presenting these numbers as online evaluation would invite the question "how
many clinicians?" and deserve it.

What this *is* is the measurement layer such an evaluation would run on, applied
to whatever traffic exists — development sessions, demos, the curl transcript.
The metrics are computed exactly as they would be in production, from records
written at serve time rather than from a curated gold set. That distinction is
the point: an offline suite measures what you thought to ask about, this
measures what actually happened.

Two properties it has that the offline suite cannot
───────────────────────────────────────────────────
* **Unfiltered distribution.** The gold sets contain the phrasings someone
  thought of. The log contains what was typed, including the messages that
  classified as `unknown` — which is where the next gold case comes from.
* **Guardrail rates over real turns.** Refusal rate, verification pass rate and
  retrieval hit rate are computed on the same records the user was served, so a
  regression shows up in traffic rather than only in a fixture.

Redaction
─────────
The audit redacts free-text patient content by default. Every field this reads —
status, intent, gate outcome, verification result, retrieved doc ids, latency —
survives redaction, so telemetry works on a log safe to keep.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath("."))

DEFAULT_LOG = Path("logs/assistant_audit.jsonl")
REPORT = Path("reports/serve_time_telemetry.md")

#: Statuses that mean a substantive answer reached the caller.
ANSWERED = {"answered"}
#: Statuses that are a deliberate refusal rather than a failure.
REFUSED = {"declined_incomplete", "declined_no_evidence", "declined_unreviewed",
           "withheld_failed_verification"}


def load(path: Path, since: Optional[str]) -> List[dict]:
    if not path.exists():
        return []
    rows: List[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue                      # a truncated tail must not kill the run
        if since and str(rec.get("timestamp", "")) < since:
            continue
        rows.append(rec)
    return rows


def _percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def analyse(rows: List[dict]) -> Dict[str, Any]:
    turns = [r for r in rows if int(r.get("turn") or 0) > 0]
    statuses = Counter(str(r.get("status") or "") for r in turns)
    intents = Counter(str(r.get("intent") or "none") for r in turns)

    answered = [r for r in turns if r.get("status") in ANSWERED]
    refused = [r for r in turns if r.get("status") in REFUSED]

    # Verification is only meaningful where an answer was composed.
    verdicts = [bool((r.get("validation") or {}).get("ok"))
                for r in answered if r.get("validation")]

    # A pass rate below 100% is an incident, not a score, so the failing turns
    # are listed with their timestamps. Clustering in time is the signal: six
    # failures spread over a month is a flaky edge case, six inside thirteen
    # minutes is a regression that was introduced and then fixed, and only the
    # timestamps tell them apart.
    failures = []
    for r in answered:
        val = r.get("validation") or {}
        if val and not val.get("ok"):
            failures.append({
                "timestamp": r.get("timestamp"),
                "intent": r.get("intent"),
                "checks": [f"{c.get('check')}:{c.get('name')}"
                           for c in val.get("checks", [])
                           if not c.get("passed") and c.get("blocking")],
            })
    failures.sort(key=lambda f: str(f["timestamp"]))

    retrieved = [len(r.get("retrieved_sources") or []) for r in answered]
    latencies = [float(r["latency_ms"]) for r in turns
                 if r.get("latency_ms") is not None]

    # A turn that asked for something is a turn the gate closed on. The fields
    # asked for most often are where a real user is most likely to give up, and
    # are the highest-value target for better extraction.
    asked = Counter()
    for r in turns:
        for fld in (r.get("missing_information") or []):
            asked[str(fld)] += 1

    gate_status = Counter(str((r.get("gate") or {}).get("status") or "none")
                          for r in turns)

    sessions: Dict[str, int] = {}
    for r in turns:
        sid = str(r.get("session_id"))
        sessions[sid] = max(sessions.get(sid, 0), int(r.get("turn") or 0))

    return {
        "n_records": len(rows),
        "n_turns": len(turns),
        "n_sessions": len(sessions),
        "turns_per_session_mean": (sum(sessions.values()) / len(sessions))
        if sessions else None,
        "status_counts": dict(statuses.most_common()),
        "intent_counts": dict(intents.most_common()),
        "gate_status_counts": dict(gate_status.most_common()),
        "answer_rate": (len(answered) / len(turns)) if turns else None,
        "refusal_rate": (len(refused) / len(turns)) if turns else None,
        "verification": {
            "n": len(verdicts),
            "pass_rate": (sum(verdicts) / len(verdicts)) if verdicts else None,
            "failures": failures,
            "first_failure": failures[0]["timestamp"] if failures else None,
            "last_failure": failures[-1]["timestamp"] if failures else None,
        },
        "retrieval": {
            "n_answers": len(answered),
            "hit_rate": (sum(1 for c in retrieved if c > 0) / len(retrieved))
            if retrieved else None,
            "mean_docs": (sum(retrieved) / len(retrieved)) if retrieved else None,
        },
        "latency_ms": {
            "n": len(latencies),
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "max": max(latencies) if latencies else None,
        },
        "most_asked_fields": asked.most_common(8),
        "unknown_intent_rate": (intents.get("unknown", 0) / len(turns))
        if turns else None,
    }


def _pct(v, p=1):
    return "n/a" if v is None else f"{v * 100:.{p}f}%"


def _num(v, p=1):
    return "n/a" if v is None else f"{v:.{p}f}"


def render(t: Dict[str, Any], log: Path) -> str:
    L: List[str] = []
    add = L.append

    add("# Serve-Time Telemetry")
    add("")
    add(f"*Generated {date.today().isoformat()} by "
        f"`scripts/evaluation/run_telemetry_eval.py` from `{log}`.*")
    add("")
    add("> **This is not online evaluation.** Online evaluation measures a "
        "system against real users — their traffic, their behaviour, and "
        "outcome signals such as whether the answer was acted on. No clinician "
        "traffic exists yet, so none of that is measured here.")
    add(">")
    add("> What this is: the same measurement layer an online evaluation would "
        "run on, applied to the traffic that does exist. The metrics are "
        "computed from records written at serve time rather than from a curated "
        "gold set, which is why they can show things the offline suite cannot — "
        "notably the messages nobody thought to write a fixture for.")
    add("")

    if not t["n_turns"]:
        add("*No turns recorded yet. Run the backend or the evaluation suite "
            "first.*")
        return "\n".join(L) + "\n"

    add("## Volume")
    add("")
    add(f"- Records: **{t['n_records']}** · turns: **{t['n_turns']}** · "
        f"sessions: **{t['n_sessions']}**")
    add(f"- Mean turns per session: **{_num(t['turns_per_session_mean'], 2)}**")
    add("")

    add("## Guardrail rates")
    add("")
    add(f"- Answer rate: **{_pct(t['answer_rate'])}** · "
        f"refusal rate: **{_pct(t['refusal_rate'])}**")
    v = t["verification"]
    add(f"- Verification pass rate: **{_pct(v['pass_rate'])}** "
        f"(over {v['n']} composed answers)")
    r = t["retrieval"]
    add(f"- Retrieval hit rate: **{_pct(r['hit_rate'])}** · "
        f"mean documents per answer: **{_num(r['mean_docs'], 2)}**")
    add("")
    if v.get("failures"):
        add("")
        add(f"### Verification failures ({len(v['failures'])})")
        add("")
        add(f"Between `{v['first_failure']}` and `{v['last_failure']}`.")
        add("")
        add("| Time | Intent | Failed checks |")
        add("| :--- | :--- | :--- |")
        for f in v["failures"]:
            add(f"| {f['timestamp']} | `{f['intent']}` | "
                f"{', '.join(f['checks']) or '—'} |")
        add("")
        add("*Clustering in time is the signal. Failures spread evenly are a "
            "flaky edge case; failures packed into a short window are a "
            "regression that was introduced and then fixed — and this is the "
            "view that distinguishes them. Re-run with `--since` after the last "
            "failure to see current health.*")
        add("")

    add("*A high refusal rate is not a fault. Most turns in a completeness-gated "
        "conversation are the system asking for what it needs; the number to "
        "watch is verification pass rate, where anything below 100% means "
        "composed output was withheld.*")
    add("")

    add("## Latency")
    add("")
    lat = t["latency_ms"]
    add(f"- p50 **{_num(lat['p50'])} ms** · p95 **{_num(lat['p95'])} ms** · "
        f"max **{_num(lat['max'])} ms** (n={lat['n']})")
    add("")
    if lat["max"] and lat["max"] > 10_000:
        add("*The tail is live evidence retrieval on a cold cache — PubMed and "
            "DailyMed are fetched over the network the first time a topic is "
            "seen, then served from a 7-day disk cache. It is bounded, not a "
            "hang, but a first-time query is slow enough to notice.*")
        add("")

    add("## Intent distribution")
    add("")
    add("| Intent | Turns |")
    add("| :--- | ---: |")
    for name, count in t["intent_counts"].items():
        add(f"| `{name}` | {count} |")
    add("")
    if t["unknown_intent_rate"]:
        add(f"*`unknown` at {_pct(t['unknown_intent_rate'])} of turns. These are "
            "the phrasings no rule covers, and they are the highest-value "
            "source of new gold cases — the offline suite cannot discover them "
            "because it only contains what someone already thought to write.*")
        add("")

    add("## Gate outcomes")
    add("")
    add("| Gate status | Turns |")
    add("| :--- | ---: |")
    for name, count in t["gate_status_counts"].items():
        add(f"| `{name}` | {count} |")
    add("")

    if t["most_asked_fields"]:
        add("## Most-requested fields")
        add("")
        add("| Field | Times asked |")
        add("| :--- | ---: |")
        for name, count in t["most_asked_fields"]:
            add(f"| `{name}` | {count} |")
        add("")
        add("*Where a real user is most likely to give up, and therefore the "
            "highest-value target for better extraction.*")
        add("")

    add("## Status breakdown")
    add("")
    add("| Status | Turns |")
    add("| :--- | ---: |")
    for name, count in t["status_counts"].items():
        add(f"| `{name}` | {count} |")
    add("")
    return "\n".join(L) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", default=str(DEFAULT_LOG))
    ap.add_argument("--since", default=None, help="ISO timestamp prefix")
    ap.add_argument("--out", default=str(REPORT))
    args = ap.parse_args(argv)

    log = Path(args.log)
    rows = load(log, args.since)
    t = analyse(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(t, log), encoding="utf-8")
    print(f"Wrote {out}")
    print(f"  turns {t['n_turns']} · sessions {t['n_sessions']} · "
          f"answer rate {_pct(t['answer_rate'])} · "
          f"verification {_pct(t['verification']['pass_rate'])} · "
          f"p95 {_num(t['latency_ms']['p95'])} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
