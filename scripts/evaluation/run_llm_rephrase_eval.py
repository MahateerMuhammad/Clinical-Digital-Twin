#!/usr/bin/env python
"""
Measure what the optional LLM rephrase stage actually buys, and what it costs.

This is step 5 of `reports/llm_layer_design.md` §5 — the LLM layer's headline
metric — and it is deliberately built *before* steps 2-4. Fine-tuning is only worth
three GPU hours if a stock instruct model fails the verifier often enough to matter,
and nobody has measured that. Running this first turns "should we fine-tune?" from a
judgement call into a number.

What is measured
────────────────
Every held-out case is generated twice: once with the LLM disabled (the
deterministic composer, which is what ships today) and once with it enabled. Because
the pipeline fails closed, three outcomes are possible:

    deterministic              the LLM was never consulted
    llm_rephrased_verified     the LLM rewrote it and every fact traced
    deterministic_llm_rejected the LLM rewrote it, the verifier caught an invention,
                               and the deterministic text was returned instead

The headline is the **verifier pass rate**: of the generations where the LLM was
consulted, the share that survived. A low rate is not a safety problem — the
deterministic text ships either way — it is a *usefulness* problem, because the LLM
stage is doing nothing but adding latency.

The second question is whether the rephrase is even an improvement. A model that
passes the verifier by returning its input unchanged scores 100% and is worthless,
so readability change and edit distance are reported alongside. Without them the
headline number cannot distinguish "safe and useful" from "safe and inert".

Usage
─────
    python scripts/evaluation/run_llm_rephrase_eval.py --backend null
    python scripts/evaluation/run_llm_rephrase_eval.py --backend ollama --cases 50
    python scripts/evaluation/run_llm_rephrase_eval.py --backend transformers \\
        --adapter models/adapters/qwen3b-cdt

`--backend null` establishes the floor: no LLM is consulted, so the pass rate is
undefined and every case is deterministic. That run is still worth doing — it proves
the harness reports "no LLM" rather than silently reporting success.
"""

from __future__ import annotations

import os as _os
import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import argparse
import difflib
import random
import re
import statistics
from collections import Counter
from datetime import datetime, timezone

import pandas as pd

from src.llm.backends import get_backend
from src.llm.model_runner import LiveModelRunner
from src.llm.pipeline import ClinicalReportPipeline

# The payload builder is imported, not reimplemented. A second copy of the field
# mapping is how the fidelity harness drifted from production; see
# reports/data_correction_notice.md.
from scripts.evaluation.run_llm_end_to_end_eval import DIAGNOSES, build_payload

ROOT = _ROOT
OUT = ROOT / "reports/tables/llm_rephrase_evaluation.md"


# ── readability ───────────────────────────────────────────────────────────────

def _flesch(text: str) -> float:
    """
    Flesch Reading Ease, computed here rather than imported.

    `textstat` is an optional dependency in this project and was made optional
    deliberately (it dominated pipeline runtime). A ~15-line implementation avoids
    reintroducing it for a reporting nicety. Higher is easier; clinical prose
    typically lands in the 20-40 band.
    """
    words = re.findall(r"[A-Za-z]+", text)
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    if not words or not sentences:
        return 0.0

    def syllables(word: str) -> int:
        word = word.lower()
        groups = re.findall(r"[aeiouy]+", word)
        n = len(groups)
        if word.endswith("e") and n > 1:
            n -= 1
        return max(1, n)

    total_syl = sum(syllables(w) for w in words)
    return (206.835
            - 1.015 * (len(words) / len(sentences))
            - 84.6 * (total_syl / len(words)))


def _similarity(a: str, b: str) -> float:
    """1.0 means the rephrase returned its input unchanged."""
    return difflib.SequenceMatcher(None, a, b).ratio()


# ── cohort ────────────────────────────────────────────────────────────────────

def load_cases(n: int, seed: int) -> pd.DataFrame:
    """
    Held-out admissions only.

    The test split, not train: if steps 2-4 are ever run, the adapter will be fitted
    on train-split reports, and measuring it on those would report memorisation as
    fidelity. Holding this harness to the test split from the outset means the number
    stays comparable before and after any fine-tune.
    """
    adm = pd.read_parquet(ROOT / "data/processed/admission_level_selected.parquet")
    if "split" not in adm.columns:
        split = pd.read_parquet(ROOT / "data/processed/patient_split.parquet")
        adm = adm.merge(split[["subject_id", "split"]], on="subject_id", how="left")
    held_out = adm[adm["split"] == "test"]
    if held_out.empty:
        raise SystemExit("No test-split admissions found.")
    return held_out.sample(min(n, len(held_out)), random_state=seed)


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(cases: pd.DataFrame, backend_name: str, adapter: str | None,
             seed: int) -> dict:
    runner = LiveModelRunner()
    backend = get_backend(backend_name, **({"adapter_path": adapter} if adapter else {}))
    pipeline = ClinicalReportPipeline(model_runner=runner, llm_backend=backend)
    rng = random.Random(seed)

    modes: Counter = Counter()
    violations: Counter = Counter()
    rows = []

    for _, row in cases.iterrows():
        payload, _ = build_payload(row, rng.choice(DIAGNOSES))

        base = pipeline.generate(payload, use_llm=False)
        if base.status != "ok" or not base.report_markdown:
            modes["skipped_not_ok"] += 1
            continue

        result = pipeline.generate(payload, use_llm=True)
        modes[result.generation_mode] += 1
        for v in result.grounding.get("violations", []):
            violations[f"{v.get('severity', '?')}/{v.get('kind', '?')}"] += 1

        rows.append({
            "mode": result.generation_mode,
            "flesch_before": _flesch(base.report_markdown),
            "flesch_after": _flesch(result.report_markdown),
            "similarity": _similarity(base.report_markdown, result.report_markdown),
            "chars_before": len(base.report_markdown),
            "chars_after": len(result.report_markdown),
            "ms": result.timings_ms.get("llm", 0.0),
        })

    return {
        "backend": backend.describe(),
        "modes": modes,
        "violations": violations,
        "rows": rows,
        "n": len(cases),
    }


def _median(values) -> float:
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else float("nan")


def render(res: dict, seed: int) -> str:
    rows = res["rows"]
    modes = res["modes"]
    consulted = modes["llm_rephrased_verified"] + modes["deterministic_llm_rejected"]
    verified = modes["llm_rephrased_verified"]

    L = [
        "# LLM Rephrase Layer — Verifier Pass Rate",
        "",
        "> [!NOTE]",
        f"> Generated by `scripts/evaluation/run_llm_rephrase_eval.py`, seed {seed}, "
        f"on {len(rows)} held-out **test**-split admissions.",
        "> This is step 5 of `reports/llm_layer_design.md` §5 — the LLM layer's",
        "> headline metric. It is measured before steps 2-4 so that the decision to",
        "> fine-tune is made from a number rather than an assumption.",
        "",
        f"Backend: `{res['backend']}`",
        "",
        "## 1. Verdict",
        "",
    ]

    if consulted == 0:
        L += [
            "**No LLM was consulted.** Every case was served by the deterministic",
            "composer, which is the shipping configuration. The pass rate is undefined",
            "rather than 100% — there were no generations to verify.",
            "",
            "This is the correct result for `--backend null`, and the reason that run",
            "exists: a harness that reported success here would be measuring nothing.",
            "",
        ]
    else:
        rate = 100.0 * verified / consulted
        sim = _median(r["similarity"] for r in rows
                      if r["mode"] == "llm_rephrased_verified")
        L += [
            f"**Verifier pass rate: {rate:.1f}%** ({verified} of {consulted} "
            "generations survived grounding).",
            "",
            "| Measure | Value | Reading |",
            "| :--- | ---: | :--- |",
            f"| Pass rate | {rate:.1f}% | share of LLM outputs that shipped |",
            f"| Median similarity to input | {sim:.3f} | 1.000 means unchanged |",
            f"| Median readability change | "
            f"{_median(r['flesch_after'] - r['flesch_before'] for r in rows):+.1f} | "
            "Flesch points; higher is easier |",
            f"| Median latency | {_median(r['ms'] for r in rows):.0f} ms | per report |",
            "",
        ]
        if sim > 0.98:
            L += [
                "> [!WARNING]",
                "> The rephrased text is near-identical to its input. A model that",
                "> passes by returning what it was given is safe and useless — the",
                "> pass rate alone would have hidden this, which is why similarity is",
                "> reported beside it.",
                "",
            ]

    L += ["## 2. Outcome distribution", "",
          "| Generation mode | N | Meaning |", "| :--- | ---: | :--- |"]
    meanings = {
        "deterministic": "LLM not consulted",
        "llm_rephrased_verified": "rewritten, every fact traced, shipped",
        "deterministic_llm_rejected": "rewritten, verifier caught an invention, discarded",
        "skipped_not_ok": "payload did not produce a baseline report",
    }
    for mode, count in modes.most_common():
        L.append(f"| `{mode}` | {count} | {meanings.get(mode, '')} |")

    L += ["", "## 3. Why rejections happened", ""]
    if res["violations"]:
        L += ["| Severity / kind | N |", "| :--- | ---: |"]
        L += [f"| `{k}` | {v} |" for k, v in res["violations"].most_common()]
        L += ["",
              "Every one of these was caught before reaching a caller. The",
              "deterministic text shipped in its place, so a rejection is a cost in",
              "latency, not a safety incident."]
    else:
        L.append("No grounding violations were recorded.")

    L += [
        "",
        "## 4. How to read this",
        "",
        "- A **high** pass rate with **low** similarity is the outcome worth having:",
        "  the model is rewriting substantively and inventing nothing.",
        "- A **high** pass rate with **high** similarity means the stage is inert.",
        "  Fine-tuning would not fix that; the prompt or the model would.",
        "- A **low** pass rate is not a safety failure. The architecture fails closed,",
        "  so the deterministic text is returned. It means the stage costs latency and",
        "  returns nothing, and is the case where fine-tuning on the output contract",
        "  (design §3) is the right remedy.",
        "",
        f"_Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC._",
        "",
    ]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "null", "ollama", "transformers"])
    ap.add_argument("--adapter", default=None, help="QLoRA adapter path")
    ap.add_argument("--cases", type=int, default=25)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    cases = load_cases(args.cases, args.seed)
    res = evaluate(cases, args.backend, args.adapter, args.seed)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(res, args.seed), encoding="utf-8")

    modes = res["modes"]
    consulted = modes["llm_rephrased_verified"] + modes["deterministic_llm_rejected"]
    print(f"\nbackend: {res['backend']}")
    for mode, count in modes.most_common():
        print(f"  {mode:30s} {count}")
    if consulted:
        print(f"\n  verifier pass rate: "
              f"{100.0 * modes['llm_rephrased_verified'] / consulted:.1f}%")
    else:
        print("\n  no LLM consulted — pass rate undefined")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
