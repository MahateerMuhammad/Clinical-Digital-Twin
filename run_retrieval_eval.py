#!/usr/bin/env python3
"""
run_retrieval_eval.py
─────────────────────
Evaluate RAG retrieval quality against the version-controlled gold sets and write
a Markdown report.

    python run_retrieval_eval.py
    python run_retrieval_eval.py --json          # machine-readable to stdout
    python run_retrieval_eval.py --fail-under 0.85

Runs fully offline — no gold-set evaluation touches the network.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Import the scientific stack before any torch stub (scipy probes torch.Tensor).
import scipy.stats  # noqa: E402,F401
import sklearn.feature_extraction.text  # noqa: E402,F401

if "torch" not in sys.modules:                      # noqa: E402
    try:
        import torch  # noqa: F401
    except ImportError:
        _t = types.ModuleType("torch")
        _t.Tensor = type("Tensor", (), {})
        _t.set_num_threads = lambda n: None
        _t.load = lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no ckpt"))
        sys.modules["torch"] = _t

from src.llm.retrieval_eval import (  # noqa: E402
    render_markdown_report,
    run_full_evaluation,
)

DEFAULT_OUT = ROOT / "reports" / "tables" / "rag_retrieval_evaluation.md"


def main() -> int:
    ap = argparse.ArgumentParser(description="RAG retrieval evaluation")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Markdown report path")
    ap.add_argument("--json", action="store_true", help="emit JSON to stdout instead")
    ap.add_argument("--fail-under", type=float, default=None,
                    help="exit non-zero if any headline metric falls below this")
    args = ap.parse_args()

    results = run_full_evaluation()

    if args.json:
        print(json.dumps({k: v.to_dict() for k, v in results.items()}, indent=2))
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(render_markdown_report(results), encoding="utf-8")

        print("\nRAG RETRIEVAL EVALUATION")
        print("=" * 62)
        for r in results.values():
            print(f"\n{r.name}  (n={r.n}, failures={len(r.failures)})")
            for k, v in r.metrics.items():
                print(f"    {k:42} {'n/a' if v is None else f'{v:.3f}'}")
        print(f"\nReport → {args.out}")

    headline = {
        "terminology": "accuracy",
        "guidelines": "ndcg@3",
        "relevance": "f1",
    }
    if args.fail_under is not None:
        for key, metric in headline.items():
            val = results[key].metrics.get(metric)
            if val is not None and val < args.fail_under:
                print(f"\nFAIL: {key}.{metric} = {val:.3f} < {args.fail_under}", file=sys.stderr)
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
