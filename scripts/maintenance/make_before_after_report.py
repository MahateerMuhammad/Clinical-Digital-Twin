#!/usr/bin/env python3
"""
scripts/maintenance/make_before_after_report.py
───────────────────────────
Build the paired pre/post correction comparison for the identifier fix.

Reads the metric tables snapshotted in ``reports/baseline_pre_id_fix/`` and the
freshly regenerated ones in ``reports/tables/``, and emits a side-by-side table.

Both sets are evaluated on the *same* held-out test patients, because
``patient_split.parquet`` was deliberately held fixed across the correction, so
the deltas reflect the data repair rather than a different split.

    python scripts/maintenance/make_before_after_report.py
"""


from __future__ import annotations


# ── repo-root bootstrap ──────────────────────────────────────────────────────
# These scripts live two levels below the project root. Python puts the *script's*
# directory on sys.path, not the working directory, so `import src...` would fail
# from here; and many of them address data with root-relative paths such as
# "models/" or "reports/tables/". Both are fixed by putting the root on the path
# and running from it, which makes execution identical from any directory.
import os as _os
import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
_os.chdir(_ROOT)
# ─────────────────────────────────────────────────────────────────────────────
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "reports" / "baseline_pre_id_fix"
CURRENT = ROOT / "reports" / "tables"
OUT = CURRENT / "model_comparison_before_after.md"

#: (file, row label substring, human name, metric column header)
#:
#: Phase 1 headlines **Run C** — the strict 24-hour observation window. That is the
#: figure used by ``models/best_models/README.md``, the Phase 6 sequence comparison
#: and the Phase 9 risk stratification. Run A retains post-hoc ICD codes and is an
#: upper bound under leakage, not a result; it is listed only so the leakage gap
#: stays visible.
TRACKED: List[Tuple[str, str, str, str]] = [
    ("mortality_model_comparison.md", "Run C (24h Window)",
     "**Phase 1 Mortality — Run C (headline, strict 24h)**", "AUROC"),
    ("mortality_model_comparison.md", "Run B (Leak-Free)",
     "Phase 1 Mortality — Run B (full-stay, leak-free)", "AUROC"),
    ("mortality_model_comparison.md", "Run A (With ICD)",
     "Phase 1 Mortality — Run A (_leaky upper bound, not a result_)", "AUROC"),
    ("readmission_model_comparison.md", "Run B (Strict 24h)",
     "**Phase 2 Readmission — strict 24h (headline)**", "AUROC"),
    ("icu_admission_model_comparison.md", "LightGBM", "Phase 3 ICU admission", "AUROC"),
    ("los_two_stage_results.md", "Hospital LOS", "Phase 4 Hospital LOS — Stage A", "AUROC"),
    ("deterioration_model_results.md", "LightGBM", "Phase 5 Deterioration", "AUROC"),
]


def parse_rows(path: Path) -> List[Dict[str, str]]:
    """Parse every markdown table row in a file into {header: cell}."""
    if not path.exists():
        return []
    rows: List[Dict[str, str]] = []
    headers: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip().startswith("|"):
            headers = []
            continue
        cells = [c.strip().strip("*`") for c in line.strip().strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):
            continue                      # separator row
        if not headers:
            headers = cells
            continue
        if len(cells) == len(headers):
            rows.append(dict(zip(headers, cells)))
    return rows


def _num(v: Optional[str]) -> Optional[float]:
    if not v:
        return None
    m = re.search(r"-?\d+\.\d+|-?\d+", str(v).replace(",", ""))
    return float(m.group()) if m else None


def find_metric(rows: List[Dict[str, str]], row_key: str, metric: str,
                model: str = "LightGBM") -> Optional[float]:
    """Best matching cell: prefers the LightGBM row (the winning model)."""
    best: Optional[float] = None
    for r in rows:
        joined = " ".join(r.values())
        if row_key.lower() not in joined.lower():
            continue
        col = next((k for k in r if metric.lower() in k.lower()), None)
        if not col:
            continue
        val = _num(r.get(col))
        if val is None:
            continue
        if model.lower() in joined.lower():
            return val
        if best is None:
            best = val
    return best


def main() -> int:
    if not BASELINE.exists():
        print(f"No baseline snapshot at {BASELINE} — cannot compare.", file=sys.stderr)
        return 1

    lines: List[str] = []
    lines.append("# Model Performance — Before vs After the Identifier Correction\n")
    lines.append(
        "Both columns are evaluated on the **same held-out test patients**: "
        "`patient_split.parquet` was held fixed across the correction, so these "
        "deltas isolate the effect of repairing the laboratory join.\n"
    )
    lines.append(
        "> [!NOTE]\n"
        "> Before the correction, 50.1% of admissions (every odd `hadm_id`) carried "
        "> **no laboratory features at all**, and some even-ID admissions held labs "
        "> rounded onto them from a neighbouring admission. See "
        "> [`data_correction_notice.md`](../data_correction_notice.md).\n"
    )

    lines.append("\n| Model | Metric | Before | After | Δ |")
    lines.append("| :--- | :--- | ---: | ---: | ---: |")

    any_found = False
    for fname, row_key, label, metric in TRACKED:
        before = find_metric(parse_rows(BASELINE / fname), row_key, metric)
        after = find_metric(parse_rows(CURRENT / fname), row_key, metric)
        if before is None and after is None:
            continue
        any_found = True
        b = f"{before:.4f}" if before is not None else "—"
        a = f"{after:.4f}" if after is not None else "—"
        if before is not None and after is not None:
            d = after - before
            delta = f"{d:+.4f}"
            if abs(d) >= 0.01:
                delta += " ⚠" if d < 0 else " ↑"
        else:
            delta = "—"
        lines.append(f"| {label} | {metric} | {b} | {a} | {delta} |")

    if not any_found:
        lines.append("| _no comparable rows found_ | | | | |")

    lines.append(
        "\n## Interpretation\n\n"
        "A metric that **falls** after correction is the expected outcome if the "
        "missing-laboratory pattern was itself carrying signal — absence of lab "
        "records correlates with shorter, less acute admissions, which a model can "
        "exploit without any physiological information.\n\n"
        "A metric that **rises** indicates the models were previously starved of "
        "genuine laboratory signal for half the cohort.\n\n"
        "Either direction is a legitimate finding and should be reported as such. "
        "The corrected figures supersede the baseline in all cases.\n"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWritten → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
