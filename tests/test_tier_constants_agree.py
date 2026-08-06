"""
The served risk tiers must match the published ones.

Tier cutoffs and per-tier mortality rates are percentiles of one specific model's
test predictions, so every Phase 1 retrain invalidates them. They have gone stale
twice: once when the cutoffs were updated by hand and SYSTEM_CONSTANTS was not, and
once when `scripts/maintenance/recompute_risk_tiers.py --patch` silently failed because it was still
searching model_runner.py for literals that had moved to report_composer.

Both failures are invisible at runtime — patients get tiered against superseded
thresholds while the report quotes different numbers, and nothing raises. These tests
compare the constants against reports/tables/risk_stratification.md, which is written
from the model itself, so disagreement means one of them is out of date.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPORT = Path("reports/tables/risk_stratification.md")

pytestmark = pytest.mark.skipif(
    not REPORT.exists(),
    reason="risk_stratification.md not generated; run scripts/maintenance/recompute_risk_tiers.py --write-report",
)


def _published():
    """Parse cutoffs and observed per-tier mortality rates out of the report."""
    text = REPORT.read_text(encoding="utf-8")

    m = re.search(r"Probability cutoffs:\s*\*\*([\d.]+)\s*/\s*([\d.]+)\s*/\s*([\d.]+)\*\*", text)
    assert m, "could not find the cutoff line in risk_stratification.md"
    cutoffs = tuple(float(g) for g in m.groups())

    # Locate the header row so the Observed Mortality column is found by name rather
    # than by a hardcoded index that silently shifts when a column is added.
    lines = [l for l in text.splitlines() if l.strip().startswith("|")]
    header = next(l for l in lines if "Observed Mortality" in l)
    cols = [c.strip() for c in header.split("|")]
    idx = cols.index("Observed Mortality")

    rates = []
    for tier in ("Tier 1: Low Risk", "Tier 2: Moderate Risk",
                 "Tier 3: High Risk", "Tier 4: Extreme Risk"):
        row = next((l for l in lines if f"**{tier}**" in l), None)
        assert row, f"no table row for {tier}"
        cells = [c.strip() for c in row.split("|")]
        rates.append(float(cells[idx].replace("*", "").replace("%", "")))
    return cutoffs, rates


def test_cutoffs_match_the_published_report():
    from src.llm.report_composer import TIER_CUTOFFS

    published, _ = _published()
    assert tuple(round(c, 4) for c in TIER_CUTOFFS) == published, (
        f"TIER_CUTOFFS {TIER_CUTOFFS} disagrees with risk_stratification.md "
        f"{published}. Run: scripts/maintenance/recompute_risk_tiers.py --patch --write-report")


def test_observed_rates_match_the_published_report():
    from src.llm.report_composer import SYSTEM_CONSTANTS

    _, published = _published()
    constants = [SYSTEM_CONSTANTS[f"phase9_tier{i}_observed_mortality_pct"]
                 for i in range(1, 5)]
    assert constants == published, (
        f"SYSTEM_CONSTANTS rates {constants} disagree with risk_stratification.md "
        f"{published}. Run: scripts/maintenance/recompute_risk_tiers.py --patch --write-report")


def test_tier_boundaries_are_ordered_and_map_correctly():
    from src.llm.report_composer import TIER_CUTOFFS, tier_for_probability

    assert list(TIER_CUTOFFS) == sorted(TIER_CUTOFFS), "cutoffs must ascend"
    assert tier_for_probability(0.0).startswith("Tier 1")
    assert tier_for_probability(TIER_CUTOFFS[0]).startswith("Tier 2")
    assert tier_for_probability(TIER_CUTOFFS[1]).startswith("Tier 3")
    assert tier_for_probability(TIER_CUTOFFS[2]).startswith("Tier 4")
    assert tier_for_probability(1.0).startswith("Tier 4")


def test_model_runner_carries_no_copy_of_the_cutoffs():
    """The literals must exist in exactly one place."""
    import inspect
    from src.llm import model_runner
    from src.llm.report_composer import TIER_CUTOFFS

    src = inspect.getsource(model_runner)
    for cutoff in TIER_CUTOFFS:
        assert f"{cutoff}" not in src, (
            f"{cutoff} is hardcoded in model_runner.py; it must import "
            "tier_for_probability from report_composer instead")


def test_twin_retrieval_auroc_matches_its_report():
    """
    The retrieval AUROC quoted in the twin section must match its own report.

    It has now gone stale twice from the same cause — a literal in prompt_builder.py.
    First at 0.7253, which came from the superseded *unconditional* metric and so was
    never right; then again the moment the retrieval harness was re-run at a different
    query count. It is quoted to clinicians as evidence of how far to trust twin
    precedent, so a wrong value there is not cosmetic.
    """
    import re

    from src.llm.report_composer import SYSTEM_CONSTANTS

    report = Path("reports/tables/twin_retrieval_evaluation.md")
    if not report.exists():
        pytest.skip("twin_retrieval_evaluation.md not generated; run "
                    "scripts/evaluation/run_twin_retrieval_eval.py")

    text = report.read_text(encoding="utf-8")
    m = re.search(r"\|\s*Mortality AUROC\s*\|\s*\*\*([\d.]+)\*\*", text)
    assert m, "could not find the Mortality AUROC row in twin_retrieval_evaluation.md"
    assert float(m.group(1)) == SYSTEM_CONSTANTS["phase7_twin_retrieval_auroc"], (
        f"SYSTEM_CONSTANTS says {SYSTEM_CONSTANTS['phase7_twin_retrieval_auroc']}, the "
        f"report says {m.group(1)}. Re-run scripts/evaluation/run_twin_retrieval_eval.py "
        "and update the constant.")


def test_prompt_builder_holds_no_literal_retrieval_auroc():
    """The figure must be read from SYSTEM_CONSTANTS, not retyped."""
    import inspect

    from src.llm import prompt_builder
    from src.llm.report_composer import SYSTEM_CONSTANTS

    src = inspect.getsource(prompt_builder)
    auroc = SYSTEM_CONSTANTS["phase7_twin_retrieval_auroc"]
    # The value may appear inside a comment explaining the history, but never as the
    # formatted literal the report would print.
    assert f"AUROC {auroc:.4f} on" not in src, (
        "prompt_builder.py hardcodes the retrieval AUROC again; format it from "
        "SYSTEM_CONSTANTS so a re-run of the harness cannot leave it stale")


def test_tier_context_prose_uses_the_current_rates():
    """The prose quoted to clinicians must not lag the constants."""
    from src.llm.report_composer import SYSTEM_CONSTANTS, TIER_CONTEXT

    for i, text in enumerate(TIER_CONTEXT.values(), start=1):
        rate = SYSTEM_CONSTANTS[f"phase9_tier{i}_observed_mortality_pct"]
        assert f"{rate}%" in text, f"tier {i} prose quotes a stale rate: {text}"
