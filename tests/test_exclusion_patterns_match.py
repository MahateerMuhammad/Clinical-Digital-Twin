"""
Every leakage-exclusion pattern must be capable of matching something.

`ICU_ADMISSION_EXCLUDE` and `LOS_EXCLUDE_STRICT` listed `"fluids_*"` and
`"vitals_*"`. The built columns are `fluid_input_total` and `vital_heart_rate_mean`
— singular — so both patterns matched nothing and both exclusions were inert for
the life of the project.

Nothing leaked, only because the admission matrix happens to contain no such column.
Had vitals or fluids ever been merged, Phase 3 would have been predicting ICU
admission from features populated almost exclusively for ICU patients — the exact
availability leakage that forced the Phase 5 rebuild — with the guard silent.

That is the failure this file exists to prevent: a pattern that protects nothing
looks identical to one that protects everything. The namespace is assembled from
every feature set the project can build, not just the columns currently selected,
so a guard for a feature family stays verifiable while that family is unadopted.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.features import leakage_filters as lf
from src.features.leakage_filters import match_column_patterns

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pq = None

#: Exclusion lists that guard a trained phase.
EXCLUSION_LISTS = [
    name for name in dir(lf)
    if name.endswith(("_EXCLUDE", "_EXCLUDE_STRICT", "_EXCLUDE_PRIMARY",
                      "_EXCLUDE_RUN_B", "_EXCLUDE_RUN_C"))
    and isinstance(getattr(lf, name), list)
]

#: Patterns that legitimately match nothing today, with the reason.
#: Kept explicit so an exemption is a decision someone wrote down, not a silence.
KNOWN_UNMATCHED = {
    "icd_embedding_placeholder": "reserved name; no embedding column is built yet",
    "readmit_*": "defensive guard against a naming convention not currently produced",
    "ed_disposition": "prefixed form; the ED builder drops `disposition` before merge",
    "ed_outtime": "same — `outtime` is dropped, only `ed_los_hours` survives",
    "disposition": (
        "dropped at source by ED_FORBIDDEN_COLUMNS in src/features/emergency.py, so "
        "it never reaches a feature file. The exclusion is defence in depth: if that "
        "drop is ever removed, this pattern is what stops the ED outcome becoming a "
        "feature"),
    "news2_*": (
        "computed in-memory by src/models/deterioration.py (the superseded "
        "case-control model) and never persisted. Retained because NEWS2 is derived "
        "from vital_* and would reappear the moment vitals are adopted"),
}


def _namespace() -> set[str]:
    """Every column name the project can produce, across all feature sets."""
    names: set[str] = set()
    for path in Path("data/interim/features").glob("*.parquet"):
        names.update(f.name for f in pq.ParquetFile(path).schema_arrow)
    for name in ("admission_level", "admission_level_selected", "icu_level"):
        path = Path(f"data/processed/{name}.parquet")
        if path.exists():
            names.update(f.name for f in pq.ParquetFile(path).schema_arrow)
    return names


pytestmark = pytest.mark.skipif(
    pq is None or not Path("data/processed").exists(),
    reason="built feature sets not present in this checkout",
)


def test_the_namespace_is_actually_populated():
    """Guards the guard: an empty namespace would make every test below vacuous."""
    names = _namespace()
    assert len(names) > 200, f"only {len(names)} columns discovered; check the paths"


def test_every_exclusion_pattern_matches_something(subtests):
    names = sorted(_namespace())
    assert EXCLUSION_LISTS, "no exclusion lists discovered in leakage_filters"

    for list_name in sorted(EXCLUSION_LISTS):
        for pattern in getattr(lf, list_name):
            if pattern in KNOWN_UNMATCHED:
                continue
            with subtests.test(list=list_name, pattern=pattern):
                assert match_column_patterns(names, [pattern]), (
                    f"`{pattern}` in {list_name} matches no column the project "
                    "builds. An exclusion that matches nothing is not protecting "
                    "anything — check for a plural/singular slip such as "
                    "`vitals_*` against the built `vital_*`.")


def test_the_families_that_caused_this_are_matched(subtests):
    """Explicit regression test for the two patterns that were wrong."""
    columns = ["fluid_input_total", "fluid_balance",
               "vital_heart_rate_mean", "vital_sbp_max"]
    for list_name in ("ICU_ADMISSION_EXCLUDE", "ICU_ADMISSION_EXCLUDE_STRICT",
                      "LOS_EXCLUDE_STRICT"):
        with subtests.test(list=list_name):
            matched = match_column_patterns(columns, getattr(lf, list_name))
            assert set(matched) == set(columns), (
                f"{list_name} fails to exclude {sorted(set(columns) - set(matched))}. "
                "These are populated only for ICU patients; excluding them is the "
                "whole point of an availability-leakage guard.")


def test_known_unmatched_entries_are_still_unmatched(subtests):
    """
    The exemption list must not outlive its reason.

    If one of these starts matching, the exemption is stale and the pattern should
    be moved back under the main check rather than left permanently excused.
    """
    names = sorted(_namespace())
    for pattern, reason in KNOWN_UNMATCHED.items():
        with subtests.test(pattern=pattern):
            assert not match_column_patterns(names, [pattern]), (
                f"`{pattern}` now matches real columns, so the exemption "
                f"({reason}) no longer applies. Remove it from KNOWN_UNMATCHED.")
