"""
A feature set that exists but reaches no model must say so in the data dictionary.

`data_dictionary.md` documents the *datasets*, so anything built and never merged is
invisible to it. That is how 66 emergency-department features came to be engineered,
tested and leakage-screened while appearing in no documentation — an audit found them
by reading the source tree, which is not a discovery mechanism worth relying on.

The section is derived, so this test guards the derivation rather than the prose: any
`*_features.parquet` whose columns reach no dataset must be listed, and anything
listed must genuinely be absent downstream. Adopting a set removes it from both sides
at once.
"""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pq = None

DICTIONARY = Path("reports/data_dictionary.md")
FEATURE_DIR = Path("data/interim/features")
SELECTED = Path("data/processed/admission_level_selected.parquet")

pytestmark = pytest.mark.skipif(
    pq is None or not DICTIONARY.exists() or not FEATURE_DIR.exists(),
    reason="data dictionary or interim feature sets not present",
)

KEYS = ("hadm_id", "subject_id", "stay_id")


def _unconsumed() -> dict[str, list[str]]:
    """{feature set: payload columns} for sets whose columns reach no dataset."""
    if not SELECTED.exists():
        pytest.skip("selected matrix not built")
    downstream = {f.name for f in pq.ParquetFile(SELECTED).schema_arrow}

    out = {}
    for path in sorted(FEATURE_DIR.glob("*_features.parquet")):
        columns = [f.name for f in pq.ParquetFile(path).schema_arrow]
        payload = [c for c in columns if c not in KEYS]
        if payload and not any(c in downstream for c in payload):
            out[path.stem.replace("_features", "")] = payload
    return out


def test_every_unconsumed_feature_set_is_listed(subtests):
    text = DICTIONARY.read_text(encoding="utf-8")
    for name in _unconsumed():
        with subtests.test(feature_set=name):
            assert name in text, (
                f"`{name}` is built and reaches no dataset, but the data dictionary "
                "does not mention it. Regenerate the dictionary — the Staged Feature "
                "Sets section is derived and will pick it up.")


def test_emergency_is_still_the_adoption_candidate():
    """
    ED is the one staged set intended for adoption; vitals and fluids are not.

    Vitals and fluids are keyed per ICU stay against a cohort that is 83% non-ICU, so
    their absence is a grain mismatch rather than pending work. Conflating the two
    would make the dictionary read as though three feature sets were queued up, when
    only one is.
    """
    unconsumed = _unconsumed()
    if "emergency" not in unconsumed:
        pytest.skip("ED features not built, or already adopted")

    assert len(unconsumed["emergency"]) > 50, (
        f"expected the full ED feature set, found {len(unconsumed['emergency'])}")
    text = DICTIONARY.read_text(encoding="utf-8")
    assert "ed_feature_coverage" in text, (
        "the dictionary lists the ED set but does not point at its coverage table, "
        "which is where adoption status is derived")


def _listed_as_staged() -> set[str]:
    """
    Feature-set names in the staged table's first column.

    Parsed from the table rather than matched as substrings: `medication` is adopted
    and also appears in the ED entry's prose ("medication reconciliation"), so a bare
    `in text` check reports a stale listing that is not there.
    """
    text = DICTIONARY.read_text(encoding="utf-8")
    if "Staged Feature Sets" not in text:
        return set()
    section = text.split("Staged Feature Sets")[1].split("\n## ")[0]

    names = set()
    for line in section.splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 6 or cells[1] in ("feature set", "") or set(cells[1]) <= {":", "-"}:
            continue
        names.add(cells[1])
    return names


def test_nothing_listed_as_staged_has_actually_been_adopted(subtests):
    """The other direction: a stale 'not consumed' claim is the worse failure."""
    if not SELECTED.exists():
        pytest.skip("selected matrix not built")
    downstream = {f.name for f in pq.ParquetFile(SELECTED).schema_arrow}
    listed = _listed_as_staged()
    if not listed:
        pytest.skip("no staged section present")

    for path in sorted(FEATURE_DIR.glob("*_features.parquet")):
        name = path.stem.replace("_features", "")
        payload = [f.name for f in pq.ParquetFile(path).schema_arrow if f.name not in KEYS]
        if payload and all(c in downstream for c in payload):
            with subtests.test(feature_set=name):
                assert name not in listed, (
                    f"`{name}` is fully present downstream but is still listed as "
                    "staged. Regenerate the data dictionary.")
