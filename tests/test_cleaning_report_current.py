"""
The cleaning report must describe the data that is actually on disk.

`reports/cleaning_report.md` was generated 2026-07-19, before the ID-collision repair
restored `labevents`, `admissions`, `chartevents` and `radiology_detail` (see
`reports/data_correction_notice.md` §3). Its age made it *look* superseded, and an
audit listed it for regeneration on that basis.

It is not superseded: the repair corrupted identifier **values**, not row counts or
column missingness, so the cleaning stage's output was unaffected. Regenerating it
would re-read 300M+ rows to reproduce identical numbers.

This test is the alternative to that. It re-derives the report's figures from the
interim parquets and fails if they diverge — which converts "is this old report still
true?" from a question somebody has to re-ask into one the suite answers. If it ever
fails, the report genuinely is stale and *then* it needs regenerating.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - pyarrow ships with the pandas stack here
    pq = None

REPORT = Path("reports/cleaning_report.md")
INTERIM = Path("data/interim")

pytestmark = pytest.mark.skipif(
    not REPORT.exists() or not INTERIM.exists() or pq is None,
    reason="cleaning report or interim tables not present in this checkout",
)


def _summary_rows() -> dict[str, int]:
    """Parse {table: rows_after} from the report's summary table."""
    rows = {}
    for line in REPORT.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.split("|")]
        # | table | n_actions | output_path | rows_before | rows_after | dup_* |
        if len(cells) < 8 or not re.fullmatch(r"[a-z_]+", cells[1]):
            continue
        if not cells[5].replace(",", "").isdigit():
            continue
        rows[cells[1]] = int(cells[5].replace(",", ""))
    assert rows, "no summary rows parsed from cleaning_report.md"
    return rows


def _missing_claims() -> list[tuple[str, str, float]]:
    """Parse (table, column, pct) from the `document_missing` action rows."""
    out = []
    for line in REPORT.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 7 or cells[2] != "document_missing":
            continue
        m = re.match(r"([\d.]+)% missing", cells[6])
        if m:
            out.append((cells[1], cells[3], float(m.group(1))))
    assert out, "no document_missing rows parsed from cleaning_report.md"
    return out


def test_row_counts_match_the_interim_tables(subtests):
    """The repair restored four tables; none of them changed row count."""
    for table, claimed in _summary_rows().items():
        path = INTERIM / f"{table}_clean.parquet"
        if not path.exists():
            continue
        with subtests.test(table=table):
            actual = pq.ParquetFile(path).metadata.num_rows
            assert actual == claimed, (
                f"{table}: report says {claimed:,} rows, data has {actual:,}. "
                "The cleaning report is stale — regenerate it via the pipeline.")


def test_documented_missingness_still_holds(subtests):
    """
    Missingness is the figure a restored table would move, so it is the sharper check.

    Compared at the report's own precision (2dp). A column that has drifted by less
    than that has not drifted in any sense the report claims.
    """
    for table, column, claimed in _missing_claims():
        path = INTERIM / f"{table}_clean.parquet"
        if not path.exists():
            continue
        with subtests.test(table=table, column=column):
            frame = pd.read_parquet(path, columns=[column])
            actual = 100.0 * frame[column].isna().mean()
            assert round(actual, 2) == pytest.approx(claimed, abs=0.01), (
                f"{table}.{column}: report says {claimed:.2f}% missing, data has "
                f"{actual:.2f}%. The cleaning report is stale.")
