"""
Every phase report must state that its model has no vital signs — while that is true.

The admission-level matrix contains zero `vital_*` columns. Vitals derive from
`chartevents`, which is ICU-only in MIMIC-IV and keyed by `stay_id`, so ~83% of
admissions have none and nothing merges to the admission grain. `merge_admission_features`
never joins a vitals feature set at all.

A reader assumes a clinical risk model uses vital signs. Phases 1, 2 and 5 said
otherwise; Phases 3 and 4 did not, and Phase 3 is precisely where the assumption is
most natural — ICU admission predicted at 0.92 AUROC without a single heart rate.

The test is two-directional on purpose. It fails if a report drops the disclosure, and
it also fails if the disclosure becomes *false* — adopting the ED module would put
admission-grain physiology into these models, and a caution insisting there is none
would then be the stale claim. Neither direction is allowed to drift silently.
"""

from __future__ import annotations

import re
from pathlib import Path

import joblib
import pytest

MODELS = Path("models/best_models")

#: report → the promoted artifact whose feature list it describes.
REPORTS = {
    "reports/phase1_mortality_report.md": "phase1_mortality_winning.pkl",
    "reports/phase2_readmission_report.md": "phase2_readmission_winning.pkl",
    "reports/phase3_icu_admission_report.md": "phase3_icu_admission_winning.pkl",
    "reports/phase4_los_two_stage_report.md": "phase4_hosp_los_stageA_winning.pkl",
}

pytestmark = pytest.mark.skipif(
    not MODELS.exists(), reason="promoted models not present in this checkout")


def _feature_names(path: Path):
    model = joblib.load(path)
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        booster = getattr(model, "booster_", None)
        names = booster.feature_name() if booster is not None else None
    if names is None:
        names = model.get_booster().feature_names
    return [str(n) for n in names]


@pytest.mark.parametrize("report, artifact", sorted(REPORTS.items()))
def test_report_discloses_the_absence_of_vitals(report, artifact):
    path = MODELS / artifact
    if not path.exists():
        pytest.skip(f"{artifact} not promoted")

    names = _feature_names(path)
    n_vitals = sum(1 for n in names if n.startswith(("vital_", "news2_")))
    text = Path(report).read_text(encoding="utf-8")

    if n_vitals == 0:
        # Any of the three spellings the reports use; what matters is that the claim
        # is present and negative, not that the wording is uniform.
        assert re.search(r"do not reach this model|Vital signs\*{0,2} \| \*{0,2}0|"
                         r"not\*{0,2} vitals|absent from the dataset", text), (
            f"{report} does not disclose that its model has zero vital-sign features. "
            "A reader assumes a clinical risk model uses vitals; this one does not.")
    else:
        assert "do not reach this model" not in text, (
            f"{report} claims vitals do not reach the model, but the promoted artifact "
            f"has {n_vitals} of them. The disclosure has gone stale — most likely the "
            "ED module was adopted, which is the intended way for this to change.")


def test_the_matrix_itself_still_has_no_vitals():
    """The fact the reports rest on. If this flips, all four disclosures need rewriting."""
    import pyarrow.parquet as pq

    matrix = Path("data/processed/admission_level_selected.parquet")
    if not matrix.exists():
        pytest.skip("selected matrix not built")
    columns = [f.name for f in pq.ParquetFile(matrix).schema_arrow]
    n = sum(1 for c in columns if c.startswith(("vital_", "news2_")))
    assert n == 0, (
        f"the selected matrix now carries {n} vital/NEWS2 columns. Phases 1-4 all "
        "state it carries none — update those reports before this passes again.")
