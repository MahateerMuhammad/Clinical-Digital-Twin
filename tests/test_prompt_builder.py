"""
Tests for ClinicalPromptBuilder.build_structured_prompt and the LiveModelRunner
primitives it depends on.

The point of most of these is to prove the report is *computed*. Before this, the
method could not run at all — it called `runner.get_patient_row()` and
`runner.run_live_inference()`, neither of which existed — and its SHAP block printed
the same four literals (+0.45, +0.38, +0.29, +0.18) for every patient under a heading
claiming they came from TreeExplainer. A report that renders without crashing is not
evidence that its numbers mean anything, so the tests below check that different
patients produce different values, and that the specific fabricated constants are
gone.
"""

from __future__ import annotations

import os
import re

import pandas as pd
import pytest

DATA = "data/processed"
MODELS = "models/best_models"

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA, "admission_level_selected.parquet"))
    or not os.path.exists(os.path.join(MODELS, "phase1_mortality_winning.pkl")),
    reason="processed data or Phase 1-5 models unavailable",
)


@pytest.fixture(scope="module")
def builder():
    from src.llm.prompt_builder import ClinicalPromptBuilder
    return ClinicalPromptBuilder()


@pytest.fixture(scope="module")
def two_patients(builder):
    """A well patient and a deranged one, so their reports must differ."""
    cols = ["hadm_id", "lab_wbc_last_24h", "lab_bicarbonate_last_24h"]
    d = pd.read_parquet(os.path.join(DATA, "admission_level_selected.parquet"),
                        columns=cols)
    sick = d[(d.lab_wbc_last_24h > 18) & (d.lab_bicarbonate_last_24h < 20)]
    well = d[(d.lab_wbc_last_24h.between(5, 9)) & (d.lab_bicarbonate_last_24h.between(23, 27))]
    if sick.empty or well.empty:
        pytest.skip("no suitable contrasting admissions")
    return int(well.iloc[0].hadm_id), int(sick.iloc[0].hadm_id)


# ── the missing primitives ──────────────────────────────────────────────────

def test_get_patient_row_returns_the_right_admission(builder, two_patients):
    hadm, _ = two_patients
    row = builder.runner.get_patient_row(hadm)
    assert isinstance(row, pd.Series)
    assert int(row["hadm_id"]) == hadm


def test_get_patient_row_rejects_unknown_admission(builder):
    with pytest.raises(KeyError):
        builder.runner.get_patient_row(1)


def test_run_live_inference_alias(builder, two_patients):
    hadm, _ = two_patients
    row = builder.runner.get_patient_row(hadm)
    a = builder.runner.run_live_inference(row)
    b = builder.runner.run_live_inference_with_uncertainty(row)
    assert a == b
    assert 0.0 <= a["p_mortality"] <= 1.0


def test_tier_cutoffs_are_shared_not_retyped():
    """model_runner must read the cutoffs, not carry its own copy."""
    import inspect
    from src.llm import model_runner
    from src.llm.report_composer import TIER_CUTOFFS, tier_for_probability

    src = inspect.getsource(model_runner)
    for cutoff in TIER_CUTOFFS:
        assert str(cutoff) not in src, f"{cutoff} is hardcoded in model_runner"

    assert tier_for_probability(TIER_CUTOFFS[0] - 1e-6).startswith("Tier 1")
    assert tier_for_probability(TIER_CUTOFFS[0]).startswith("Tier 2")
    assert tier_for_probability(TIER_CUTOFFS[-1]).startswith("Tier 4")


# ── the report ──────────────────────────────────────────────────────────────

def test_report_builds(builder, two_patients):
    well, _ = two_patients
    text = builder.build_structured_prompt(well)
    for heading in ("PRESENTATION LABS", "MULTI-TASK PREDICTIVE SUITE",
                    "LOCAL SHAP RISK DRIVERS", "RETRIEVED DIGITAL TWINS",
                    "RETRIEVED GUIDELINES"):
        assert heading in text


def test_fabricated_shap_literals_are_gone(builder, two_patients):
    """The four hardcoded SHAP values must not appear for any patient."""
    for hadm in two_patients:
        text = builder.build_structured_prompt(hadm)
        for fake in ("+0.45", "+0.38", "+0.29", "+0.18"):
            assert fake not in text, f"fabricated SHAP value {fake} still rendered"


def test_no_hardcoded_developer_path(builder, two_patients):
    well, _ = two_patients
    text = builder.build_structured_prompt(well)
    assert "/Users/apple" not in text
    assert "file:///" not in text


def test_shap_is_patient_specific(builder, two_patients):
    """Different patients must yield different SHAP drivers."""
    well, sick = two_patients
    a = builder._shap_drivers(builder.runner.get_patient_row(well))
    b = builder._shap_drivers(builder.runner.get_patient_row(sick))
    assert a and b, "SHAP unavailable — cannot verify it is computed"
    assert [d["shap"] for d in a] != [d["shap"] for d in b]


def test_predictions_differ_between_patients(builder, two_patients):
    well, sick = two_patients
    p_well = builder.runner.run_live_inference(builder.runner.get_patient_row(well))
    p_sick = builder.runner.run_live_inference(builder.runner.get_patient_row(sick))
    assert p_well["p_mortality"] != p_sick["p_mortality"]


def test_rendered_labs_match_the_source_row(builder, two_patients):
    """Every printed lab value must equal the value in the admission row."""
    _, sick = two_patients
    row = builder.runner.get_patient_row(sick)
    text = builder.build_structured_prompt(sick)

    checked = 0
    for cands, label, unit, _ref in builder.PRESENTATION_LABS:
        col = next((c for c in cands if c in row.index), None)
        if col is None:
            continue
        v = pd.to_numeric(row.get(col), errors="coerce")
        if pd.isna(v):
            continue
        assert f"**{label}:** {v:.1f} {unit}" in text
        checked += 1
    assert checked >= 3, "too few labs rendered to be a meaningful check"


def test_absent_features_are_not_reported_as_missing_measurements(builder, two_patients):
    """
    An analyte with no windowed feature in the build must not be described as
    'not recorded', which would blame the patient's chart for a pipeline gap.
    """
    _, sick = two_patients
    row = builder.runner.get_patient_row(sick)
    text = builder.build_structured_prompt(sick)

    for cands, label, _unit, _ref in builder.PRESENTATION_LABS:
        if any(c in row.index for c in cands):
            continue
        line = next(l for l in text.splitlines() if l.startswith(f"- **{label}:**"))
        assert "feature build" in line
        assert "not recorded" not in line


def test_unavailable_blocks_say_so(builder, two_patients, monkeypatch):
    """With SHAP suppressed the block must refuse, not fall back to constants."""
    well, _ = two_patients
    monkeypatch.setattr(builder, "_shap_drivers", lambda *a, **k: [])
    text = builder.build_structured_prompt(well)
    assert "Unavailable" in text
    assert not re.search(r"SHAP [+-]\d", text)
