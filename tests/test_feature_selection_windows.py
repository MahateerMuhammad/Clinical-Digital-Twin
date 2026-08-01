"""
Tests for the observation-window protections in feature selection.

Background
──────────
Phases 1-5 were trained on a frame with no creatinine, BUN or haematocrit *values* —
only draw counts, missing-ratios and abnormal-count flags. The measurements existed
in admission_level.parquet and were removed by this module before any model saw them.

Two defects combined:

1. The correlation step dropped whichever column happened to be the row index of the
   pair. When `lab_bun_max_24h` was compared against the whole-stay `lab_bun_median`,
   the leak-free windowed column lost — and the whole-stay winner was then deleted by
   the Run C leakage filter, leaving nothing.
2. Nothing guaranteed a correlated cluster kept a survivor. Each member could be
   dropped as the partner of a different column, so all five value aggregates for an
   analyte could disappear in one pass.

The twin-pair exemption did not help: it protects `X` against `X_24h`, but these
columns were dropped against *different aggregates* of the same analyte.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.feature_selection import (
    VALUE_AGGREGATES,
    _drop_choice,
    _windowed_value_analyte,
    _windowed_value_analytes,
    prepare_features,
)


# ── the naming helpers ──────────────────────────────────────────────────────

@pytest.mark.parametrize("name,expected", [
    ("lab_creatinine_max_24h", "lab_creatinine"),
    ("lab_bun_median_24h", "lab_bun"),
    ("lab_wbc_last_24h", "lab_wbc"),
    ("lab_creatinine_max", None),            # whole-stay, not windowed
    ("lab_creatinine_count_24h", None),      # a counter, not a value
    ("lab_creatinine_missing_ratio_24h", None),
    ("lab_creatinine_abnormal_count_24h", None),
    ("anchor_age", None),
])
def test_windowed_value_analyte(name, expected):
    assert _windowed_value_analyte(name) == expected


def test_counters_are_not_mistaken_for_values():
    """An analyte with only counters must not count as having a value feature."""
    cols = ["lab_creatinine_count_24h", "lab_creatinine_missing_ratio_24h",
            "lab_creatinine_abnormal_count_24h"]
    assert _windowed_value_analytes(cols) == set()


# ── the drop preference ─────────────────────────────────────────────────────

def test_windowed_column_always_wins():
    miss = pd.Series({"lab_bun_max_24h": 0.31, "lab_bun_median": 0.05})
    # The windowed column is *worse* observed here and must still survive: the
    # whole-stay column is removed later by the leakage filter regardless.
    assert _drop_choice("lab_bun_max_24h", "lab_bun_median", miss) == "lab_bun_median"
    assert _drop_choice("lab_bun_median", "lab_bun_max_24h", miss) == "lab_bun_median"


def test_falls_back_to_missing_rate_between_like_columns():
    miss = pd.Series({"lab_wbc_max_24h": 0.40, "lab_wbc_min_24h": 0.10})
    assert _drop_choice("lab_wbc_max_24h", "lab_wbc_min_24h", miss) == "lab_wbc_max_24h"


def test_choice_is_symmetric_and_deterministic():
    miss = pd.Series({"a_24h": 0.2, "b_24h": 0.2})
    assert _drop_choice("a_24h", "b_24h", miss) == _drop_choice("b_24h", "a_24h", miss)


# ── end-to-end on synthetic data ────────────────────────────────────────────

def _analyte_frame(n=400, seed=0):
    """
    One analyte with five mutually correlated whole-stay aggregates and their
    windowed twins — the exact shape that lost every value column.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(1.2, 0.4, n)
    df = pd.DataFrame({"hadm_id": np.arange(n), "hospital_expire_flag": rng.integers(0, 2, n)})
    for agg in VALUE_AGGREGATES:
        jitter = rng.normal(0, 0.01, n)
        df[f"lab_creatinine{agg}"] = base + jitter
        df[f"lab_creatinine{agg}_24h"] = base + jitter + rng.normal(0, 0.01, n)
    df["lab_creatinine_count_24h"] = rng.integers(1, 5, n).astype(float)
    df["unrelated"] = rng.normal(0, 1, n)
    return df


def test_analyte_keeps_a_windowed_value_feature():
    selected, report = prepare_features(_analyte_frame())
    survivors = _windowed_value_analytes(report.kept_features)
    assert "lab_creatinine" in survivors, (
        "every windowed creatinine value feature was dropped; "
        f"kept: {report.kept_features}")


def test_correlated_cluster_keeps_a_survivor():
    """A fully correlated group must never be eliminated entirely."""
    _, report = prepare_features(_analyte_frame())
    cluster = [c for c in report.kept_features if c.startswith("lab_creatinine")]
    assert cluster, "the whole correlated cluster was dropped"


def test_windowed_beats_whole_stay_end_to_end():
    """Where one of a twin pair must go, the whole-stay column is the one to go."""
    _, report = prepare_features(_analyte_frame())
    kept = set(report.kept_features)
    windowed = {c for c in kept if c.startswith("lab_creatinine") and c.endswith("_24h")}
    assert windowed, f"no windowed creatinine column survived; kept: {sorted(kept)}"


def test_invariant_raises_when_an_analyte_would_be_stripped(monkeypatch):
    """
    The guard must fire rather than silently returning a frame with no values.

    Simulated by forcing the correlation threshold low enough that everything
    correlates, with the twin protection disabled.
    """
    import src.features.feature_selection as fs

    monkeypatch.setattr(fs, "_twin_pairs", lambda cols: set())
    monkeypatch.setattr(fs, "_drop_choice",
                        lambda a, b, m: a if a.endswith("_24h") else b)

    with pytest.raises(ValueError, match="windowed value feature"):
        fs.prepare_features(_analyte_frame())


def test_invariant_silent_when_feature_never_existed():
    """Data with no windowed value columns must pass, not trip the guard."""
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        "hadm_id": np.arange(200),
        "hospital_expire_flag": rng.integers(0, 2, 200),
        "lab_creatinine_count_24h": rng.integers(1, 5, 200).astype(float),
        "anchor_age": rng.integers(20, 90, 200).astype(float),
    })
    selected, report = prepare_features(df)
    assert report.n_features_out >= 1


# ── Run C must not see any whole-admission lab aggregate ────────────────────

LAB_AGGREGATE_VOCABULARY = (
    "count", "mean", "median", "min", "max", "std",
    "missing_ratio", "abnormal_count", "first", "last", "slope", "change",
)


def test_every_lab_aggregate_suffix_is_excluded_from_run_c():
    """
    Enumerate the aggregate vocabulary rather than trusting the list.

    `lab_*_mean` was absent from FULL_STAY_LAB_AGGREGATES and leaked whole-admission
    means into Run C — means that include values measured after the 24h window and
    up to the moment of death. It survived review because the list *looked*
    complete. This asserts against the vocabulary the lab builder actually emits.
    """
    from src.features.leakage_filters import (
        MORTALITY_EXCLUDE_RUN_C, match_column_patterns,
    )
    columns = [f"lab_creatinine_{agg}" for agg in LAB_AGGREGATE_VOCABULARY]
    excluded = set(match_column_patterns(columns, MORTALITY_EXCLUDE_RUN_C))
    missed = [c for c in columns if c not in excluded]
    assert not missed, f"whole-admission aggregates visible to Run C: {missed}"


def test_run_c_keeps_the_windowed_counterparts():
    """The 24h variants must survive the same filter, or Run C has no labs."""
    from src.features.leakage_filters import (
        MORTALITY_EXCLUDE_RUN_C, match_column_patterns,
    )
    windowed = [f"lab_creatinine_{agg}_24h" for agg in ("mean", "min", "max", "first")]
    excluded = set(match_column_patterns(windowed, MORTALITY_EXCLUDE_RUN_C))
    assert not excluded, f"Run C wrongly excluded windowed labs: {sorted(excluded)}"


STRICT_PROTOCOLS = (
    "MORTALITY_EXCLUDE_RUN_C",
    "READMISSION_EXCLUDE_STRICT",
    "ICU_ADMISSION_EXCLUDE_STRICT",
    "LOS_EXCLUDE_STRICT",
    "DETERIORATION_EXCLUDE_STRICT",
)


@pytest.mark.parametrize("protocol", STRICT_PROTOCOLS)
def test_no_strict_protocol_sees_whole_stay_labs(protocol):
    """
    Every 24h protocol must exclude the entire aggregate vocabulary.

    All five build on FULL_STAY_LAB_AGGREGATES, so the `lab_*_mean` omission
    affected all of them at once — Phase 1 is simply where it was noticed. This
    parametrises the check so a future addition to one list cannot quietly leave
    the others behind.
    """
    import src.features.leakage_filters as lf
    from src.features.leakage_filters import match_column_patterns

    columns = [f"lab_creatinine_{agg}" for agg in LAB_AGGREGATE_VOCABULARY]
    excluded = set(match_column_patterns(columns, getattr(lf, protocol)))
    missed = [c for c in columns if c not in excluded]
    assert not missed, f"{protocol} leaks whole-admission labs: {missed}"
