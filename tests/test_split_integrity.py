"""
The patient split must stay the same split.

Every published metric in this project is conditioned on one specific
train/val/test partition. That partition is a deterministic function of
``(sorted subject_ids, seed, ratios)`` — but only the seed is obviously fixed. The
cohort is not: any rebuild that adds or drops a patient reshuffles the entire
population.

The magnitude is not subtle. Dropping a **single** patient from the current cohort of
223,452 moves **16,361** patients to a different split. Every model on disk would then
be evaluated on patients it trained on, with no error raised anywhere and metrics that
move in the flattering direction.

These tests pin the three things that keep that from happening quietly: the split has
no patient overlap, it regenerates identically from its recorded inputs, and
regeneration refuses to overwrite an incompatible split.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SPLIT = Path("data/processed/patient_split.parquet")
COHORT = Path("data/processed/admission_level_selected.parquet")

pytestmark = pytest.mark.skipif(
    not (SPLIT.exists() and COHORT.exists()),
    reason="processed cohort not built; run scripts/pipelines/run_pipeline.py",
)

SEED = 42
RATIOS = (0.70, 0.15, 0.15)


@pytest.fixture(scope="module")
def split():
    return pd.read_parquet(SPLIT)


@pytest.fixture(scope="module")
def cohort_subjects():
    adm = pd.read_parquet(COHORT, columns=["subject_id"])
    return np.unique(adm["subject_id"].dropna().to_numpy())


# ── the split itself ────────────────────────────────────────────────────────

def test_no_patient_appears_in_two_splits(split):
    counts = split.groupby("subject_id")["split"].nunique()
    offenders = counts[counts > 1]
    assert offenders.empty, f"{len(offenders)} patients appear in more than one split"


def test_split_labels_are_the_expected_three(split):
    assert set(split["split"]) == {"train", "val", "test"}


def test_split_covers_the_cohort_exactly(split, cohort_subjects):
    assert set(split["subject_id"]) == set(cohort_subjects), (
        "the split and the cohort disagree about which patients exist; "
        "one of them was rebuilt without the other")


def test_ratios_are_approximately_as_configured(split):
    frac = split["split"].value_counts(normalize=True)
    for name, expected in zip(("train", "val", "test"), RATIOS):
        assert abs(frac[name] - expected) < 0.005, (
            f"{name} is {frac[name]:.3f}, expected ~{expected}")


# ── reproducibility ─────────────────────────────────────────────────────────

def test_split_regenerates_identically_from_seed(split, cohort_subjects):
    """Same cohort + same seed must reproduce the file on disk, exactly."""
    rng = np.random.default_rng(SEED)
    shuffled = cohort_subjects.copy()
    rng.shuffle(shuffled)
    n = len(cohort_subjects)
    n_train, n_val = int(n * RATIOS[0]), int(n * RATIOS[1])

    expected = pd.DataFrame({
        "subject_id": np.concatenate(
            [shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]]),
        "expected": ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val),
    })
    merged = split.merge(expected, on="subject_id", how="outer")
    mismatched = int((merged["split"] != merged["expected"]).sum())
    assert mismatched == 0, (
        f"{mismatched} patients differ from a seed-{SEED} regeneration. Either the "
        "cohort changed after the split was written, or the split was hand-edited.")


# ── the fingerprint guard ───────────────────────────────────────────────────

def test_split_records_its_cohort_fingerprint(split):
    assert "cohort_fingerprint" in split.columns, (
        "patient_split.parquet carries no cohort fingerprint, so a cohort rebuild "
        "cannot be detected. Regenerate with src/data/splits.py.")
    assert split["cohort_fingerprint"].nunique() == 1


def test_fingerprint_matches_the_current_cohort(split, cohort_subjects):
    from src.data.splits import cohort_fingerprint

    assert split["cohort_fingerprint"].iloc[0] == cohort_fingerprint(cohort_subjects), (
        "the split was generated for a different cohort than the one on disk. "
        "Models trained against it are being evaluated on the wrong patients.")


def test_fingerprint_is_order_and_dtype_stable(cohort_subjects):
    """It must identify the patient set, not how it happened to be read."""
    from src.data.splits import cohort_fingerprint

    base = cohort_fingerprint(cohort_subjects)
    assert cohort_fingerprint(cohort_subjects[::-1]) == base, "order changed the digest"
    assert cohort_fingerprint(pd.Series(cohort_subjects, dtype="Int64")) == base, \
        "dtype changed the digest"


def test_fingerprint_changes_when_the_cohort_does(cohort_subjects):
    from src.data.splits import cohort_fingerprint

    assert cohort_fingerprint(cohort_subjects[:-1]) != cohort_fingerprint(cohort_subjects)


def test_dropping_one_patient_reshuffles_thousands(split, cohort_subjects):
    """
    Documents why the guard exists, with the real number.

    This is not a hypothetical: removing one patient moves five figures' worth of
    other patients across the train/test boundary.
    """
    from src.data.splits import _count_moved

    assert _count_moved(split, cohort_subjects, SEED, RATIOS) == 0, \
        "the unchanged cohort must move nobody"
    moved = _count_moved(split, cohort_subjects[:-1], SEED, RATIOS)
    assert moved > 1000, (
        f"expected a large reshuffle from a one-patient change, got {moved}; "
        "if this is now small the split algorithm changed and the guard may be moot")


def test_regeneration_refuses_an_incompatible_overwrite(split, cohort_subjects, tmp_path):
    """The guard must raise rather than silently rewrite."""
    from src.data.splits import SplitCohortMismatch, _refuse_incompatible_overwrite

    target = tmp_path / "patient_split.parquet"
    split.to_parquet(target, index=False)

    # same cohort → allowed
    _refuse_incompatible_overwrite(target, cohort_subjects, SEED, RATIOS)

    # cohort short by one → refused
    with pytest.raises(SplitCohortMismatch, match="Refusing to overwrite"):
        _refuse_incompatible_overwrite(target, cohort_subjects[:-1], SEED, RATIOS)


def test_guard_is_a_noop_when_no_split_exists(cohort_subjects, tmp_path):
    from src.data.splits import _refuse_incompatible_overwrite

    _refuse_incompatible_overwrite(tmp_path / "absent.parquet", cohort_subjects, SEED, RATIOS)
