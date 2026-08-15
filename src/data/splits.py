"""
src/data/splits.py
──────────────────
Patient-level split generator for Clinical Digital Twin.
Creates reproducible train/val/test splits (70/15/15) keyed on subject_id
to prevent data leakage across admissions belonging to the same patient.

Why the split is fingerprinted
──────────────────────────────
The assignment is a deterministic function of ``(sorted subject_ids, seed, ratios)``.
Hold all three fixed and it regenerates bit-identically — verified over the current
223,452 patients.

Change the *cohort*, though, and the shuffle re-partitions everyone. A rebuild that
adds or drops a single patient reshuffles the whole population, so admissions that
were in `test` land in `train` and vice versa. Every model already on disk would then
be scored on patients it was trained on. Nothing about that fails loudly: the metrics
simply become invalid, and they move in the flattering direction.

So the cohort's identity is hashed into the file, and regeneration refuses to
overwrite an existing split when patients would move. This is the guard that makes
"zero patient overlap" a durable property rather than one that held on the day it was
last checked.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.utils.config import CFG
from src.utils.io_utils import save_parquet
from src.utils.logger import get_logger

log = get_logger(__name__)


class SplitCohortMismatch(RuntimeError):
    """Raised when regenerating the split would move patients between splits."""


def cohort_fingerprint(subject_ids: Sequence) -> str:
    """
    Stable 16-hex-char digest of the patient set.

    Hashes the *sorted* ids so it identifies the cohort rather than the order it
    happened to be read in, and casts to int64 so a dtype change between builds
    (Int64 vs int64 vs float64) does not read as a different cohort.
    """
    arr = np.sort(np.asarray(subject_ids, dtype="int64"))
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def create_patient_splits(
    admission_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    seed: Optional[int] = None,
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    force: bool = False,
) -> pd.DataFrame:
    """
    Generate patient-level train/val/test split keyed on subject_id.

    Parameters
    ----------
    admission_path : Path, optional
        Path to admission_level_selected.parquet or patient_level.parquet.
    output_path : Path, optional
        Path where patient_split.parquet will be saved.
    seed : int, optional
        Random seed for reproducibility. Defaults to CFG.random_seed.
    ratios : tuple of float
        (train_ratio, val_ratio, test_ratio). Must sum to 1.0.
    force : bool
        Overwrite an existing split even when patients would move between splits.
        Doing so invalidates every model already trained against it; see
        :class:`SplitCohortMismatch`.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['subject_id', 'split'].
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {ratios} (sum={sum(ratios)})")

    seed = seed if seed is not None else CFG.random_seed
    admission_path = admission_path or Path(CFG.resolve(CFG.paths.processed)) / "admission_level_selected.parquet"
    if not admission_path.exists():
        admission_path = Path(CFG.resolve(CFG.paths.processed)) / "patient_level.parquet"

    if not admission_path.exists():
        raise FileNotFoundError(f"Input file not found for split generation: {admission_path}")

    log.info("Reading unique subject_ids from %s...", admission_path.name)
    df = pd.read_parquet(admission_path, columns=["subject_id"])
    unique_subjects = np.unique(df["subject_id"].dropna().to_numpy())
    n_subjects = len(unique_subjects)

    log.info("Found %d unique patients for split generation (seed=%d)", n_subjects, seed)

    rng = np.random.default_rng(seed)
    shuffled_subjects = unique_subjects.copy()
    rng.shuffle(shuffled_subjects)

    n_train = int(n_subjects * ratios[0])
    n_val = int(n_subjects * ratios[1])
    # remaining goes to test to avoid rounding issues
    n_test = n_subjects - n_train - n_val

    train_ids = set(shuffled_subjects[:n_train])
    val_ids = set(shuffled_subjects[n_train : n_train + n_val])
    test_ids = set(shuffled_subjects[n_train + n_val :])

    # Hard Assertions — Zero Overlap Check
    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids

    assert len(train_val_overlap) == 0, f"Leakage detected: {len(train_val_overlap)} subjects in both train and val!"
    assert len(train_test_overlap) == 0, f"Leakage detected: {len(train_test_overlap)} subjects in both train and test!"
    assert len(val_test_overlap) == 0, f"Leakage detected: {len(val_test_overlap)} subjects in both val and test!"
    assert len(train_ids) + len(val_ids) + len(test_ids) == n_subjects, "Mismatch in subject total count!"

    log.info("PASSED ZERO-OVERLAP ASSERTION: No subject_id shared across splits.")
    log.info("Split Sizes (Patients): Train=%d (%.1f%%), Val=%d (%.1f%%), Test=%d (%.1f%%)",
             len(train_ids), 100 * len(train_ids) / n_subjects,
             len(val_ids), 100 * len(val_ids) / n_subjects,
             len(test_ids), 100 * len(test_ids) / n_subjects)

    # Assign split label
    split_map = {}
    for s_id in train_ids:
        split_map[s_id] = "train"
    for s_id in val_ids:
        split_map[s_id] = "val"
    for s_id in test_ids:
        split_map[s_id] = "test"

    split_df = pd.DataFrame({
        "subject_id": unique_subjects,
        "split": [split_map[s_id] for s_id in unique_subjects],
    })
    split_df["cohort_fingerprint"] = cohort_fingerprint(unique_subjects)

    output_path = output_path or Path(CFG.resolve(CFG.paths.processed)) / "patient_split.parquet"

    # Refuse to silently re-partition patients that trained models have already
    # seen. See the module docstring for what this prevents.
    if not force:
        _refuse_incompatible_overwrite(output_path, unique_subjects, seed, ratios)

    save_parquet(split_df, output_path, optimise_memory=False)
    log.info("Saved patient splits → %s", output_path)

    return split_df


def _refuse_incompatible_overwrite(output_path, unique_subjects, seed, ratios) -> None:
    """Raise if overwriting would move patients between splits."""
    if not Path(output_path).exists():
        return

    existing = pd.read_parquet(output_path)
    old_fp = (str(existing["cohort_fingerprint"].iloc[0])
              if "cohort_fingerprint" in existing.columns and len(existing) else None)
    new_fp = cohort_fingerprint(unique_subjects)

    if old_fp == new_fp:
        return                      # same cohort, same seed → same assignment

    if old_fp is None:
        log.warning(
            "%s predates cohort fingerprinting, so compatibility cannot be proven. "
            "Verifying by direct comparison instead.", Path(output_path).name)

    moved = _count_moved(existing, unique_subjects, seed, ratios)
    if moved == 0:
        log.info("Cohort changed but no patient changes split; overwrite is safe.")
        return

    raise SplitCohortMismatch(
        f"Refusing to overwrite {output_path}.\n\n"
        f"The cohort has changed ({len(existing):,} patients on disk, "
        f"{len(unique_subjects):,} now), and regenerating would move "
        f"{moved:,} patients to a different split.\n\n"
        "Every model already trained against the existing split would then be "
        "evaluated on patients it was trained on, and no error would be raised — "
        "the metrics would simply be wrong and would look better.\n\n"
        "Do one of:\n"
        "  • keep the existing split (rebuild the cohort but leave this file alone), or\n"
        "  • retrain every phase after regenerating, and re-run scripts/evaluation/, or\n"
        "  • pass force=True / --force if you accept invalidating all existing models."
    )


def _count_moved(existing, unique_subjects, seed, ratios) -> int:
    """How many patients present in both cohorts would land in a different split."""
    rng = np.random.default_rng(seed)
    shuffled = unique_subjects.copy()
    rng.shuffle(shuffled)
    n = len(unique_subjects)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    proposed = pd.DataFrame({
        "subject_id": np.concatenate(
            [shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]]),
        "new_split": (["train"] * n_train + ["val"] * n_val
                      + ["test"] * (n - n_train - n_val)),
    })
    both = existing[["subject_id", "split"]].merge(proposed, on="subject_id", how="inner")
    return int((both["split"] != both["new_split"]).sum())


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=create_patient_splits.__doc__)
    ap.add_argument("--force", action="store_true",
                    help="overwrite even if patients would change split "
                         "(invalidates every trained model)")
    create_patient_splits(force=ap.parse_args().force)
