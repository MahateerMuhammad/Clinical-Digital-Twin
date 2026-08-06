"""
feature_space.py
────────────────
Build a fitted model's design matrix from an admission frame, and say honestly
which of its features were actually present.

Why absent features are NaN, not 0.0
────────────────────────────────────
Every serving path used to fill unsupplied features with ``0.0``. That is not a
neutral placeholder — it is a measurement. ``lab_creatinine_first_24h = 0.0``
asserts a creatinine of zero was drawn; ``admit_year = 0.0`` places the admission
in year zero; ``lab_*_count_24h = 0.0`` asserts no bloods were sent. The boosters
split on those values like any other, so an unseen patient was scored against a
vector of impossible measurements rather than against absent ones.

LightGBM and XGBoost both represent missingness natively: a NaN takes the default
direction learned at each split, which is the behaviour the model was fitted with.
Passing NaN says "not observed"; passing 0.0 says "observed to be zero". Only one
of those is true for a feature nobody supplied.

The effect is not cosmetic. Measured on the held-out test split
(``scripts/evaluation/run_payload_fidelity_eval.py``), switching the fill from
0.0 to NaN moves payload-based mortality AUROC from 0.8180 to 0.8470, and takes
a fully normal payload's deterioration estimate from 15.24% — above the value it
gave for diabetic ketoacidosis — down to 2.30%.

The categorical trap
────────────────────
The Phase 1-4 boosters were fitted on one-hot dummies (``admission_type_URGENT``
and ~85 more). A row read straight from ``admission_level_selected.parquet`` still
carries the *source* columns (``admission_type = "URGENT"``), so a numeric coercion
turned all of them into NaN and none of the dummies were ever set. Real admissions
therefore reached the model with 78 of 164 features populated — the row path was
quietly as degraded as the payload path it was meant to be the reference for.
``encode_admission_frame`` does the one-hot expansion so those features land.

Phase 5 (deterioration, XGBoost) is the deliberate exception: ``src/models/
deterioration.py`` coerces its object columns with ``errors="coerce"``, so
``admission_type`` and eight others were all-NaN *during training* too. Encoding
drops them from the frame and the reindex restores them as NaN, which is exactly
what the model was fitted against.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

#: Outcome-bearing or post-discharge columns. Present in the parquet, never inputs.
OUTCOME_COLUMNS: tuple[str, ...] = (
    "hospital_expire_flag", "readmission_30d", "has_icu_stay", "los_days",
    "los_hours", "icu_los_days", "n_icu_stays", "deathtime", "dod", "dischtime",
    "next_admittime", "days_to_readmission", "discharge_location",
    "first_careunit", "last_careunit", "intime", "outtime", "time_to_icu_hrs",
    "clinical_deterioration",
)

#: Identifiers and free text. `text_clean`/`text_tfidf_ready` are ~96% of the bytes.
IDENTIFIER_COLUMNS: tuple[str, ...] = (
    "subject_id", "hadm_id", "note_id", "admit_provider_id", "split",
    "text_clean", "text_tfidf_ready", "primary_icd_code", "note_type",
)

#: Object columns with more levels than this were excluded at fit time.
MAX_CATEGORY_LEVELS = 100


def _is_categorical(series: pd.Series) -> bool:
    """
    True for anything that needs one-hot expansion rather than numeric coercion.

    Defined by exclusion rather than by testing for ``object``/``CategoricalDtype``.
    The parquet stores these columns as ``CategoricalDtype`` while a frame built in
    memory gets ``object`` on pandas 2 and ``str`` on pandas 3 — an allow-list of
    dtypes silently stopped expanding categoricals the moment the pandas version
    moved, and the only symptom would have been a quietly worse feature vector.
    """
    return not (
        pd.api.types.is_numeric_dtype(series)
        or pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_datetime64_any_dtype(series)
    )


def encode_admission_frame(
    frame: pd.DataFrame,
    drop_outcomes: bool = True,
) -> pd.DataFrame:
    """
    One-hot encode an admission frame into the boosters' feature namespace.

    Column names are space-sanitised because LightGBM rewrites spaces to
    underscores at fit time: ``admission_type_EW EMER.`` in the frame and
    ``admission_type_EW_EMER.`` in the booster are the same feature under two
    spellings, and skipping the rewrite silently drops every multi-word category.

    Dummies are built with ``drop_first=False``. On a single row ``drop_first=True``
    drops *that row's own* category, so a URGENT admission would encode identically
    to the reference level; the reindex in :func:`align_to_model` discards the
    reference column instead, which reproduces the training encoding for any number
    of rows.
    """
    df = frame.copy()

    drop = list(IDENTIFIER_COLUMNS) + (list(OUTCOME_COLUMNS) if drop_outcomes else [])
    df = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
    df = df.drop(columns=[c for c in df.columns
                          if pd.api.types.is_datetime64_any_dtype(df[c])])

    cats, oversized = [], []
    for c in df.columns:
        if _is_categorical(df[c]):
            (cats if df[c].nunique(dropna=False) <= MAX_CATEGORY_LEVELS
             else oversized).append(c)
    df = df.drop(columns=oversized)

    if cats:
        df = pd.get_dummies(df, columns=cats, drop_first=False, dtype=float)

    df.columns = [str(c).replace(" ", "_") for c in df.columns]
    return df.loc[:, ~df.columns.duplicated()]


def align_to_model(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """
    Reindex ``frame`` onto ``feature_names``, leaving absent features as NaN.

    The fill is NaN and not 0.0 deliberately; see the module docstring. Values that
    are present but non-numeric are coerced, which also yields NaN — a string in a
    numeric feature is not evidence of a zero either.
    """
    absent = pd.Series(np.nan, index=frame.index, dtype=float)
    # Built in one pass rather than by repeated insertion: these matrices run to ~170
    # columns and per-column assignment fragments the block manager badly enough that
    # pandas warns about it.
    return pd.concat(
        [(pd.to_numeric(frame[name], errors="coerce") if name in frame.columns
          else absent).rename(name) for name in feature_names],
        axis=1,
    ).astype(float)


def feature_coverage(frame: pd.DataFrame, feature_names: Sequence[str]) -> float:
    """
    Fraction of ``feature_names`` genuinely present in ``frame``.

    Must be measured *before* :func:`align_to_model` runs: once the matrix is built
    an absent feature and a supplied NaN are the same value, and the distinction
    this number reports no longer exists.
    """
    have = set(frame.columns)
    return sum(f in have for f in feature_names) / len(feature_names)


def design_matrix(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    encode: bool = True,
) -> tuple[pd.DataFrame, float]:
    """Encode, measure coverage, then align. Returns ``(X, coverage)``."""
    encoded = encode_admission_frame(frame) if encode else frame
    return align_to_model(encoded, feature_names), feature_coverage(encoded, feature_names)


def looks_like_admission_row(obj: Iterable) -> bool:
    """
    True if ``obj`` is a stored admission row rather than an unseen-patient payload.

    The two travel through the same entry points but must be treated differently:
    a stored row populates the whole feature set and every task is served from it,
    while a payload populates under a fifth and most tasks are withheld. The
    discriminator is the presence of the parquet's own identifier columns, which a
    payload never carries.
    """
    try:
        index = set(obj.index)
    except AttributeError:
        return False
    return "hadm_id" in index or "subject_id" in index
