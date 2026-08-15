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

``src/models/deterioration.py`` coerces its object columns with ``errors="coerce"``,
so ``admission_type`` and eight others were all-NaN during *its* training. That
model is superseded: the promoted Phase 5 artifact is the landmark model, and
``scripts/pipelines/run_deterioration_landmark.py`` builds its matrix by calling
``encode_admission_frame`` here. Its ``race_*`` columns are therefore real one-hot
dummies, exactly like Phases 1-4. There is no categorical exception any more.

Absent dummies are 0.0, not NaN
───────────────────────────────
The rule at the top of this docstring inverts for one-hot families, and getting
that backwards cost more than getting the numeric fill backwards did.

``pd.get_dummies`` emits only the levels a frame actually contains. A payload
carrying ``race = "WHITE"`` produces ``race_WHITE`` alone, so the other thirty-one
``race_*`` features the booster expects are absent — and filling them with NaN says
"we do not know whether this patient is Black", when the payload just said they are
White. Every sibling of a supplied category is a **known zero**. The training
matrix, one-hot encoded over the whole cohort, contained those zeros; withholding
them at serving time is a train/serve mismatch, not conservatism.

The distinction that makes this safe is between a category that was *supplied* and
one that was *not*. Only the former licences the zeros, and only for the rows where
the source value is non-null — so the mask is tracked per column and per row rather
than assumed. A frame that never mentions ``race`` still gets NaN across all
thirty-two, which is the honest answer for it.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

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

#: ``DataFrame.attrs`` key holding ``{dummy_prefix: row-mask of supplied values}``.
#: Written by :func:`encode_admission_frame`, consumed by :func:`align_to_model` to
#: decide which absent dummies are known zeros. Passed explicitly between the two in
#: :func:`design_matrix`; ``attrs`` is a convenience for callers that encode
#: separately, and pandas does not guarantee it survives arbitrary operations.
ONEHOT_KNOWN = "onehot_known"


def onehot_known(frame: pd.DataFrame) -> Dict[str, np.ndarray]:
    """The one-hot provenance recorded on ``frame``, or empty if it was not encoded."""
    known = frame.attrs.get(ONEHOT_KNOWN) or {}
    return {k: v for k, v in known.items() if len(v) == len(frame)}


def as_frame(series: pd.Series) -> pd.DataFrame:
    """
    A one-row frame from ``series``, carrying its one-hot provenance across.

    ``Series.to_frame().T`` drops ``attrs``, and every caller that scores a payload
    does exactly that before aligning. Without the carry-over the provenance is
    silently lost between the encoder and the consumer, and the zeros it licenses
    revert to NaN — the failure is invisible, because a NaN-filled matrix scores
    perfectly happily.
    """
    frame = series.to_frame().T
    known = series.attrs.get(ONEHOT_KNOWN)
    if known:
        frame.attrs[ONEHOT_KNOWN] = known
    return frame


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

    # Captured before the expansion, because afterwards the source columns are gone
    # and there is no way to tell a dummy whose sibling is a known zero from a
    # numeric feature that simply was not supplied.
    known = {
        f"{str(c).replace(' ', '_')}_": df[c].notna().to_numpy(dtype=bool)
        for c in cats
    }

    if cats:
        df = pd.get_dummies(df, columns=cats, drop_first=False, dtype=float)

    df.columns = [str(c).replace(" ", "_") for c in df.columns]
    out = df.loc[:, ~df.columns.duplicated()]
    out.attrs[ONEHOT_KNOWN] = known
    return out


def _known_zero_column(
    name: str,
    known: Dict[str, np.ndarray],
    index: pd.Index,
) -> Optional[pd.Series]:
    """
    The fill for an absent dummy ``name``, or None if it is not a dummy at all.

    Longest prefix wins. ``admission_type`` and ``admission_location`` are both live
    source columns, and matching the shorter one first would attribute
    ``admission_location_EMERGENCY_ROOM`` to the wrong family — harmless while both
    are supplied, wrong the moment only one is.
    """
    for prefix in sorted(known, key=len, reverse=True):
        if name.startswith(prefix):
            # 0.0 where the category was supplied for that row, NaN where it was not.
            return pd.Series(np.where(known[prefix], 0.0, np.nan),
                             index=index, dtype=float)
    return None


def align_to_model(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    known: Optional[Dict[str, np.ndarray]] = None,
) -> pd.DataFrame:
    """
    Reindex ``frame`` onto ``feature_names``, filling absent features honestly.

    Absent numeric features are NaN and not 0.0; absent *dummies* of a category that
    was supplied are 0.0 and not NaN. Both directions are argued in the module
    docstring. ``known`` maps dummy prefix → per-row mask of supplied categories and
    comes from :func:`encode_admission_frame`; without it every absent feature is
    NaN, which is the old behaviour and still the correct one for an unencoded frame.

    Values that are present but non-numeric are coerced, which also yields NaN — a
    string in a numeric feature is not evidence of a zero either.
    """
    known = onehot_known(frame) if known is None else known
    absent = pd.Series(np.nan, index=frame.index, dtype=float)

    def column(name: str) -> pd.Series:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce")
        # Explicit `is None`: a Series has no usable truth value, and an all-zero
        # fill is exactly the case `or` would have silently discarded.
        fill = _known_zero_column(name, known, frame.index)
        return absent if fill is None else fill

    # Built in one pass rather than by repeated insertion: these matrices run to ~170
    # columns and per-column assignment fragments the block manager badly enough that
    # pandas warns about it.
    return pd.concat(
        [column(name).rename(name) for name in feature_names], axis=1,
    ).astype(float)


def feature_coverage(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    known: Optional[Dict[str, np.ndarray]] = None,
) -> float:
    """
    Fraction of ``feature_names`` whose value is genuinely known for ``frame``.

    A dummy absent because its category was supplied as something else counts as
    known — the payload determined it to be zero just as surely as if it had been
    written out. Counting only literally-present columns understated coverage badly:
    supplying ``race`` informs thirty-two features and was credited with one.

    Must be measured *before* :func:`align_to_model` runs: once the matrix is built
    an absent feature and a supplied NaN are the same value, and the distinction
    this number reports no longer exists.
    """
    known = onehot_known(frame) if known is None else known
    have = set(frame.columns)
    # Only families supplied for *every* row are credited. Coverage is one number for
    # the whole frame, so a category present on some rows and null on others cannot be
    # reported as known without overstating it for the rows that lack it.
    prefixes = tuple(p for p, mask in known.items() if mask.all())
    return sum(
        f in have or (bool(prefixes) and f.startswith(prefixes))
        for f in feature_names
    ) / len(feature_names)


def design_matrix(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    encode: bool = True,
) -> tuple[pd.DataFrame, float]:
    """Encode, measure coverage, then align. Returns ``(X, coverage)``."""
    encoded = encode_admission_frame(frame) if encode else frame
    # Read once and passed explicitly: `attrs` is not guaranteed to survive whatever
    # a caller does to `frame` before this, and both consumers must see the same map.
    known = onehot_known(encoded)
    return (align_to_model(encoded, feature_names, known),
            feature_coverage(encoded, feature_names, known))


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
