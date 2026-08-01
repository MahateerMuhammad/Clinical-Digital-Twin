"""
src/features/feature_selection.py
─────────────────────────────────
Feature selection preparation: variance, correlation, missingness reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

from src.utils.config import CFG
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class FeatureSelectionReport:
    """Results of feature selection preparation."""

    n_features_in: int
    n_features_out: int
    constant_features: List[str] = field(default_factory=list)
    duplicate_features: List[str] = field(default_factory=list)
    near_zero_variance: List[str] = field(default_factory=list)
    high_missing: List[str] = field(default_factory=list)
    #: ``(kept, dropped, |r|)`` — which column of each over-correlated pair survived.
    highly_correlated_pairs: List[Tuple[str, str, float]] = field(default_factory=list)
    #: Columns removed by the correlation step alone.
    correlated_dropped: List[str] = field(default_factory=list)
    dropped_features: List[str] = field(default_factory=list)
    kept_features: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_features_in": self.n_features_in,
            "n_features_out": self.n_features_out,
            "n_constant": len(self.constant_features),
            "n_duplicate": len(self.duplicate_features),
            "n_near_zero_variance": len(self.near_zero_variance),
            "n_high_missing": len(self.high_missing),
            "n_highly_correlated_pairs": len(self.highly_correlated_pairs),
            "n_correlated_dropped": len(self.correlated_dropped),
            "n_dropped": len(self.dropped_features),
        }


#: Suffixes marking an observation-window variant of another column, e.g.
#: ``lab_wbc_max`` (whole admission) and ``lab_wbc_max_24h`` (first 24 hours).
WINDOW_SUFFIXES: Tuple[str, ...] = ("_24h",)

#: Correlation is estimated on at most this many rows. At 546k rows the exact
#: pairwise computation is O(cols^2 x rows) inside a single-threaded Cython loop
#: and ran for over five hours on ~500 columns. A quarter-million-row sample
#: estimates a correlation to far better than the third decimal place, which is
#: all a 0.95 near-duplicate threshold needs.
CORR_SAMPLE_ROWS = 250_000


def _twin_of(name: str, columns: Set[str]) -> Optional[str]:
    """Return the counterpart column if ``name`` is a windowed variant, else None."""
    for suffix in WINDOW_SUFFIXES:
        if name.endswith(suffix):
            base = name[: -len(suffix)]
            if base in columns:
                return base
    return None


def _twin_pairs(columns: List[str]) -> Set[FrozenSet[str]]:
    """
    Pairs of columns that encode the same measurement over different windows.

    These are deliberate parallel encodings consumed by different protocols: the
    full-stay column belongs to mortality Run A/B, the ``_24h`` column to the
    strict early-window protocols. Feature selection must not eliminate one in
    favour of the other — which list applies is decided by
    :mod:`src.features.leakage_filters`, not by a correlation threshold.

    Without this, `lab_creatinine_first` and `lab_creatinine_first_24h` (identical
    for the 92% of admissions whose first draw falls inside the window) would be
    collapsed to whichever happened to be merged first, silently stripping labs
    from either Run B or Run C depending on merge order.
    """
    col_set = set(columns)
    return {
        frozenset((c, twin))
        for c in columns
        if (twin := _twin_of(c, col_set)) is not None
    }


#: Aggregate suffixes that carry an actual measured value, as opposed to a counter
#: (``_count``), a completeness statistic (``_missing_ratio``) or a flag tally
#: (``_abnormal_count``). Losing every one of these for an analyte means the models
#: can see how often it was drawn but never what it said.
#:
#: ``_std``, ``_slope`` and ``_change`` are excluded deliberately: they describe how a
#: value moved, not what it was, and Run C removes them as trajectory leakage anyway.
VALUE_AGGREGATES: Tuple[str, ...] = (
    "_mean", "_median", "_min", "_max", "_first", "_last",
)

#: Alternative assays for a measurement the panel already reports: ``_wb`` is the
#: blood-gas analyser, ``_poc`` the bedside point-of-care meter. They measure the same
#: analyte on different instruments and are ordered far less often, so they are
#: folded into the base analyte for the survival check — losing whole-blood sodium
#: while serum sodium survives is not a loss of information.
ASSAY_VARIANT_SUFFIXES: Tuple[str, ...] = ("_wb", "_poc")

#: Analytes permitted to disappear because another analyte carries the same
#: information, mapped to the one that must survive in their place.
#:
#: Haemoglobin and haematocrit are the same measurement twice — haematocrit is
#: conventionally about three times haemoglobin, and they correlate at r = 0.957 in
#: this cohort. Dropping one is genuine de-duplication, not signal loss.
#:
#: Entries here are deliberate and reviewed. Adding one is how you tell the guard
#: that a loss is acceptable; the guard exists so that no *unreviewed* loss can
#: happen quietly.
ACCEPTED_ABSORPTIONS: Dict[str, str] = {
    "lab_hemoglobin": "lab_hematocrit",
}


def _is_windowed(name: str) -> bool:
    return name.endswith(WINDOW_SUFFIXES)


def _windowed_value_analyte(name: str) -> Optional[str]:
    """
    For ``lab_creatinine_max_24h`` return ``lab_creatinine``; else None.

    Only windowed value columns qualify. Those are the sole lab features that
    survive the Run C leakage filter, so they are the ones whose loss silently
    strips an analyte from every leak-free model.
    """
    if not name.startswith("lab_") or not _is_windowed(name):
        return None
    stem = name
    for suffix in WINDOW_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    for agg in VALUE_AGGREGATES:
        if stem.endswith(agg):
            analyte = stem[: -len(agg)]
            # Fold the alternative-assay variants into the base analyte, so
            # lab_sodium_wb_max_24h counts as sodium rather than as its own
            # measurement that must independently survive.
            for variant in ASSAY_VARIANT_SUFFIXES:
                if analyte.endswith(variant):
                    analyte = analyte[: -len(variant)]
                    break
            return analyte
    return None


def _windowed_value_analytes(columns) -> Set[str]:
    return {a for c in columns if (a := _windowed_value_analyte(c)) is not None}


def _drop_choice(a: str, b: str, miss_rate: pd.Series) -> str:
    """
    Of two over-correlated columns, return the one to discard.

    Order of preference, highest first:

    1. **Keep the windowed variant.** A whole-stay aggregate and its ``_24h``
       counterpart are near-identical by construction, so whichever the correlation
       step happens to drop looks arbitrary — but it is not. The whole-stay column
       is deleted later by the Run C leakage filter, so dropping the ``_24h`` one
       here removes the measurement from every leak-free model. That is how
       creatinine, BUN and haematocrit ended up with no value feature at all: the
       leaky twin won the correlation contest, then was itself removed.
    2. **Keep the better-observed column**, by missing rate.
    3. **Lexicographic**, purely so repeated runs give identical datasets.

    Note this compares *any* two correlated columns, not just exact twins.
    ``lab_bun_max_24h`` was dropped against ``lab_bun_median`` — a different
    aggregate, so the twin-pair exemption never applied to it.
    """
    a_win, b_win = _is_windowed(a), _is_windowed(b)
    if a_win != b_win:
        return b if a_win else a

    a_miss = float(miss_rate.get(a, 0.0))
    b_miss = float(miss_rate.get(b, 0.0))
    if a_miss != b_miss:
        return a if a_miss > b_miss else b

    return max(a, b)


def _abs_corr_matrix(numeric: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Absolute correlation matrix, mean-imputed and computed via a matrix product.

    ``DataFrame.corr()`` dispatches to ``pandas._libs.algos.nancorr``, which walks
    every column pair in Python-level Cython with NaN masking. Imputing each
    column at its own mean makes the problem a single BLAS ``X.T @ X``: seconds
    rather than hours, at the cost of treating missing values as neutral instead
    of using pairwise-complete observations. For near-duplicate detection that
    trade is sound — mean imputation can only pull a correlation toward zero, so
    it never invents a spurious duplicate.
    """
    if numeric.shape[1] < 2:
        return pd.DataFrame(index=numeric.columns, columns=numeric.columns, dtype=float)

    frame = numeric
    if len(frame) > CORR_SAMPLE_ROWS:
        frame = frame.sample(CORR_SAMPLE_ROWS, random_state=seed)

    X = frame.to_numpy(dtype=np.float64, copy=True)
    means = np.nanmean(np.where(np.isinf(X), np.nan, X), axis=0)
    means = np.nan_to_num(means, nan=0.0)
    bad = ~np.isfinite(X)
    X[bad] = np.take(means, np.where(bad)[1])

    X -= X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    X /= sd

    corr = (X.T @ X) / len(X)
    np.clip(corr, -1.0, 1.0, out=corr)
    return pd.DataFrame(np.abs(corr), index=numeric.columns, columns=numeric.columns)


def _find_duplicate_columns(
    df: pd.DataFrame,
    protected_pairs: Optional[Set[FrozenSet[str]]] = None,
) -> List[str]:
    """Identify duplicate feature columns, preserving windowed/full-stay twins."""
    protected_pairs = protected_pairs or set()
    duplicates = []
    seen: Dict[int, str] = {}
    for col in df.columns:
        col_hash = int(pd.util.hash_pandas_object(df[col].fillna(-999), index=False).sum())
        first = seen.get(col_hash)
        if first is None:
            seen[col_hash] = col
        elif frozenset((first, col)) in protected_pairs:
            continue
        else:
            duplicates.append(col)
    return duplicates


def prepare_features(
    df: pd.DataFrame,
    id_cols: Optional[List[str]] = None,
    target_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, FeatureSelectionReport]:
    """
    Remove constant, duplicate, near-zero variance, and highly correlated features.

    ID and target columns are preserved.
    """
    id_cols = id_cols or ["subject_id", "hadm_id", "stay_id", "note_id"]
    target_cols = target_cols or [
        CFG.targets.mortality_inhosp,
        CFG.targets.los_days,
        CFG.targets.icu_admission,
        CFG.targets.icu_los_days,
        CFG.targets.readmission_30d,
        "hospital_expire_flag",
    ]

    preserve = []
    for c in id_cols + target_cols:
        if c in df.columns and c not in preserve:
            preserve.append(c)
    feature_cols = [c for c in df.columns if c not in preserve]

    numeric = df[feature_cols].select_dtypes(include=[np.number])
    report = FeatureSelectionReport(n_features_in=len(feature_cols), n_features_out=0)

    # Constant features
    nunique = numeric.nunique(dropna=True)
    report.constant_features = nunique[nunique <= 1].index.tolist()

    # Near-zero variance
    if len(numeric.columns) > 0:
        try:
            vt = VarianceThreshold(threshold=CFG.feature_selection.variance_threshold)
            vt.fit(numeric.fillna(0))
            mask = vt.get_support()
            report.near_zero_variance = numeric.columns[~mask].tolist()
        except ValueError:
            report.near_zero_variance = []

    # High missing
    miss_rate = numeric.isna().mean()
    report.high_missing = miss_rate[
        miss_rate > CFG.feature_selection.missing_rate_threshold
    ].index.tolist()

    # Observation-window twins are exempt from both de-duplication steps below.
    protected = _twin_pairs(list(numeric.columns))
    if protected:
        log.info("Feature selection: protecting %d observation-window twin pairs", len(protected))

    # Highly correlated pairs.
    #
    # Two fixes over the original, which dropped `partner` unconditionally:
    #
    #   * The loser is now chosen by _drop_choice rather than by whichever column
    #     happened to be the row index, so a leak-free `_24h` column always beats
    #     its whole-stay counterpart.
    #   * A column already dropped is skipped on both sides. Previously every
    #     member of a mutually correlated cluster could be dropped — each as the
    #     partner of a different survivor — which is how an analyte lost all five
    #     of its value aggregates at once.
    if numeric.shape[1] >= 2:
        corr = _abs_corr_matrix(numeric)
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        dropped_corr: Set[str] = set()
        for col in upper.columns:
            if col in dropped_corr:
                continue
            high = upper.index[upper[col] > CFG.feature_selection.correlation_threshold].tolist()
            for partner in high:
                if col in dropped_corr or partner in dropped_corr:
                    continue
                if frozenset((col, partner)) in protected:
                    continue
                loser = _drop_choice(col, partner, miss_rate)
                keeper = col if loser == partner else partner
                report.highly_correlated_pairs.append(
                    (keeper, loser, float(upper.loc[partner, col])))
                dropped_corr.add(loser)
        report.correlated_dropped = sorted(dropped_corr)

    # Duplicate columns
    report.duplicate_features = _find_duplicate_columns(numeric, protected_pairs=protected)

    drop_set = set(report.constant_features)
    drop_set.update(report.near_zero_variance)
    drop_set.update(report.high_missing)
    drop_set.update(report.duplicate_features)
    drop_set.update(report.correlated_dropped)

    report.dropped_features = sorted(drop_set)
    report.kept_features = [c for c in feature_cols if c not in drop_set]
    report.n_features_out = len(report.kept_features)

    # ── invariant: no analyte may lose every windowed value feature ──────────
    #
    # Windowed value columns are the only lab features that survive the Run C
    # leakage filter. An analyte that enters selection with one and leaves without
    # any is invisible to every leak-free model — the models keep its draw counts
    # and abnormal flags, but never a result. That is what happened to creatinine,
    # BUN and haematocrit, and nothing downstream noticed: the frame still had
    # hundreds of lab columns, the pipeline logged a normal reduction, and the loss
    # only surfaced when a clinical report tried to print a creatinine.
    #
    # The check compares before against after, so it cannot fire on data that never
    # carried the feature — a smoke run on a small sample stays quiet.
    before = _windowed_value_analytes(numeric.columns)
    after = _windowed_value_analytes(report.kept_features)
    lost = sorted(
        a for a in before - after
        # An analyte may go if a reviewed stand-in survived in its place.
        if ACCEPTED_ABSORPTIONS.get(a) not in after
    )
    if lost:
        raise ValueError(
            "Feature selection removed every windowed value feature for "
            f"{len(lost)} analyte(s): {', '.join(lost)}.\n"
            "These are the only lab features that survive Run C leakage filtering, "
            "so the affected measurements would be entirely invisible to Phases 1-5. "
            "Check the correlation threshold "
            f"({CFG.feature_selection.correlation_threshold}) and the missing-rate "
            f"threshold ({CFG.feature_selection.missing_rate_threshold}) — an analyte "
            "whose columns are all above the missing-rate cutoff is dropped before "
            "the correlation step ever runs."
        )

    result = df[preserve + report.kept_features].copy()
    log.info(
        "Feature selection: %d → %d features (%d dropped)",
        report.n_features_in, report.n_features_out, len(report.dropped_features),
    )
    return result, report


def generate_correlation_report(df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """Save full correlation matrix for numeric features."""
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return pd.DataFrame()
    corr = numeric.corr()
    from src.utils.io_utils import save_parquet
    save_parquet(corr.reset_index().rename(columns={"index": "feature"}), output_path)
    return corr


def generate_missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Missing value report per feature."""
    df = df.loc[:, ~df.columns.duplicated()]
    records = []
    for col in df.columns:
        n_miss = int(df[col].isna().sum())
        records.append({
            "feature": col,
            "dtype": str(df[col].dtype),
            "n_missing": n_miss,
            "pct_missing": round(100 * n_miss / max(len(df), 1), 2),
        })
    return pd.DataFrame(records).sort_values("pct_missing", ascending=False)
