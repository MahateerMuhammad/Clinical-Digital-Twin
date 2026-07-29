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
    highly_correlated_pairs: List[Tuple[str, str, float]] = field(default_factory=list)
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

    # Highly correlated pairs
    if numeric.shape[1] >= 2:
        corr = _abs_corr_matrix(numeric)
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop_corr = set()
        for col in upper.columns:
            high = upper.index[upper[col] > CFG.feature_selection.correlation_threshold].tolist()
            for partner in high:
                if frozenset((col, partner)) in protected:
                    continue
                report.highly_correlated_pairs.append((col, partner, float(upper.loc[partner, col])))
                to_drop_corr.add(partner)

    # Duplicate columns
    report.duplicate_features = _find_duplicate_columns(numeric, protected_pairs=protected)

    drop_set = set(report.constant_features)
    drop_set.update(report.near_zero_variance)
    drop_set.update(report.high_missing)
    drop_set.update(report.duplicate_features)
    for _, partner, _ in report.highly_correlated_pairs:
        drop_set.add(partner)

    report.dropped_features = sorted(drop_set)
    report.kept_features = [c for c in feature_cols if c not in drop_set]
    report.n_features_out = len(report.kept_features)

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
