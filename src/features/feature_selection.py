"""
src/features/feature_selection.py
─────────────────────────────────
Feature selection preparation: variance, correlation, missingness reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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


def _find_duplicate_columns(df: pd.DataFrame) -> List[str]:
    """Identify duplicate feature columns."""
    duplicates = []
    seen_hashes: set[int] = set()
    for col in df.columns:
        col_hash = int(pd.util.hash_pandas_object(df[col].fillna(-999), index=False).sum())
        if col_hash in seen_hashes:
            duplicates.append(col)
        else:
            seen_hashes.add(col_hash)
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

    # Highly correlated pairs
    if numeric.shape[1] >= 2:
        corr = numeric.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop_corr = set()
        for col in upper.columns:
            high = upper.index[upper[col] > CFG.feature_selection.correlation_threshold].tolist()
            for partner in high:
                report.highly_correlated_pairs.append((col, partner, float(upper.loc[partner, col])))
                to_drop_corr.add(partner)

    # Duplicate columns
    report.duplicate_features = _find_duplicate_columns(numeric)

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
