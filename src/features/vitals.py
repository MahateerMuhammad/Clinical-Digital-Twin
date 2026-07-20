"""
src/features/vitals.py
──────────────────────
ICU vital sign feature engineering from chartevents.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.config import CFG
from src.utils.io_utils import aggregate_chunked
from src.utils.logger import get_logger
from src.utils.validation import VITAL_RANGES

log = get_logger(__name__)


def _itemid_to_vital(itemid: int) -> Optional[str]:
    for vital_name, ids in CFG.vitals.items():
        if itemid in ids:
            return vital_name
    return None


def build_vital_features_from_df(chartevents: pd.DataFrame) -> pd.DataFrame:
    """Aggregate vital sign features at stay_id level."""
    if chartevents.empty:
        return pd.DataFrame(columns=["stay_id"])

    ce = chartevents.copy()
    ce["valuenum"] = pd.to_numeric(ce.get("valuenum"), errors="coerce")
    if "charttime" in ce.columns:
        ce["charttime"] = pd.to_datetime(ce["charttime"], errors="coerce")

    vital_itemids = set()
    for ids in CFG.vitals.values():
        vital_itemids.update(ids)
    ce = ce[ce["itemid"].isin(vital_itemids)] if "itemid" in ce.columns else ce

    records = {}
    for (stay_id, itemid), sub in ce.groupby(["stay_id", "itemid"], observed=True):
        if stay_id not in records:
            records[stay_id] = {"stay_id": stay_id}
            
        vital = _itemid_to_vital(int(itemid))
        if not vital:
            continue
        # Order chronologically so _latest / rolling / trend reflect real time,
        # not raw file order (which is not guaranteed to be sorted by charttime).
        if "charttime" in sub.columns:
            sub = sub.sort_values("charttime")
        vals = sub["valuenum"].dropna()
        if vals.empty:
            continue

        lo, hi = VITAL_RANGES.get(vital, (-np.inf, np.inf))
        vals = vals[(vals >= lo) & (vals <= hi)]

        if vals.empty:
            continue

        prefix = f"vital_{vital}"
        records[stay_id][f"{prefix}_mean"] = vals.mean()
        records[stay_id][f"{prefix}_median"] = vals.median()
        records[stay_id][f"{prefix}_min"] = vals.min()
        records[stay_id][f"{prefix}_max"] = vals.max()
        records[stay_id][f"{prefix}_std"] = vals.std()
        records[stay_id][f"{prefix}_latest"] = vals.iloc[-1]
        records[stay_id][f"{prefix}_count"] = len(vals)

        if len(vals) >= 3:
            records[stay_id][f"{prefix}_rolling_mean_3"] = vals.rolling(3).mean().iloc[-1]
            records[stay_id][f"{prefix}_rolling_std_3"] = vals.rolling(3).std().iloc[-1]
            records[stay_id][f"{prefix}_trend"] = np.polyfit(np.arange(len(vals)), vals, 1)[0]
            records[stay_id][f"{prefix}_variance"] = vals.var()

    records_list = list(records.values())
    if not records_list:
        return pd.DataFrame(columns=["stay_id"])
    result = pd.DataFrame(records_list)
    log.info("Vital features: %d ICU stays × %d cols", len(result), result.shape[1])
    return result


def build_vital_features_chunked(filepath: str) -> pd.DataFrame:
    """Chunked aggregation of vital signs per stay_id."""
    vital_itemids = set()
    for ids in CFG.vitals.values():
        vital_itemids.update(ids)

    def filter_vitals(chunk: pd.DataFrame) -> pd.DataFrame:
        chunk["valuenum"] = pd.to_numeric(chunk.get("valuenum"), errors="coerce")
        return chunk[chunk["itemid"].isin(vital_itemids)]

    partial = aggregate_chunked(
        filepath=filepath,
        group_col="stay_id",
        agg_funcs={"valuenum": ["count", "mean", "min", "max", "std"]},
        date_cols=["charttime"],
        usecols=["stay_id", "hadm_id", "itemid", "valuenum", "charttime"],
        filter_fn=filter_vitals,
    )
    if partial.empty:
        return pd.DataFrame(columns=["stay_id"])

    partial.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in partial.columns]
    log.info("Chunked vital features: %d ICU stays", len(partial))
    return partial
