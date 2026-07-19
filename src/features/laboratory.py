"""
src/features/laboratory.py
──────────────────────────
Laboratory feature engineering with chunked aggregation support.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.config import CFG
from src.utils.io_utils import aggregate_chunked
from src.utils.logger import get_logger

log = get_logger(__name__)


def _lab_stats(group: pd.DataFrame, prefix: str) -> pd.Series:
    if group.empty or "valuenum" not in group.columns:
        return pd.Series({
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_first": np.nan,
            f"{prefix}_last": np.nan,
            f"{prefix}_count": 0,
            f"{prefix}_abnormal_count": 0,
            f"{prefix}_missing_ratio": 1.0,
        })

    raw_vals = pd.to_numeric(group["valuenum"], errors="coerce")
    vals = raw_vals.dropna()
    if vals.empty:
        return pd.Series({
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_first": np.nan,
            f"{prefix}_last": np.nan,
            f"{prefix}_count": len(raw_vals),
            f"{prefix}_abnormal_count": 0,
            f"{prefix}_missing_ratio": 1.0,
        })

    sorted_g = group.sort_values("charttime") if "charttime" in group.columns else group
    sorted_vals = pd.to_numeric(sorted_g["valuenum"], errors="coerce")

    abnormal = 0
    if "flag" in group.columns:
        abnormal = int(group["flag"].astype(str).str.lower().isin(["abnormal", "high", "low"]).sum())

    return pd.Series({
        f"{prefix}_mean": vals.mean(),
        f"{prefix}_median": vals.median(),
        f"{prefix}_min": vals.min(),
        f"{prefix}_max": vals.max(),
        f"{prefix}_std": vals.std(),
        f"{prefix}_first": sorted_vals.dropna().iloc[0] if sorted_vals.notna().any() else np.nan,
        f"{prefix}_last": sorted_vals.dropna().iloc[-1] if sorted_vals.notna().any() else np.nan,
        f"{prefix}_count": len(raw_vals),
        f"{prefix}_abnormal_count": abnormal,
        f"{prefix}_missing_ratio": 1 - len(vals) / max(len(raw_vals), 1),
    })


def build_lab_features_from_df(labevents: pd.DataFrame) -> pd.DataFrame:
    """Aggregate lab features at hadm_id level from an in-memory labevents slice."""
    if labevents.empty:
        return pd.DataFrame(columns=["hadm_id"])

    labs = labevents.copy()
    labs["valuenum"] = pd.to_numeric(labs.get("valuenum"), errors="coerce")
    if "charttime" in labs.columns:
        labs["charttime"] = pd.to_datetime(labs["charttime"], errors="coerce")

    key_labs = CFG.key_labs
    reverse_key_labs = {}
    for k, v in key_labs.items():
        if isinstance(v, list):
            for item_id in v:
                reverse_key_labs[item_id] = k
        else:
            reverse_key_labs[v] = k
    records = {}
    for (hadm_id, itemid), subset in labs.groupby(["hadm_id", "itemid"], observed=True):
        if hadm_id not in records:
            records[hadm_id] = {"hadm_id": hadm_id, "lab_total_count": 0, "lab_unique_items": 0}
            for lab_name in key_labs.keys():
                records[hadm_id].update(_lab_stats(pd.DataFrame(), f"lab_{lab_name}").to_dict())
        
        lab_name = reverse_key_labs.get(itemid)
        if not lab_name:
            continue
            
        records[hadm_id]["lab_total_count"] += len(subset)
        records[hadm_id]["lab_unique_items"] += 1
        
        stats = _lab_stats(subset, f"lab_{lab_name}")
        records[hadm_id].update(stats.to_dict())

        if len(subset) >= 2 and "charttime" in subset.columns:
            ordered = subset.sort_values("charttime")
            y = pd.to_numeric(ordered["valuenum"], errors="coerce").dropna()
            if len(y) >= 2:
                x = np.arange(len(y))
                slope = np.polyfit(x, y, 1)[0]
                records[hadm_id][f"lab_{lab_name}_slope"] = slope
                records[hadm_id][f"lab_{lab_name}_change"] = y.iloc[-1] - y.iloc[0]

    records_list = list(records.values())
    if not records_list:
        return pd.DataFrame(columns=["hadm_id"])
    result = pd.DataFrame(records_list)
    log.info("Lab features: %d admissions × %d cols", len(result), result.shape[1])
    return result


def build_lab_features_chunked(
    filepath: str,
    max_chunks: Optional[int] = None,
) -> pd.DataFrame:
    """Stream labevents.csv and aggregate key lab stats per hadm_id."""
    key_itemids = set(CFG.key_labs.values())

    def filter_key_labs(chunk: pd.DataFrame) -> pd.DataFrame:
        chunk["valuenum"] = pd.to_numeric(chunk.get("valuenum"), errors="coerce")
        if "itemid" in chunk.columns:
            return chunk[chunk["itemid"].isin(key_itemids)]
        return chunk.iloc[0:0]

    partial = aggregate_chunked(
        filepath=filepath,
        group_col="hadm_id",
        agg_funcs={"valuenum": ["count", "mean", "min", "max", "std"]},
        date_cols=["charttime"],
        usecols=["hadm_id", "itemid", "valuenum", "charttime", "flag"],
        filter_fn=filter_key_labs,
    )

    if partial.empty:
        return pd.DataFrame(columns=["hadm_id"])

    partial.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in partial.columns]
    partial = partial.rename(columns={"hadm_id_": "hadm_id"} if "hadm_id_" in partial.columns else {})
    if "hadm_id" not in partial.columns and partial.index.name == "hadm_id":
        partial = partial.reset_index()

    log.info("Chunked lab features: %d admissions", len(partial))
    return partial
