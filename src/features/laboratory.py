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


def _reverse_key_labs() -> Dict[int, str]:
    out: Dict[int, str] = {}
    for name, ids in CFG.key_labs.items():
        for item_id in (ids if isinstance(ids, list) else [ids]):
            out[item_id] = name
    return out


def build_lab_features_vectorised(labevents: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorised equivalent of :func:`build_lab_features_from_df`.

    The original iterates over every ``(hadm_id, itemid)`` group in Python and
    accumulates a dict per admission. On the full cohort that is ~8.4M groups and
    ~430k dicts of ~200 float keys, which exceeded 32 GB of RAM and spent 13 of 16
    hours paging rather than computing. This produces the same numbers with
    ``groupby().agg()`` in minutes and roughly 1 GB.

    Multi-itemid analytes are pooled, not overwritten
    ─────────────────────────────────────────────────
    When a lab maps to several ``itemid``s (only ``creatinine_wb`` does here), the
    original ``records[hadm].update(stats)`` let the highest ``itemid`` present for an
    admission **overwrite** the others, silently discarding the rest. That behaviour
    was reproduced verbatim through the identifier correction so the
    corrected-vs-baseline comparison isolated one change at a time.

    That comparison is published, so the quirk is now fixed: readings are pooled
    across every ``itemid`` mapping to the same analyte, which is what the statistics
    claim to describe. On the current cohort this changes nothing measurable —
    ``creatinine_wb`` appears in 4,007 admissions and **none** carries both 52024 and
    52546, so there was never anything to discard. The fix is for the rebuild where
    that stops being true, which nothing would otherwise report.
    """
    if labevents.empty:
        return pd.DataFrame(columns=["hadm_id"])

    rev = _reverse_key_labs()
    cols = [c for c in ("hadm_id", "itemid", "valuenum", "charttime", "flag")
            if c in labevents.columns]
    labs = labevents[cols].copy()
    labs["valuenum"] = pd.to_numeric(labs["valuenum"], errors="coerce")
    if "charttime" in labs.columns:
        labs["charttime"] = pd.to_datetime(labs["charttime"], errors="coerce")
    if "itemid" in labs.columns:
        labs = labs[labs["itemid"].isin(rev.keys())]
    if labs.empty:
        return pd.DataFrame(columns=["hadm_id"])
    labs = labs[labs["hadm_id"].notna()]
    labs["lab_name"] = labs["itemid"].map(rev)

    # ── per-admission totals, over every key-lab row ─────────────────────
    totals = labs.groupby("hadm_id", observed=True).size().rename("lab_total_count")
    uniq = (labs.groupby(["hadm_id", "itemid"], observed=True).size()
                .groupby("hadm_id", observed=True).size().rename("lab_unique_items"))

    # ── pool every itemid mapping to the same analyte ────────────────────
    # Previously: keep only rows whose itemid equalled the per-admission max, i.e.
    # `sel = labs[labs["itemid"] == labs.groupby([...])["itemid"].transform("max")]`.
    # That silently dropped readings from the lower-numbered assay. See the docstring.
    sel = labs

    keys = ["hadm_id", "lab_name"]
    g = sel.groupby(keys, observed=True)

    agg = g["valuenum"].agg(
        mean="mean", median="median", min="min", max="max", std="std",
        count="size", n_valid="count",
    )
    agg["missing_ratio"] = 1.0 - agg["n_valid"] / agg["count"].clip(lower=1)

    if "flag" in sel.columns:
        abnormal = (sel["flag"].astype(str).str.lower().isin(["abnormal", "high", "low"])
                    .groupby([sel["hadm_id"], sel["lab_name"]], observed=True).sum())
        agg["abnormal_count"] = abnormal
    else:
        agg["abnormal_count"] = 0
    agg["abnormal_count"] = agg["abnormal_count"].fillna(0)
    # the original returns 0 abnormal when no numeric value survived
    agg.loc[agg["n_valid"] == 0, "abnormal_count"] = 0

    # ── first / last / slope / change over charttime-ordered valid values ─
    ordered = sel[sel["valuenum"].notna()]
    if "charttime" in ordered.columns:
        ordered = ordered.sort_values(keys + ["charttime"], na_position="last",
                                      kind="mergesort")
    og = ordered.groupby(keys, observed=True)
    agg["first"] = og["valuenum"].first()
    agg["last"] = og["valuenum"].last()

    # slope of an ordinary least squares fit against position 0..n-1, computed in
    # closed form: with x = 0..n-1, Sxx = n(n^2-1)/12 and Sxy = sum(x*y) - n*xbar*ybar
    x = og.cumcount()
    y = ordered["valuenum"].to_numpy(dtype="float64")
    tmp = pd.DataFrame({"hadm_id": ordered["hadm_id"].to_numpy(),
                        "lab_name": ordered["lab_name"].to_numpy(),
                        "x": x.to_numpy(dtype="float64"), "y": y})
    tmp["xy"] = tmp["x"] * tmp["y"]
    s = tmp.groupby(keys, observed=True).agg(n=("y", "size"), sy=("y", "sum"),
                                             sxy=("xy", "sum"))
    n = s["n"].to_numpy(dtype="float64")
    xbar = (n - 1.0) / 2.0
    ybar = s["sy"].to_numpy(dtype="float64") / np.where(n > 0, n, 1.0)
    sxx = n * (n * n - 1.0) / 12.0
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = (s["sxy"].to_numpy(dtype="float64") - n * xbar * ybar) / sxx
    s["slope"] = np.where(n >= 2, slope, np.nan)
    agg["slope"] = s["slope"]
    agg["change"] = agg["last"] - agg["first"]
    # original only records slope/change when at least two valid values exist
    too_short = agg["n_valid"] < 2
    agg.loc[too_short, ["slope", "change"]] = np.nan

    agg = agg.drop(columns=["n_valid"])

    # ── pivot to one row per admission, one column per lab statistic ─────
    wide = agg.unstack("lab_name")
    wide.columns = [f"lab_{lab}_{stat}" for stat, lab in wide.columns]

    result = wide.join(totals, how="outer").join(uniq, how="outer").reset_index()

    # guarantee a column for every configured lab, as the original did
    for lab in CFG.key_labs:
        for stat in ("mean", "median", "min", "max", "std", "first", "last",
                     "count", "abnormal_count", "missing_ratio"):
            col = f"lab_{lab}_{stat}"
            if col not in result.columns:
                result[col] = np.nan
    for lab in CFG.key_labs:
        for stat, default in (("count", 0), ("abnormal_count", 0), ("missing_ratio", 1.0)):
            col = f"lab_{lab}_{stat}"
            result[col] = result[col].fillna(default)

    for c in ("lab_total_count", "lab_unique_items"):
        result[c] = result[c].fillna(0).astype("int64")

    # match the original's plain int64 key: a nullable Int32 index would change
    # the dtype of every downstream merge against admissions
    result["hadm_id"] = result["hadm_id"].astype("int64")

    log.info("Lab features (vectorised): %d admissions × %d cols",
             len(result), result.shape[1])
    return result


def build_lab_features_from_df(labevents: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate lab features at hadm_id level from an in-memory labevents slice.

    Delegates to :func:`build_lab_features_vectorised`, which is ~230x faster and
    uses a fraction of the memory. The original per-group implementation is kept
    as :func:`build_lab_features_reference` and is used by the tests to prove the
    two agree; it is not viable on the full cohort (it exhausted 32 GB of RAM).

    .. warning::
       The aggregates produced here span the **entire admission**. ``charttime``
       is used only to order records, never to filter them, so ``lab_*_min``,
       ``lab_*_max`` and ``lab_*_median`` include values charted days after
       admission. For an early-observation-window model use
       :func:`build_lab_features_windowed`.
    """
    return build_lab_features_vectorised(labevents)


def restrict_to_observation_window(
    labevents: pd.DataFrame,
    admissions: pd.DataFrame,
    hours: float = 24.0,
) -> pd.DataFrame:
    """
    Keep only lab draws charted within ``hours`` of ``admittime``.

    Draws charted *before* admission are retained deliberately: 15% of first labs
    in this cohort have a negative offset because they were drawn in the emergency
    department before the inpatient admission was registered. Those results are
    on the chart at hour zero, so they are legitimately available to an
    admission-time model; excluding them would discard genuine early physiology.

    Only the columns the aggregator consumes are carried through, and non-key
    itemids are dropped up front, because the full labevents table is ~768 MB on
    disk and materialising a merge over all of it is what previously drove this
    pipeline into swap.
    """
    if labevents.empty or admissions.empty or "charttime" not in labevents.columns:
        return labevents.iloc[0:0]

    cols = [c for c in ("hadm_id", "itemid", "valuenum", "charttime", "flag")
            if c in labevents.columns]
    labs = labevents[cols].copy()

    if "itemid" in labs.columns:
        labs = labs[labs["itemid"].isin(_reverse_key_labs().keys())]
    labs = labs[labs["hadm_id"].notna()]
    if labs.empty:
        return labs

    adm = admissions[["hadm_id", "admittime"]].dropna(subset=["hadm_id"])
    admit = pd.to_datetime(adm["admittime"], errors="coerce")
    admit.index = adm["hadm_id"].astype("int64").values

    labs["charttime"] = pd.to_datetime(labs["charttime"], errors="coerce")
    offset_h = (
        labs["charttime"] - labs["hadm_id"].astype("int64").map(admit)
    ).dt.total_seconds() / 3600.0

    kept = labs.loc[offset_h.notna() & (offset_h <= hours)]
    log.info(
        "Observation window (<=%.0fh): %d of %d key-lab rows retained (%.1f%%)",
        hours, len(kept), len(labs), 100.0 * len(kept) / max(len(labs), 1),
    )
    return kept


def build_lab_features_windowed(
    labevents: pd.DataFrame,
    admissions: pd.DataFrame,
    hours: float = 24.0,
    suffix: str = "_24h",
) -> pd.DataFrame:
    """
    Lab aggregates restricted to the first ``hours`` of the admission.

    Emits the same statistics as :func:`build_lab_features_vectorised` with every
    column suffixed (``lab_wbc_max`` -> ``lab_wbc_max_24h``), so the windowed and
    full-stay feature sets can coexist on one row. The full-stay columns remain
    correct for the full-stay protocols (mortality Run A/B); the suffixed columns
    are what a strict early-window protocol should consume.
    """
    windowed = restrict_to_observation_window(labevents, admissions, hours=hours)
    feats = build_lab_features_vectorised(windowed)
    if feats.empty:
        return pd.DataFrame(columns=["hadm_id"])
    return feats.rename(
        columns={c: f"{c}{suffix}" for c in feats.columns if c != "hadm_id"}
    )


def build_lab_features_reference(labevents: pd.DataFrame) -> pd.DataFrame:
    """Original per-(hadm_id, itemid) implementation. Reference only — slow.

    Retained so the vectorised version can be checked against it. Validated
    identical across 690 column comparisons on real data.
    """
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
    
    # Fast filter: drop non-key lab items before running groupby
    if "itemid" in labs.columns:
        labs = labs[labs["itemid"].isin(reverse_key_labs.keys())]

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
