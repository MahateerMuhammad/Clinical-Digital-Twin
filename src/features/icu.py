"""
src/features/icu.py
───────────────────
ICU-specific features: duration, fluids, ventilation proxies.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.utils.io_utils import aggregate_chunked
from src.utils.logger import get_logger

log = get_logger(__name__)


def build_icu_stay_features(icustays: pd.DataFrame) -> pd.DataFrame:
    """ICU stay-level features."""
    if icustays.empty:
        return pd.DataFrame(columns=["stay_id"])

    icu = icustays.copy()
    icu["intime"] = pd.to_datetime(icu["intime"], errors="coerce")
    icu["outtime"] = pd.to_datetime(icu["outtime"], errors="coerce")

    icu["icu_duration_hours"] = (icu["outtime"] - icu["intime"]).dt.total_seconds() / 3600.0
    icu["icu_duration_days"] = icu["los"]

    adm_counts = icu.groupby("hadm_id", observed=True).size().rename("n_icu_stays_per_admission")
    icu = icu.merge(adm_counts.reset_index(), on="hadm_id", how="left")

    log.info("ICU stay features: %d stays", len(icu))
    return icu


def build_fluid_features(inputevents: pd.DataFrame, outputevents: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fluid intake/output at stay_id level."""
    inp_df = pd.DataFrame(columns=["stay_id"])
    out_df = pd.DataFrame(columns=["stay_id"])

    if not inputevents.empty and "stay_id" in inputevents.columns:
        inp = inputevents.copy()
        inp["amount"] = pd.to_numeric(inp.get("amount"), errors="coerce")
        inp_df = inp.groupby("stay_id", observed=True)["amount"].agg(
            fluid_input_total="sum",
            fluid_input_count="count"
        ).reset_index()

    if not outputevents.empty and "stay_id" in outputevents.columns:
        out = outputevents.copy()
        out["value"] = pd.to_numeric(out.get("value"), errors="coerce")
        out_df = out.groupby("stay_id", observed=True)["value"].agg(
            fluid_output_total="sum",
            fluid_output_count="count"
        ).reset_index()

    if inp_df.empty and out_df.empty:
        return pd.DataFrame(columns=["stay_id"])

    result = pd.merge(inp_df, out_df, on="stay_id", how="outer") if not inp_df.empty and not out_df.empty else (inp_df if not inp_df.empty else out_df)

    if "fluid_input_total" in result.columns and "fluid_output_total" in result.columns:
        result["fluid_balance"] = result["fluid_input_total"] - result["fluid_output_total"]

    log.info("Fluid features: %d ICU stays", len(result))
    return result


def build_fluid_features_chunked(input_path: str, output_path: str) -> pd.DataFrame:
    """Chunked fluid aggregation."""
    inp = aggregate_chunked(
        input_path,
        group_col="stay_id",
        agg_funcs={"amount": ["sum", "count"]},
        usecols=["stay_id", "amount"],
    )
    out = aggregate_chunked(
        output_path,
        group_col="stay_id",
        agg_funcs={"value": ["sum", "count"]},
        usecols=["stay_id", "value"],
    )

    if inp.empty and out.empty:
        return pd.DataFrame(columns=["stay_id"])

    inp.columns = ["stay_id", "fluid_input_total", "fluid_input_count"] if len(inp.columns) == 3 else inp.columns
    out.columns = ["stay_id", "fluid_output_total", "fluid_output_count"] if len(out.columns) == 3 else out.columns

    result = inp.merge(out, on="stay_id", how="outer") if not inp.empty and not out.empty else inp if not inp.empty else out
    if "fluid_input_total" in result.columns and "fluid_output_total" in result.columns:
        result["fluid_balance"] = result["fluid_input_total"].fillna(0) - result["fluid_output_total"].fillna(0)

    return result
