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


#: Volume units in MIMIC-IV `amountuom`/`valueuom`, and their value in millilitres.
#:
#: A conversion table rather than an allow-list. Accepting `l` as if it were `ml`
#: would understate those rows a thousandfold — a subtler version of the very bug
#: this replaces, and one that an allow-list quietly invites. Every unit the cohort
#: actually contains is enumerated here; anything absent is dropped, not guessed.
#:
#: `ml/hr` and `/hour` are rates, not volumes, and are deliberately excluded: a rate
#: needs a duration before it means anything, and that is a different feature.
_ML_PER_UNIT_RAW: Dict[str, float] = {
    "ml": 1.0, "milliliters": 1.0, "millilitres": 1.0,
    "cc": 1.0, "cm3": 1.0,          # 1 cm³ is 1 mL by definition
    "l": 1000.0, "liters": 1000.0, "litres": 1000.0,
    "ul": 1e-3, "µl": 1e-3, "mcl": 1e-3, "mm^3": 1e-3,
    "nl": 1e-6, "pl": 1e-9,
    "ounces": 29.5735,              # US fluid ounce
}

#: Keys casefolded the same way the data is, so every entry is reachable.
#:
#: Not cosmetic: `str.casefold()` maps the MICRO SIGN (U+00B5, which is what MIMIC
#: actually stores) to GREEK SMALL LETTER MU (U+03BC). A literal "µl" key written
#: with the micro sign therefore never matches a casefolded lookup, and those rows
#: would have been dropped as unrecognised units while looking perfectly correct in
#: the source. Normalising both sides through the same function removes the class of
#: bug rather than this one instance.
ML_PER_UNIT: Dict[str, float] = {k.casefold(): v for k, v in _ML_PER_UNIT_RAW.items()}


def _to_millilitres(frame: pd.DataFrame, unit_col: str, value_col: str) -> pd.DataFrame:
    """
    Keep volume rows and express them all in millilitres.

    `inputevents` records 22 distinct units and only 54% of rows are millilitres.
    Summing the raw `amount` column added milligrams, micrograms, doses and mEq into
    the same total — and mcg and grams differ by a factor of a million, so a single
    microgram-dosed infusion could dominate a patient's apparent fluid intake. The
    result was not a slightly noisy volume; it was not a volume at all.

    If the unit column is missing the frame is returned unconverted, because an
    extract without `amountuom` cannot be checked and silently dropping every row
    would be a worse failure than the one being fixed.
    """
    out = frame.copy()
    out[value_col] = pd.to_numeric(out.get(value_col), errors="coerce")
    if unit_col not in out.columns:
        log.warning("%s absent; fluid totals cannot be unit-checked", unit_col)
        return out

    units = out[unit_col].astype("string").str.strip().str.casefold()
    factor = units.map(ML_PER_UNIT)
    keep = factor.notna()

    dropped = int((~keep).sum())
    if dropped:
        log.info("Fluid: dropped %d of %d rows in non-volume units (%s)",
                 dropped, len(out), ", ".join(sorted(set(units[~keep].dropna()))[:6]))

    out = out.loc[keep].copy()
    out[value_col] = out[value_col] * factor.loc[keep].astype(float)
    return out


def build_fluid_features(inputevents: pd.DataFrame, outputevents: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fluid intake/output at stay_id level, in millilitres only."""
    inp_df = pd.DataFrame(columns=["stay_id"])
    out_df = pd.DataFrame(columns=["stay_id"])

    if not inputevents.empty and "stay_id" in inputevents.columns:
        inp = _to_millilitres(inputevents, "amountuom", "amount")
        inp_df = inp.groupby("stay_id", observed=True)["amount"].agg(
            fluid_input_total="sum",
            fluid_input_count="count"
        ).reset_index()

    if not outputevents.empty and "stay_id" in outputevents.columns:
        out = _to_millilitres(outputevents, "valueuom", "value")
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
    """
    Chunked fluid aggregation, in millilitres only.

    This is the path the pipeline actually takes for `inputevents` (10.9M rows), so
    the unit filter has to be applied here and not only in the in-memory builder —
    fixing one and not the other would leave the production artifact wrong while the
    tests, which use small in-memory frames, passed.

    The unit column is added to `usecols` so `filter_fn` has something to filter on.
    """
    inp = aggregate_chunked(
        input_path,
        group_col="stay_id",
        agg_funcs={"amount": ["sum", "count"]},
        usecols=["stay_id", "amount", "amountuom"],
        filter_fn=lambda chunk: _to_millilitres(chunk, "amountuom", "amount"),
    )
    out = aggregate_chunked(
        output_path,
        group_col="stay_id",
        agg_funcs={"value": ["sum", "count"]},
        usecols=["stay_id", "value", "valueuom"],
        filter_fn=lambda chunk: _to_millilitres(chunk, "valueuom", "value"),
    )

    if inp.empty and out.empty:
        return pd.DataFrame(columns=["stay_id"])

    inp.columns = ["stay_id", "fluid_input_total", "fluid_input_count"] if len(inp.columns) == 3 else inp.columns
    out.columns = ["stay_id", "fluid_output_total", "fluid_output_count"] if len(out.columns) == 3 else out.columns

    result = inp.merge(out, on="stay_id", how="outer") if not inp.empty and not out.empty else inp if not inp.empty else out
    if "fluid_input_total" in result.columns and "fluid_output_total" in result.columns:
        result["fluid_balance"] = result["fluid_input_total"].fillna(0) - result["fluid_output_total"].fillna(0)

    return result
