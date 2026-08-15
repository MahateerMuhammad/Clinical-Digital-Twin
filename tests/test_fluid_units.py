"""
Fluid totals must be volumes, not the sum of whatever units a row happened to use.

`inputevents` records 22 distinct units and only 54% of rows are millilitres. The
builder summed the raw `amount` column, so milligrams, micrograms, doses, grams and
mEq all landed in `fluid_input_total`. Since mcg and grams differ by 10^6, one
microgram-dosed infusion could dominate a patient's apparent fluid intake.

The feature reaches no model today, which is the only reason this was harmless. It
would become a garbage input the moment fluids are adopted — so it is fixed now,
while the cost is one filter rather than a retrain.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.icu import ML_PER_UNIT, _to_millilitres, build_fluid_features


def _inputs() -> pd.DataFrame:
    return pd.DataFrame([
        {"stay_id": 1, "amount": 500.0, "amountuom": "ml"},
        {"stay_id": 1, "amount": 250.0, "amountuom": "mL"},     # case variation
        {"stay_id": 1, "amount": 80.0, "amountuom": "mg"},      # mass, must not count
        {"stay_id": 1, "amount": 4000.0, "amountuom": "mcg"},   # the 10^6 hazard
        {"stay_id": 1, "amount": 2.0, "amountuom": "dose"},
        {"stay_id": 2, "amount": 100.0, "amountuom": "ml"},
    ])


def _outputs() -> pd.DataFrame:
    return pd.DataFrame([
        {"stay_id": 1, "value": 300.0, "valueuom": "ml"},
        {"stay_id": 1, "value": 5.0, "valueuom": "mEq"},
        {"stay_id": 2, "value": 50.0, "valueuom": "ml"},
    ])


def test_only_volumes_are_summed():
    out = build_fluid_features(_inputs(), _outputs()).set_index("stay_id")
    assert out.loc[1, "fluid_input_total"] == 750.0, (
        "expected 500 + 250 mL; a larger number means mass units were added in")
    assert out.loc[1, "fluid_output_total"] == 300.0


def test_counts_exclude_non_volume_rows():
    """A count that includes mg rows misstates how much fluid charting occurred."""
    out = build_fluid_features(_inputs(), _outputs()).set_index("stay_id")
    assert out.loc[1, "fluid_input_count"] == 2
    assert out.loc[1, "fluid_output_count"] == 1


def test_balance_is_computed_from_filtered_totals():
    out = build_fluid_features(_inputs(), _outputs()).set_index("stay_id")
    assert out.loc[1, "fluid_balance"] == 450.0     # 750 in - 300 out
    assert out.loc[2, "fluid_balance"] == 50.0


def test_the_microgram_hazard_specifically():
    """4000 mcg is 0.004 g. Summed as a volume it dwarfs the real 750 mL."""
    naive = _inputs()["amount"].sum()
    filtered = build_fluid_features(_inputs(), _outputs()).set_index("stay_id")
    assert naive > 4 * filtered.loc[1, "fluid_input_total"], (
        "fixture no longer demonstrates the defect it was written for")


@pytest.mark.parametrize("unit", ["ml", "mL", "ML", " ml ", "cc", "cm3"])
def test_millilitre_spellings_pass_through_unscaled(unit):
    frame = pd.DataFrame([{"stay_id": 1, "amount": 10.0, "amountuom": unit}])
    out = _to_millilitres(frame, "amountuom", "amount")
    assert len(out) == 1 and out["amount"].iloc[0] == 10.0


@pytest.mark.parametrize("unit, expected", [
    ("L", 10_000.0), ("liters", 10_000.0),      # 10 L is 10,000 mL
    ("ul", 0.01), ("µl", 0.01),
    ("ounces", 295.735),
])
def test_other_volume_units_are_converted_not_just_accepted(unit, expected):
    """
    The trap in the first version of this fix.

    Treating `l` as a volume and summing it unchanged understates those rows by a
    factor of a thousand — a quieter version of the bug being fixed. Volumes must be
    converted to a common unit, not merely allowed through.
    """
    frame = pd.DataFrame([{"stay_id": 1, "amount": 10.0, "amountuom": unit}])
    out = _to_millilitres(frame, "amountuom", "amount")
    assert out["amount"].iloc[0] == pytest.approx(expected)


@pytest.mark.parametrize("unit", ["ml/hr", "/hour"])
def test_rates_are_not_treated_as_volumes(unit):
    """A rate needs a duration before it is a volume. Both appear in inputevents."""
    frame = pd.DataFrame([{"stay_id": 1, "amount": 10.0, "amountuom": unit}])
    assert _to_millilitres(frame, "amountuom", "amount").empty


@pytest.mark.parametrize("unit", ["mg", "mcg", "grams", "dose", "units", "mEq"])
def test_non_volume_units_are_rejected(unit):
    frame = pd.DataFrame([{"stay_id": 1, "amount": 10.0, "amountuom": unit}])
    assert _to_millilitres(frame, "amountuom", "amount").empty


def test_missing_unit_column_does_not_silently_drop_everything():
    """
    An extract without `amountuom` cannot be checked.

    Returning nothing would be a worse failure than the one being fixed — a silent
    zero for every patient — so the builder warns and passes the rows through.
    """
    frame = pd.DataFrame([{"stay_id": 1, "amount": 10.0}])
    assert len(_to_millilitres(frame, "amountuom", "amount")) == 1


def test_non_numeric_values_become_missing_not_zero():
    frame = pd.DataFrame([{"stay_id": 1, "amount": "n/a", "amountuom": "ml"}])
    assert _to_millilitres(frame, "amountuom", "amount")["amount"].isna().all()


def test_conversion_table_holds_no_mass_or_activity_unit():
    """Cheap guard against someone adding `mg` to the table by hand."""
    forbidden = {"mg", "mcg", "g", "grams", "dose", "units", "meq", "mmol",
                 "ng", "pg", "international units"}
    assert not (set(ML_PER_UNIT) & forbidden)


def test_every_factor_is_positive():
    assert all(v > 0 for v in ML_PER_UNIT.values())


def test_units_are_stored_casefolded():
    """Lookup casefolds the data, so a capitalised key would never be reached."""
    assert all(k == k.casefold() for k in ML_PER_UNIT)
