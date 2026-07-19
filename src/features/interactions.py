"""
src/features/interactions.py
────────────────────────────
Clinical interaction features.
"""

from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


def build_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction terms from existing engineered features."""
    out = df.copy()

    interactions = [
        ("anchor_age", "dx_diabetes", "age_x_diabetes"),
        ("anchor_age", "lab_sodium_mean", "age_x_sodium"),
        ("anchor_age", "lab_creatinine_mean", "creatinine_x_age"),
        ("vital_heart_rate_mean", "vital_temperature_c_mean", "hr_x_temperature"),
        ("lab_creatinine_mean", "dx_ckd", "creatinine_x_ckd"),
        ("charlson_comorbidity_index", "anchor_age", "cci_x_age"),
    ]

    created = 0
    for col_a, col_b, name in interactions:
        if col_a in out.columns and col_b in out.columns:
            out[name] = out[col_a] * out[col_b]
            created += 1

    log.info("Created %d interaction features", created)
    return out
