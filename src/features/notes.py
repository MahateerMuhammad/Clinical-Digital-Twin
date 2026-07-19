"""
src/features/notes.py
─────────────────────
Clinical notes preprocessing and text features.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)

MEDICAL_KEYWORDS = [
    "diabetes", "hypertension", "pneumonia", "sepsis", "heart failure",
    "copd", "stroke", "myocardial infarction", "renal failure", "cancer",
    "antibiotic", "insulin", "ventilator", "intubation", "dialysis",
]
NEGATION_PATTERNS = re.compile(r"\b(no|not|denies|without|negative for)\b", re.I)
NON_ALPHA = re.compile(r"[^a-zA-Z0-9\s]")


def clean_note_text(text: str) -> str:
    """Basic NLP preprocessing for clinical notes."""
    if not isinstance(text, str) or not text.strip():
        return ""
    t = text.lower()
    t = NON_ALPHA.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def extract_note_features(notes: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Extract structured features from clinical notes."""
    if notes.empty or text_col not in notes.columns:
        return pd.DataFrame(columns=["note_id"])

    df = notes.copy()
    df["text_clean"] = df[text_col].astype(str).map(clean_note_text)
    df["char_count"] = df[text_col].astype(str).str.len()
    df["word_count"] = df["text_clean"].str.split().str.len()
    df["sentence_count"] = df[text_col].astype(str).str.count(r"[.!?]+").clip(lower=1)

    def keyword_count(text: str) -> int:
        return sum(1 for kw in MEDICAL_KEYWORDS if kw in text)

    df["medical_keyword_count"] = df["text_clean"].map(keyword_count)
    df["negation_count"] = df[text_col].astype(str).str.count(NEGATION_PATTERNS)

    try:
        import textstat
        df["readability_flesch"] = df[text_col].astype(str).map(
            lambda x: textstat.flesch_reading_ease(x) if len(x) > 20 else np.nan
        )
    except ImportError:
        df["readability_flesch"] = np.nan

    # TF-IDF ready text stored separately
    df["text_tfidf_ready"] = df["text_clean"]

    keep = [c for c in df.columns if c in (
        "note_id", "subject_id", "hadm_id", "note_type", "charttime",
        "char_count", "word_count", "sentence_count", "medical_keyword_count",
        "negation_count", "readability_flesch", "text_clean", "text_tfidf_ready",
    )]
    result = df[keep]
    log.info("Note features: %d notes", len(result))
    return result


def aggregate_note_features(note_features: pd.DataFrame) -> pd.DataFrame:
    """Aggregate note features to admission level."""
    if note_features.empty or "hadm_id" not in note_features.columns:
        return pd.DataFrame(columns=["hadm_id"])

    agg = note_features.groupby("hadm_id", observed=True).agg(
        note_count=("note_id", "count") if "note_id" in note_features.columns else ("hadm_id", "count"),
        note_char_count_mean=("char_count", "mean"),
        note_word_count_mean=("word_count", "mean"),
        note_word_count_sum=("word_count", "sum"),
        note_sentence_count_mean=("sentence_count", "mean"),
        note_medical_keyword_mean=("medical_keyword_count", "mean"),
        note_negation_mean=("negation_count", "mean"),
    ).reset_index()

    log.info("Aggregated note features: %d admissions", len(agg))
    return agg
