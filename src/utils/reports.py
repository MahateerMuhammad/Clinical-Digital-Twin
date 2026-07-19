"""Shared report formatting utilities."""

from __future__ import annotations

import pandas as pd


def df_to_markdown(df: pd.DataFrame) -> str:
    """Convert DataFrame to markdown table."""
    try:
        return df.to_markdown(index=False)
    except ImportError:
        header = "| " + " | ".join(df.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
        rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.values]
        return "\n".join([header, sep] + rows)
