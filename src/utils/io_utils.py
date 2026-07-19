"""
src/utils/io_utils.py
─────────────────────
I/O utilities for reading large CSVs in chunks, saving/loading Parquet
files with efficient dtypes, and reporting memory usage.

Usage
-----
    from src.utils.io_utils import read_csv_chunked, save_parquet, load_parquet
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.config import CFG
from src.utils.logger import get_logger

log = get_logger(__name__)


# ── Memory helpers ────────────────────────────────────────────────────────────

def get_memory_mb(df: pd.DataFrame) -> float:
    """Return DataFrame memory usage in megabytes (deep inspection)."""
    return df.memory_usage(deep=True).sum() / 1_048_576


def optimise_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast numeric columns to the smallest safe dtype and convert
    low-cardinality object columns to ``category``.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Same data with reduced memory footprint.
    """
    original_mb = get_memory_mb(df)

    for col in df.select_dtypes(include=["int64"]).columns:
        col_min, col_max = df[col].min(), df[col].max()
        if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
            df[col] = df[col].astype(np.int8)
        elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
            df[col] = df[col].astype(np.int16)
        elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
            df[col] = df[col].astype(np.int32)

    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype(np.float32)

    for col in df.select_dtypes(include=["object"]).columns:
        n_unique = df[col].nunique(dropna=False)
        if n_unique / max(len(df), 1) < 0.5:   # <50% unique → category
            df[col] = df[col].where(df[col].isna(), df[col].astype(str)).astype("category")

    new_mb = get_memory_mb(df)
    log.debug("dtype optimisation: %.1f MB → %.1f MB (%.0f%% reduction)",
              original_mb, new_mb, 100 * (1 - new_mb / original_mb))
    return df


# ── CSV reading ───────────────────────────────────────────────────────────────

def read_csv_chunked(
    filepath: Union[str, Path],
    chunk_size: Optional[int] = None,
    date_cols: Optional[List[str]] = None,
    dtype: Optional[dict] = None,
    usecols: Optional[List[str]] = None,
    row_processor: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    max_chunks: Optional[int] = None,
    show_progress: bool = True,
    low_memory: bool = False,
) -> pd.DataFrame:
    """
    Read a (potentially very large) CSV file in chunks and concatenate the
    result into a single DataFrame.

    Parameters
    ----------
    filepath : str | Path
        Path to the CSV file.
    chunk_size : int, optional
        Number of rows per chunk. Defaults to ``CFG.io.chunk_size``.
    date_cols : list[str], optional
        Column names to parse as datetime.
    dtype : dict, optional
        Column dtype overrides passed to ``pd.read_csv``.
    usecols : list[str], optional
        Subset of columns to load.
    row_processor : callable, optional
        Function applied to each chunk *before* concatenation, useful for
        lightweight per-chunk transformations.
    max_chunks : int, optional
        Hard limit on the number of chunks (for quick smoke tests).
    show_progress : bool
        Show a tqdm progress bar.
    low_memory : bool
        Pass ``low_memory`` to ``pd.read_csv``.

    Returns
    -------
    pd.DataFrame
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    chunk_size = chunk_size or CFG.io.chunk_size
    date_cols = date_cols or []

    log.info("Reading %s  (chunk_size=%d)", filepath.name, chunk_size)

    compression = "gzip" if filepath.suffix == ".gz" else None
    reader = pd.read_csv(
        filepath,
        compression=compression,
        chunksize=chunk_size,
        parse_dates=date_cols if date_cols else False,
        dtype=dtype,
        usecols=usecols,
        low_memory=low_memory,
        on_bad_lines="warn",      # don't crash on truncated files (labevents)
    )

    chunks: List[pd.DataFrame] = []
    file_size_mb = filepath.stat().st_size / 1_048_576
    estimated_chunks = max(1, int(file_size_mb / (chunk_size * 0.1)))   # rough estimate

    with tqdm(
        total=max_chunks or estimated_chunks,
        unit="chunk",
        desc=filepath.stem,
        disable=not show_progress,
    ) as pbar:
        for i, chunk in enumerate(reader):
            if max_chunks and i >= max_chunks - 1:
                log.warning("%s is a PARTIAL FILE (~%d rows loaded). max_chunks=%d reached.", filepath.name, (i + 1) * chunk_size, max_chunks)
                break

            if row_processor is not None:
                chunk = row_processor(chunk)

            chunks.append(chunk)
            pbar.update(1)

    if not chunks:
        log.warning("No data read from %s — returning empty DataFrame.", filepath.name)
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    log.info(
        "Loaded %s: %d rows × %d cols (%.1f MB in RAM)",
        filepath.name, len(df), df.shape[1], get_memory_mb(df),
    )
    return df


def read_csv_full(
    filepath: Union[str, Path],
    date_cols: Optional[List[str]] = None,
    dtype: Optional[dict] = None,
    usecols: Optional[List[str]] = None,
    low_memory: bool = False,
) -> pd.DataFrame:
    """
    Read a small-to-medium CSV file in a single pass (no chunking).

    Parameters
    ----------
    filepath : str | Path
    date_cols : list[str], optional
    dtype : dict, optional
    usecols : list[str], optional
    low_memory : bool

    Returns
    -------
    pd.DataFrame
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    log.info("Reading (full) %s", filepath.name)
    compression = "gzip" if filepath.suffix == ".gz" else None
    df = pd.read_csv(
        filepath,
        compression=compression,
        parse_dates=date_cols if date_cols else False,
        dtype=dtype,
        usecols=usecols,
        low_memory=low_memory,
        on_bad_lines="warn",
    )
    log.info(
        "Loaded %s: %d rows × %d cols (%.1f MB)",
        filepath.name, len(df), df.shape[1], get_memory_mb(df),
    )
    return df


# ── Parquet I/O ───────────────────────────────────────────────────────────────

def save_parquet(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    compression: Optional[str] = None,
    engine: Optional[str] = None,
    index: bool = False,
    optimise_memory: bool = True,
) -> Path:
    """
    Save a DataFrame as a Parquet file with optional dtype optimisation.

    Parameters
    ----------
    df : pd.DataFrame
    output_path : str | Path
        Destination file path (parent directories are created automatically).
    compression : str, optional
        Defaults to ``CFG.io.parquet_compression``.
    engine : str, optional
        Defaults to ``CFG.io.parquet_engine``.
    index : bool
        Whether to include the DataFrame index.
    optimise_memory : bool
        If True, downcast dtypes before saving.

    Returns
    -------
    Path
        Resolved path to the saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    compression = compression or CFG.io.parquet_compression
    engine = engine or CFG.io.parquet_engine

    if optimise_memory:
        df = optimise_dtypes(df.copy())

    before_mb = get_memory_mb(df)
    df.to_parquet(
        output_path,
        engine=engine,
        compression=compression,
        index=index,
    )
    file_size_mb = output_path.stat().st_size / 1_048_576

    log.info(
        "Saved Parquet → %s  (%.1f MB RAM → %.1f MB on disk, compression=%s)",
        output_path.name, before_mb, file_size_mb, compression,
    )
    return output_path


def load_parquet(
    filepath: Union[str, Path],
    columns: Optional[List[str]] = None,
    engine: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load a Parquet file with optional column pruning.

    Parameters
    ----------
    filepath : str | Path
    columns : list[str], optional
        Subset of columns to load (columnar skip for speed).
    engine : str, optional

    Returns
    -------
    pd.DataFrame
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Parquet file not found: {filepath}")

    engine = engine or CFG.io.parquet_engine
    df = pd.read_parquet(filepath, engine=engine, columns=columns)

    log.info(
        "Loaded Parquet %s: %d rows × %d cols (%.1f MB)",
        filepath.name, len(df), df.shape[1], get_memory_mb(df),
    )
    return df


# ── Chunked Parquet aggregation (for giant tables) ────────────────────────────

def aggregate_chunked(
    filepath: Union[str, Path],
    group_col: str,
    agg_funcs: dict,
    date_cols: Optional[List[str]] = None,
    usecols: Optional[List[str]] = None,
    chunk_size: Optional[int] = None,
    filter_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
) -> pd.DataFrame:
    """
    Stream through a large CSV in chunks and build an aggregation on the fly
    without loading the entire file into memory.

    Parameters
    ----------
    filepath : str | Path
    group_col : str
        Column to group by (e.g. 'subject_id', 'hadm_id').
    agg_funcs : dict
        Aggregation spec passed to ``groupby().agg()``.
    date_cols : list[str], optional
    usecols : list[str], optional
        Must include ``group_col``.
    chunk_size : int, optional
    filter_fn : callable, optional
        Optional row-level filter applied to each chunk before aggregating.

    Returns
    -------
    pd.DataFrame
        Final aggregated result (concatenated and re-aggregated over chunks).
    """
    filepath = Path(filepath)
    chunk_size = chunk_size or CFG.io.chunk_size

    if usecols and group_col not in usecols:
        usecols = [group_col] + list(usecols)

    log.info(
        "Chunked aggregation on %s  (group_col=%s, chunk_size=%d)",
        filepath.name, group_col, chunk_size,
    )

    reader = pd.read_csv(
        filepath,
        compression="gzip" if filepath.suffix == ".gz" else None,
        chunksize=chunk_size,
        parse_dates=date_cols if date_cols else False,
        usecols=usecols,
        low_memory=False,
        on_bad_lines="warn",
    )

    partials: List[pd.DataFrame] = []
    file_size_mb = filepath.stat().st_size / 1_048_576
    estimated_chunks = max(1, int(file_size_mb / (chunk_size * 0.1)))

    with tqdm(total=estimated_chunks, unit="chunk", desc=f"agg:{filepath.stem}") as pbar:
        for chunk in reader:
            if filter_fn is not None:
                chunk = filter_fn(chunk)
            if len(chunk) == 0:
                pbar.update(1)
                continue
            partial = chunk.groupby(group_col, observed=True).agg(agg_funcs)
            partials.append(partial)
            pbar.update(1)

    if not partials:
        log.warning("No data aggregated from %s", filepath.name)
        return pd.DataFrame()

    combined = pd.concat(partials)
    result = combined.groupby(level=0).agg(agg_funcs)

    log.info("Aggregation complete: %d groups", len(result))
    return result.reset_index()
