"""
src/utils/logger.py
───────────────────
Centralised logging configuration for the Clinical Digital Twin pipeline.

Usage
-----
    from src.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Pipeline started")
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


# ── Try Rich handler for colourful console output ────────────────────────────
try:
    from rich.logging import RichHandler
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


# ── Module-level registry so we don't create duplicate handlers ──────────────
_LOGGERS: dict[str, logging.Logger] = {}


def setup_root_logger(
    log_file: str = "logs/pipeline.log",
    level: str = "INFO",
    max_bytes: int = 10_485_760,   # 10 MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Configure the root logger with a rotating file handler and a console
    handler. Call this once at pipeline startup.

    Parameters
    ----------
    log_file : str
        Path to the log file (parent directory is created if absent).
    level : str
        Logging level string, e.g. 'DEBUG', 'INFO', 'WARNING'.
    max_bytes : int
        Maximum size of each log file before rotation.
    backup_count : int
        Number of rotated backup files to keep.

    Returns
    -------
    logging.Logger
        Configured root logger.
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger("cdt")      # "cdt" = Clinical Digital Twin
    root.setLevel(numeric_level)

    if root.handlers:
        return root   # already configured

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Rotating file handler ────────────────────────────────────────────────
    fh = RotatingFileHandler(
        filename=str(log_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    fh.setLevel(numeric_level)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # ── Console handler ──────────────────────────────────────────────────────
    if _RICH_AVAILABLE:
        ch = RichHandler(
            level=numeric_level,
            show_path=False,
            rich_tracebacks=True,
            markup=True,
        )
    else:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(numeric_level)
        ch.setFormatter(fmt)

    root.addHandler(ch)

    return root


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Return a child logger under the 'cdt' namespace.

    Parameters
    ----------
    name : str
        Module name, typically ``__name__``.
    log_file : str, optional
        If provided, overrides the default log file path.

    Returns
    -------
    logging.Logger
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    # Ensure root is configured (uses defaults if not yet set up)
    if not logging.getLogger("cdt").handlers:
        _log_file = log_file or os.environ.get("CDT_LOG_FILE", "logs/pipeline.log")
        setup_root_logger(log_file=_log_file)

    logger = logging.getLogger(f"cdt.{name}")
    _LOGGERS[name] = logger
    return logger
