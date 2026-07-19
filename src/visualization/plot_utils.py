"""
src/visualization/plot_utils.py
───────────────────────────────
Shared plotting utilities for publication-quality figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from src.utils.config import CFG
from src.utils.logger import get_logger

log = get_logger(__name__)

# Publication-style defaults
mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "figure.figsize": (10, 6),
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

PALETTE = sns.color_palette("colorblind")


def get_figures_dir() -> Path:
    return Path(CFG.resolve(CFG.paths.figures))


def save_figure(fig: plt.Figure, name: str, subdir: str = "") -> Path:
    """Save matplotlib figure to reports/figures."""
    out_dir = get_figures_dir() / subdir if subdir else get_figures_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.debug("Saved figure → %s", path)
    return path


def save_plotly(fig, name: str, subdir: str = "") -> Path:
    """Save plotly figure as HTML and static PNG."""
    out_dir = get_figures_dir() / subdir if subdir else get_figures_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / f"{name}.html"
    fig.write_html(str(html_path))
    try:
        png_path = out_dir / f"{name}.png"
        fig.write_image(str(png_path), scale=2)
    except Exception as exc:  # noqa: BLE001
        log.debug("Plotly PNG export skipped for %s: %s", name, exc)
    return html_path
