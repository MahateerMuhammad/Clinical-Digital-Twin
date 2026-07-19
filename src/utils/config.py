"""
src/utils/config.py
───────────────────
Loads and validates configs/config.yaml and exposes a singleton ``CFG``
object used across the entire pipeline.

Usage
-----
    from src.utils.config import CFG
    raw_hosp = CFG.paths.raw.hosp
    chunk_size = CFG.io.chunk_size
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.utils.logger import get_logger

log = get_logger(__name__)

# Default config location — can be overridden via CDT_CONFIG env variable
_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


# ── Nested config dataclasses ─────────────────────────────────────────────────

@dataclass
class RawPaths:
    hosp: str = "data/raw/hosp"
    icu: str = "data/raw/icu"
    notes: str = "data/raw/note"


@dataclass
class Paths:
    raw: RawPaths = field(default_factory=RawPaths)
    interim: str = "data/interim"
    processed: str = "data/processed"
    external: str = "data/external"
    reports: str = "reports"
    figures: str = "reports/figures"
    tables: str = "reports/tables"
    logs: str = "logs"


@dataclass
class Logging:
    level: str = "INFO"
    log_file: str = "logs/pipeline.log"
    max_bytes: int = 10_485_760
    backup_count: int = 5


@dataclass
class IO:
    chunk_size: int = 500_000
    parquet_compression: str = "snappy"
    parquet_engine: str = "pyarrow"


@dataclass
class FeatureSelection:
    variance_threshold: float = 0.01
    correlation_threshold: float = 0.95
    missing_rate_threshold: float = 0.70


@dataclass
class Targets:
    mortality_inhosp: str = "hospital_expire_flag"
    los_days: str = "los_days"
    icu_admission: str = "has_icu_stay"
    icu_los_days: str = "icu_los_days"
    readmission_30d: str = "readmission_30d"


@dataclass
class PipelineSettings:
    full_load: bool = True
    max_chunks_large_tables: Optional[int] = None
    eda_sample_rows: int = 50_000
    eda_max_chunks_per_large_table: int = 3
    skip_tables: List[str] = field(default_factory=list)


@dataclass
class Config:
    """Top-level singleton configuration object."""

    project_name: str = "Clinical Digital Twin"
    version: str = "1.0.0"
    random_seed: int = 42

    paths: Paths = field(default_factory=Paths)
    logging: Logging = field(default_factory=Logging)
    io: IO = field(default_factory=IO)
    feature_selection: FeatureSelection = field(default_factory=FeatureSelection)
    targets: Targets = field(default_factory=Targets)
    pipeline: PipelineSettings = field(default_factory=PipelineSettings)

    # Raw YAML content for arbitrary access (e.g., table configs, vitals, charlson)
    tables: Dict[str, Any] = field(default_factory=dict)
    vitals: Dict[str, List[int]] = field(default_factory=dict)
    key_labs: Dict[str, int] = field(default_factory=dict)
    charlson: Dict[str, Any] = field(default_factory=dict)

    # Resolved project root
    project_root: Path = field(default_factory=lambda: Path.cwd())

    def resolve(self, relative_path: str) -> Path:
        """Resolve a path relative to the project root."""
        p = self.project_root / relative_path
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def table_file(self, table_name: str) -> Path:
        """Return the absolute path for a named table's raw CSV/CSV.GZ."""
        t = self.tables[table_name]
        src = t["source"]
        fname = t["file"]
        if src == "hosp":
            base = self.paths.raw.hosp
        elif src == "icu":
            base = self.paths.raw.icu
        elif src in ("notes", "note"):
            base = self.paths.raw.notes
        else:
            raise ValueError(f"Unknown source '{src}' for table '{table_name}'")

        root = self.project_root / base
        direct = root / fname
        if direct.exists():
            return direct

        stem = Path(fname).stem
        for candidate in (root / f"{stem}.csv.gz", root / f"{stem}.gz"):
            if candidate.exists():
                return candidate

        return direct

    def __repr__(self) -> str:
        return (
            f"Config(project='{self.project_name}', version='{self.version}', "
            f"root='{self.project_root}')"
        )


# ── Loader ────────────────────────────────────────────────────────────────────

def _deep_get(d: dict, *keys: str, default: Any = None) -> Any:
    """Safely navigate nested dict."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def load_config(config_path: Optional[Path] = None) -> Config:
    """
    Parse the YAML config file and return a fully-populated ``Config`` object.

    Parameters
    ----------
    config_path : Path, optional
        Override the default config location. Falls back to the CDT_CONFIG
        environment variable, then to ``configs/config.yaml``.

    Returns
    -------
    Config
    """
    path = config_path or Path(os.environ.get("CDT_CONFIG", str(_DEFAULT_CONFIG)))

    if not path.exists():
        log.warning("Config file not found at %s — using defaults.", path)
        return Config()

    with open(path, "r", encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh)

    cfg = Config()

    # Project metadata
    proj = raw.get("project", {})
    cfg.project_name = proj.get("name", cfg.project_name)
    cfg.version = proj.get("version", cfg.version)
    cfg.random_seed = proj.get("random_seed", cfg.random_seed)

    # Resolve project root as the directory two levels above this file
    cfg.project_root = Path(__file__).resolve().parents[2]

    # Paths
    p = raw.get("paths", {})
    raw_p = p.get("raw", {})
    cfg.paths = Paths(
        raw=RawPaths(
            hosp=raw_p.get("hosp", cfg.paths.raw.hosp),
            icu=raw_p.get("icu", cfg.paths.raw.icu),
            notes=raw_p.get("notes", cfg.paths.raw.notes),
        ),
        interim=p.get("interim", cfg.paths.interim),
        processed=p.get("processed", cfg.paths.processed),
        external=p.get("external", cfg.paths.external),
        reports=p.get("reports", cfg.paths.reports),
        figures=p.get("figures", cfg.paths.figures),
        tables=p.get("tables", cfg.paths.tables),
        logs=p.get("logs", cfg.paths.logs),
    )

    # Logging
    lg = raw.get("logging", {})
    cfg.logging = Logging(
        level=lg.get("level", cfg.logging.level),
        log_file=lg.get("log_file", cfg.logging.log_file),
        max_bytes=int(lg.get("max_bytes", cfg.logging.max_bytes)),
        backup_count=int(lg.get("backup_count", cfg.logging.backup_count)),
    )

    # IO
    io = raw.get("io", {})
    cfg.io = IO(
        chunk_size=int(io.get("chunk_size", cfg.io.chunk_size)),
        parquet_compression=io.get("parquet_compression", cfg.io.parquet_compression),
        parquet_engine=io.get("parquet_engine", cfg.io.parquet_engine),
    )

    # Feature selection
    fs = raw.get("feature_selection", {})
    cfg.feature_selection = FeatureSelection(
        variance_threshold=float(fs.get("variance_threshold", cfg.feature_selection.variance_threshold)),
        correlation_threshold=float(fs.get("correlation_threshold", cfg.feature_selection.correlation_threshold)),
        missing_rate_threshold=float(fs.get("missing_rate_threshold", cfg.feature_selection.missing_rate_threshold)),
    )

    # Targets
    tg = raw.get("targets", {})
    cfg.targets = Targets(
        mortality_inhosp=tg.get("mortality_inhosp", cfg.targets.mortality_inhosp),
        los_days=tg.get("los_days", cfg.targets.los_days),
        icu_admission=tg.get("icu_admission", cfg.targets.icu_admission),
        icu_los_days=tg.get("icu_los_days", cfg.targets.icu_los_days),
        readmission_30d=tg.get("readmission_30d", cfg.targets.readmission_30d),
    )

    pl = raw.get("pipeline", {})
    max_chunks = pl.get("max_chunks_large_tables")
    cfg.pipeline = PipelineSettings(
        full_load=bool(pl.get("full_load", cfg.pipeline.full_load)),
        max_chunks_large_tables=int(max_chunks) if max_chunks is not None else None,
        eda_sample_rows=int(pl.get("eda_sample_rows", cfg.pipeline.eda_sample_rows)),
        eda_max_chunks_per_large_table=int(
            pl.get("eda_max_chunks_per_large_table", cfg.pipeline.eda_max_chunks_per_large_table)
        ),
        skip_tables=list(pl.get("skip_tables", cfg.pipeline.skip_tables)),
    )

    # Table definitions, vitals, key labs, Charlson (kept as raw dicts)
    cfg.tables = raw.get("tables", {})
    cfg.vitals = raw.get("vitals", {})
    cfg.key_labs = raw.get("key_labs", {})
    cfg.charlson = raw.get("charlson", {})

    log.info("Config loaded from %s (project root: %s)", path, cfg.project_root)
    return cfg


# ── Singleton ─────────────────────────────────────────────────────────────────
CFG: Config = load_config()
