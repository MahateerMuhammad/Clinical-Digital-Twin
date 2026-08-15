"""
src/data/pipeline.py
────────────────────
Main orchestrator for the Clinical Digital Twin preprocessing pipeline.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.data.cleaner import DataCleaner
from src.data.loader import DataLoader
from src.features.build_features import FeatureBuilder
from src.features.dataset_builder import DatasetBuilder
from src.utils.config import CFG, load_config
from src.utils.io_utils import save_parquet
from src.utils.logger import get_logger, setup_root_logger
from src.utils.schema import infer_table_schema, schemas_to_dataframe
from src.visualization.eda import EDAAnalyzer

log = get_logger(__name__)

# Tables loaded in full (non-chunked or manageable size)
SMALL_TABLES = [
    "patients", "admissions", "diagnoses_icd", "d_icd_diagnoses",
    "procedures_icd", "d_icd_procedures", "d_labitems",
    "icustays", "d_items",
    "edstays", "triage",
]

# Large tables — loaded with chunk limits unless full mode
LARGE_TABLES = [
    "labevents", "prescriptions", "pharmacy", "emar", "emar_detail",
    "chartevents", "inputevents", "outputevents", "discharge", "radiology_detail",
    "ed_vitalsign", "medrecon",
]


class ClinicalDigitalTwinPipeline:
    """
    End-to-end preprocessing pipeline.

    Steps
    -----
    1. Load all datasets + schema inference
    2. Clean all tables → interim Parquet
    3. EDA + visualizations
    4. Feature engineering
    5. Merge tables
    6. Feature selection preparation
    7. Dataset creation → processed Parquet
    8. Documentation generation
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        full_mode: bool = False,
        skip_eda: bool = False,
        skip_large: bool = False,
        steps: Optional[List[str]] = None,
    ) -> None:
        if config_path:
            global CFG
            CFG = load_config(Path(config_path))

        setup_root_logger(
            log_file=CFG.logging.log_file,
            level=CFG.logging.level,
            max_bytes=CFG.logging.max_bytes,
            backup_count=CFG.logging.backup_count,
        )

        self.cfg = CFG
        self.full_mode = full_mode or getattr(self.cfg.pipeline, "full_load", True)
        self.skip_eda = skip_eda
        self.skip_large = skip_large
        self.steps = steps or ["load", "clean", "eda", "features", "datasets"]
        self.loader = DataLoader()
        self.cleaner = DataCleaner()
        self.feature_builder = FeatureBuilder()
        self.dataset_builder = DatasetBuilder()
        self.tables: Dict[str, pd.DataFrame] = {}
        self.cleaned: Dict[str, pd.DataFrame] = {}
        self.features: Dict[str, pd.DataFrame] = {}
        self.schemas: Dict = {}
        self.summaries: Dict = {}

    def run(self) -> None:
        """Execute the full pipeline."""
        start = time.time()
        log.info("=" * 60)
        log.info("Clinical Digital Twin Pipeline — START")
        log.info("Mode: %s | Steps: %s", "FULL" if self.full_mode else "STANDARD", self.steps)
        log.info("=" * 60)

        if "load" in self.steps:
            self.step_load()

        if "clean" in self.steps:
            self.step_clean()

        if "eda" in self.steps and not self.skip_eda:
            self.step_eda()

        if "features" in self.steps:
            self.step_features()

        if "datasets" in self.steps:
            self.step_datasets()

        elapsed = time.time() - start
        log.info("Pipeline completed in %.1f minutes", elapsed / 60)
        self._write_pipeline_summary(elapsed)

    def step_load(self) -> None:
        """Step 1: Load all datasets."""
        log.info("── STEP 1: Loading datasets ──")
        max_chunks = None if self.full_mode else self.cfg.pipeline.max_chunks_large_tables

        for name in SMALL_TABLES:
            if name in self.cfg.pipeline.skip_tables:
                continue
            loader_fn = getattr(self.loader, f"load_{name}", None)
            if loader_fn:
                df, summary = loader_fn()
                self.tables[name] = df
                self.summaries[name] = summary

        if not self.skip_large:
            for name in LARGE_TABLES:
                if name in self.cfg.pipeline.skip_tables:
                    continue
                loader_fn = getattr(self.loader, f"load_{name}", None)
                if loader_fn:
                    kwargs = {}
                    if not self.full_mode:
                        kwargs["max_chunks"] = max_chunks or self.cfg.pipeline.eda_max_chunks_per_large_table
                    df, summary = loader_fn(**kwargs)
                    self.tables[name] = df
                    self.summaries[name] = summary

        # Schema inference
        for name, df in self.tables.items():
            if df is not None and not df.empty:
                t_cfg = self.cfg.tables.get(name, {})
                self.schemas[name] = infer_table_schema(
                    df, name,
                    config_date_cols=t_cfg.get("date_cols"),
                    config_cat_cols=t_cfg.get("cat_cols"),
                    config_id_cols=t_cfg.get("id_cols"),
                    text_col=t_cfg.get("text_col"),
                )

        summary_df = self.loader.generate_dataset_summary(self.summaries)
        schema_df = schemas_to_dataframe(self.schemas)
        tables_dir = Path(self.cfg.resolve(self.cfg.paths.tables))
        save_parquet(summary_df, tables_dir / "load_summary.parquet")
        save_parquet(schema_df, tables_dir / "schema_summary.parquet")

        log.info("Loaded %d tables", len(self.tables))

    def step_clean(self) -> None:
        """Step 2: Clean all loaded tables."""
        log.info("── STEP 2: Cleaning datasets ──")
        reports = {}

        for name, df in self.tables.items():
            t_cfg = self.cfg.tables.get(name, {})
            # save=True only here: the real pipeline is the one entity allowed to
            # write data/interim/. clean_table now defaults to save=False so that
            # tests calling it with synthetic fixtures cannot overwrite real data.
            cleaned, report = self.cleaner.clean_table(
                df, name, id_cols=t_cfg.get("id_cols"), save=True,
            )
            self.cleaned[name] = cleaned
            reports[name] = report

        ref_report = self.cleaner.check_all_referential_integrity(self.cleaned)
        save_parquet(ref_report, Path(self.cfg.resolve("reports/tables")) / "referential_integrity.parquet")
        self.cleaner.save_cleaning_report(reports)
        log.info("Cleaned %d tables", len(self.cleaned))

    def _get_source_tables(self) -> Dict[str, pd.DataFrame]:
        """Return in-memory cleaned/raw tables, or load interim cleaned parquet files if empty."""
        if self.cleaned:
            return self.cleaned
        if self.tables:
            return self.tables

        interim_dir = Path(self.cfg.resolve(self.cfg.paths.interim))
        log.info("Loading pre-cleaned interim tables from %s...", interim_dir)
        loaded = {}
        for parquet_path in interim_dir.glob("*_clean.parquet"):
            table_name = parquet_path.name.replace("_clean.parquet", "")
            try:
                log.info("Loading interim table '%s' (%s)...", table_name, parquet_path.name)
                loaded[table_name] = pd.read_parquet(parquet_path)
            except Exception as e:
                log.warning("Could not load interim table '%s': %s", table_name, e)

        self.cleaned = loaded
        return self.cleaned

    def _get_features(self) -> Dict[str, pd.DataFrame]:
        """Return in-memory features, or load saved feature parquet files from disk."""
        if self.features:
            return self.features

        features_dir = Path(self.cfg.resolve(self.cfg.paths.interim)) / "features"
        if features_dir.exists():
            log.info("Loading saved feature sets from %s...", features_dir)
            loaded = {}
            for parquet_path in features_dir.glob("*_features.parquet"):
                feat_name = parquet_path.name.replace("_features.parquet", "")
                try:
                    loaded[feat_name] = pd.read_parquet(parquet_path)
                    log.info("Loaded feature set '%s'", feat_name)
                except Exception as e:
                    log.warning("Could not load feature set '%s': %s", feat_name, e)
            self.features = loaded
            return self.features

        return {}

    def step_eda(self) -> None:
        """Step 3: Exploratory data analysis."""
        log.info("── STEP 3: Exploratory Data Analysis ──")
        eda = EDAAnalyzer()
        source = self._get_source_tables()
        eda.run_all(source)

    def step_features(self) -> None:
        """Step 4-5: Feature engineering."""
        log.info("── STEP 4-5: Feature Engineering ──")
        source = self._get_source_tables()
        self.features = self.feature_builder.build_all(source, use_chunked=not self.full_mode)
        self.feature_builder.save_features(self.features)

    def step_datasets(self) -> None:
        """Step 6-7: Merge, feature selection, dataset creation."""
        log.info("── STEP 6-7: Dataset Creation ──")
        source = self._get_source_tables()
        features = self._get_features()
        self.dataset_builder.build_all(source, features)

    def _write_pipeline_summary(self, elapsed_sec: float) -> None:
        path = Path(self.cfg.resolve("reports/pipeline_summary.md"))
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("# Pipeline Summary\n\n")
            fh.write(f"Completed: {datetime.utcnow().isoformat()}Z\n")
            fh.write(f"Duration: {elapsed_sec / 60:.1f} minutes\n")
            fh.write(f"Mode: {'full' if self.full_mode else 'standard'}\n\n")
            fh.write("## Tables Processed\n\n")
            for name, summary in self.summaries.items():
                fh.write(f"- **{name}**: {summary.n_rows:,} rows, {summary.memory_mb:.1f} MB")
                if summary.is_partial:
                    fh.write(f" ⚠ {summary.partial_note}")
                fh.write("\n")
        log.info("Pipeline summary → %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clinical Digital Twin Preprocessing Pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--full", action="store_true", help="Full mode: load entire large tables")
    parser.add_argument("--skip-eda", action="store_true", help="Skip EDA step")
    parser.add_argument("--skip-large", action="store_true", help="Skip large table loading")
    parser.add_argument(
        "--steps", nargs="+",
        default=["load", "clean", "eda", "features", "datasets"],
        choices=["load", "clean", "eda", "features", "datasets"],
    )
    args = parser.parse_args()

    pipeline = ClinicalDigitalTwinPipeline(
        config_path=args.config,
        full_mode=args.full,
        skip_eda=args.skip_eda,
        skip_large=args.skip_large,
        steps=args.steps,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
