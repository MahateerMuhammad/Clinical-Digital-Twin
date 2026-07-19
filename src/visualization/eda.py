"""
src/visualization/eda.py
────────────────────────
Comprehensive exploratory data analysis for MIMIC-IV tables.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from scipy import stats

from src.features.diagnosis import build_comorbidity_pairs
from src.utils.config import CFG
from src.utils.io_utils import save_parquet
from src.utils.logger import get_logger
from src.visualization.plot_utils import PALETTE, save_figure, save_plotly

log = get_logger(__name__)


class EDAAnalyzer:
    """Run EDA across all tables and save publication-quality figures."""

    def __init__(self, sample_rows: Optional[int] = None) -> None:
        self.cfg = CFG
        self.sample_rows = sample_rows or self.cfg.pipeline.eda_sample_rows
        self.figures_created: List[str] = []

    def run_all(self, tables: Dict[str, pd.DataFrame]) -> Path:
        """Execute full EDA pipeline."""
        for name, df in tables.items():
            if df is None or df.empty:
                log.warning("Skipping EDA for empty table: %s", name)
                continue
            sample = df.sample(min(len(df), self.sample_rows), random_state=self.cfg.random_seed)
            self.analyze_table(sample, name)

        if "admissions" in tables:
            self._temporal_analysis(tables["admissions"])
            self._patient_analysis(tables)

        if "diagnoses_icd" in tables:
            self._disease_analysis(tables["diagnoses_icd"], tables.get("admissions"))

        if "labevents" in tables:
            self._lab_analysis(tables["labevents"], tables.get("admissions"))

        if "prescriptions" in tables:
            self._medication_analysis(tables["prescriptions"])

        if "icustays" in tables:
            self._icu_analysis(tables.get("icustays"), tables.get("chartevents"))

        if "discharge" in tables:
            self._notes_analysis(tables["discharge"])

        return self._write_eda_report()

    def analyze_table(self, df: pd.DataFrame, table_name: str) -> None:
        """General EDA for any table."""
        subdir = table_name
        self._plot_missingness(df, table_name, subdir)
        self._plot_summary_stats(df, table_name, subdir)

        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()

        for col in numeric[:8]:
            self._plot_numeric_distribution(df, col, table_name, subdir)

        for col in categorical[:6]:
            self._plot_categorical_distribution(df, col, table_name, subdir)

        if len(numeric) >= 2:
            self._plot_correlation_matrix(df[numeric], table_name, subdir)

    def _plot_missingness(self, df: pd.DataFrame, table_name: str, subdir: str) -> None:
        miss = df.isna().mean().sort_values(ascending=False)
        miss = miss[miss > 0]
        if miss.empty:
            return
        fig, ax = plt.subplots(figsize=(10, max(4, len(miss) * 0.3)))
        miss.head(30).mul(100).plot.barh(ax=ax, color=PALETTE[0])
        ax.set_xlabel("Missing (%)")
        ax.set_title(f"{table_name}: Missing Values")
        self.figures_created.append(str(save_figure(fig, f"{table_name}_missingness", subdir)))

    def _plot_summary_stats(self, df: pd.DataFrame, table_name: str, subdir: str) -> None:
        stats_df = df.describe(include="all").T.reset_index().rename(columns={"index": "column"})
        stats_df = stats_df.astype(str)
        save_parquet(stats_df, Path(self.cfg.resolve("reports/tables")) / f"{table_name}_summary_stats.parquet")

    def _plot_numeric_distribution(self, df: pd.DataFrame, col: str, table_name: str, subdir: str) -> None:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"{table_name}: {col}", fontsize=14)

        sns.histplot(series, kde=True, ax=axes[0, 0], color=PALETTE[0])
        axes[0, 0].set_title("Histogram + Density")

        sns.boxplot(x=series, ax=axes[0, 1], color=PALETTE[1])
        axes[0, 1].set_title("Boxplot")

        sns.violinplot(x=series, ax=axes[1, 0], color=PALETTE[2])
        axes[1, 0].set_title("Violin Plot")

        stats.probplot(series, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title("QQ Plot")

        series = pd.to_numeric(series, errors='coerce').astype(np.float64)
        skew = stats.skew(series, nan_policy='omit')
        kurt = stats.kurtosis(series, nan_policy='omit')
        fig.text(0.02, 0.02, f"Skewness={skew:.2f}, Kurtosis={kurt:.2f}", fontsize=10)
        self.figures_created.append(str(save_figure(fig, f"{table_name}_{col}_numeric", subdir)))

    def _plot_categorical_distribution(self, df: pd.DataFrame, col: str, table_name: str, subdir: str) -> None:
        counts = df[col].astype(str).value_counts().head(20)
        counts.index = counts.index.str.replace(r'[\x92]', "'", regex=True).str.replace(r'[\x93\x94]', '"', regex=True).str.replace(r'[\x95]', '-', regex=True)
        if counts.empty:
            return
        fig, ax = plt.subplots(figsize=(10, 6))
        counts.plot.bar(ax=ax, color=PALETTE[3])
        ax.set_title(f"{table_name}: Top Categories — {col}")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        self.figures_created.append(str(save_figure(fig, f"{table_name}_{col}_categories", subdir)))

        # Plotly interactive
        pct = (counts / counts.sum() * 100).reset_index()
        pct.columns = [col, "pct"]
        fig_p = px.bar(pct, x=col, y="pct", title=f"{table_name}: {col} Distribution (%)")
        self.figures_created.append(str(save_plotly(fig_p, f"{table_name}_{col}_categories_interactive", subdir)))

    def _plot_correlation_matrix(self, numeric_df: pd.DataFrame, table_name: str, subdir: str) -> None:
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, cmap="RdBu_r", center=0, ax=ax, annot=len(corr) <= 12)
        ax.set_title(f"{table_name}: Correlation Matrix")
        self.figures_created.append(str(save_figure(fig, f"{table_name}_correlation", subdir)))

    def _temporal_analysis(self, admissions: pd.DataFrame) -> None:
        adm = admissions.copy()
        adm["admittime"] = pd.to_datetime(adm["admittime"], errors="coerce")
        adm["dischtime"] = pd.to_datetime(adm["dischtime"], errors="coerce")
        adm = adm.dropna(subset=["admittime"])

        adm["year"] = adm["admittime"].dt.year
        adm["month"] = adm["admittime"].dt.month
        adm["dow"] = adm["admittime"].dt.dayofweek
        adm["hour"] = adm["admittime"].dt.hour

        fig_p = px.line(adm.groupby("year").size().reset_index(name="count"), x="year", y="count",
                        title="Admissions Over Time (Yearly)")
        self.figures_created.append(str(save_plotly(fig_p, "temporal_admissions_yearly", "temporal")))

        fig_p = px.bar(adm.groupby("month").size().reset_index(name="count"), x="month", y="count",
                       title="Monthly Admissions")
        self.figures_created.append(str(save_plotly(fig_p, "temporal_admissions_monthly", "temporal")))

        fig_p = px.bar(adm.groupby("hour").size().reset_index(name="count"), x="hour", y="count",
                       title="Hourly Admissions")
        self.figures_created.append(str(save_plotly(fig_p, "temporal_admissions_hourly", "temporal")))

        if adm["dischtime"].notna().any():
            fig_p = px.line(
                adm.dropna(subset=["dischtime"]).groupby(adm["dischtime"].dt.date).size().reset_index(name="count"),
                x="dischtime", y="count", title="Discharges Over Time",
            )
            self.figures_created.append(str(save_plotly(fig_p, "temporal_discharges", "temporal")))

    def _patient_analysis(self, tables: Dict[str, pd.DataFrame]) -> None:
        patients = tables.get("patients", pd.DataFrame())
        admissions = tables.get("admissions", pd.DataFrame())

        if not patients.empty and "anchor_age" in patients.columns:
            fig, ax = plt.subplots()
            sns.histplot(patients["anchor_age"], kde=True, ax=ax, color=PALETTE[0])
            ax.set_title("Patient Age Distribution")
            self.figures_created.append(str(save_figure(fig, "patient_age_distribution", "patient")))

        if not patients.empty and "gender" in patients.columns:
            fig_p = px.pie(patients, names="gender", title="Gender Distribution")
            self.figures_created.append(str(save_plotly(fig_p, "patient_gender_distribution", "patient")))

        if not admissions.empty:
            if "hospital_expire_flag" in admissions.columns:
                fig_p = px.bar(
                    admissions["hospital_expire_flag"].value_counts().reset_index(),
                    x="hospital_expire_flag", y="count", title="In-Hospital Mortality",
                )
                self.figures_created.append(str(save_plotly(fig_p, "mortality_distribution", "patient")))

            adm = admissions.copy()
            adm["admittime"] = pd.to_datetime(adm["admittime"], errors="coerce")
            adm["dischtime"] = pd.to_datetime(adm["dischtime"], errors="coerce")
            adm["los_days"] = (adm["dischtime"] - adm["admittime"]).dt.total_seconds() / 86400
            fig, ax = plt.subplots()
            sns.histplot(adm["los_days"].dropna().clip(0, 60), kde=True, ax=ax, color=PALETTE[4])
            ax.set_title("Hospital Length of Stay (days, capped at 60)")
            self.figures_created.append(str(save_figure(fig, "los_distribution", "patient")))

        if "icustays" in tables and not tables["icustays"].empty:
            fig, ax = plt.subplots()
            sns.histplot(tables["icustays"]["los"].dropna(), kde=True, ax=ax, color=PALETTE[5])
            ax.set_title("ICU Length of Stay (days)")
            self.figures_created.append(str(save_figure(fig, "icu_los_distribution", "patient")))

    def _disease_analysis(self, diagnoses: pd.DataFrame, admissions: Optional[pd.DataFrame]) -> None:
        top_dx = diagnoses["icd_code"].value_counts().head(20)
        fig, ax = plt.subplots(figsize=(12, 6))
        top_dx.plot.bar(ax=ax, color=PALETTE[0])
        ax.set_title("Top 20 ICD Diagnosis Codes")
        plt.xticks(rotation=45, ha="right")
        self.figures_created.append(str(save_figure(fig, "top_diagnoses", "disease")))

        pairs = build_comorbidity_pairs(diagnoses, top_n=15)
        if not pairs.empty:
            save_parquet(pairs.head(100), Path(self.cfg.resolve("reports/tables")) / "disease_cooccurrence.parquet")

        if admissions is not None and "hospital_expire_flag" in admissions.columns:
            dx_mort = diagnoses.merge(
                admissions[["hadm_id", "hospital_expire_flag"]], on="hadm_id", how="left",
            )
            mort_rate = dx_mort.groupby("icd_code", observed=True)["hospital_expire_flag"].mean()
            top_mort = mort_rate.loc[top_dx.index].dropna()
            if not top_mort.empty:
                fig, ax = plt.subplots(figsize=(12, 6))
                top_mort.mul(100).plot.bar(ax=ax, color=PALETTE[1])
                ax.set_title("Mortality Rate by Top Diagnosis (%)")
                ax.set_ylabel("Mortality (%)")
                plt.xticks(rotation=45, ha="right")
                self.figures_created.append(str(save_figure(fig, "mortality_by_diagnosis", "disease")))

    def _lab_analysis(self, labevents: pd.DataFrame, admissions: Optional[pd.DataFrame]) -> None:
        labs = labevents.copy()
        labs["valuenum"] = pd.to_numeric(labs.get("valuenum"), errors="coerce")
        key_ids = list(CFG.key_labs.values())

        if "itemid" in labs.columns:
            labs = labs[labs["itemid"].isin(key_ids)]

        if labs.empty:
            return

        fig, ax = plt.subplots()
        sns.histplot(labs["valuenum"].dropna(), kde=True, ax=ax, color=PALETTE[2])
        ax.set_title("Key Lab Values Distribution")
        self.figures_created.append(str(save_figure(fig, "lab_value_distribution", "laboratory")))

        if admissions is not None and "hospital_expire_flag" in admissions.columns:
            merged = labs.merge(admissions[["hadm_id", "hospital_expire_flag"]], on="hadm_id", how="left")
            if merged["hospital_expire_flag"].notna().any():
                fig, ax = plt.subplots()
                for flag, grp in merged.groupby("hospital_expire_flag"):
                    sns.kdeplot(grp["valuenum"].dropna(), ax=ax, label=f"mortality={flag}")
                ax.set_title("Lab Values by Mortality")
                ax.legend()
                self.figures_created.append(str(save_figure(fig, "lab_mortality_correlation", "laboratory")))

    def _medication_analysis(self, prescriptions: pd.DataFrame) -> None:
        if "drug" not in prescriptions.columns:
            return
        top_drugs = prescriptions["drug"].value_counts().head(20)
        fig, ax = plt.subplots(figsize=(12, 6))
        top_drugs.plot.bar(ax=ax, color=PALETTE[3])
        ax.set_title("Top 20 Prescribed Medications")
        plt.xticks(rotation=45, ha="right")
        self.figures_created.append(str(save_figure(fig, "top_medications", "medication")))

    def _icu_analysis(self, icustays: Optional[pd.DataFrame], chartevents: Optional[pd.DataFrame]) -> None:
        if icustays is not None and not icustays.empty:
            fig_p = px.bar(
                icustays["first_careunit"].value_counts().head(10).reset_index(),
                x="first_careunit", y="count", title="ICU Unit Distribution",
            )
            self.figures_created.append(str(save_plotly(fig_p, "icu_unit_distribution", "icu")))

        if chartevents is not None and not chartevents.empty:
            ce = chartevents.copy()
            ce["valuenum"] = pd.to_numeric(ce.get("valuenum"), errors="coerce")
            vital_ids = []
            for name, ids in CFG.vitals.items():
                vital_ids.extend(ids)
            ce = ce[ce["itemid"].isin(vital_ids)] if "itemid" in ce.columns else ce
            if not ce.empty:
                fig, ax = plt.subplots()
                sns.histplot(ce["valuenum"].dropna(), kde=True, ax=ax, color=PALETTE[4])
                ax.set_title("ICU Vital Signs Distribution")
                self.figures_created.append(str(save_figure(fig, "icu_vitals_distribution", "icu")))

    def _notes_analysis(self, discharge: pd.DataFrame) -> None:
        if "text" not in discharge.columns:
            return

        text_lens = discharge["text"].astype(str).str.len()
        fig, ax = plt.subplots()
        sns.histplot(text_lens, kde=True, ax=ax, color=PALETTE[5])
        ax.set_title("Discharge Note Length Distribution")
        self.figures_created.append(str(save_figure(fig, "note_length_distribution", "notes")))

        try:
            from wordcloud import WordCloud
            sample_text = " ".join(discharge["text"].dropna().astype(str).sample(
                min(500, len(discharge)), random_state=42,
            ))
            wc = WordCloud(width=1200, height=600, background_color="white", max_words=100).generate(sample_text)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title("Discharge Notes Word Cloud (sample)")
            self.figures_created.append(str(save_figure(fig, "notes_wordcloud", "notes")))
        except ImportError:
            log.debug("wordcloud not available")

    def _write_eda_report(self) -> Path:
        path = Path(self.cfg.resolve("reports/eda_report.md"))
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("# Exploratory Data Analysis Report\n\n")
            fh.write(f"Generated: {datetime.utcnow().isoformat()}Z\n\n")
            fh.write(f"Total figures created: {len(self.figures_created)}\n\n")
            fh.write("## Figures\n\n")
            for fig in self.figures_created:
                fh.write(f"- `{fig}`\n")
        log.info("EDA report saved → %s (%d figures)", path, len(self.figures_created))
        return path
