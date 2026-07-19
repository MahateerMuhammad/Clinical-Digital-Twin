# Clinical Digital Twin

AI-powered Personalized Patient Simulator using MIMIC-IV.

## Overview

This project implements a production-quality data engineering and preprocessing pipeline for the Clinical Digital Twin research prototype. It loads, cleans, explores, merges, and engineers features from MIMIC-IV EHR data, producing ML-ready Parquet datasets for downstream modeling (XGBoost, LightGBM, LSTM, Transformers, patient embeddings, explainable AI).

**This stage covers preprocessing only — no ML models are trained here.**

## Project Structure

```
Clinical Digital Twin/
├── configs/config.yaml          # Central configuration
├── data/
│   ├── raw/hosp|icu|notes/      # Raw MIMIC-IV CSV files
│   ├── interim/                 # Cleaned intermediate Parquet
│   └── processed/               # Final ML-ready datasets
├── notebooks/                   # Step-by-step Jupyter notebooks
├── src/
│   ├── data/                    # Loading, cleaning, merging, pipeline
│   ├── features/                # Feature engineering modules
│   ├── visualization/           # EDA and plotting
│   └── utils/                   # Config, logging, I/O, validation
├── reports/
│   ├── figures/                 # Publication-quality plots
│   └── tables/                  # Summary tables (Parquet)
├── logs/                        # Pipeline logs
├── run_pipeline.py              # CLI entry point
└── requirements.txt
```

## Quick Start

### 1. Environment Setup

```bash
cd "Clinical Digital Twin"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Place MIMIC-IV Data

Copy MIMIC-IV files into:

- `data/raw/hosp/` — patients, admissions, diagnoses, labs, prescriptions, etc.
- `data/raw/icu/` — icustays, chartevents, inputevents, outputevents
- `data/raw/notes/` — discharge summaries, radiology notes

Both `.csv` and `.csv.gz` formats are supported.

### 3. Run the Pipeline

```bash
# Fast smoke test (small tables only, ~5 min)
python run_pipeline.py --skip-large

# Standard mode (samples large tables via chunked reading)
python run_pipeline.py

# Full mode (loads entire large tables — may take hours)
python run_pipeline.py --full
```

### 4. Run Specific Steps

```bash
python run_pipeline.py --steps load clean
python run_pipeline.py --steps eda features datasets
```

## Pipeline Steps

| Step | Module | Output |
|------|--------|--------|
| 1. Load | `src/data/loader.py` | Schema inference, load summary |
| 2. Clean | `src/data/cleaner.py` | `data/interim/*_clean.parquet`, cleaning report |
| 3. EDA | `src/visualization/eda.py` | `reports/figures/`, EDA report |
| 4. Features | `src/features/` | Engineered features in `data/interim/features/` |
| 5. Datasets | `src/features/dataset_builder.py` | `data/processed/*.parquet` |

## Output Datasets

| Dataset | Granularity | File |
|---------|-------------|------|
| Patient Level | One row per patient | `data/processed/patient_level.parquet` |
| Admission Level | One row per hospital admission | `data/processed/admission_level.parquet` |
| ICU Level | One row per ICU stay | `data/processed/icu_level.parquet` |
| Time Series | Chronological events | `data/processed/time_series.parquet` |
| Clinical Notes | Cleaned text + features | `data/processed/clinical_notes.parquet` |
| Similarity | Embedding-ready vectors | `data/processed/similarity.parquet` |

## Feature Groups

- **Demographic**: age, gender, race, insurance, admission type
- **Diagnosis**: ICD counts, Charlson Comorbidity Index, chronic disease flags
- **Laboratory**: mean/median/min/max/slope/abnormal counts per key lab
- **Vitals**: rolling stats, trends for HR, BP, RR, SpO₂, temperature
- **Medication**: counts, drug classes, duration
- **Procedure**: counts, major procedure indicators
- **ICU**: duration, fluid balance
- **Temporal**: weekend/night admission, seasonality
- **Notes**: length, keyword counts, TF-IDF-ready text
- **Interactions**: age×diabetes, creatinine×age, etc.

## Documentation

Auto-generated reports in `reports/`:

- `cleaning_report.md` — all cleaning decisions documented
- `eda_report.md` — EDA figure index
- `feature_engineering_report.md` — feature selection summary
- `data_dictionary.md` — feature dictionary
- `pipeline_summary.md` — run metadata

## Configuration

Edit `configs/config.yaml` to control:

- File paths and table definitions
- Vital sign and lab item IDs
- Charlson comorbidity ICD mappings
- Feature selection thresholds
- Pipeline chunk sizes and sampling

## Notebooks

Interactive walkthrough in `notebooks/`:

1. `01_data_loading.ipynb`
2. `02_data_cleaning.ipynb`
3. `03_eda.ipynb`
4. `04_feature_engineering.ipynb`
5. `05_dataset_creation.ipynb`

## Design Principles

- **No silent data removal** — all cleaning actions are logged
- **Parquet only** — no CSV outputs for processed data
- **Modular & reusable** — all logic in `src/`, notebooks are thin wrappers
- **Chunked I/O** — handles 40GB+ chartevents via streaming aggregation
- **Reproducible** — central config, logging, random seeds

## Known Data Limitations

- `labevents.csv` may be a partial download (~14M of ~122M rows) — flagged in all reports
- `chartevents.csv` is ~39GB — use `--full` only when sufficient RAM/time available
- `radiology.csv` is optional; `radiology_detail.csv` used if main file absent

## Next Steps (Model Development)

This pipeline prepares data for:

- Mortality / readmission / ICU admission prediction
- Length of stay regression
- Patient embedding and similar patient retrieval
- Clinical note NLP (Transformers)
- Time-series LSTM models
- Explainable AI (SHAP, LIME)

## License

MIMIC-IV data use requires PhysioNet credentialing and DUA compliance.
