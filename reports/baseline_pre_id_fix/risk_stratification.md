# Phase 9 — In-Hospital Mortality Risk Stratification & Clinical Resource Planning Audit

## 1. Executive Summary & Audit Objectives
This report presents the **Phase 9 Patient Risk Stratification Framework** constructed from the winning **Phase 1 Calibrated LightGBM Mortality Model** (`phase1_mortality_calibrated.pkl`, AUROC 0.9490) evaluated on the held-out test split ($N = 82,806$ admissions).

### Audit Verification Scope:
* **Zero Model Retraining**: No new model parameters were fit. Risk probabilities are evaluated strictly to establish a clinical risk stratification framework.
* **Held-Out Test Set Isolation**: Evaluated exclusively on $N = 82,806$ held-out test admissions (Base Mortality Rate: **2.16%**, $1,787$ observed deaths).
* **Statistical Rigor (1,000 Bootstrap Resamples)**: All observed mortality rates and risk enrichment metrics include 95% Bootstrap Confidence Intervals (2.5th–97.5th percentiles) to evaluate statistical stability and adjacent-tier separation.
* **Monotonicity Context**: Because the underlying model has a known AUROC of 0.9490, monotonic ordering of observed mortality across probability-sorted quantiles is mathematically expected. Monotonicity confirms that the binning mechanism functions correctly without rank inversions, but it is a expected property of rank-ordered sorting rather than an independent validation that these specific percentile cutoffs are optimal.

---

## 2. Primary Clinical 4-Tier Stratification Scheme

The **Clinical 4-Tier Scheme** uses asymmetric quantile cutoffs ($0\text{--}50\text{th}\%$, $50\text{--}80\text{th}\%$, $80\text{--}95\text{th}\%$, and Top $5\%$) specifically designed to isolate the low-risk majority for routine floor care while concentrating high-risk admissions into actionable clinical tiers:

| Risk Tier | Percentile Range | Predicted Probability Cutoff ($P_{\text{mortality}}$) | Admissions ($N$) | Cohort Share (%) | Observed Deaths | Share of Total Deaths (%) | Observed Mortality Rate (%) | 95% Bootstrap CI (Mortality Rate) | Risk Enrichment vs Base | 95% Bootstrap CI (Enrichment) | Recommended Clinical Action |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Tier 1: Low Risk** | 0 – 50th% | $[0.00\% - 0.94\%)$ | 41,049 | **49.6%** | 90 | 5.0% | **0.22%** | **0.17% – 0.27%** | **0.10x** | **0.08x – 0.13x** | General Ward / Routine Floor Care |
| **Tier 2: Moderate Risk** | 50 – 80th% | $[0.94\% - 11.19\%)$ | 24,922 | **30.1%** | 260 | 14.5% | **1.04%** | **0.92% – 1.16%** | **0.48x** | **0.43x – 0.54x** | Standard Telemetry & Continuous Vitals |
| **Tier 3: High Risk** | 80 – 95th% | $[11.19\% - 21.71\%)$ | 10,278 | **12.4%** | 450 | 25.2% | **4.38%** | **3.98% – 4.77%** | **2.03x** | **1.88x – 2.19x** | Step-Down / Progressive Care Unit |
| **Tier 4: Extreme Risk** | Top 5% (95–100th%) | $[21.71\% - 100.00\%]$ | 6,557 | **7.9%** | 987 | **55.2%** | **15.05%** | **14.16% – 15.92%** | **6.98x** | **6.68x – 7.28x** | Immediate ICU Consultation & Rapid Response |
| **Total Test Cohort** | **0 – 100th%** | **$[0.00\% - 100.00\%]$** | **82,806** | **100.0%** | **1,787** | **100.0%** | **2.16%** | **2.07% – 2.26%** | **1.00x** | **1.00x – 1.00x** | Population Baseline |

### Statistical Separation & Monotonicity Audit:
* **Non-Overlapping Confidence Intervals**: Across all adjacent tiers, the 95% Bootstrap CIs for observed mortality rate are strictly non-overlapping (Tier 1: $0.17\%\text{--}0.27\%$, Tier 2: $0.92\%\text{--}1.16\%$, Tier 3: $3.98\%\text{--}4.77\%$, Tier 4: $14.16\%\text{--}15.92\%$). This confirms that the risk separations between adjacent tiers represent statistically significant, distinct clinical risk categories.
* **Monotonic Ordering**: Observed mortality increases strictly monotonically ($0.22\% \to 1.04\% \to 4.38\% \to 15.05\%$), confirming that probability sorting creates a well-ordered progression without tier inversions.

---

## 3. Secondary Equal-Quartile Statistical Benchmark

As a standard statistical reference, the table below evaluates equal 25% cohort quartiles:

| Quartile | Percentile Range | Predicted Probability Cutoff ($P_{\text{mortality}}$) | Admissions ($N$) | Cohort Share (%) | Observed Deaths | Share of Total Deaths (%) | Observed Mortality Rate (%) | 95% Bootstrap CI (Mortality Rate) | Risk Enrichment vs Base | 95% Bootstrap CI (Enrichment) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q1: Low Risk** | 0 – 25th% | $[0.00\% - 0.13\%)$ | 20,449 | 24.7% | 22 | 1.2% | **0.11%** | **0.06% – 0.15%** | **0.05x** | **0.03x – 0.07x** |
| **Q2: Moderate-Low Risk** | 25 – 50th% | $[0.13\% - 0.94\%)$ | 20,600 | 24.9% | 68 | 3.8% | **0.33%** | **0.26% – 0.41%** | **0.15x** | **0.12x – 0.19x** |
| **Q3: Moderate-High Risk** | 50 – 75th% | $[0.94\% - 5.48\%)$ | 19,549 | 23.6% | 163 | 9.1% | **0.83%** | **0.70% – 0.96%** | **0.39x** | **0.33x – 0.44x** |
| **Q4: High Risk** | 75 – 100th% | $[5.48\% - 100.00\%]$ | 22,208 | 26.8% | 1,534 | 85.8% | **6.91%** | **6.58% – 7.25%** | **3.20x** | **3.14x – 3.27x** |

### Statistical Audit (Quartile Scheme):
All four quartile 95% Bootstrap CIs are strictly non-overlapping (Q1: $0.06\%\text{--}0.15\%$, Q2: $0.26\%\text{--}0.41\%$, Q3: $0.70\%\text{--}0.96\%$, Q4: $6.58\%\text{--}7.25\%$), confirming robust separation across equal cohort divisions.

---

## 4. Clinical Resource Planning & Quantified False-Negative Trade-offs

### 1. Asymmetric Quantile Rationale vs Equal Quartiles
* Equal quartiles lump all admissions above the 75th percentile into Q4 ($26.8\%$ of cohort), diluting the highest-risk patients.
* Isolating the **Extreme Risk Tier (Tier 4, $P \ge 21.71\%$)** flags **$7.9\%$ of admissions** ($N = 6,557$). This single top tier captures **$55.2\%$ of ALL in-hospital deaths** ($987$ of $1,787$ deaths), with an observed mortality rate of **$15.05\%$** (95% CI: $14.16\%\text{--}15.92\%$, **$6.98\times$ baseline enrichment**).

### 2. Quantified False-Negative Costs & Clinical Risk Trade-offs
While Tier 4 captures the majority of hospital deaths ($55.2\%$), **clinical teams must explicitly recognize the false-negative cost of lower tiers**:
* **Combined Lower Tier Mortality (Tiers 1–3)**: Tiers 1–3 combined still account for **$44.8\%$ of all observed hospital deaths** ($800$ out of $1,787$ total deaths).
* **Tier 1 False-Negative Cost**: Routing Tier 1 ($P < 0.94\%$) to routine floor care without continuous telemetry has a low observed mortality rate of **$0.22\%$** (95% CI: $0.17\%\text{--}0.27\%$). However, because Tier 1 represents nearly half the hospital population ($N = 41,049$), this corresponds to **90 unmonitored in-hospital deaths** ($5.0\%$ of all hospital deaths).
* **Clinical Triage Implication**: Risk stratification does not eliminate mortality risk in low-tier admissions. Discharging telemetry or lowering monitoring intensity for Tier 1 patients carries an explicit, quantified false-negative burden of 90 deaths per 82,800 admissions that must be balanced against hospital bed availability and monitoring capacity.
