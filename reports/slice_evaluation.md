# Subgroup (Slice) Evaluation

*Generated 2026-08-14 by `scripts/evaluation/run_slice_eval.py`.*

The headline evaluation reports one calibration figure per task. That figure is an average, and an average is what hides a model that is well calibrated overall and poorly calibrated for one group. **The number to read here is the gap between slices, not the mean.**

- Held-out test rows: **82,806**
- Probabilities: **isotonic-calibrated** (what the API returns)
- Support floor: a slice needs **n ≥ 500** and **≥ 25 events** to be measured; below that it is reported as unmeasured rather than as a finding
- Race categories: 33 raw → 14 after grouping on MIMIC's hierarchical delimiter

## mortality

Overall — n 82,806 · base rate 2.16% · AUROC 0.919 · Brier 0.0178 · ECE 0.0009

### by sex

**AUROC gap 0.019** — best `F` 0.928, worst `M` 0.909. **ECE gap 0.0004** — worst `F` 0.0019.

| Slice | n | Events | Base rate | Mean pred | AUROC | Brier | ECE |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `F` | 43,281 | 807 | 1.86% | 2.00% | 0.928 | 0.0154 | 0.0019 |
| `M` | 39,525 | 980 | 2.48% | 2.44% | 0.909 | 0.0203 | 0.0015 |

### by age_band

**AUROC gap 0.128** — best `18-39` 0.963, worst `85+` 0.834. **ECE gap 0.0052** — worst `85+` 0.0067.

| Slice | n | Events | Base rate | Mean pred | AUROC | Brier | ECE |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `55-69` | 24,006 | 538 | 2.24% | 2.30% | 0.906 | 0.0186 | 0.0016 |
| `40-54` | 17,894 | 232 | 1.30% | 1.25% | 0.936 | 0.0105 | 0.0016 |
| `70-84` | 17,640 | 629 | 3.57% | 3.67% | 0.881 | 0.0290 | 0.0019 |
| `18-39` | 17,560 | 95 | 0.54% | 0.52% | 0.963 | 0.0046 | 0.0015 |
| `85+` | 5,706 | 293 | 5.13% | 5.54% | 0.834 | 0.0423 | 0.0067 |

### by race

**AUROC gap 0.062** — best `BLACK` 0.934, worst `UNKNOWN` 0.872. **ECE gap 0.0615** — worst `UNABLE TO OBTAIN` 0.0635.

| Slice | n | Events | Base rate | Mean pred | AUROC | Brier | ECE |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `WHITE` | 55,247 | 1119 | 2.03% | 2.34% | 0.910 | 0.0171 | 0.0032 |
| `BLACK` | 13,429 | 175 | 1.30% | 1.48% | 0.934 | 0.0110 | 0.0021 |
| `HISPANIC` | 3,482 | 35 | 1.01% | 1.34% | 0.933 | 0.0084 | 0.0039 |
| `ASIAN` | 2,981 | 56 | 1.88% | 1.52% | 0.931 | 0.0157 | 0.0052 |
| `OTHER` | 2,887 | 61 | 2.11% | 1.78% | 0.929 | 0.0173 | 0.0051 |
| `UNKNOWN` | 2,068 | 247 | 11.94% | 6.48% | 0.872 | 0.0867 | 0.0564 |
| `HISPANIC OR LATINO` | 1,057 | 14 | 1.32% | — | — | — | *unmeasured* |
| `UNABLE TO OBTAIN` | 504 | 61 | 12.10% | 5.98% | 0.892 | 0.0865 | 0.0635 |
| `PATIENT DECLINED TO ANSWER` | 319 | 4 | 1.25% | — | — | — | *unmeasured* |
| `PORTUGUESE` | 282 | 5 | 1.77% | — | — | — | *unmeasured* |
| `AMERICAN INDIAN` | 252 | 1 | 0.40% | — | — | — | *unmeasured* |
| `SOUTH AMERICAN` | 139 | 3 | 2.16% | — | — | — | *unmeasured* |
| `NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER` | 88 | 5 | 5.68% | — | — | — | *unmeasured* |
| `MULTIPLE RACE` | 71 | 1 | 1.41% | — | — | — | *unmeasured* |

*7 of 14 slices fell below the support floor and are not scored.*

### by insurance

**AUROC gap 0.056** — best `Private` 0.948, worst `Medicare` 0.891. **ECE gap 0.0287** — worst `Other` 0.0299.

| Slice | n | Events | Base rate | Mean pred | AUROC | Brier | ECE |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Medicare` | 36,805 | 1096 | 2.98% | 3.35% | 0.891 | 0.0245 | 0.0040 |
| `Private` | 26,235 | 310 | 1.18% | 1.16% | 0.948 | 0.0098 | 0.0012 |
| `Medicaid` | 16,185 | 228 | 1.41% | 1.43% | 0.940 | 0.0116 | 0.0020 |
| `Other` | 2,072 | 93 | 4.49% | 1.84% | 0.894 | 0.0396 | 0.0299 |
| `No charge` | 61 | 1 | 1.64% | — | — | — | *unmeasured* |
| `nan` | 0 | 0 | n/a | — | — | — | *unmeasured* |

*2 of 6 slices fell below the support floor and are not scored.*

### by language

**AUROC gap 0.035** — best `Chinese` 0.930, worst `Russian` 0.895. **ECE gap 0.0097** — worst `Chinese` 0.0109.

| Slice | n | Events | Base rate | Mean pred | AUROC | Brier | ECE |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `English` | 74,665 | 1586 | 2.12% | 2.20% | 0.919 | 0.0175 | 0.0012 |
| `Spanish` | 2,798 | 44 | 1.57% | 1.83% | 0.917 | 0.0134 | 0.0033 |
| `Russian` | 1,225 | 34 | 2.78% | 3.20% | 0.895 | 0.0226 | 0.0104 |
| `Chinese` | 1,164 | 31 | 2.66% | 1.85% | 0.930 | 0.0223 | 0.0109 |
| `Kabuverdianu` | 715 | 12 | 1.68% | — | — | — | *unmeasured* |
| `Portuguese` | 466 | 14 | 3.00% | — | — | — | *unmeasured* |
| `Haitian` | 336 | 8 | 2.38% | — | — | — | *unmeasured* |
| `Other` | 210 | 5 | 2.38% | — | — | — | *unmeasured* |
| `Modern Greek (1453-)` | 176 | 8 | 4.55% | — | — | — | *unmeasured* |
| `Vietnamese` | 165 | 7 | 4.24% | — | — | — | *unmeasured* |
| `Italian` | 155 | 3 | 1.94% | — | — | — | *unmeasured* |
| `American Sign Language` | 141 | 0 | 0.00% | — | — | — | *unmeasured* |
| `Arabic` | 86 | 2 | 2.33% | — | — | — | *unmeasured* |
| `Persian` | 64 | 1 | 1.56% | — | — | — | *unmeasured* |
| `Khmer` | 63 | 5 | 7.94% | — | — | — | *unmeasured* |
| `Polish` | 60 | 2 | 3.33% | — | — | — | *unmeasured* |
| `Korean` | 56 | 2 | 3.57% | — | — | — | *unmeasured* |
| `Thai` | 36 | 1 | 2.78% | — | — | — | *unmeasured* |
| `Amharic` | 25 | 1 | 4.00% | — | — | — | *unmeasured* |
| `Japanese` | 20 | 0 | 0.00% | — | — | — | *unmeasured* |
| `French` | 16 | 0 | 0.00% | — | — | — | *unmeasured* |
| `Hindi` | 16 | 1 | 6.25% | — | — | — | *unmeasured* |
| `Armenian` | 14 | 2 | 14.29% | — | — | — | *unmeasured* |
| `Bengali` | 8 | 0 | 0.00% | — | — | — | *unmeasured* |
| `Somali` | 6 | 0 | 0.00% | — | — | — | *unmeasured* |
| `nan` | 0 | 0 | n/a | — | — | — | *unmeasured* |

*22 of 26 slices fell below the support floor and are not scored.*

### by marital_status

**AUROC gap 0.077** — best `SINGLE` 0.941, worst `WIDOWED` 0.864. **ECE gap 0.0048** — worst `WIDOWED` 0.0062.

| Slice | n | Events | Base rate | Mean pred | AUROC | Brier | ECE |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `MARRIED` | 34,610 | 713 | 2.06% | 2.22% | 0.917 | 0.0171 | 0.0022 |
| `SINGLE` | 31,317 | 428 | 1.37% | 1.50% | 0.941 | 0.0114 | 0.0015 |
| `WIDOWED` | 8,741 | 280 | 3.20% | 3.73% | 0.864 | 0.0272 | 0.0062 |
| `DIVORCED` | 6,138 | 122 | 1.99% | 2.26% | 0.883 | 0.0171 | 0.0028 |
| `nan` | 0 | 0 | n/a | — | — | — | *unmeasured* |

*1 of 5 slices fell below the support floor and are not scored.*

### by admission_type

**AUROC gap 0.016** — best `DIRECT EMER.` 0.908, worst `EW EMER.` 0.893. **ECE gap 0.0072** — worst `URGENT` 0.0104.

| Slice | n | Events | Base rate | Mean pred | AUROC | Brier | ECE |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `EW EMER.` | 26,760 | 991 | 3.70% | 3.01% | 0.893 | 0.0302 | 0.0079 |
| `EU OBSERVATION` | 18,390 | 25 | 0.14% | 0.53% | 0.902 | 0.0014 | 0.0041 |
| `OBSERVATION ADMIT` | 12,892 | 295 | 2.29% | 3.16% | 0.895 | 0.0197 | 0.0089 |
| `URGENT` | 8,253 | 362 | 4.39% | 3.50% | 0.904 | 0.0345 | 0.0104 |
| `SURGICAL SAME DAY ADMISSION` | 6,471 | 15 | 0.23% | — | — | — | *unmeasured* |
| `DIRECT OBSERVATION` | 3,786 | 2 | 0.05% | — | — | — | *unmeasured* |
| `DIRECT EMER.` | 3,251 | 79 | 2.43% | 2.41% | 0.908 | 0.0198 | 0.0032 |
| `ELECTIVE` | 1,910 | 18 | 0.94% | — | — | — | *unmeasured* |
| `AMBULATORY OBSERVATION` | 1,093 | 0 | 0.00% | — | — | — | *unmeasured* |

*4 of 9 slices fell below the support floor and are not scored.*

## icu_admission

Overall — n 82,806 · base rate 15.55% · AUROC 0.888 · Brier 0.0805 · ECE 0.0020

### by sex

**AUROC gap 0.005** — best `M` 0.889, worst `F` 0.884. **ECE gap 0.0028** — worst `M` 0.0081.

| Slice | n | Events | Base rate | Mean pred | AUROC | Brier | ECE |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `F` | 43,281 | 5728 | 13.23% | 13.62% | 0.884 | 0.0738 | 0.0053 |
| `M` | 39,525 | 7149 | 18.09% | 17.36% | 0.889 | 0.0879 | 0.0081 |

### by age_band

**AUROC gap 0.086** — best `18-39` 0.920, worst `85+` 0.833. **ECE gap 0.0162** — worst `85+` 0.0207.

| Slice | n | Events | Base rate | Mean pred | AUROC | Brier | ECE |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `55-69` | 24,006 | 4312 | 17.96% | 18.15% | 0.879 | 0.0904 | 0.0045 |
| `40-54` | 17,894 | 2272 | 12.70% | 12.11% | 0.886 | 0.0700 | 0.0082 |
| `70-84` | 17,640 | 3778 | 21.42% | 20.84% | 0.866 | 0.1058 | 0.0108 |
| `18-39` | 17,560 | 1384 | 7.88% | 7.72% | 0.920 | 0.0431 | 0.0073 |
| `85+` | 5,706 | 1131 | 19.82% | 21.01% | 0.833 | 0.1091 | 0.0207 |

### by race

**AUROC gap 0.032** — best `ASIAN` 0.911, worst `WHITE` 0.878. **ECE gap 0.1813** — worst `UNKNOWN` 0.1851.

| Slice | n | Events | Base rate | Mean pred | AUROC | Brier | ECE |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `WHITE` | 55,247 | 8671 | 15.69% | 15.98% | 0.878 | 0.0843 | 0.0038 |
| `BLACK` | 13,429 | 1403 | 10.45% | 11.31% | 0.892 | 0.0605 | 0.0100 |
| `HISPANIC` | 3,482 | 342 | 9.82% | 11.57% | 0.898 | 0.0560 | 0.0206 |
| `ASIAN` | 2,981 | 358 | 12.01% | 12.33% | 0.911 | 0.0620 | 0.0139 |
| `OTHER` | 2,887 | 423 | 14.65% | 14.29% | 0.890 | 0.0772 | 0.0192 |
| `UNKNOWN` | 2,068 | 1140 | 55.13% | 36.62% | 0.903 | 0.1739 | 0.1851 |
| `HISPANIC OR LATINO` | 1,057 | 95 | 8.99% | 8.42% | 0.906 | 0.0528 | 0.0211 |
| `UNABLE TO OBTAIN` | 504 | 263 | 52.18% | 35.64% | 0.888 | 0.1701 | 0.1654 |
| `PATIENT DECLINED TO ANSWER` | 319 | 72 | 22.57% | — | — | — | *unmeasured* |
| `PORTUGUESE` | 282 | 45 | 15.96% | — | — | — | *unmeasured* |
| `AMERICAN INDIAN` | 252 | 27 | 10.71% | — | — | — | *unmeasured* |
| `SOUTH AMERICAN` | 139 | 11 | 7.91% | — | — | — | *unmeasured* |
| `NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER` | 88 | 18 | 20.45% | — | — | — | *unmeasured* |
| `MULTIPLE RACE` | 71 | 9 | 12.68% | — | — | — | *unmeasured* |

*6 of 14 slices fell below the support floor and are not scored.*

### by insurance

**AUROC gap 0.035** — best `Private` 0.902, worst `Medicare` 0.866. **ECE gap 0.0210** — worst `Other` 0.0255.

| Slice | n | Events | Base rate | Mean pred | AUROC | Brier | ECE |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Medicare` | 36,805 | 7044 | 19.14% | 19.52% | 0.866 | 0.0984 | 0.0060 |
| `Private` | 26,235 | 3266 | 12.45% | 12.15% | 0.902 | 0.0655 | 0.0045 |
| `Medicaid` | 16,185 | 2009 | 12.41% | 12.05% | 0.900 | 0.0658 | 0.0061 |
| `Other` | 2,072 | 335 | 16.17% | 13.74% | 0.896 | 0.0765 | 0.0255 |
| `No charge` | 61 | 1 | 1.64% | — | — | — | *unmeasured* |
| `nan` | 0 | 0 | n/a | — | — | — | *unmeasured* |

*2 of 6 slices fell below the support floor and are not scored.*

### by language

**AUROC gap 0.054** — best `Chinese` 0.911, worst `Russian` 0.857. **ECE gap 0.0254** — worst `Chinese` 0.0277.

| Slice | n | Events | Base rate | Mean pred | AUROC | Brier | ECE |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `English` | 74,665 | 11670 | 15.63% | 15.45% | 0.888 | 0.0808 | 0.0023 |
| `Spanish` | 2,798 | 357 | 12.76% | 13.82% | 0.887 | 0.0684 | 0.0141 |
| `Russian` | 1,225 | 165 | 13.47% | 14.97% | 0.857 | 0.0795 | 0.0259 |
| `Chinese` | 1,164 | 147 | 12.63% | 13.20% | 0.911 | 0.0625 | 0.0277 |
| `Kabuverdianu` | 715 | 83 | 11.61% | 12.44% | 0.884 | 0.0650 | 0.0211 |
| `Portuguese` | 466 | 87 | 18.67% | — | — | — | *unmeasured* |
| `Haitian` | 336 | 85 | 25.30% | — | — | — | *unmeasured* |
| `Other` | 210 | 36 | 17.14% | — | — | — | *unmeasured* |
| `Modern Greek (1453-)` | 176 | 44 | 25.00% | — | — | — | *unmeasured* |
| `Vietnamese` | 165 | 26 | 15.76% | — | — | — | *unmeasured* |
| `Italian` | 155 | 32 | 20.65% | — | — | — | *unmeasured* |
| `American Sign Language` | 141 | 6 | 4.26% | — | — | — | *unmeasured* |
| `Arabic` | 86 | 8 | 9.30% | — | — | — | *unmeasured* |
| `Persian` | 64 | 7 | 10.94% | — | — | — | *unmeasured* |
| `Khmer` | 63 | 13 | 20.63% | — | — | — | *unmeasured* |
| `Polish` | 60 | 13 | 21.67% | — | — | — | *unmeasured* |
| `Korean` | 56 | 14 | 25.00% | — | — | — | *unmeasured* |
| `Thai` | 36 | 4 | 11.11% | — | — | — | *unmeasured* |
| `Amharic` | 25 | 4 | 16.00% | — | — | — | *unmeasured* |
| `Japanese` | 20 | 2 | 10.00% | — | — | — | *unmeasured* |
| `French` | 16 | 3 | 18.75% | — | — | — | *unmeasured* |
| `Hindi` | 16 | 6 | 37.50% | — | — | — | *unmeasured* |
| `Armenian` | 14 | 3 | 21.43% | — | — | — | *unmeasured* |
| `Bengali` | 8 | 2 | 25.00% | — | — | — | *unmeasured* |
| `Somali` | 6 | 0 | 0.00% | — | — | — | *unmeasured* |
| `nan` | 0 | 0 | n/a | — | — | — | *unmeasured* |

*21 of 26 slices fell below the support floor and are not scored.*

### by marital_status

**AUROC gap 0.051** — best `SINGLE` 0.898, worst `WIDOWED` 0.847. **ECE gap 0.0098** — worst `WIDOWED` 0.0125.

| Slice | n | Events | Base rate | Mean pred | AUROC | Brier | ECE |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `MARRIED` | 34,610 | 5712 | 16.50% | 16.51% | 0.883 | 0.0849 | 0.0027 |
| `SINGLE` | 31,317 | 3661 | 11.69% | 11.96% | 0.898 | 0.0638 | 0.0041 |
| `WIDOWED` | 8,741 | 1495 | 17.10% | 18.30% | 0.847 | 0.0995 | 0.0125 |
| `DIVORCED` | 6,138 | 989 | 16.11% | 16.21% | 0.871 | 0.0874 | 0.0108 |
| `nan` | 0 | 0 | n/a | — | — | — | *unmeasured* |

*1 of 5 slices fell below the support floor and are not scored.*

### by admission_type

**AUROC gap 0.258** — best `URGENT` 0.927, worst `DIRECT OBSERVATION` 0.670. **ECE gap 0.0238** — worst `EW EMER.` 0.0522.

| Slice | n | Events | Base rate | Mean pred | AUROC | Brier | ECE |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `EW EMER.` | 26,760 | 6617 | 24.73% | 19.53% | 0.840 | 0.1275 | 0.0522 |
| `EU OBSERVATION` | 18,390 | 88 | 0.48% | 3.32% | 0.776 | 0.0076 | 0.0284 |
| `OBSERVATION ADMIT` | 12,892 | 1893 | 14.68% | 19.42% | 0.826 | 0.0937 | 0.0474 |
| `URGENT` | 8,253 | 2032 | 24.62% | 19.94% | 0.927 | 0.0958 | 0.0468 |
| `SURGICAL SAME DAY ADMISSION` | 6,471 | 1368 | 21.14% | 23.16% | 0.874 | 0.0990 | 0.0373 |
| `DIRECT OBSERVATION` | 3,786 | 33 | 0.87% | 4.65% | 0.670 | 0.0127 | 0.0378 |
| `DIRECT EMER.` | 3,251 | 443 | 13.63% | 17.53% | 0.885 | 0.0782 | 0.0397 |
| `ELECTIVE` | 1,910 | 400 | 20.94% | 24.60% | 0.911 | 0.0855 | 0.0433 |
| `AMBULATORY OBSERVATION` | 1,093 | 3 | 0.27% | — | — | — | *unmeasured* |

*1 of 9 slices fell below the support floor and are not scored.*

## readmission

Overall — n 82,806 · base rate 20.03% · AUROC 0.615 · Brier 0.1561 · ECE 0.0050

### by sex

**AUROC gap 0.034** — best `F` 0.631, worst `M` 0.596. **ECE gap 0.0098** — worst `M` 0.0149.

| Slice | n | Events | Base rate | Mean pred | AUROC | Brier | ECE |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `F` | 43,281 | 8114 | 18.75% | 19.16% | 0.631 | 0.1475 | 0.0051 |
| `M` | 39,525 | 8473 | 21.44% | 20.35% | 0.596 | 0.1656 | 0.0149 |

### by age_band

**AUROC gap 0.100** — best `18-39` 0.666, worst `85+` 0.566. **ECE gap 0.0079** — worst `40-54` 0.0142.

| Slice | n | Events | Base rate | Mean pred | AUROC | Brier | ECE |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `55-69` | 24,006 | 4930 | 20.54% | 20.72% | 0.610 | 0.1594 | 0.0092 |
| `40-54` | 17,894 | 4016 | 22.44% | 21.31% | 0.589 | 0.1715 | 0.0142 |
| `70-84` | 17,640 | 3377 | 19.14% | 18.62% | 0.588 | 0.1527 | 0.0064 |
| `18-39` | 17,560 | 3341 | 19.03% | 19.20% | 0.666 | 0.1464 | 0.0069 |
| `85+` | 5,706 | 923 | 16.18% | 15.67% | 0.566 | 0.1349 | 0.0062 |

### by race

**AUROC gap 0.098** — best `HISPANIC OR LATINO` 0.655, worst `UNKNOWN` 0.557. **ECE gap 0.1428** — worst `UNKNOWN` 0.1498.

| Slice | n | Events | Base rate | Mean pred | AUROC | Brier | ECE |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `WHITE` | 55,247 | 11293 | 20.44% | 19.75% | 0.613 | 0.1587 | 0.0088 |
| `BLACK` | 13,429 | 2943 | 21.92% | 19.71% | 0.624 | 0.1660 | 0.0223 |
| `HISPANIC` | 3,482 | 678 | 19.47% | 19.83% | 0.611 | 0.1535 | 0.0070 |
| `ASIAN` | 2,981 | 584 | 19.59% | 18.56% | 0.629 | 0.1524 | 0.0115 |
| `OTHER` | 2,887 | 500 | 17.32% | 19.37% | 0.649 | 0.1378 | 0.0211 |
| `UNKNOWN` | 2,068 | 135 | 6.53% | 21.50% | 0.557 | 0.0874 | 0.1498 |
| `HISPANIC OR LATINO` | 1,057 | 168 | 15.89% | 17.86% | 0.655 | 0.1287 | 0.0236 |
| `UNABLE TO OBTAIN` | 504 | 35 | 6.94% | 21.52% | 0.617 | 0.0881 | 0.1457 |
| `PATIENT DECLINED TO ANSWER` | 319 | 37 | 11.60% | — | — | — | *unmeasured* |
| `PORTUGUESE` | 282 | 78 | 27.66% | — | — | — | *unmeasured* |
| `AMERICAN INDIAN` | 252 | 74 | 29.37% | — | — | — | *unmeasured* |
| `SOUTH AMERICAN` | 139 | 31 | 22.30% | — | — | — | *unmeasured* |
| `NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER` | 88 | 17 | 19.32% | — | — | — | *unmeasured* |
| `MULTIPLE RACE` | 71 | 14 | 19.72% | — | — | — | *unmeasured* |

*6 of 14 slices fell below the support floor and are not scored.*

### by insurance

**AUROC gap 0.043** — best `Private` 0.641, worst `Medicare` 0.597. **ECE gap 0.0231** — worst `Medicaid` 0.0357.

| Slice | n | Events | Base rate | Mean pred | AUROC | Brier | ECE |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Medicare` | 36,805 | 7664 | 20.82% | 19.68% | 0.597 | 0.1620 | 0.0126 |
| `Private` | 26,235 | 4434 | 16.90% | 19.19% | 0.641 | 0.1362 | 0.0229 |
| `Medicaid` | 16,185 | 3894 | 24.06% | 20.78% | 0.606 | 0.1794 | 0.0357 |
| `Other` | 2,072 | 346 | 16.70% | 19.90% | 0.634 | 0.1357 | 0.0320 |
| `No charge` | 61 | 7 | 11.48% | — | — | — | *unmeasured* |
| `nan` | 0 | 0 | n/a | — | — | — | *unmeasured* |

*2 of 6 slices fell below the support floor and are not scored.*

### by language

**AUROC gap 0.089** — best `Kabuverdianu` 0.627, worst `Russian` 0.537. **ECE gap 0.0313** — worst `Kabuverdianu` 0.0358.

| Slice | n | Events | Base rate | Mean pred | AUROC | Brier | ECE |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `English` | 74,665 | 14995 | 20.08% | 19.83% | 0.616 | 0.1564 | 0.0046 |
| `Spanish` | 2,798 | 577 | 20.62% | 19.66% | 0.624 | 0.1590 | 0.0127 |
| `Russian` | 1,225 | 219 | 17.88% | 17.39% | 0.537 | 0.1469 | 0.0206 |
| `Chinese` | 1,164 | 219 | 18.81% | 17.62% | 0.592 | 0.1508 | 0.0133 |
| `Kabuverdianu` | 715 | 103 | 14.41% | 17.66% | 0.627 | 0.1221 | 0.0358 |
| `Portuguese` | 466 | 96 | 20.60% | — | — | — | *unmeasured* |
| `Haitian` | 336 | 66 | 19.64% | — | — | — | *unmeasured* |
| `Other` | 210 | 40 | 19.05% | — | — | — | *unmeasured* |
| `Modern Greek (1453-)` | 176 | 46 | 26.14% | — | — | — | *unmeasured* |
| `Vietnamese` | 165 | 33 | 20.00% | — | — | — | *unmeasured* |
| `Italian` | 155 | 36 | 23.23% | — | — | — | *unmeasured* |
| `American Sign Language` | 141 | 53 | 37.59% | — | — | — | *unmeasured* |
| `Arabic` | 86 | 23 | 26.74% | — | — | — | *unmeasured* |
| `Persian` | 64 | 7 | 10.94% | — | — | — | *unmeasured* |
| `Khmer` | 63 | 25 | 39.68% | — | — | — | *unmeasured* |
| `Polish` | 60 | 12 | 20.00% | — | — | — | *unmeasured* |
| `Korean` | 56 | 6 | 10.71% | — | — | — | *unmeasured* |
| `Thai` | 36 | 10 | 27.78% | — | — | — | *unmeasured* |
| `Amharic` | 25 | 1 | 4.00% | — | — | — | *unmeasured* |
| `Japanese` | 20 | 4 | 20.00% | — | — | — | *unmeasured* |
| `French` | 16 | 0 | 0.00% | — | — | — | *unmeasured* |
| `Hindi` | 16 | 2 | 12.50% | — | — | — | *unmeasured* |
| `Armenian` | 14 | 3 | 21.43% | — | — | — | *unmeasured* |
| `Bengali` | 8 | 0 | 0.00% | — | — | — | *unmeasured* |
| `Somali` | 6 | 0 | 0.00% | — | — | — | *unmeasured* |
| `nan` | 0 | 0 | n/a | — | — | — | *unmeasured* |

*21 of 26 slices fell below the support floor and are not scored.*

### by marital_status

**AUROC gap 0.040** — best `MARRIED` 0.633, worst `WIDOWED` 0.593. **ECE gap 0.0175** — worst `WIDOWED` 0.0261.

| Slice | n | Events | Base rate | Mean pred | AUROC | Brier | ECE |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `MARRIED` | 34,610 | 6452 | 18.64% | 19.37% | 0.633 | 0.1468 | 0.0086 |
| `SINGLE` | 31,317 | 6821 | 21.78% | 20.31% | 0.610 | 0.1663 | 0.0161 |
| `WIDOWED` | 8,741 | 1811 | 20.72% | 18.14% | 0.593 | 0.1620 | 0.0261 |
| `DIVORCED` | 6,138 | 1363 | 22.21% | 20.53% | 0.600 | 0.1699 | 0.0206 |
| `nan` | 0 | 0 | n/a | — | — | — | *unmeasured* |

*1 of 5 slices fell below the support floor and are not scored.*

### by admission_type

**AUROC gap 0.085** — best `URGENT` 0.664, worst `AMBULATORY OBSERVATION` 0.580. **ECE gap 0.0876** — worst `ELECTIVE` 0.0942.

| Slice | n | Events | Base rate | Mean pred | AUROC | Brier | ECE |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `EW EMER.` | 26,760 | 5483 | 20.49% | 20.79% | 0.603 | 0.1598 | 0.0065 |
| `EU OBSERVATION` | 18,390 | 3625 | 19.71% | 17.24% | 0.592 | 0.1562 | 0.0248 |
| `OBSERVATION ADMIT` | 12,892 | 2745 | 21.29% | 21.80% | 0.624 | 0.1625 | 0.0114 |
| `URGENT` | 8,253 | 1255 | 15.21% | 18.62% | 0.664 | 0.1253 | 0.0341 |
| `SURGICAL SAME DAY ADMISSION` | 6,471 | 911 | 14.08% | 18.46% | 0.599 | 0.1216 | 0.0456 |
| `DIRECT OBSERVATION` | 3,786 | 756 | 19.97% | 17.73% | 0.587 | 0.1581 | 0.0224 |
| `DIRECT EMER.` | 3,251 | 1035 | 31.84% | 23.93% | 0.613 | 0.2160 | 0.0790 |
| `ELECTIVE` | 1,910 | 602 | 31.52% | 22.34% | 0.592 | 0.2198 | 0.0942 |
| `AMBULATORY OBSERVATION` | 1,093 | 175 | 16.01% | 17.08% | 0.580 | 0.1331 | 0.0123 |

---

## Remediation applied

The numbers above already include **age-band isotonic calibration**, fitted by `scripts/maintenance/fit_group_calibrators.py`.

**This is not retraining.** No booster was refitted and no feature set changed. A second isotonic regression is fitted per age band on the *validation* split — the same estimator, on the same split, as the global calibrators already used — and applied on top of the global value at serve time. Only the mapping from score to probability moves; the model's ranking is untouched, which is why AUROC is essentially unchanged while ECE falls sharply.

A band calibrator is kept **only if it improves calibration on the test split it was not fitted on**. An isotonic fit always improves its own data, so that check is what separates a real correction from memorisation.

Fitted 2026-08-14 on the `val` split; floors n ≥ 1000 and events ≥ 50.

| Task | Band | val n | val events | ECE before | ECE after |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `mortality` | 18-39 | 17,477 | 78 | 0.0018 | 0.0015 |
| `mortality` | 40-54 | 17,762 | 216 | 0.0023 | 0.0016 |
| `mortality` | 55-69 | 23,659 | 542 | 0.0033 | 0.0016 |
| `mortality` | 70-84 | 17,276 | 602 | 0.0062 | 0.0019 |
| `mortality` | 85+ | 5,645 | 324 | 0.0118 | 0.0067 |
| `icu_admission` | 18-39 | 17,477 | 1302 | 0.0157 | 0.0073 |
| `icu_admission` | 40-54 | 17,762 | 2188 | 0.0164 | 0.0082 |
| `icu_admission` | 55-69 | 23,659 | 4330 | 0.0169 | 0.0045 |
| `icu_admission` | 70-84 | 17,276 | 3535 | 0.0284 | 0.0108 |
| `icu_admission` | 85+ | 5,645 | 1242 | 0.0427 | 0.0207 |
| `readmission` | 18-39 | 17,477 | 3348 | 0.0562 | 0.0069 |
| `readmission` | 40-54 | 17,762 | 3807 | 0.0743 | 0.0142 |
| `readmission` | 55-69 | 23,659 | 4935 | 0.0471 | 0.0092 |
| `readmission` | 70-84 | 17,276 | 3197 | 0.0369 | 0.0064 |
| `readmission` | 85+ | 5,645 | 889 | 0.0193 | 0.0062 |

**Still open, and requiring a retrain rather than a recalibration:** the `UNKNOWN` / `UNABLE TO OBTAIN` race groups, whose observed mortality is several times the cohort average while the model predicts roughly half of it. Those labels mark patients too unwell for demographics to be collected — signal the model currently sees only as another category value. An explicit `demographics_incomplete` feature would let it learn what the label means. Race is also an optional payload field, so a calibrator keyed on it would apply to a minority of real requests and would conflate "the clinician did not type it" with "the hospital could not record it" — different patients.

## Reading this

A large **AUROC gap** means the model separates cases better in some groups than others — it ranks well for one population and poorly for another. A large **ECE gap** means the probability means different things depending on the group: "5% risk" may be accurate for one and an underestimate for another, which is the more dangerous of the two because the number looks identical on screen.

Base-rate differences between slices are **not** themselves a model defect. Emergency admissions genuinely die more often than elective ones. What matters is whether the model is equally *accurate* and equally *honest* across groups, which is what AUROC and ECE measure and base rate does not.

