# Serve-Time Telemetry

*Generated 2026-08-14 by `scripts/evaluation/run_telemetry_eval.py` from `logs/assistant_audit.jsonl`.*

> **This is not online evaluation.** Online evaluation measures a system against real users — their traffic, their behaviour, and outcome signals such as whether the answer was acted on. No clinician traffic exists yet, so none of that is measured here.
>
> What this is: the same measurement layer an online evaluation would run on, applied to the traffic that does exist. The metrics are computed from records written at serve time rather than from a curated gold set, which is why they can show things the offline suite cannot — notably the messages nobody thought to write a fixture for.

## Volume

- Records: **137** · turns: **74** · sessions: **46**
- Mean turns per session: **1.61**

## Guardrail rates

- Answer rate: **28.4%** · refusal rate: **71.6%**
- Verification pass rate: **71.4%** (over 21 composed answers)
- Retrieval hit rate: **85.7%** · mean documents per answer: **3.48**


### Verification failures (6)

Between `2026-08-13T16:05:57+00:00` and `2026-08-13T16:18:40+00:00`.

| Time | Intent | Failed checks |
| :--- | :--- | :--- |
| 2026-08-13T16:05:57+00:00 | `risk_assessment` | 1:only provided information |
| 2026-08-13T16:07:31+00:00 | `risk_assessment` | 1:only provided information |
| 2026-08-13T16:09:55+00:00 | `risk_assessment` | 1:only provided information |
| 2026-08-13T16:14:25+00:00 | `counterfactual` | 1:only provided information |
| 2026-08-13T16:16:48+00:00 | `counterfactual` | 1:only provided information |
| 2026-08-13T16:18:40+00:00 | `counterfactual` | 1:only provided information |

*Clustering in time is the signal. Failures spread evenly are a flaky edge case; failures packed into a short window are a regression that was introduced and then fixed — and this is the view that distinguishes them. Re-run with `--since` after the last failure to see current health.*

*A high refusal rate is not a fault. Most turns in a completeness-gated conversation are the system asking for what it needs; the number to watch is verification pass rate, where anything below 100% means composed output was withheld.*

## Latency

- p50 **n/a ms** · p95 **n/a ms** · max **n/a ms** (n=0)

## Intent distribution

| Intent | Turns |
| :--- | ---: |
| `risk_assessment` | 50 |
| `guideline_lookup` | 12 |
| `capabilities` | 7 |
| `counterfactual` | 3 |
| `lab_result_interpretation` | 1 |
| `drug_dosing` | 1 |

## Gate outcomes

| Gate status | Turns |
| :--- | ---: |
| `INCOMPLETE` | 46 |
| `COMPLETE` | 21 |
| `none` | 7 |

## Most-requested fields

| Field | Times asked |
| :--- | ---: |
| `sbp_min` | 43 |
| `hr_max` | 43 |
| `potassium_max` | 38 |
| `sodium_min` | 36 |
| `platelets_min` | 36 |
| `hematocrit_min` | 36 |
| `glucose_max` | 36 |
| `creatinine_max` | 29 |

*Where a real user is most likely to give up, and therefore the highest-value target for better extraction.*

## Status breakdown

| Status | Turns |
| :--- | ---: |
| `declined_incomplete` | 53 |
| `answered` | 21 |

