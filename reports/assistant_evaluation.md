# Clinician Assistant — Evaluation Report

*Generated 2026-08-14 by `scripts/evaluation/run_assistant_eval.py`.*

Every gold set below was authored by the same engineering effort that wrote the code. These numbers measure **internal consistency and regression safety**, not clinical correctness: a correctly-retrieved guideline that does not apply to the patient passes every check here.

## 1. Retrieval

Concept-anchored lexical retrieval over the curated guideline corpus. `normalise_diagnosis` maps the query to canonical concepts, the concept index returns candidates, and query-term overlap re-scores them. **No embedding retrieval and no reranker** are involved at this tier.

- Queries: **45**
- MRR: **1.000**
- Context recall (unbounded): **100.0%**
- Median retrieval latency: **0.02 ms**

| k | Precision@k | Recall@k | nDCG@k | Hit rate@k | Context precision@k |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 100.0% | 71.1% | 0.976 | 100.0% | 100.0% |
| 3 | 82.9% | 98.0% | 0.982 | 100.0% | 99.6% |
| 5 | 80.5% | 100.0% | 0.990 | 100.0% | 98.8% |

*Context precision is the rank-sensitive metric: it penalises a relevant document returned late, which plain precision@k cannot see. It is the number a reranker would move.*

## 2. Intent routing

- Cases: **46** · Accuracy: **100.0%** · Macro F1: **1.000**
- Model-vs-evidence routing accuracy: **100.0%**

| Intent | Precision | Recall | F1 | Support |
| :--- | ---: | ---: | ---: | ---: |
| `capabilities` | 100.0% | 100.0% | 1.000 | 3 |
| `counterfactual` | 100.0% | 100.0% | 1.000 | 6 |
| `drug_dosing` | 100.0% | 100.0% | 1.000 | 7 |
| `guideline_lookup` | 100.0% | 100.0% | 1.000 | 12 |
| `lab_result_interpretation` | 100.0% | 100.0% | 1.000 | 4 |
| `risk_assessment` | 100.0% | 100.0% | 1.000 | 10 |
| `terminology` | 100.0% | 100.0% | 1.000 | 4 |

## 3. Fact extraction

Backend: **deterministic**

- Precision **100.0%** · Recall **100.0%** · F1 **1.000**
- **Fabrication rate: 0.0%** (0 of 28 extracted values)

*Fabrication rate is the one that must be zero: it counts values written into patient state that do not appear in the message. Recall below 100% means the gate asks more questions — friction, not danger.*

## 4. Abstention

- Cases: **15** · Accuracy: **100.0%**
- **Under-refusal rate: 0.0%** (answered when it should have refused — the unsafe direction)
- Over-refusal rate: 0.0% (refused when it could have answered — friction)

## 5. Faithfulness

- Answers produced: **34**
- Verified: **100.0%**
- Carrying at least one citation: **100.0%**

*Computed over answered turns only. Including refusals would let a system that never answers score 100%.*

