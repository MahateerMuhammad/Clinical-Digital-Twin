# RAG Retrieval Evaluation

Generated: 2026-07-30T16:57:09+00:00

Gold sets: `tests/gold/*.json` · Harness: `src/llm/retrieval_eval.py`


> [!WARNING]
> **All gold sets are `review_status: unreviewed`.** They were authored
> alongside the code they measure, so these numbers demonstrate internal
> consistency and guard against regression. They are **not** independent
> clinical validation, and must not be cited as retrieval accuracy against
> clinician ground truth until each row has been reviewed.


Guideline corpus: **23 records** across **15 concepts** (0 clinician-reviewed).


## 1. Summary

| Evaluation | N | Headline metric | Value |
| :--- | ---: | :--- | ---: |
| Terminology normalisation | 123 | Concept accuracy | 1.000 |
| Level 1 guideline retrieval | 45 | nDCG@3 | 0.982 |
| Topical relevance judgement | 64 | F1 | 1.000 |

## Terminology normalisation

Cases: **123** · Failures: **0**

| Metric | Value |
| :--- | ---: |
| accuracy | 1.000 |
| false_positive_rate_on_null_terms | 0.000 |
| false_negative_rate_on_real_terms | 0.000 |
| composite_all_concepts_accuracy | 1.000 |

<details><summary>Breakdown</summary>


| Group | N | Accuracy |
| :--- | ---: | ---: |
| abbreviation | 28 | 1.000 |
| canonical | 25 | 1.000 |
| composite | 8 | 1.000 |
| composite_term | 1 | 1.000 |
| electrolyte_group | 2 | 1.000 |
| empty | 1 | 1.000 |
| hard_negative_pe | 8 | 1.000 |
| junk | 4 | 1.000 |
| lay_term | 2 | 1.000 |
| legacy_term | 3 | 1.000 |
| out_of_scope | 5 | 1.000 |
| partial | 1 | 1.000 |
| qualified | 1 | 1.000 |
| regional_variant | 5 | 1.000 |
| related_term | 10 | 1.000 |
| sign | 2 | 1.000 |
| spelling_variant | 2 | 1.000 |
| subtype | 9 | 1.000 |
| synonym | 3 | 1.000 |
| umbrella_term | 2 | 1.000 |
| variant | 1 | 1.000 |

</details>


## Level 1 guideline retrieval

Cases: **45** · Failures: **0**

| Metric | Value |
| :--- | ---: |
| recall@1 | 0.711 |
| precision@1 | 1.000 |
| ndcg@1 | 0.976 |
| recall@3 | 0.980 |
| precision@3 | 0.829 |
| ndcg@3 | 0.982 |
| recall@5 | 1.000 |
| precision@5 | 0.805 |
| ndcg@5 | 0.990 |
| mrr | 1.000 |
| out_of_scope_correctly_empty | 1.000 |

## Topical relevance judgement

Cases: **64** · Failures: **0**

| Metric | Value |
| :--- | ---: |
| precision | 1.000 |
| recall | 1.000 |
| f1 | 1.000 |
| accuracy | 1.000 |
| false_positive_rate | 0.000 |
| pe_substring_trap_accuracy | 1.000 |

<details><summary>Breakdown</summary>


| Group | N | Accuracy |
| :--- | ---: | ---: |
| confusion | | {'tp': 40, 'fp': 0, 'tn': 24, 'fn': 0} |

</details>

