# Payload Fidelity Evaluation (Phase 11 serving gate)

_Generated 2026-08-06 10:17 UTC by `scripts/evaluation/run_payload_fidelity_eval.py`._

Held-out **test** split, 8,000 sampled admissions, seed 0, 400 bootstrap rounds.

## What this measures

Each model is scored twice on the same admissions: once from the complete
feature row it was validated on, and once from only the fields an unseen-patient
payload can carry, with everything else NaN. **Retention** is the fraction of
the validated model's discriminative lift that survives the restriction,
`(AUROC_payload - 0.5) / (AUROC_reference - 0.5)`.

A task is served from a payload only if retention reaches **66.7%**. The reference AUROCs below reproduce each
phase's published figure, which is what validates the harness itself.

## Results

| Task | n | Base rate | Payload coverage | AUROC (full record) | AUROC (payload) | Retention | 95% CI | Spearman ρ | Served |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | :---: |
| mortality | 8,000 | 1.99% | 18.3% | 0.9448 | 0.8470 | 78.0% | [0.716, 0.835] | +0.727 | **yes** |
| readmission | 8,000 | 19.65% | 17.6% | 0.7062 | 0.5673 | 32.6% | [0.250, 0.397] | +0.255 | withheld |
| icu_admission | 8,000 | 15.28% | 17.6% | 0.9209 | 0.6012 | 24.0% | [0.193, 0.284] | +0.004 | withheld |
| hospital_los | 8,000 | 25.39% | 17.6% | 0.8997 | 0.4894 | -2.7% | [-0.066, 0.011] | -0.076 | withheld |
| deterioration | 5,271 | 2.05% | 23.3% | 0.7858 | 0.4976 | -0.8% | [-0.197, 0.191] | -0.166 | withheld |

## Reading the table

- **Spearman ρ** is the rank correlation between the payload prediction and the
  same model's full-record prediction on the same patient. It is the sharpest
  statement of the problem: for ICU admission it is near zero, so the payload
  figure is not a degraded version of the validated prediction — it is
  unrelated to it.
- A retention **below zero** means the payload prediction is anti-correlated
  with the outcome; the model, denied the features it relies on, ranks patients
  backwards.
- **Payload coverage** is the share of trained features a payload populates.
  It is low for every task and is *not* the gate — coverage says how much input
  is missing, retention says whether what remains still discriminates.

## Consequence

1 of 5 tasks (mortality) may be served from a presentation payload. The rest are withheld by
`LiveModelRunner.run_live_inference_with_uncertainty`, which returns `None`
and a reason rather than a number. Predictions from a stored admission row are
unaffected — that path supplies the full feature set.

Regenerate after any Phase 1-5 retrain, then `--patch` to update
`PAYLOAD_FIDELITY` in `src/llm/model_runner.py`. `tests/test_payload_fidelity.py`
fails if the two disagree.
