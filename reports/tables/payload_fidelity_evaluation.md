# Payload Fidelity Evaluation (Phase 11 serving gate)

_Generated 2026-08-10 06:18 UTC by `scripts/evaluation/run_payload_fidelity_eval.py`._

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
| mortality | 8,000 | 1.99% | 67.7% | 0.9448 | 0.8809 | 85.6% | [0.805, 0.898] | +0.833 | **yes** |
| readmission | 8,000 | 19.65% | 67.6% | 0.7062 | 0.6684 | 81.7% | [0.772, 0.860] | +0.744 | **yes** |
| icu_admission | 8,000 | 15.28% | 67.6% | 0.9209 | 0.8183 | 75.6% | [0.734, 0.781] | +0.793 | **yes** |
| hospital_los | 8,000 | 25.39% | 67.6% | 0.8997 | 0.7314 | 57.9% | [0.556, 0.604] | +0.717 | withheld |
| deterioration | 5,271 | 2.05% | 91.5% | 0.7858 | 0.7617 | 91.6% | [0.828, 1.010] | +0.898 | **yes** |

## Reading the table

- **Spearman ρ** is the rank correlation between the payload prediction and the
  same model's full-record prediction on the same patient. A ρ near zero means
  the payload figure is not a degraded version of the validated prediction but
  an unrelated one; a high ρ means the same patients are ranked the same way.
  Here ρ ranges +0.72 to +0.90, lowest for hospital_los.
- A retention **below zero** would mean the payload prediction is anti-correlated
  with the outcome — the model, denied the features it relies on, ranking
  patients backwards. No task does so here.
- **Payload coverage** is the share of trained features a payload populates.
  It is *not* the gate — coverage says how much input is missing, retention says
  whether what remains still discriminates. The two move together but not
  reliably: a payload can cover little and retain much, or the reverse.

## Consequence

4 of 5 tasks (mortality, readmission, icu_admission, deterioration) may be served from a presentation payload. The rest are withheld by
`LiveModelRunner.run_live_inference_with_uncertainty`, which returns `None`
and a reason rather than a number. Predictions from a stored admission row are
unaffected — that path supplies the full feature set.

Regenerate after any Phase 1-5 retrain, then `--patch` to update
`PAYLOAD_FIDELITY` in `src/llm/model_runner.py`. `tests/test_payload_fidelity.py`
fails if the two disagree.
