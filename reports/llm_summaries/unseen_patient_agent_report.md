# Clinical Decision Support Summary

**Presentation:** 72-year-old M — acute kidney injury  
**Mapped concept:** Acute Kidney Injury

## 1. Observed values (as supplied)

- peak creatinine 4.8 mg/dL
- peak BUN 88 mg/dL
- peak WBC 18.5 K/uL
- lowest bicarbonate 17 mEq/L
- lowest sodium 132 mEq/L
- peak potassium 5.8 mEq/L
- lowest platelets 96 K/uL
- lowest haematocrit 28 %
- peak glucose 194 mg/dL
- lowest systolic BP 90 mmHg
- peak heart rate 118 bpm

These are the values provided as input; they are restated here without interpretation beyond the model outputs below.

## 2. Model risk estimates

_No model predictions were supplied._

## 3. Retrieved evidence

**Level 1: Clinical Practice Guidelines**

- [KDIGO Clinical Practice Guideline for AKI 2012, Section 5: Renal Replacement Therapy] KDIGO Clinical Practice Guideline for AKI (2012) — Section 5: Renal Replacement Therapy
  > Initiate renal replacement therapy emergently for life-threatening changes in fluid, electrolyte and acid-base balance. Base the decision on the broader clinical context rather than the serum creatinine or urea value alone.
  Source: https://kdigo.org/guidelines/acute-kidney-injury/
  _paraphrased summary — verify against source document_
- [KDIGO Clinical Practice Guideline for AKI 2012, Section 3.9: Drug Dosing] KDIGO Clinical Practice Guideline for AKI (2012) — Section 3.9: Drug Dosing
  > Adjust renally-cleared drug doses for the current level of kidney function. Serum creatinine-based estimates are unreliable in non-steady-state AKI, so dose adjustment should account for changing function rather than a single eGFR value.
  Source: https://kdigo.org/guidelines/acute-kidney-injury/
  _paraphrased summary — verify against source document_
- [KDIGO Clinical Practice Guideline for AKI 2012, Section 3: Prevention and Treatment] KDIGO Clinical Practice Guideline for AKI (2012) — Section 3: Prevention and Treatment
  > In patients with or at risk of AKI, discontinue nephrotoxic agents where possible, ensure volume status and perfusion pressure are optimised, and monitor serum creatinine and urine output closely. Avoid hyperglycaemia.
  Source: https://kdigo.org/guidelines/acute-kidney-injury/
  _paraphrased summary — verify against source document_
- [ACC/AHA/HFSA Guideline for the Management of Heart Failure 2022, Cardiorenal Considerations] ACC/AHA/HFSA Guideline for the Management of Heart Failure (2022) — Cardiorenal Considerations
  > A modest rise in serum creatinine during effective decongestion does not by itself mandate stopping diuresis. Assess volume status directly rather than treating the creatinine change in isolation.
  Source: https://www.ahajournals.org/doi/10.1161/CIR.0000000000001063
  _paraphrased summary — verify against source document_

**Level 2: FDA Medication Labels**

- [NIH DailyMed FDA Label: FUROSEMIDE] Official FDA Label: FUROSEMIDE INJECTION [CIVICA, INC.]
  > Official NIH DailyMed Package Insert for FUROSEMIDE: Evaluate continuous infusion vs bolus diuresis, renal clearance, and electrolyte panel prior to administration.
- [NIH DailyMed FDA Label: ENOXAPARIN] Official FDA Label: LOVENOX (ENOXAPARIN SODIUM) INJECTION [SANOFI-AVENTIS U.S. LLC]
  > Official NIH DailyMed Package Insert for ENOXAPARIN: Evaluate continuous infusion vs bolus diuresis, renal clearance, and electrolyte panel prior to administration.

## 4. Active medications, by mechanistic relevance

- **enoxaparin** (anticoagulant) — relevance 6.0: anticoagulant without a matching indication in this presentation
- **furosemide** (loop_diuretic) — relevance 4.0: loop_diuretic without a matching indication in this presentation
- **vancomycin** (antibiotic) — relevance 3.0: antibiotic without a matching indication in this presentation

Relevance reflects the link between drug class and the stated presentation. It is not a recommendation to start, stop or change any therapy.

## 5. Uncertainty and confidence

- Probabilities are calibrated estimates from models trained on MIMIC-IV; they describe populations, not individual certainty.
- No causal claim is made. These models identify association, not treatment effect.
- Historical twin evidence unavailable (projection_unavailable: Level 5 twin retrieval needs a full admission feature row, not an unseen-patient payload: the Phase 7 encoder takes the full debiased feature set and a payload supplies only labs, vitals and demographics. Use ClinicalPromptBuilder.get_digital_twins with a hadm_id, or src.llm.twin_projection.PatientProjector with an admission-level frame. No surrogate embedding will be produced.); no similar-patient comparison is included.

## 6. Limitations

- This summary restates supplied values, model outputs and retrieved guideline text. It does not constitute a clinical assessment.
- Guideline records are paraphrased summaries pending clinician review; verify wording against the cited source before acting.
- Model estimates derive from a single-centre US ICU/hospital dataset and may not transfer to other populations or care settings.

## 7. Provenance

- Evidence documents cited: 6
- Retrieval status: ok
- Twin retrieval: projection_unavailable: Level 5 twin retrieval needs a full admission feature row, not an unseen-patient payload: the Phase 7 encoder takes the full debiased feature set and a payload supplies only labs, vitals and demographics. Use ClinicalPromptBuilder.get_digital_twins with a hadm_id, or src.llm.twin_projection.PatientProjector with an admission-level frame. No surrogate embedding will be produced.
- Generated deterministically from structured inputs; every number above appears in the payload, the model outputs, or a cited document.
