# LLM Layer — Design and Training Plan

## 1. Architectural decision: the LLM does not reason

The models produce the risk numbers (Phases 1–5). The RAG produces the evidence.
**The LLM only restates that material more readably.** It is never asked what the
patient has, what the risk is, or what to do.

This is the decision that makes every other requirement achievable:

| Requirement | How the architecture satisfies it |
| :--- | :--- |
| No hallucination | The LLM cannot introduce facts, because everything it emits is checked against a closed fact store built from the inputs. |
| Refuse incomplete input | A deterministic validator runs *before* the LLM is loaded. The model never decides whether input is sufficient. |
| Handle synonyms | Resolved in `terminology.py` before retrieval. The LLM never sees an unnormalised diagnosis. |
| Small enough for Kaggle | Rephrasing is a far easier task than clinical reasoning, so a 3B model suffices. |

The pipeline:

```
payload
  → completeness gate       refuse + ask if insufficient        (deterministic)
  → model inference         calibrated + feature-coverage guard (Phase 1–5)
  → evidence retrieval      tiered, cited                       (RAG)
  → deterministic composer  grounded by construction            (no LLM)
  → LLM rephrase            readability only                    (optional)
  → grounding verifier      reject if anything was invented     (deterministic)
```

The LLM stage is the only optional one, and the only one that can fail closed.
If its output does not verify, the deterministic text is returned. **There is no
path in which unverified model prose reaches a caller.**

---

## 2. Model selection

**Recommendation: `Qwen2.5-3B-Instruct`, QLoRA 4-bit.**

| Candidate | Params | Fits T4 16GB (4-bit) | Verdict |
| :--- | ---: | :---: | :--- |
| **Qwen2.5-3B-Instruct** | 3.1B | ✅ comfortable | **Recommended.** Strong instruction-following at its size; leaves headroom for seq-4096. |
| Llama-3.2-3B-Instruct | 3.2B | ✅ comfortable | Equivalent choice. Pick on licence preference. |
| Llama-3.1-8B-Instruct | 8B | ⚠️ tight | Fits at seq-2048 with grad-checkpointing, but no headroom and ~3× the training time for a task that does not need it. |
| Qwen2.5-0.5B | 0.5B | ✅ trivial | Too weak — degrades section structure and drops citations. |

Why not larger: the task is *constrained rewriting of supplied text*. Capability
beyond instruction-following buys nothing here, and every extra parameter costs
Kaggle GPU hours you cannot get back.

### Kaggle budget
- 30 GPU-hours/week, 12h max per session, ~2 × T4 (16 GB each) or 1 × P100
- 3B QLoRA, ~3k examples, seq-2048, 2 epochs ≈ **2–3 hours on one T4**
- Comfortably inside one session with room to iterate

---

## 3. Fine-tuning: format, not medicine

**Do not fine-tune on clinical reasoning.** Train only the output contract:
section structure, citation retention, hedged non-causal phrasing, and refusal.

### Dataset generation (free, from this repo)

Sample real cohort rows → run the existing pipeline → the deterministic composer
output *is* the target. Input is the structured bundle; target is the composed
report.

```python
# ~3,000 examples, generated offline, no annotation cost
for row in cohort.sample(3000):
    payload  = row_to_payload(row)
    result   = pipeline.generate(payload, use_llm=False)   # deterministic target
    example  = {"input": structured_bundle(result), "target": result.report_markdown}
```

Add three adversarial slices, roughly 15% of the set:

1. **Incomplete payloads** → target is the `question_for_user` text, teaching the
   model to ask rather than fill gaps.
2. **Missing evidence** → target states "no evidence retrieved", teaching it not
   to substitute recalled knowledge.
3. **Withheld predictions** (low feature coverage) → target omits risk numbers
   entirely rather than estimating them.

### QLoRA configuration

```python
LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
           task_type="CAUSAL_LM",
           target_modules=["q_proj","k_proj","v_proj","o_proj",
                           "gate_proj","up_proj","down_proj"])

TrainingArguments(per_device_train_batch_size=2, gradient_accumulation_steps=8,
                  learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.03,
                  num_train_epochs=2, bf16=False, fp16=True,
                  gradient_checkpointing=True, optim="paged_adamw_8bit")
```

Inference is greedy (`do_sample=False`) — this stage must not be creative.

---

## 4. Guardrails, and what each actually covers

| Guardrail | Implemented in | Catches |
| :--- | :--- | :--- |
| Completeness gate | `payload_validation.py` | Missing/implausible/uninterpretable input |
| Feature-coverage guard | `pipeline.py` | Predictions from mostly-absent features |
| Concept confidence floor | `terminology.py` | Low-confidence diagnosis guesses → asks instead |
| Numeric grounding | `grounding.py` | Any number not in payload/predictions/evidence |
| Citation grounding | `grounding.py` | Citations to documents never retrieved |
| Claim patterns | `grounding.py` | SHAP/statistical/causal claims without support |
| Drug lexicon check | `grounding.py` | Medications not on the list or in the evidence |
| Deterministic fallback | `report_composer.py` | Total LLM failure |

**What this does not do.** The verifier proves *traceability*, not *correctness*.
It cannot detect a sentence that is grammatical, fully grounded, and clinically
misleading — for example correctly quoting a guideline that does not apply to this
patient. Preventing that needs clinician review of the guideline corpus and the
retrieval gold set, which is the outstanding work item.

---

## 5. Build order

1. ~~**Expand the payload schema**~~ — **done, 2026-08-10.** The figure quoted here
   was "~8%"; the measured value was 18.3%, which is the sort of drift this repo
   keeps producing and the reason step 5 exists.

   Two causes, and only the second was the one named:

   - `_convert_payload_to_series` reconstructed dummy names by hand (`gender_M = 0.0`)
     instead of emitting source categoricals and letting `encode_admission_frame`
     expand them. A female patient therefore reached Phase 5 as `gender_M = 0.0`
     with `gender_F` absent — "not male, sex unknown".
   - The schema never asked for race, language, insurance, marital status,
     admission type/location or prior utilisation. Those one-hot expansions are 86
     of the mortality model's 164 features, and `prior_*` is the Expansion A&D block
     Phase 2 was built on — the readmission model was being asked its question with
     the patient's own readmission history withheld.

   A complete payload now reaches **67.7%** of the mortality feature space (91.5%
   for deterioration). Retention against the 66.7% floor, before → after:

   | Task | Before | After | Served |
   | :--- | ---: | ---: | :--- |
   | mortality | 78.0% | **85.6%** | yes |
   | readmission | 32.6% | **81.7%** | now served |
   | icu_admission | 24.0% | **75.6%** | now served |
   | deterioration | −0.8% | **91.6%** | now served |
   | hospital_los | −2.7% | 57.9% | still withheld |

   Reference AUROCs are unchanged (0.9448 / 0.7062 / 0.9209 / 0.8997 / 0.7858), which
   is what makes the comparison valid: the stored-row path was never affected. It
   already worked, because the parquet holds these columns as `CategoricalDtype`
   carrying every level, so `get_dummies` emitted the full family. Only the payload
   path, which built plain strings, lost the siblings.

   `hospital_los` stays withheld at 57.9%. It is close, and it is the honest place to
   stop: length of stay depends on disposition and social factors a presentation
   payload does not carry, and the floor is not there to be lowered until things pass.

2. Generate the ~3k-example dataset from the cohort.
3. QLoRA fine-tune 3B on Kaggle (~3h).
4. Export the adapter, load via `TransformersBackend(adapter_path=...)`.
5. ~~Measure: % of generations passing the verifier, on a held-out set.~~
   **Harness built 2026-08-12**, and moved ahead of steps 2-4 — see below.
   `scripts/evaluation/run_llm_rephrase_eval.py`.

> [!WARNING]
> **Step 3's training pairs do not match what inference asks for.**
>
> §3 above specifies examples of the form *(structured bundle → composed report)*.
> But `pipeline.py` calls `backend.rephrase(deterministic_md, system_prompt)` — the
> model receives the **finished markdown**, not a bundle. Training would teach
> composition; inference asks for rewriting. That is a train/serve mismatch, and it
> is the same defect this project keeps producing: two descriptions of one thing,
> maintained separately, drifting apart.
>
> There is a second problem underneath it. If the target is the deterministic report
> and the input becomes that same report, the pair teaches the identity function —
> a model that reproduces its input perfectly, which is what the composer already
> does for free. Producing genuinely more readable targets needs a teacher model
> (distillation) or human rewriting. Neither is free, and neither is in §3.

### Why step 5 was built first

Fine-tuning is worth three GPU hours only if a stock instruct model fails the
verifier often enough to matter. Nobody had measured that, so steps 2-4 rested on an
assumption. The harness turns it into a number, and it needs no GPU.

It reports three things together, because the headline alone is misleading:

| Measure | Why it is there |
| :--- | :--- |
| Verifier pass rate | the headline: share of LLM outputs that survive grounding |
| Similarity to input | a model that passes by echoing its input scores 100% and is useless |
| Readability change | the only reason this stage exists |

The decision rule the harness encodes:

- **High pass rate, low similarity** — the stage works. No fine-tune needed.
- **High pass rate, high similarity** — the stage is inert. Fine-tuning will not fix
  that; the prompt or the model choice will.
- **Low pass rate** — not a safety failure, since the pipeline fails closed and the
  deterministic text ships. It means the stage costs latency and returns nothing,
  and *this* is the case where fine-tuning on the output contract is the right
  remedy — with the input/target correction above applied first.

Nothing about the architecture changes either way: the LLM never reasons, and the
verifier gates it. This decides only whether the optional stage earns its place.

> [!NOTE]
> Steps 2-5 are deferred pending a decision on whether to run them at all. They
> change how the report *reads*; step 1 changed what the system can *answer*. With
> four of five tasks now servable, the fine-tune is polish on a working deterministic
> composer, and the composer currently passes every published requirement at 100%
> with no LLM loaded.
