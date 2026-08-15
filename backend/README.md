# Clinician backend

FastAPI service over the MIMIC-IV risk models and the curated guideline corpus.

```sh
.venv/bin/pip install -r backend/requirements.txt
cp .env.example .env          # then fill in OPENROUTER_API_KEY if you want one
.venv/bin/uvicorn backend.main:app --host 127.0.0.1 --port 8010
```

`.env` is gitignored; `.env.example` documents every variable and ships empty.
Nothing in it is required — with no key at all, extraction falls back to
deterministic patterns that score 100% precision and recall on the gold set.

Interactive docs at `http://127.0.0.1:8010/docs`. Port 8010, not 8000 — 8000 is
often already taken by another local service.

---

## ⚠ Scope

**Localhost only. No authentication. Not for real patient data.**

There is no login, no rate limiting, no transport security and no access
control. Sessions live in process memory. Nothing here would protect a real
patient's record, so do not put one in it. MIMIC-IV and synthetic cases are
fine — MIMIC is de-identified and already published.

Run with a single worker. Session state is in-memory, so a second worker would
serve half the requests from a process that has never heard of the conversation.

---

## Endpoints

| | |
| :-- | :-- |
| `GET /api/health` | what loaded, corpus counts, how many records are clinician-reviewed |
| `POST /api/models/validate` | check a payload without scoring it — needs no models |
| `POST /api/models/predict` | multi-task risk with calibration and uncertainty |
| `POST /api/models/report` | the full grounded report |
| `POST /api/models/whatif` | re-score with an input changed |
| `GET /api/models/describe` | loaded models, features, promoted metrics |
| `POST /api/assistant/sessions` | open a conversation |
| `POST /api/assistant/sessions/{id}/messages` | one turn |
| `GET /api/assistant/sessions/{id}` | current state, for a UI restoring |
| `DELETE /api/assistant/sessions/{id}` | discard a conversation |

### Refusals are 200s

`POST /api/assistant/.../messages` returns `200` with
`status: "declined_incomplete"` and a populated `questions` array when the
completeness gate is closed. `POST /api/models/report` returns `200` with
`status: "incomplete_input"` and the missing fields named.

These are the system working, not errors. Returning 4xx would push callers
towards retrying past a gate that exists to stop exactly that.

`POST /api/models/predict` is the exception: it returns `422` for an incomplete
payload, because there is no partial prediction to return.

---

## Environment

Set in `.env` (preferred) or the shell. Both are read at construction, not at
import, so editing `.env` takes effect without worrying about import order.

| variable | effect |
| :-- | :-- |
| `OPENROUTER_API_KEY` | enables LLM fact extraction from free text. Absent → deterministic patterns only: fewer fields per turn, more questions asked, nothing guessed. |
| `CDT_OPENROUTER_MODEL` | default `nvidia/nemotron-3-super-120b-a12b:free` — 120B mixture-of-experts, ~12B active, so large-model instruction-following at small-model latency. Alternatives if the free tier queues: `openai/gpt-oss-20b:free`, `google/gemma-4-31b-it:free`. |
| `CDT_OPENROUTER_BASE_URL` | default `https://openrouter.ai/api/v1`; override for a proxy or compatible gateway |
| `CDT_OPENROUTER_TIMEOUT` | seconds, default 30 |
| `CDT_ASSISTANT_DEBUG` | `1` attaches the gate decision, extraction proposals and faithfulness checks to each turn. Off by default — spec 28 keeps the reasoning trace out of what the reader receives. |
| `CDT_CORS_ORIGINS` | comma-separated; defaults to the usual Vite and CRA dev ports |

**OpenRouter sends what it is given to a third party.** Fine for MIMIC and
synthetic cases. Not fine for a real patient's labs.

---

## Known limitations

- **The guideline corpus is 23 records over 15 ICU conditions, none clinician-reviewed.** Sepsis, AKI, ARDS, DKA, hyperkalaemia, GI bleed, liver failure, stroke, heart failure, pneumonia, COPD, PE, pancreatitis, MI, hypertensive emergency. Anything else retrieves nothing and the assistant declines.
- **`hospital_los` is withheld** at 57.9% payload retention, below the two-thirds floor. By design. Four of five tasks are servable from a payload.
- **Readmission needs prior utilisation.** Without admission counts the model predicts readmission with the patient's own readmission history hidden.
- **Triage is disabled for this audience.** The red-flag rules are written for a person describing their own symptoms; a clinician asking about septic shock would be told to call an ambulance. Rewriting them as a clinician acuity flag is separate work.
- **Grounding proves traceability, not correctness.** A correctly-quoted guideline that does not apply to this patient passes every check. Clinician review of the corpus is the open item.
- **66 ED features exist and reach no model.** Adoption needs a dataset rebuild and a Phase 1–5 retrain.
