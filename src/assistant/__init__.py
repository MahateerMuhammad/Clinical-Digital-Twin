"""
src/assistant/
──────────────
Patient-facing clinical assistant.

Relationship to ``src/llm/``
───────────────────────────
``src/llm/`` serves *clinicians* a single-shot report about an admitted patient:
a payload of ICU-grade laboratory values goes in, MIMIC-trained risk models run,
and a grounded report comes out.

This package serves *patients* a multi-turn conversation. It deliberately does
**not** call the risk models. Those models were fitted on fourteen inpatient
laboratory values — creatinine, BUN, bicarbonate, platelets, haematocrit and the
rest — none of which a person at home can supply. Running them on a payload that
is mostly absent would return a mortality percentage derived from imputation
rather than from the patient, which is the precise failure mode the grounding
layer exists to prevent.

What the two share is the safety substrate, not the models:

* ``src.llm.grounding``   closed-world fact checking for generated text
* ``src.llm.guidelines``  the curated guideline corpus and its citations
* ``src.llm.terminology`` drug and diagnosis normalisation
* ``src.llm.rag_corpus``  evidence retrieval

Design rule inherited unchanged from ``src/llm/``: **the language model never
decides whether it has enough information to answer.** Application code does,
before any generation call, and the model cannot override it.
"""

from __future__ import annotations

__all__ = ["state", "intents", "requirements", "gate"]
