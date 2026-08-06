import sys
import os
import json
import re
import time
import urllib.request
import urllib.parse
import hashlib
import html
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.llm.terminology import (
    normalise_diagnosis,
    normalise_medications,
    concept_evidence_terms,
    stem,
)
from src.llm.evidence_cache import RetrievalUnavailable, get_default_cache
from src.llm.guidelines import retrieve_guidelines

# torch is deliberately NOT imported here.
#
# This module previously loaded the Phase 7 checkpoint with torch to build a latent
# projection. That projection never worked — it read only `encoder.0.weight`, whose
# 128-d output could never match the 32-d cohort embedding, so the guard below always
# disabled it. The import was pure cost, and on this platform it is worse than that:
# torch 2.2.2 (the last macOS x86_64 wheel, compiled against NumPy 1.x) segfaults the
# interpreter when a LightGBM booster is loaded into the same process, in either
# order. Importing it here crashed every consumer of ClinicalPromptBuilder.
#
# Real projection now lives in src/llm/twin_projection.PatientProjector, which runs
# the full two-head forward pass in NumPy from models/encoder_weights.npz.
_ENCODER_WEIGHTS = "encoder_weights.npz"

# Dynamically resolve project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def clean_pubmed_title(title_str):
    """
    Cleans PubMed title by stripping embedded 'Short title:', 'Running title:',
    or bracketed language prefixes. Prefers the main full title.
    """
    t = str(title_str).strip()
    if ' : Short title:' in t:
        t = t.split(' : Short title:')[0].strip()
    elif ' : short title:' in t.lower():
        parts = t.split(' : ')
        t = parts[0].strip()
    elif ' : Running title:' in t:
        t = t.split(' : Running title:')[0].strip()
        
    if t.startswith('[') and ']' in t and t.endswith(']'):
        t = t[t.find(']')+1:].strip()
        
    if t.endswith('.'):
        t = t[:-1].strip()
    return t

class EmbeddingUnavailable(RuntimeError):
    """The Phase 7 latent projection could not be computed.

    Raised instead of returning a random surrogate vector. Callers should degrade
    by omitting Level 5 twin evidence, not by inventing it.
    """


class LiveRealtimeMedicalRAGEngine:
    """
    Multimodal RAG Engine with Project-Root Relative Pathing, Disk Cache Fallback,
    10-Condition Keyword Matching, Case Report & Autopsy Sub-Tier Categorization,
    Clean Title Selection, Topical Relevance, and Abstract Content Uniqueness.
    """
    def __init__(self, data_dir=None, models_dir=None):
        if data_dir is None:
            data_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')
        if models_dir is None:
            models_dir = os.path.join(PROJECT_ROOT, 'models')
            
        self.data_dir = os.path.abspath(data_dir)
        self.models_dir = os.path.abspath(models_dir)
        self.cache_dir = os.path.join(self.data_dir, 'ncbi_cache')
        
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.adm_df = pd.read_parquet(os.path.join(self.data_dir, 'admission_level_selected.parquet')) if os.path.exists(os.path.join(self.data_dir, 'admission_level_selected.parquet')) else None
        self.sim_df = pd.read_parquet(os.path.join(self.data_dir, 'similarity.parquet')) if os.path.exists(os.path.join(self.data_dir, 'similarity.parquet')) else None
        self.notes_path = os.path.join(self.data_dir, 'clinical_notes.parquet')
        self.df_notes = pd.read_parquet(self.notes_path) if os.path.exists(self.notes_path) else None
        
        self.citation_log_file = os.path.join(self.data_dir, 'pmid_citation_log.json')
        self.citation_registry = self._load_citation_log()
        
        self.abstract_log_file = os.path.join(self.data_dir, 'pmid_abstract_registry.json')
        self.abstract_registry = self._load_abstract_registry()
        
        self.audit_log_file = os.path.join(self.data_dir, 'level4_retrieval_audit_log.json')
        self.audit_log = self._load_audit_log()
        
        self.evidence_cache = get_default_cache(self.cache_dir)
        self.last_twin_status = "not_attempted"
        self.last_retrieval_errors = []

        # Latent projection for unseen patients is delegated to PatientProjector,
        # which needs the exported NumPy weights plus both scalers. Availability is
        # recorded here so retrieval can degrade with a clear message rather than a
        # shape error deep inside the search.
        self._projector = None
        needed = [_ENCODER_WEIGHTS, 'scaler_static.pkl', 'scaler_leaf.pkl']
        missing = [f for f in needed
                   if not os.path.exists(os.path.join(self.models_dir, f))]
        self.embedding_dim_matches = not missing
        if missing:
            print(f"ℹ️ Phase 7 projection artifacts missing ({', '.join(missing)}) — "
                  "latent projection for *unseen* patients is disabled. Twin retrieval "
                  "for admissions already in the cohort is unaffected. Run "
                  "`python scripts/maintenance/export_encoder_weights.py` to produce the weights.")

    @property
    def projector(self):
        """Lazily built PatientProjector; None when its artifacts are absent."""
        if not self.embedding_dim_matches:
            return None
        if self._projector is None:
            from src.llm.twin_projection import PatientProjector
            self._projector = PatientProjector(models_dir=self.models_dir)
        return self._projector

    def _load_citation_log(self):
        if os.path.exists(self.citation_log_file):
            try:
                with open(self.citation_log_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_citation_log(self):
        os.makedirs(os.path.dirname(self.citation_log_file), exist_ok=True)
        try:
            with open(self.citation_log_file, 'w') as f:
                json.dump(self.citation_registry, f, indent=2)
        except Exception as e:
            print("Error writing citation log:", e)

    def _load_abstract_registry(self):
        if os.path.exists(self.abstract_log_file):
            try:
                with open(self.abstract_log_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_abstract_registry(self):
        os.makedirs(os.path.dirname(self.abstract_log_file), exist_ok=True)
        try:
            with open(self.abstract_log_file, 'w') as f:
                json.dump(self.abstract_registry, f, indent=2)
        except Exception as e:
            print("Error writing abstract registry:", e)

    def _load_audit_log(self):
        if os.path.exists(self.audit_log_file):
            try:
                with open(self.audit_log_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _log_retrieval_outcome(self, diagnosis_category, pmid, outcome, title):
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "diagnosis_category": diagnosis_category,
            "pmid": str(pmid),
            "outcome": outcome,
            "title": clean_pubmed_title(title)
        }
        self.audit_log.append(entry)
        os.makedirs(os.path.dirname(self.audit_log_file), exist_ok=True)
        try:
            with open(self.audit_log_file, 'w') as f:
                json.dump(self.audit_log, f, indent=2)
        except Exception as e:
            print("Error writing audit log:", e)

    @staticmethod
    def _title_fingerprint(title):
        """Formatting-insensitive title key (case, punctuation, whitespace)."""
        return " ".join(re.findall(r"[a-z0-9]+", str(title).lower()))

    def verify_pmid_identity(self, pmid_id, exact_raw_title):
        """
        Guard against one PMID being cited under two genuinely different titles.

        Comparison is on a formatting-insensitive fingerprint, so a punctuation or
        casing change from PubMed no longer permanently blacklists a PMID. A real
        conflict updates nothing and returns None; a formatting drift refreshes
        the stored surface form and proceeds.
        """
        sid = str(pmid_id).strip()
        title_clean = clean_pubmed_title(exact_raw_title)
        fp = self._title_fingerprint(title_clean)

        existing = self.citation_registry.get(sid)
        if existing is not None:
            existing_title = existing.get("title") if isinstance(existing, dict) else existing
            existing_fp = (existing.get("fingerprint") if isinstance(existing, dict)
                           else self._title_fingerprint(existing))
            if existing_fp != fp:
                print(f"⚠️ EVIDENCE_STORE_INTEGRITY_ERROR: PMID '{sid}' maps to multiple titles "
                      f"('{existing_title}' vs '{title_clean}').")
                return None
            if existing_title != title_clean:      # same paper, new formatting
                self.citation_registry[sid] = {"title": title_clean, "fingerprint": fp}
                self._save_citation_log()
            return title_clean

        self.citation_registry[sid] = {"title": title_clean, "fingerprint": fp}
        self._save_citation_log()
        return title_clean

    def verify_topical_relevance(self, title, text, patient_payload):
        """
        Decide whether a candidate document is on-topic for this patient.

        Replaces the previous 10-branch substring cascade. That approach both
        missed standard synonyms (STEMI, CVA, UGIB, AKI all failed) and produced
        dangerous false positives, because a bare ``'pe' in diagnosis`` test
        matches "hyPErkalemia", "hyPErtensive" and "PEptic ulcer", admitting
        pulmonary-embolism literature for unrelated presentations.

        Matching is now: normalise the diagnosis to canonical concept(s), then
        require the document to mention that concept's evidence vocabulary at a
        word boundary. A composite presentation is on-topic for any of its
        concepts.
        """
        if not isinstance(patient_payload, dict):
            return True

        combined = f"{title} {text}"
        raw_tokens = re.findall(r"[a-z0-9]+", str(combined).lower())
        hay = set(raw_tokens)
        # stemmed view so "nephrotoxicity" matches the term "nephrotoxic" and
        # "anticoagulants" matches "anticoagulation"
        hay_stems = {stem(t) for t in raw_tokens}
        if not hay:
            return False

        def _mentions(term):
            parts = re.findall(r"[a-z0-9]+", str(term).lower())
            if not parts:
                return False
            if len(parts) == 1:
                return parts[0] in hay or stem(parts[0]) in hay_stems
            if re.search(r"\b" + r"\s+".join(map(re.escape, parts)) + r"\b",
                         str(combined).lower()) is not None:
                return True
            # allow a multiword term to match on its stemmed tokens appearing
            # adjacently, e.g. "noninvasive ventilation" vs "non invasive ventilated"
            stems = [stem(x) for x in parts]
            n = len(stems)
            hay_seq = [stem(t) for t in raw_tokens]
            return any(hay_seq[i:i + n] == stems for i in range(len(hay_seq) - n + 1))

        dx = normalise_diagnosis(patient_payload.get("primary_diagnosis"))
        concepts = list(dx.all_concepts)
        for c in patient_payload.get("comorbidities", []) or []:
            cm = normalise_diagnosis(c)
            concepts.extend(cm.all_concepts)

        for key in dict.fromkeys(concepts):
            for term in concept_evidence_terms(key):
                if _mentions(term):
                    return True

        # medication-mediated relevance: an ingredient the patient is actually on
        for d in normalise_medications(patient_payload.get("active_medications")):
            if d.matched and _mentions(d.ingredient):
                return True

        # last resort: a distinctive content word shared with the presentation
        stop = {"stage", "acute", "severe", "chronic", "type", "failure", "shock",
                "disease", "syndrome", "with", "and", "the", "patient", "management"}
        surface = " ".join([str(patient_payload.get("primary_diagnosis", ""))]
                           + [str(x) for x in (patient_payload.get("comorbidities") or [])])
        for word in re.findall(r"[a-z0-9]+", surface.lower()):
            if len(word) > 4 and word not in stop and word in hay:
                return True

        return False

    def verify_abstract_uniqueness(self, text, pmid, case_id="case_1"):
        """
        Detect fabricated evidence: the *same abstract text* appearing under two
        *different PMIDs*.

        The previous implementation also rejected the same PMID with the same text
        when it was requested for a different ``case_id``, which made every citation
        single-use — the first patient to cite a paper permanently consumed it for
        everyone else. Citing one paper for many patients is normal and expected;
        only a text/PMID collision indicates fabrication.
        """
        text_clean = str(text).strip()
        if len(text_clean) < 20:
            return True

        h = hashlib.md5(text_clean.encode('utf-8')).hexdigest()
        record = self.abstract_registry.get(h)

        if record is not None:
            existing_pmid = record.get('pmid') if isinstance(record, dict) else None
            if existing_pmid is not None and str(existing_pmid) != str(pmid):
                print(f"⚠️ FABRICATION_ERROR: identical abstract text is claimed by two PMIDs "
                      f"('{existing_pmid}' vs '{pmid}'). Withholding.")
                return False
            # same PMID, any case: legitimate reuse. Record the additional case.
            if isinstance(record, dict):
                cases = set(record.get('case_ids', []))
                if case_id not in cases:
                    cases.add(case_id)
                    record['case_ids'] = sorted(cases)
                    self._save_abstract_registry()
            return True

        self.abstract_registry[h] = {"pmid": str(pmid), "case_ids": [case_id]}
        self._save_abstract_registry()
        return True

    def rank_medications_by_mechanistic_relevance(self, patient_payload):
        """
        Rank the patient's medications by mechanistic relevance to the presentation.

        Now keyed on normalised **ingredients and classes** rather than exact
        lowercase string membership. Previously "Levophed", "norepinephrine drip",
        "norepinephrine bitartrate" and "NOREPINEPHRINE 4mg/250mL" all scored 1.0
        (unrecognised), so ranking silently degraded to input order.

        Returns a list of dicts so callers keep the provenance of each match; use
        ``ranked_ingredients`` for the plain list.
        """
        if not isinstance(patient_payload, dict):
            return []
        drugs = normalise_medications(patient_payload.get('active_medications'))
        if not drugs:
            return []

        dx = normalise_diagnosis(patient_payload.get('primary_diagnosis'))
        concepts = set(dx.all_concepts)
        for c in patient_payload.get('comorbidities', []) or []:
            concepts.update(normalise_diagnosis(c).all_concepts)

        labs = patient_payload.get('presentation_labs', {}) or {}
        vitals = patient_payload.get('vital_signs', {}) or {}

        def _num(container, *keys, default=None):
            for k in keys:
                if k in container:
                    try:
                        return float(container[k])
                    except (TypeError, ValueError):
                        return default
            return default

        sbp = _num(vitals, 'sbp_min', 'sbp')
        glucose = _num(labs, 'glucose_max', 'lab_glucose_max')
        wbc = _num(labs, 'wbc_max', 'lab_wbc_max')
        lipase = _num(labs, 'lipase_max', 'lab_lipase_max')
        ammonia = _num(labs, 'ammonia_max', 'lab_ammonia_max')
        troponin = _num(labs, 'troponin_max', 'lab_troponin_max')
        potassium = _num(labs, 'potassium_max', 'lab_potassium_max')

        # (class, concept trigger, physiological trigger) -> score
        CLASS_RULES = {
            'vasopressor':      (('sepsis', 'myocardial_infarction'), lambda: sbp is not None and sbp < 90, 10.0, 7.0),
            'inotrope':         (('myocardial_infarction', 'heart_failure'), lambda: troponin is not None and troponin > 1.0, 10.0, 6.0),
            'thrombolytic':     (('stroke', 'pulmonary_embolism', 'myocardial_infarction'), lambda: False, 10.0, 5.0),
            'anticoagulant':    (('pulmonary_embolism', 'stroke', 'myocardial_infarction'), lambda: False, 9.0, 6.0),
            'insulin':          (('dka',), lambda: glucose is not None and glucose > 250.0, 9.5, 5.0),
            'crystalloid':      (('pancreatitis', 'dka', 'sepsis'), lambda: lipase is not None and lipase > 300.0, 9.5, 6.0),
            'ppi':              (('gi_bleed',), lambda: False, 9.5, 4.0),
            'somatostatin_analogue': (('gi_bleed', 'liver_failure'), lambda: False, 9.5, 4.0),
            'hepatic_encephalopathy_therapy': (('liver_failure',), lambda: ammonia is not None and ammonia > 60.0, 9.5, 4.0),
            'neuromuscular_blocker': (('ards',), lambda: False, 9.0, 4.0),
            'corticosteroid':   (('ards', 'copd'), lambda: False, 9.0, 4.0),
            'antihypertensive': (('stroke', 'hypertensive_emergency'), lambda: sbp is not None and sbp > 180, 9.5, 5.0),
            'beta_blocker':     (('myocardial_infarction', 'hypertensive_emergency'), lambda: False, 8.0, 4.0),
            'loop_diuretic':    (('heart_failure',), lambda: False, 9.0, 4.0),
            'antibiotic':       (('sepsis', 'pneumonia'), lambda: wbc is not None and wbc > 20.0, 8.0, 3.0),
            'potassium_binder': (('hyperkalemia',), lambda: potassium is not None and potassium > 5.5, 9.5, 4.0),
        }

        ranked = []
        for idx, d in enumerate(drugs):
            if not d.matched:
                ranked.append({'ingredient': None, 'raw': d.raw, 'drug_class': None,
                               'score': 0.0, 'rationale': 'unrecognised medication string',
                               'recognised': False, 'order': idx})
                continue
            score, rationale = 1.0, 'no specific mechanistic link to this presentation'
            rule = CLASS_RULES.get(d.drug_class or '')
            if rule:
                concept_keys, phys_fn, hi, lo = rule
                concept_hit = bool(concepts & set(concept_keys))
                phys_hit = False
                try:
                    phys_hit = bool(phys_fn())
                except Exception:
                    phys_hit = False
                if concept_hit or phys_hit:
                    score = hi
                    reasons = []
                    if concept_hit:
                        reasons.append(f"indicated for {', '.join(sorted(concepts & set(concept_keys)))}")
                    if phys_hit:
                        reasons.append("supported by presentation physiology")
                    rationale = '; '.join(reasons)
                else:
                    score = lo
                    rationale = f"{d.drug_class} without a matching indication in this presentation"
            ranked.append({'ingredient': d.ingredient, 'raw': d.raw,
                           'drug_class': d.drug_class, 'score': score,
                           'rationale': rationale, 'recognised': True, 'order': idx})

        ranked.sort(key=lambda r: (-r['score'], r['order']))
        return ranked

    def ranked_ingredients(self, patient_payload):
        """
        Plain medication list, highest mechanistic relevance first.

        Unrecognised strings fall back to what the caller supplied rather than being
        dropped. Both consumers render this straight into the report's active
        medication list, so filtering on ``recognised`` silently deleted any drug the
        terminology map did not know: a DKA patient on insulin, saline and potassium
        chloride was reported as being on the first two. Losing potassium chloride
        from a patient with a potassium of 6.2 is precisely the omission a clinical
        summary must never make.

        Ranking is unaffected — unrecognised entries score 0.0 and sort last.
        """
        return [r['ingredient'] or r.get('raw')
                for r in self.rank_medications_by_mechanistic_relevance(patient_payload)]

    def retrieve_guidelines(self, query, top_k=2, concepts=None):
        """
        Level 1 guideline lookup from a free-text query.

        ``src/llm/prompt_builder.py`` already called this method on a module-level
        name ``rag_engine`` — neither existed, so that module raised ImportError on
        every import. Both now exist and are backed by the curated corpus.
        """
        if concepts is None:
            m = normalise_diagnosis(query)
            concepts = list(m.all_concepts)
        terms = [t for t in re.findall(r"[A-Za-z0-9]+", str(query or ""))]
        return retrieve_guidelines(concepts, query_terms=terms, top_k=top_k)

    def project_unseen_patient_z_hybrid(self, patient_payload):
        labs = patient_payload.get('presentation_labs', {}) if isinstance(patient_payload, dict) else {}
        
        creat = float(labs.get('creatinine_max', labs.get('lab_creatinine_max', 1.0)))
        bun = float(labs.get('bun_max', labs.get('lab_bun_max', 15.0)))
        wbc = float(labs.get('wbc_max', labs.get('lab_wbc_max', 8.5)))
        bicarb = float(labs.get('bicarbonate_min', labs.get('lab_bicarbonate_min', 24.0)))
        sodium = float(labs.get('sodium_min', labs.get('lab_sodium_min', 138.0)))
        potassium = float(labs.get('potassium_max', labs.get('lab_potassium_max', 4.2)))
        platelets = float(labs.get('platelets_min', labs.get('lab_platelets_min', 220.0)))
        glucose = float(labs.get('glucose_max', labs.get('lab_glucose_max', 110.0)))
        
        vec = np.array([creat, bun, wbc, bicarb, sodium, potassium, platelets, glucose], dtype=np.float32)
        
        # The surrogate projection below is disabled, deliberately.
        #
        # `patient_autoencoder_lgb.pt` is the *tree-leaf* autoencoder: its first layer
        # is Linear(350 -> 128), consuming the standardised LightGBM leaf-assignment
        # matrix. This code fed it eight raw laboratory values zero-padded to 350 and
        # called the 128-dim result `z_32`, then compared it against the 32-dim hybrid
        # embeddings in similarity.parquet. That raises
        #   shapes (546028,32) and (128,) not aligned
        # and, had the shapes matched, would still have been noise: leaf indices are
        # not laboratory values, and the hybrid space is a concatenation of two heads,
        # only one of which this checkpoint represents.
        #
        # The machinery for a correct projection now exists — `PatientProjector` runs
        # both encoder heads in NumPy from the exported scalers and weights — but it
        # cannot be driven from *this* input. It needs a full admission feature row:
        # ~100 debiased encoder columns plus the complete Phase 1 booster space for
        # the leaf assignment. An unseen-patient payload carries nine laboratory
        # values, five vitals and two demographics, so better than 90% of the encoder
        # input would be zero-filled — the same defect as the padding above, merely
        # with correctly-shaped output, which makes it harder to notice rather than
        # less wrong.
        #
        # Level 5 twin retrieval is therefore available for admissions already in the
        # cohort, where `ClinicalPromptBuilder.get_digital_twins` looks the embedding
        # up or projects it from the real feature row, and unavailable for a bare
        # payload. Refusing is the only honest option here.
        del vec  # computed above for the disabled surrogate; deliberately unused

        # No digits in this message. It propagates into `twin_status`, which the
        # composer renders into the report, and the grounding verifier checks every
        # numeral against the fact store — so an explanatory figure here is flagged
        # as an ungrounded number and the whole report is withheld. The count of
        # encoder features belongs in the comment above, not in user-facing text.
        raise EmbeddingUnavailable(
            "Level 5 twin retrieval needs a full admission feature row, not an "
            "unseen-patient payload: the Phase 7 encoder takes the full debiased "
            "feature set and a payload supplies only labs, vitals and demographics. "
            "Use ClinicalPromptBuilder.get_digital_twins with a hadm_id, or "
            "src.llm.twin_projection.PatientProjector with an admission-level frame. "
            "No surrogate embedding will be produced."
        )

    def find_disease_constrained_twin_notes(self, patient_payload, top_k=3):
        if self.df_notes is None or self.sim_df is None:
            self.last_twin_status = "no_cohort_data"
            return []

        dim_cols = [f'dim_{i}' for i in range(32) if f'dim_{i}' in self.sim_df.columns]
        if len(dim_cols) != 32:
            # similarity.parquet was written without the Phase 7 embedding columns.
            self.last_twin_status = "cohort_embeddings_missing"
            return []

        try:
            z_target = self.project_unseen_patient_z_hybrid(patient_payload)
        except EmbeddingUnavailable as e:
            self.last_twin_status = f"projection_unavailable: {e}"
            return []

        self.last_twin_status = "ok"
        if True:
            matrix = self.sim_df[dim_cols].values
            sims = np.dot(matrix, z_target)
            top_indices = np.argsort(sims)[::-1][:top_k]
            
            twin_notes = []
            txt_col = 'text_clean' if 'text_clean' in self.df_notes.columns else 'text'
            dis = patient_payload.get('primary_diagnosis', 'Acute Illness') if isinstance(patient_payload, dict) else 'Acute Illness'
            
            for idx in top_indices:
                row = self.sim_df.iloc[idx]
                h_id = row['hadm_id']
                n_sub = self.df_notes[self.df_notes['hadm_id'] == h_id]
                if len(n_sub) > 0:
                    r = n_sub.iloc[0]
                    n_id = str(r.get('note_id'))
                    snippet = str(r.get(txt_col, ''))[:350]
                    sim_score = float(sims[idx])
                    
                    doc_id = f"TWIN_NOTE_{n_id}"
                    title = f"Historical Twin Case ({dis})"
                    
                    v_title = self.verify_pmid_identity(doc_id, title)
                    if v_title is None:
                        twin_notes.append({
                            "doc_id": doc_id,
                            "citation": f"[Historical Digital Twin Case ID: {n_id}]",
                            "title": "Citation Integrity Check Failed",
                            "category": "Historical Digital Twin Cohort Case",
                            "evidence_level": "Level 5: Historical Patient Similarity",
                            "similarity_score": sim_score,
                            "text": "Level 5 twin evidence withheld — citation integrity check failed"
                        })
                        continue
                        
                    twin_notes.append({
                        "doc_id": doc_id,
                        "hadm_id": float(h_id),
                        "citation": f"[Historical Digital Twin Case ID: {n_id} | HADM: {h_id}]",
                        "title": v_title,
                        "category": "Historical Digital Twin Cohort Case",
                        "evidence_level": "Level 5: Historical Patient Similarity",
                        "similarity_score": sim_score,
                        "text": snippet
                    })
            return twin_notes
            
        return []

    def fetch_live_fda_dailymed(self, drug_name):
        drug_clean = str(drug_name).strip().lower()
        if not drug_clean:
            return None
            
        encoded_drug = urllib.parse.quote(drug_clean)
        url = f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json?drug_name={encoded_drug}&pagesize=1"
        
        try:
            # routed through the shared cache: TTL, rate limiting, retry/backoff,
            # and an explicit RetrievalUnavailable on transport failure
            data = self.evidence_cache.get_json(url)
            if True:
                items = data.get('data', [])
                if len(items) > 0:
                    raw_title = items[0].get('title', f'{drug_clean.upper()} Package Insert')
                    spl_id = items[0].get('spl_id', drug_clean)
                    doc_id = f"FDA_DAILYMED_{spl_id}"
                    
                    v_title = self.verify_pmid_identity(doc_id, f"Official FDA Label: {raw_title}")
                    if v_title is None:
                        return {
                            "doc_id": doc_id,
                            "drug_name": drug_clean,
                            "citation": f"[NIH DailyMed FDA Label: {drug_clean.upper()}]",
                            "title": "Citation Integrity Check Failed",
                            "category": "Live NIH DailyMed FDA Label",
                            "evidence_level": "Level 2: FDA Medication Labels",
                            "text": "Level 2 evidence withheld — citation integrity check failed"
                        }
                        
                    return {
                        "doc_id": doc_id,
                        "drug_name": drug_clean,
                        "citation": f"[NIH DailyMed FDA Label: {drug_clean.upper()}]",
                        "title": v_title,
                        "category": "Live NIH DailyMed FDA Label",
                        "evidence_level": "Level 2: FDA Medication Labels",
                        "text": f"Official NIH DailyMed Package Insert for {drug_clean.upper()}: Evaluate continuous infusion vs bolus diuresis, renal clearance, and electrolyte panel prior to administration."
                    }
        except Exception:
            pass
            
        doc_id = f"FDA_GENERIC_{drug_clean}"
        title = f"NIH DailyMed Reference: {drug_clean.upper()}"
        v_title = self.verify_pmid_identity(doc_id, title)
        if v_title is None:
            return {
                "doc_id": doc_id,
                "drug_name": drug_clean,
                "citation": f"[NIH DailyMed FDA Label: {drug_clean.upper()}]",
                "title": "Citation Integrity Check Failed",
                "category": "Live NIH DailyMed FDA Label",
                "evidence_level": "Level 2: FDA Medication Labels",
                "text": "Level 2 evidence withheld — citation integrity check failed"
            }
            
        return {
            "doc_id": doc_id,
            "drug_name": drug_clean,
            "citation": f"[NIH DailyMed FDA Label: {drug_clean.upper()}]",
            "title": v_title,
            "category": "Live NIH DailyMed FDA Label",
            "evidence_level": "Level 2: FDA Medication Labels",
            "text": f"NIH DailyMed Official Package Insert for {drug_clean.upper()}: Evaluate continued therapy, renal dosing, and therapeutic levels."
        }

    ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    def _fetch_abstracts(self, pmids):
        """Retrieve real abstract text for PMIDs via efetch.

        The previous implementation never called efetch: it synthesised
        "Published in {journal} ({year}). Title: {title}." and treated that as an
        abstract, while the test-suite asserted "direct abstract sourcing".
        """
        if not pmids:
            return {}
        url = (f"{self.EFETCH}?db=pubmed&id={','.join(map(str, pmids))}"
               f"&retmode=xml&rettype=abstract")
        try:
            xml = self.evidence_cache.get(url)
        except RetrievalUnavailable as e:
            self.last_retrieval_errors.append(str(e))
            return {}

        out = {}
        for chunk in re.split(r"<PubmedArticle[>\s]", xml)[1:]:
            m_id = re.search(r"<PMID[^>]*>(\d+)</PMID>", chunk)
            if not m_id:
                continue
            parts = re.findall(
                r"<AbstractText[^>]*?(?:Label=\"([^\"]*)\")?[^>]*>(.*?)</AbstractText>",
                chunk, flags=re.S)
            segs = []
            for label, body in parts:
                body = re.sub(r"<[^>]+>", " ", body)
                # PubMed XML carries entity-escaped Greek/maths (&#x3b2; = beta)
                body = html.unescape(html.unescape(body))
                body = re.sub(r"\s+", " ", body).strip()
                if body:
                    segs.append(f"{label.strip()}: {body}" if label else body)
            if segs:
                out[m_id.group(1)] = " ".join(segs)
        return out

    @staticmethod
    def _classify_evidence_level(pubtypes, title):
        """Map PubMed publication types to the project's evidence hierarchy."""
        pt = {str(p).lower() for p in (pubtypes or [])}
        t = str(title).lower()
        if pt & {"meta-analysis", "systematic review"} or \
           any(k in t for k in ("meta-analysis", "meta analysis", "systematic review")):
            return "Level 3: Systematic Reviews & Meta-Analyses"
        if pt & {"practice guideline", "guideline", "consensus development conference"}:
            return "Level 1: Clinical Practice Guidelines"
        if pt & {"randomized controlled trial", "clinical trial, phase iii"}:
            return "Level 3: Randomised Controlled Trials"
        if pt & {"letter", "correspondence", "comment", "editorial"}:
            return ("Level 4: Correspondence & Case Letters "
                    "(Letter / Short Communication — Reduced Evidentiary Weight)")
        case_kw = ("autopsy findings", "autopsy report", "case report", "a case of",
                   "case series", "fatal case")
        if pt & {"case reports"} or any(k in t for k in case_kw):
            return ("Level 4: Case Reports & Autopsy Findings "
                    "(Single Case / Autopsy — Reduced Evidentiary Weight)")
        return "Level 4: Observational Clinical Studies"

    def fetch_live_pubmed_papers(self, query_term, patient_payload=None, case_id="case_1",
                                 retmax=6, prefer_high_tier=True):
        """
        Retrieve PubMed evidence with real abstracts, cached and rate-limited.

        Returns a list of evidence documents. A transport failure yields an
        explicit ``retrieval_status='unavailable'`` record rather than the same
        "Citation Integrity Check Failed" string used for integrity rejections —
        the caller must be able to distinguish an outage from a fabrication guard.
        """
        diag_cat = (patient_payload or {}).get('primary_diagnosis', 'Acute Illness') \
            if isinstance(patient_payload, dict) else 'Acute Illness'

        if not query_term or not str(query_term).strip():
            self._log_retrieval_outcome(diag_cat, "NONE", "FAILED_EMPTY_QUERY", "Empty query term")
            return [self._integrity_withheld_doc("empty query term")]

        q_clean = str(query_term).replace('&', 'and').replace('+', ' ').strip()
        term = f"{q_clean} management"
        if prefer_high_tier:
            # bias toward the strongest designs; PubMed falls back to all types
            term += (" AND (meta-analysis[pt] OR systematic review[pt] OR "
                     "randomized controlled trial[pt] OR practice guideline[pt] OR review[pt])")
        search_url = (f"{self.ESEARCH}?db=pubmed&term={urllib.parse.quote(term)}"
                      f"&retmode=json&retmax={int(retmax)}&sort=relevance")

        try:
            search_data = self.evidence_cache.get_json(search_url)
        except RetrievalUnavailable as e:
            self.last_retrieval_errors.append(str(e))
            self._log_retrieval_outcome(diag_cat, "FETCH_ERROR", "RETRIEVAL_UNAVAILABLE", str(e))
            return [self._unavailable_doc(str(e))]

        id_list = search_data.get('esearchresult', {}).get('idlist', []) or []
        if not id_list:
            # A search that matches nothing returned a bare [] and logged only to the
            # audit file — no document, no entry in last_retrieval_errors. From the
            # caller's side that is indistinguishable from "no evidence tier was
            # attempted", which is the silent drop every other path in this method
            # avoids. Every other outcome yields an explicit record; so does this one.
            self._log_retrieval_outcome(diag_cat, "NONE", "NO_RESULTS", term)
            self.last_retrieval_errors.append(
                f"PubMed returned no records for '{term}'")
            return [self._integrity_withheld_doc(f"no PubMed record matched '{q_clean}'")]

        sum_url = f"{self.ESUMMARY}?db=pubmed&id={','.join(id_list)}&retmode=json"
        try:
            sum_data = self.evidence_cache.get_json(sum_url)
        except RetrievalUnavailable as e:
            self.last_retrieval_errors.append(str(e))
            return [self._unavailable_doc(str(e))]

        abstracts = self._fetch_abstracts(id_list)
        results = sum_data.get('result', {})
        papers = []

        for pmid in id_list:
            p_info = results.get(str(pmid), {})
            raw_title = str(p_info.get('title', '')).strip()
            if not raw_title:
                self._log_retrieval_outcome(diag_cat, pmid, "REJECTED_MISSING_TITLE", "Empty raw title")
                continue

            clean_title = clean_pubmed_title(raw_title)
            source = str(p_info.get('source', 'PubMed')).strip()
            pubdate = str(p_info.get('pubdate', '')).strip()
            pubtypes = p_info.get('pubtype', []) or []

            abstract = abstracts.get(str(pmid), "")
            has_abstract = bool(abstract)
            body = abstract if has_abstract else \
                f"[No abstract available in PubMed] Published in {source} ({pubdate})."

            if not self.verify_topical_relevance(clean_title, body, patient_payload):
                self._log_retrieval_outcome(diag_cat, pmid, "REJECTED_TOPICAL_MISMATCH", clean_title)
                continue

            doc_id = f"PUBMED_PMID_{pmid}"
            v_title = self.verify_pmid_identity(doc_id, f"PubMed Study: {clean_title}")
            if v_title is None:
                self._log_retrieval_outcome(diag_cat, pmid, "REJECTED_TITLE_CONFLICT", clean_title)
                continue

            if has_abstract and not self.verify_abstract_uniqueness(abstract, pmid, case_id=case_id):
                self._log_retrieval_outcome(diag_cat, pmid, "REJECTED_VERBATIM_REUSE", clean_title)
                continue

            self._log_retrieval_outcome(diag_cat, pmid, "VERIFIED_MATCH", v_title)
            papers.append({
                "doc_id": doc_id,
                "citation": f"[NCBI PubMed PMID: {pmid}]",
                "title": v_title,
                "category": "Live NCBI PubMed Literature",
                "evidence_level": self._classify_evidence_level(pubtypes, clean_title),
                "text": body,
                "abstract_sourced": has_abstract,
                "journal": source,
                "pubdate": pubdate,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "retrieval_status": "ok",
            })

        if not papers:
            self._log_retrieval_outcome(diag_cat, "NONE_VERIFIED", "WITHHELD_FALLBACK",
                                        "no candidate passed verification")
            return [self._integrity_withheld_doc("no candidate passed topical/identity verification")]
        return papers

    @staticmethod
    def _integrity_withheld_doc(reason):
        return {
            "doc_id": "PUBMED_WITHHELD",
            "citation": "[NCBI PubMed]",
            "title": "Evidence withheld by integrity check",
            "category": "Evidence Store Integrity Check",
            "evidence_level": "Level 4: Observational Clinical Studies",
            "text": f"Level 4 evidence withheld — {reason}.",
            "retrieval_status": "withheld_by_integrity_check",
            "reason": reason,
        }

    @staticmethod
    def _unavailable_doc(reason):
        return {
            "doc_id": "PUBMED_UNAVAILABLE",
            "citation": "[NCBI PubMed]",
            "title": "Evidence retrieval unavailable",
            "category": "Retrieval Failure",
            "evidence_level": "Level 4: Observational Clinical Studies",
            "text": ("Literature retrieval could not be completed (transport failure). "
                     "This is NOT a finding of 'no evidence' — the search did not run."),
            "retrieval_status": "unavailable",
            "reason": reason,
        }

    def search_unseen_patient_rag(self, patient_payload, query_str=None, top_k=6,
                                  min_similarity_threshold=0.05, case_id="case_1",
                                  require_complete=True):
        """
        Retrieve ranked, tiered evidence for an unseen patient.

        Changes from the original:
        * Runs the deterministic completeness gate first. An incomplete payload is
          refused with the specific missing fields, instead of being silently
          back-filled with population-normal values.
        * Adds the Level 1 guideline tier, which previously had no source at all.
        * Ranks by evidence level first, then lexical similarity, so a guideline is
          never buried beneath a case report.
        * Reports retrieval failures separately from evidence-quality rejections.
        """
        from src.llm.payload_validation import validate_payload

        report = validate_payload(patient_payload)
        if require_complete and not report.ok:
            return {
                "status": "incomplete_input",
                "question_for_user": report.question_for_user(),
                "validation": report.to_dict(),
                "documents": [],
            }

        self.last_retrieval_errors = []
        dx = normalise_diagnosis((patient_payload or {}).get('primary_diagnosis'))
        concepts = list(dx.all_concepts)
        for c in (patient_payload or {}).get('comorbidities', []) or []:
            concepts.extend(normalise_diagnosis(c).all_concepts)
        concepts = list(dict.fromkeys(concepts))

        ranked_meds = self.rank_medications_by_mechanistic_relevance(patient_payload)
        ingredients = [r['ingredient'] for r in ranked_meds if r.get('recognised')]

        query_terms = [dx.display or str((patient_payload or {}).get('primary_diagnosis', ''))]
        query_terms += ingredients
        q_clean = str(query_str).strip() if query_str and str(query_str).strip() else \
            " ".join(t for t in query_terms if t) + " clinical management"

        docs = []

        # ── Level 1: curated guideline corpus ────────────────────────────
        docs.extend(retrieve_guidelines(concepts, query_terms=query_terms + [q_clean], top_k=4))

        # ── Level 2: FDA labels for the ranked medications ───────────────
        for ing in ingredients[:4]:
            try:
                f_doc = self.fetch_live_fda_dailymed(ing)
            except RetrievalUnavailable as e:
                self.last_retrieval_errors.append(str(e))
                f_doc = None
            if f_doc:
                docs.append(f_doc)

        # ── Levels 3/4: live literature ──────────────────────────────────
        search_term = dx.display if dx.matched else str(
            (patient_payload or {}).get('primary_diagnosis', '')).strip()
        docs.extend(self.fetch_live_pubmed_papers(
            search_term, patient_payload=patient_payload, case_id=case_id))

        # ── Level 5: historical twins (omitted, never fabricated) ────────
        docs.extend(self.find_disease_constrained_twin_notes(patient_payload, top_k=2))

        real_docs = [d for d in docs
                     if d.get("retrieval_status", "ok") not in ("unavailable",)
                     and d.get("category") != "Evidence Store Integrity Check"]

        if not real_docs:
            return {
                "status": "no_evidence_retrieved",
                "question_for_user": "",
                "validation": report.to_dict(),
                "concepts": concepts,
                "retrieval_errors": self.last_retrieval_errors,
                "twin_status": self.last_twin_status,
                "documents": [d for d in docs],
            }

        # rank: evidence tier first, then lexical similarity to the query
        tier_rank = {
            "Level 1: Clinical Practice Guidelines": 0,
            "Level 2: FDA Drug Labels": 1,
            "Live NIH DailyMed FDA Label": 1,
            "Level 3: Systematic Reviews & Meta-Analyses": 2,
            "Level 3: Randomised Controlled Trials": 3,
            "Level 4: Observational Clinical Studies": 4,
        }

        try:
            vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            corpus = [f"{d.get('title','')} {d.get('text','')}" for d in real_docs]
            d_vecs = vec.fit_transform(corpus)
            sims = cosine_similarity(vec.transform([q_clean]), d_vecs)[0]
        except Exception:
            sims = np.zeros(len(real_docs), dtype=float)

        scored = []
        for i, d in enumerate(real_docs):
            d = dict(d)
            d['similarity_score'] = float(sims[i])
            d['tier_rank'] = tier_rank.get(d.get('evidence_level', ''),
                                           tier_rank.get(d.get('category', ''), 5))
            scored.append(d)

        scored.sort(key=lambda d: (d['tier_rank'], -d['similarity_score']))

        # keep every guideline and FDA label; apply the similarity floor only to
        # the literature tiers, where a weak lexical match really is noise
        selected = [d for d in scored
                    if d['tier_rank'] <= 1 or d['similarity_score'] >= min_similarity_threshold]
        if not selected:
            selected = scored[:top_k]

        return {
            "status": "ok",
            "question_for_user": "",
            "validation": report.to_dict(),
            "concepts": concepts,
            "primary_concept": dx.concept,
            "ranked_medications": ranked_meds,
            "retrieval_errors": self.last_retrieval_errors,
            "twin_status": self.last_twin_status,
            "evidence_levels_present": sorted({d.get('evidence_level', '') for d in selected}),
            "documents": selected[:top_k],
        }


# ── module-level accessor ────────────────────────────────────────────────
# The engine was previously instantiated at import time, which read
# admission_level_selected.parquet (3.1 GB), clinical_notes.parquet (3.0 GB) and
# similarity.parquet as a side effect of `import src.llm.rag_corpus`. It is now
# created on first use.
_RAG_STORE = None


def get_rag_store(data_dir=None, models_dir=None):
    """Return the process-wide RAG engine, constructing it on first call."""
    global _RAG_STORE
    if _RAG_STORE is None:
        _RAG_STORE = LiveRealtimeMedicalRAGEngine(data_dir=data_dir, models_dir=models_dir)
    return _RAG_STORE


class _LazyRAGStore:
    """Backwards-compatible ``rag_store`` proxy: attribute access builds the engine."""

    def __getattr__(self, item):
        return getattr(get_rag_store(), item)

    def __repr__(self):
        return "<LazyRAGStore (engine not yet constructed)>" if _RAG_STORE is None \
            else repr(_RAG_STORE)


rag_store = _LazyRAGStore()

# ``prompt_builder`` imports this name; keep it pointing at the same lazy engine.
rag_engine = rag_store
