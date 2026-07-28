"""
tests/test_rag_hardened.py
──────────────────────────
Adversarial regression suite for the hardened RAG layer.

Every case here corresponds to a defect found by audit. Offline by default: no
test reaches the network. Run with::

    .venv/bin/python -m pytest tests/test_rag_hardened.py -q
    .venv/bin/python tests/test_rag_hardened.py          # standalone runner
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Import the scientific stack before any torch stub so scipy does not probe it.
import scipy.stats  # noqa: E402,F401
import sklearn.feature_extraction.text  # noqa: E402,F401

if "torch" not in sys.modules:
    try:
        import torch  # noqa: F401
    except ImportError:
        _t = types.ModuleType("torch")
        _t.Tensor = type("Tensor", (), {})
        _t.set_num_threads = lambda n: None
        _t.load = lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no ckpt"))
        sys.modules["torch"] = _t

from src.llm import rag_corpus as rc                       # noqa: E402
from src.llm.evidence_cache import EvidenceCache, RetrievalUnavailable  # noqa: E402
from src.llm.guidelines import corpus_stats, retrieve_guidelines        # noqa: E402
from src.llm.payload_validation import validate_payload                 # noqa: E402
from src.llm.terminology import (                                       # noqa: E402
    normalise_diagnosis,
    normalise_medication,
)

COMPLETE_PAYLOAD = {
    "demographics": {"age": 68, "gender": "M"},
    "primary_diagnosis": "Septic Shock",
    "comorbidities": ["COPD"],
    "presentation_labs": {
        "creatinine_max": 3.2, "bun_max": 54.0, "wbc_max": 26.5,
        "bicarbonate_min": 16.0, "sodium_min": 132.0, "potassium_max": 5.1,
        "platelets_min": 90.0, "hematocrit_min": 28.0, "glucose_max": 180.0,
    },
    "vital_signs": {"sbp_min": 82, "hr_max": 132},
    "active_medications": ["norepinephrine", "vancomycin", "cefepime"],
}


def _engine(tmpdir, **kw):
    """Engine instance with no cohort parquet loads and an isolated cache."""
    e = rc.LiveRealtimeMedicalRAGEngine.__new__(rc.LiveRealtimeMedicalRAGEngine)
    e.data_dir = tmpdir
    e.models_dir = os.path.join(tmpdir, "models")
    e.cache_dir = os.path.join(tmpdir, "cache")
    os.makedirs(e.cache_dir, exist_ok=True)
    e.adm_df = None
    e.sim_df = pd.DataFrame({"subject_id": [1], "hadm_id": [2.0],
                             "embedding_placeholder": [""]})
    e.df_notes = pd.DataFrame({"note_id": ["n1"], "hadm_id": [2.0],
                               "text_clean": ["sepsis and aki"]})
    e.notes_path = ""
    e.citation_log_file = os.path.join(tmpdir, "cite.json")
    e.citation_registry = {}
    e.abstract_log_file = os.path.join(tmpdir, "abs.json")
    e.abstract_registry = {}
    e.audit_log_file = os.path.join(tmpdir, "audit.json")
    e.audit_log = []
    e.evidence_cache = EvidenceCache(e.cache_dir, offline=True)
    e.last_twin_status = "not_attempted"
    e.last_retrieval_errors = []
    e.w0_numpy = None
    e.b0_numpy = None
    for k, v in kw.items():
        setattr(e, k, v)
    return e


class TerminologyTests(unittest.TestCase):
    """Clinical synonyms must resolve; substrings must not create false hits."""

    SYNONYMS = [
        ("STEMI", "myocardial_infarction"), ("NSTEMI", "myocardial_infarction"),
        ("MI", "myocardial_infarction"), ("Acute coronary syndrome", "myocardial_infarction"),
        ("CHF exacerbation", "heart_failure"), ("Congestive cardiac failure", "heart_failure"),
        ("Cardiorenal syndrome", "heart_failure"),
        ("Septicaemia", "sepsis"), ("Urosepsis", "sepsis"), ("Bacteraemia", "sepsis"),
        ("CVA", "stroke"), ("Brain attack", "stroke"), ("Cerebral infarction", "stroke"),
        ("UGIB", "gi_bleed"), ("Haematemesis", "gi_bleed"), ("Melena", "gi_bleed"),
        ("AKI", "aki"), ("Renal insufficiency", "aki"), ("Acute tubular necrosis", "aki"),
        ("DKA", "dka"), ("Hyperosmolar hyperglycemic state", "dka"),
        ("ARDS", "ards"), ("Acute respiratory failure", "ards"),
        ("Hepatic encephalopathy", "liver_failure"),
        ("Pulmonary embolus", "pulmonary_embolism"),
        ("AECOPD", "copd"),
    ]

    def test_synonyms_resolve(self):
        for surface, expected in self.SYNONYMS:
            with self.subTest(surface=surface):
                self.assertEqual(normalise_diagnosis(surface).concept, expected)

    def test_pe_substring_does_not_false_positive(self):
        """'pe' must never match inside hyPErkalemia / hyPErtensive / PEptic."""
        for surface in ["Hyperkalemia", "Hypertensive emergency", "Hyperglycemia",
                        "Hypernatremia", "Peptic ulcer disease"]:
            with self.subTest(surface=surface):
                self.assertNotEqual(normalise_diagnosis(surface).concept,
                                    "pulmonary_embolism")

    def test_composite_diagnosis_keeps_all_concepts(self):
        m = normalise_diagnosis("Septic Shock & Acute Respiratory Failure")
        self.assertEqual(m.concept, "sepsis")           # first-mentioned is primary
        self.assertIn("ards", m.all_concepts)

    def test_medication_surface_forms(self):
        for surface in ["norepinephrine", "Norepinephrine", "norepinephrine drip",
                        "Levophed", "norepinephrine bitartrate",
                        "NOREPINEPHRINE 4mg/250mL", "norepi"]:
            with self.subTest(surface=surface):
                d = normalise_medication(surface)
                self.assertEqual(d.ingredient, "norepinephrine")
                self.assertEqual(d.drug_class, "vasopressor")

    def test_medication_brands(self):
        for surface, ing in [("Lasix 40mg IV push", "furosemide"),
                             ("Protonix", "pantoprazole"), ("vanc", "vancomycin"),
                             ("Zosyn IVPB 3.375g", "piperacillin tazobactam"),
                             ("0.9% NaCl", "normal saline")]:
            with self.subTest(surface=surface):
                self.assertEqual(normalise_medication(surface).ingredient, ing)

    def test_unknown_terms_are_unmatched_not_guessed(self):
        for junk in ["asdfqwer", "", None, "zzzz-999"]:
            with self.subTest(junk=junk):
                self.assertIsNone(normalise_diagnosis(junk).concept)


class ValidationGateTests(unittest.TestCase):
    """Incomplete input must be refused with specifics, never back-filled."""

    def test_complete_payload_passes(self):
        self.assertTrue(validate_payload(COMPLETE_PAYLOAD).ok)

    def test_empty_payload_refused_with_question(self):
        r = validate_payload({})
        self.assertFalse(r.ok)
        self.assertGreaterEqual(len(r.missing_required), 10)
        self.assertIn("Peak serum creatinine", r.question_for_user())

    def test_missing_single_field_is_named(self):
        p = json.loads(json.dumps(COMPLETE_PAYLOAD))
        del p["presentation_labs"]["creatinine_max"]
        r = validate_payload(p)
        self.assertFalse(r.ok)
        self.assertTrue(any("creatinine" in m["path"] for m in r.missing_required))

    def test_implausible_values_rejected(self):
        for path, value in [("creatinine_max", -50.0), ("creatinine_max", 1e12),
                            ("potassium_max", 99.0), ("sodium_min", 5.0)]:
            with self.subTest(path=path, value=value):
                p = json.loads(json.dumps(COMPLETE_PAYLOAD))
                p["presentation_labs"][path] = value
                r = validate_payload(p)
                self.assertFalse(r.ok)
                self.assertTrue(r.implausible)

    def test_non_numeric_lab_does_not_crash(self):
        p = json.loads(json.dumps(COMPLETE_PAYLOAD))
        p["presentation_labs"]["creatinine_max"] = "unknown"
        r = validate_payload(p)          # must not raise ValueError
        self.assertFalse(r.ok)
        self.assertTrue(r.uninterpretable)

    def test_numeric_string_accepted(self):
        p = json.loads(json.dumps(COMPLETE_PAYLOAD))
        p["presentation_labs"]["creatinine_max"] = "3.2"
        self.assertTrue(validate_payload(p).ok)

    def test_unrecognised_diagnosis_refused(self):
        p = json.loads(json.dumps(COMPLETE_PAYLOAD))
        p["primary_diagnosis"] = "zzzz not a disease"
        self.assertFalse(validate_payload(p).ok)

    def test_hostile_payloads_do_not_crash(self):
        for bad in [None, "sepsis", ["sepsis"], 42, {"primary_diagnosis": None},
                    {"primary_diagnosis": "sepsis " * 20000},
                    {"primary_diagnosis": "sepsis 🦠 敗血症"},
                    {"active_medications": "norepinephrine"},
                    {"presentation_labs": {"creatinine_max": float("nan")}},
                    {"presentation_labs": {"creatinine_max": float("inf")}}]:
            with self.subTest(bad=str(bad)[:40]):
                r = validate_payload(bad)
                self.assertFalse(r.ok)


class IntegrityGuardTests(unittest.TestCase):
    """Fabrication guards must block fabrication without destroying real evidence."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ragtest_")
        self.e = _engine(self.tmp)

    def test_same_paper_reusable_across_patients(self):
        txt = "Early antibiotics reduced mortality in this septic shock cohort."
        self.assertTrue(self.e.verify_abstract_uniqueness(txt, "111", case_id="patient_A"))
        self.assertTrue(self.e.verify_abstract_uniqueness(txt, "111", case_id="patient_B"))
        self.assertTrue(self.e.verify_abstract_uniqueness(txt, "111", case_id="patient_C"))

    def test_same_text_two_pmids_still_blocked(self):
        txt = "Identical abstract body claimed by two different identifiers entirely."
        self.assertTrue(self.e.verify_abstract_uniqueness(txt, "111", case_id="a"))
        self.assertFalse(self.e.verify_abstract_uniqueness(txt, "222", case_id="a"))

    def test_title_reformatting_does_not_blacklist_pmid(self):
        self.assertIsNotNone(self.e.verify_pmid_identity("222", "Sepsis management in the ICU"))
        self.assertIsNotNone(self.e.verify_pmid_identity("222", "Sepsis Management in the ICU."))
        self.assertIsNotNone(self.e.verify_pmid_identity("222", "sepsis  management   in the icu"))

    def test_genuinely_different_title_still_blocked(self):
        self.assertIsNotNone(self.e.verify_pmid_identity("333", "Sepsis management in the ICU"))
        self.assertIsNone(self.e.verify_pmid_identity("333", "Prone positioning in ARDS"))


class RelevanceTests(unittest.TestCase):
    """Topical relevance must accept synonyms and reject unrelated literature."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ragtest_")
        self.e = _engine(self.tmp)

    def test_accepts_on_topic_under_synonym(self):
        cases = [
            ("STEMI", "Troponin dynamics after myocardial infarction"),
            ("CVA", "Alteplase thrombolysis in acute ischemic stroke"),
            ("UGIB", "Endoscopy timing in gastrointestinal bleeding"),
            ("AKI", "Renal replacement therapy initiation timing"),
            ("CHF exacerbation", "Diuretic strategies in acute heart failure"),
        ]
        for dx, title in cases:
            with self.subTest(dx=dx):
                self.assertTrue(self.e.verify_topical_relevance(
                    title, title, {"primary_diagnosis": dx}))

    def test_rejects_off_topic(self):
        cases = [
            ("Hyperkalemia", "Alteplase thrombolysis for pulmonary embolism"),
            ("Hypertensive emergency", "Anticoagulation for venous thromboembolism"),
            ("Peptic ulcer disease", "Heparin bridging in thromboembolism"),
            ("Septic Shock", "Orthopaedic outcomes after elective knee arthroplasty"),
        ]
        for dx, title in cases:
            with self.subTest(dx=dx):
                self.assertFalse(self.e.verify_topical_relevance(
                    title, title, {"primary_diagnosis": dx}))


class MedicationRankingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ragtest_")
        self.e = _engine(self.tmp)

    def test_brand_and_dose_forms_are_ranked(self):
        p = dict(COMPLETE_PAYLOAD, active_medications=["Levophed 4mg/250mL", "Lasix 40mg IV"])
        ranked = self.e.rank_medications_by_mechanistic_relevance(p)
        self.assertTrue(all(r["recognised"] for r in ranked))
        self.assertEqual(ranked[0]["ingredient"], "norepinephrine")   # shock outranks diuretic
        self.assertGreaterEqual(ranked[0]["score"], 9.0)

    def test_unrecognised_medication_flagged_not_silently_dropped(self):
        p = dict(COMPLETE_PAYLOAD, active_medications=["zzz-mystery-drug"])
        ranked = self.e.rank_medications_by_mechanistic_relevance(p)
        self.assertEqual(len(ranked), 1)
        self.assertFalse(ranked[0]["recognised"])

    def test_every_ranking_carries_a_rationale(self):
        ranked = self.e.rank_medications_by_mechanistic_relevance(COMPLETE_PAYLOAD)
        self.assertTrue(all(r.get("rationale") for r in ranked))


class GuidelineCorpusTests(unittest.TestCase):
    """Level 1 must be a real retrievable source, not a label."""

    def test_corpus_is_populated(self):
        stats = corpus_stats()
        self.assertGreaterEqual(stats["n_records"], 15)
        self.assertGreaterEqual(stats["n_concepts_covered"], 10)

    def test_retrieval_by_concept(self):
        docs = retrieve_guidelines(["sepsis"], query_terms=["antibiotic"])
        self.assertTrue(docs)
        self.assertEqual(docs[0]["evidence_level"], "Level 1: Clinical Practice Guidelines")

    def test_unknown_concept_returns_empty_not_fabricated(self):
        self.assertEqual(retrieve_guidelines(["not_a_concept"]), [])

    def test_every_record_carries_provenance(self):
        for doc in retrieve_guidelines(list(corpus_stats()["concepts"]), top_k=100):
            self.assertTrue(doc["url"])
            self.assertFalse(doc["verbatim"])       # paraphrase, honestly marked
            self.assertIn("provenance", doc)


class DegradationTests(unittest.TestCase):
    """Failures must be explicit and never fabricated."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ragtest_")
        self.e = _engine(self.tmp)

    def test_embedding_refused_when_weights_absent(self):
        with self.assertRaises(rc.EmbeddingUnavailable):
            self.e.project_unseen_patient_z_hybrid(COMPLETE_PAYLOAD)

    def test_twin_retrieval_degrades_with_status(self):
        twins = self.e.find_disease_constrained_twin_notes(COMPLETE_PAYLOAD)
        self.assertEqual(twins, [])
        self.assertEqual(self.e.last_twin_status, "cohort_embeddings_missing")

    def test_offline_raises_retrieval_unavailable(self):
        cache = EvidenceCache(os.path.join(self.tmp, "c2"), offline=True)
        with self.assertRaises(RetrievalUnavailable):
            cache.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed")

    def test_outage_is_distinguishable_from_integrity_rejection(self):
        docs = self.e.fetch_live_pubmed_papers("septic shock", COMPLETE_PAYLOAD)
        self.assertEqual(docs[0]["retrieval_status"], "unavailable")
        self.assertNotEqual(docs[0]["retrieval_status"], "withheld_by_integrity_check")

    def test_cache_actually_persists(self):
        cache = EvidenceCache(os.path.join(self.tmp, "c3"))
        url = "https://example.invalid/x"
        cache._write(url, '{"ok": true}')
        self.assertEqual(cache.get(url), '{"ok": true}')
        self.assertEqual(cache.stats["hit"], 1)


class OrchestratorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ragtest_")
        self.e = _engine(self.tmp)

    def test_incomplete_payload_asks_instead_of_retrieving(self):
        out = self.e.search_unseen_patient_rag({"primary_diagnosis": "Septic Shock"})
        self.assertEqual(out["status"], "incomplete_input")
        self.assertTrue(out["question_for_user"])
        self.assertEqual(out["documents"], [])

    def test_complete_payload_returns_guidelines_offline(self):
        """With the network down, Level 1 must still deliver evidence."""
        out = self.e.search_unseen_patient_rag(COMPLETE_PAYLOAD)
        self.assertEqual(out["status"], "ok")
        self.assertTrue(out["documents"])
        self.assertIn("Level 1: Clinical Practice Guidelines", out["evidence_levels_present"])

    def test_guidelines_rank_above_literature(self):
        out = self.e.search_unseen_patient_rag(COMPLETE_PAYLOAD)
        self.assertEqual(out["documents"][0]["evidence_level"],
                         "Level 1: Clinical Practice Guidelines")

    def test_concepts_are_reported(self):
        out = self.e.search_unseen_patient_rag(COMPLETE_PAYLOAD)
        self.assertEqual(out["primary_concept"], "sepsis")
        self.assertIn("copd", out["concepts"])

    def test_no_fabricated_twin_evidence(self):
        out = self.e.search_unseen_patient_rag(COMPLETE_PAYLOAD)
        for d in out["documents"]:
            self.assertNotIn("Level 5", d.get("evidence_level", ""))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class GoldSetThresholdTests(unittest.TestCase):
    """Regression guard: retrieval quality must not silently degrade.

    Thresholds sit just below the measured baseline, so an unrelated change that
    damages retrieval fails here rather than being discovered in a report months
    later. Raise them when the gold sets grow.
    """

    @classmethod
    def setUpClass(cls):
        from src.llm.retrieval_eval import run_full_evaluation
        cls.results = run_full_evaluation()

    def test_terminology_accuracy(self):
        self.assertGreaterEqual(self.results["terminology"].metrics["accuracy"], 0.97)

    def test_terminology_does_not_guess(self):
        fpr = self.results["terminology"].metrics["false_positive_rate_on_null_terms"]
        self.assertLessEqual(fpr, 0.05)

    def test_guideline_ndcg(self):
        self.assertGreaterEqual(self.results["guidelines"].metrics["ndcg@3"], 0.90)

    def test_guideline_out_of_scope_returns_nothing(self):
        self.assertEqual(self.results["guidelines"].metrics["out_of_scope_correctly_empty"], 1.0)

    def test_relevance_f1(self):
        self.assertGreaterEqual(self.results["relevance"].metrics["f1"], 0.93)

    def test_relevance_no_false_positives(self):
        """A false positive here means off-topic evidence reaches a clinician."""
        self.assertLessEqual(self.results["relevance"].metrics["false_positive_rate"], 0.02)

    def test_pe_substring_traps_all_caught(self):
        self.assertEqual(self.results["relevance"].metrics["pe_substring_trap_accuracy"], 1.0)
