import sys
import os
import unittest

sys.path.insert(0, os.path.abspath('.'))

from src.llm.rag_corpus import LiveRealtimeMedicalRAGEngine

class TestPhase12RAGLevel4Integrity(unittest.TestCase):
    """
    Phase 12 Verification Suite for RAG Level 4 Evidence Rules 1, 2, and 3.
    """
    @classmethod
    def setUpClass(cls):
        print("\n=======================================================")
        print("  RUNNING RAG LEVEL 4 EVIDENCE INTEGRITY SUITE         ")
        print("=======================================================\n")
        cls.rag = LiveRealtimeMedicalRAGEngine()
        
        cls.sepsis_payload = {
            "primary_diagnosis": "Septic Shock & Acute Respiratory Failure",
            "comorbidities": ["COPD", "Severe Leukocytosis"],
            "presentation_labs": {"creatinine_max": 3.2, "bun_max": 54.0, "wbc_max": 26.5},
            "vital_signs": {"sbp_min": 82, "hr_max": 132},
            "active_medications": ["norepinephrine", "vancomycin", "cefepime"],
            "chief_complaint": "High fever, altered mental status"
        }

    # Rule 1 Test: Direct Abstract Sourcing (No synthetic/constructed fallback abstract)
    def test_01_direct_abstract_sourcing_rule1(self):
        # When PMID direct abstract cannot be fetched or is absent, it MUST trigger withholding fallback
        papers = self.rag.fetch_live_pubmed_papers("non_existent_search_query_99999", patient_payload=self.sepsis_payload)
        self.assertGreater(len(papers), 0)
        # The withheld-document title was renamed.
        self.assertIn("withheld", papers[0]["title"].lower())
        # The withheld document now names the specific reason rather than repeating a
        # fixed phrase. Assert the withholding and that a reason accompanies it.
        self.assertIn("Level 4 evidence withheld", papers[0]["text"])
        self.assertGreater(len(papers[0]["text"].split("—", 1)[-1].strip()), 5,
                           "withheld document must state why")
        print("  [TEST 1 PASSED] Rule 1 Direct Abstract Sourcing verified (No synthetic fallback abstracts).")

    # Rule 2 Test: Topical Relevance Verification against patient primary diagnosis & medications
    def test_02_topical_relevance_verification_rule2(self):
        off_topic_title = "PubMed Study: Management of Distal Radius Fractures in Elderly Patients"
        off_topic_abstract = "Surgical fixation options for closed forearm fractures."
        
        # Check that off-topic title/abstract fails relevance for septic shock payload
        is_relevant = self.rag.verify_topical_relevance(off_topic_title, off_topic_abstract, self.sepsis_payload)
        self.assertFalse(is_relevant)
        
        on_topic_title = "PubMed Study: Hemodynamic Management in Septic Shock"
        on_topic_abstract = "Vasopressor therapy with norepinephrine improves perfusion."
        is_relevant_on = self.rag.verify_topical_relevance(on_topic_title, on_topic_abstract, self.sepsis_payload)
        self.assertTrue(is_relevant_on)
        print("  [TEST 2 PASSED] Rule 2 Topical Relevance Verification verified.")

    # Rule 3 Test: Verbatim Abstract Duplication & Fabrication Detection across PMIDs
    def test_03_verbatim_abstract_duplication_rule3(self):
        pmid_1 = "TEST_PMID_90001"
        pmid_2 = "TEST_PMID_90002"
        duplicate_abstract = "Observational study investigating fluid resuscitation protocols in ICU patients."
        
        # Signature is verify_abstract_uniqueness(text, pmid) — the arguments were
        # passed the other way round, so each call registered the *pmid* as the
        # abstract text. The two calls therefore never collided and the duplication
        # guard was never actually exercised by this test.
        res1 = self.rag.verify_abstract_uniqueness(duplicate_abstract, pmid_1)
        self.assertTrue(res1)

        # The identical abstract under a different PMID must be rejected.
        res2 = self.rag.verify_abstract_uniqueness(duplicate_abstract, pmid_2)
        self.assertFalse(res2)
        print("  [TEST 3 PASSED] Rule 3 Verbatim Abstract Duplication Detection verified.")

if __name__ == '__main__':
    unittest.main()
