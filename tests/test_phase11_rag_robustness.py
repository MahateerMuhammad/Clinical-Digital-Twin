import sys
import os
import unittest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath('.'))

from src.llm.rag_corpus import LiveRealtimeMedicalRAGEngine
from src.llm.model_runner import LiveModelRunner
from src.llm.clinical_agent import EnterpriseClinicalAgent

class TestPhase11RAGRobustness(unittest.TestCase):
    """
    Master Rigorous Test Suite for Phase 11 Medical RAG & Clinical AI Engine.
    Executes 10 comprehensive unit and stress tests.
    """
    @classmethod
    def setUpClass(cls):
        print("\n=======================================================")
        print("  RUNNING PHASE 11 MASTER RIGOROUS TEST SUITE (10/10) ")
        print("=======================================================\n")
        cls.rag = LiveRealtimeMedicalRAGEngine()
        cls.runner = LiveModelRunner()
        cls.agent = EnterpriseClinicalAgent()
        cls.test_hadm = 22595853
        # The engine validates a *nested* payload — demographics / presentation_labs /
        # vital_signs — and refuses a flat one. The flat dict below was the pre-
        # validation shape and no longer describes any supported input.
        cls.new_patient_payload = {
            'primary_diagnosis': 'acute kidney injury',
            'demographics': {'age': 68, 'gender': 'M'},
            'presentation_labs': {
                'creatinine_max': 4.5, 'bun_max': 82.0, 'wbc_max': 21.0,
                'bicarbonate_min': 16.0, 'sodium_min': 133.0, 'potassium_max': 5.6,
                'platelets_min': 105.0, 'hematocrit_min': 29.0, 'glucose_max': 180.0,
            },
            'vital_signs': {'sbp_min': 92.0, 'hr_max': 112.0, 'rr_max': 24.0,
                            'spo2_min': 93.0, 'temp_max': 38.2},
            'active_medications': ['vancomycin', 'enoxaparin'],
        }

    # Test 1: Live NIH DailyMed FDA API Fetching
    def test_01_live_dailymed_api(self):
        doc = self.rag.fetch_live_fda_dailymed('vancomycin')
        self.assertIsNotNone(doc)
        self.assertIn('citation', doc)
        self.assertIn('NIH DailyMed FDA Label', doc['citation'])
        print("  [TEST 1 PASSED] Live NIH DailyMed FDA API fetch successful.")

    # Test 2: Live NCBI PubMed API Fetching
    def test_02_live_pubmed_api(self):
        docs = self.rag.fetch_live_pubmed_papers('acute kidney injury creatinine')
        self.assertGreater(len(docs), 0)
        self.assertIn('PMID', docs[0]['citation'])
        print("  [TEST 2 PASSED] Live NCBI PubMed API search successful.")

    # Test 3: RAG Search for NEW UNSEEN Patient Payload
    def test_03_new_patient_rag(self):
        # Renamed: search_unseen_patient_rag, and it returns a result envelope whose
        # documents are under 'documents' rather than a bare list.
        res = self.rag.search_unseen_patient_rag(
            self.new_patient_payload, query_str="vancomycin renal failure", top_k=3)
        self.assertEqual(res['status'], 'ok', res.get('question_for_user', ''))
        self.assertGreater(len(res['documents']), 0)
        self.assertIn('citation', res['documents'][0])
        print("  [TEST 3 PASSED] NEW unseen patient payload RAG search successful.")

    # Test 4: Similar Historical Digital Twin Case Notes Retrieval
    def test_04_digital_twin_case_notes(self):
        # Renamed to find_disease_constrained_twin_notes. Level 5 retrieval requires
        # projecting the patient into the Phase 7 space, which a payload cannot do —
        # it carries a fraction of the encoder's inputs. The contract is that it
        # refuses and records why, never that it invents a twin.
        twin_notes = self.rag.find_disease_constrained_twin_notes(
            self.new_patient_payload, top_k=2)
        self.assertIsInstance(twin_notes, list)
        if not twin_notes:
            self.assertTrue(self.rag.last_twin_status,
                            "twin retrieval returned nothing without recording a status")
        else:
            self.assertIn('category', twin_notes[0])
        print("  [TEST 4 PASSED] Similar historical Digital Twin case notes retrieval verified.")

    # Test 5: Invalid/Missing Input Handling
    def test_05_invalid_input_handling(self):
        res = self.rag.search_unseen_patient_rag(None, query_str="sepsis", top_k=2)
        self.assertIsInstance(res, dict)
        self.assertEqual(res['status'], 'incomplete_input',
                         "a None payload must be refused, not processed")
        print("  [TEST 5 PASSED] Invalid/None input handled without crashing.")

    # Test 6: Blank Query String Handling
    def test_06_blank_query(self):
        # A blank query must fall back to the payload's own concepts, not fail.
        res = self.rag.search_unseen_patient_rag(
            self.new_patient_payload, query_str="", top_k=2)
        self.assertEqual(res['status'], 'ok')
        self.assertGreater(len(res['documents']), 0)
        print("  [TEST 6 PASSED] Blank query string handled safely.")

    # Test 7: Live Model Runner 5-Task Feature Column Alignment
    def test_07_live_model_runner(self):
        p_row = self.runner.get_patient_row(self.test_hadm)
        preds = self.runner.run_live_inference(p_row)
        self.assertIn('p_mortality', preds)
        self.assertIn('risk_tier', preds)
        self.assertGreaterEqual(preds['p_mortality'], 0.0)
        self.assertLessEqual(preds['p_mortality'], 1.0)
        print("  [TEST 7 PASSED] Multi-task live model inference verified.")

    # Test 8: Real-Time SHAP TreeExplainer Local Vector Calculation
    def test_08_shap_treeexplainer(self):
        shap_out = self.agent.tool_explain_shap(self.test_hadm, task='mortality', top_k=3)
        self.assertIn('top_shap_features', shap_out)
        self.assertGreater(len(shap_out['top_shap_features']), 0)
        print("  [TEST 8 PASSED] Real-time SHAP TreeExplainer calculation verified.")

    # Test 9: Counterfactual Treatment Simulator Precision
    def test_09_counterfactual_simulator(self):
        sim_res = self.runner.simulate_what_if(self.test_hadm, {
            'lab_bun_max': 10.0,
            'lab_creatinine_max': 0.8
        })
        self.assertIn('deltas', sim_res)
        self.assertIn('delta_p_mortality', sim_res['deltas'])
        print(f"  [TEST 9 PASSED] Counterfactual Simulator verified (Delta Mort = {sim_res['deltas']['delta_p_mortality']*100:.2f}%).")

    # Test 10: End-to-End Agentic Workflow Execution across Test Cohort
    def test_10_agentic_workflow_cohort(self):
        test_hadms = [22595853, 29668384, 21095812]
        for h in test_hadms:
            rep = self.agent.execute_agentic_workflow(h)
            # The workflow now composes through the fail-closed pipeline and appends
            # the tool output, rather than concatenating unverified markdown under a
            # banner heading.
            self.assertIn('Appendix — tool outputs', rep)
            self.assertIn('Local SHAP drivers', rep)
            self.assertIn('Phase 7 digital twins', rep)
            self.assertRegex(rep, r'\*\*Status:\*\* (ok|ok_no_evidence|incomplete_input|refused)')
        print("  [TEST 10 PASSED] End-to-End Agentic Workflow cohort execution verified.")

if __name__ == '__main__':
    unittest.main()
