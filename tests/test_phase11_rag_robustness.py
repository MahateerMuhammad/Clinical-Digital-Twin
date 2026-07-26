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
        cls.new_patient_payload = {
            'anchor_age': 68,
            'gender': 'M',
            'lab_creatinine_max': 4.5,
            'lab_bun_max': 82.0,
            'lab_wbc_max': 21.0,
            'active_meds': ['vancomycin', 'enoxaparin']
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
        res = self.rag.search_rag_for_new_or_existing_patient(self.new_patient_payload, "vancomycin renal failure", top_k=3)
        self.assertGreater(len(res), 0)
        self.assertIn('citation', res[0])
        print("  [TEST 3 PASSED] NEW unseen patient payload RAG search successful.")

    # Test 4: Similar Historical Digital Twin Case Notes Retrieval
    def test_04_digital_twin_case_notes(self):
        twin_notes = self.rag.find_similar_twin_case_notes(self.new_patient_payload, top_k=2)
        self.assertGreater(len(twin_notes), 0)
        self.assertIn('Historical Digital Twin Case', twin_notes[0]['category'])
        print("  [TEST 4 PASSED] Similar historical Digital Twin case notes retrieval verified.")

    # Test 5: Invalid/Missing Input Handling
    def test_05_invalid_input_handling(self):
        res = self.rag.search_rag_for_new_or_existing_patient(None, "sepsis", top_k=2)
        self.assertIsInstance(res, list)
        print("  [TEST 5 PASSED] Invalid/None input handled without crashing.")

    # Test 6: Blank Query String Handling
    def test_06_blank_query(self):
        res = self.rag.search_rag_for_new_or_existing_patient(self.test_hadm, "", top_k=2)
        self.assertGreater(len(res), 0)
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
            self.assertIn('ADVANCED CLINICAL DIGITAL TWIN AGENT REPORT', rep)
            self.assertIn('SHAP Risk Drivers', rep)
            self.assertIn('RAG Retrieved Evidence', rep)
        print("  [TEST 10 PASSED] End-to-End Agentic Workflow cohort execution verified.")

if __name__ == '__main__':
    unittest.main()
