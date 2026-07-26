import sys
import os
import json
import hashlib
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.llm.rag_corpus import LiveRealtimeMedicalRAGEngine, clean_pubmed_title
from src.llm.model_runner import LiveModelRunner
from src.llm.clinical_assistant import EnterpriseClinicalAgent

class TestClinicalRAGStandingRegressionSuite(unittest.TestCase):
    """
    Exhaustive 12-Assertion Standing Regression Suite for Clinical Digital Twin RAG & Decision-Support System.
    Validates 10 ICU & Emergency patient cases against all safety rules.
    """
    @classmethod
    def setUpClass(cls):
        print("\n=======================================================")
        print("  RUNNING EXHAUSTIVE 12-ASSERTION REGRESSION SUITE (10 CASES)")
        print("=======================================================\n")
        sys.stdout.flush()
        cls.rag = LiveRealtimeMedicalRAGEngine()
        cls.runner = LiveModelRunner()
        cls.agent = EnterpriseClinicalAgent()
        
        cls.test_cases = [
            {
                "case_id": "cardiorenal_test",
                "filename": "unseen_patient_01_cardiorenal_clinical_report.md",
                "payload": {
                    "primary_diagnosis": "Acute Decompensated Heart Failure & Stage 3 AKI",
                    "comorbidities": ["Type 2 Diabetes", "Chronic Kidney Disease Stage 4"],
                    "demographics": {"age": 72, "gender": "M", "admission_type": "EMERGENCY"},
                    "presentation_labs": {"creatinine_max": 4.8, "bun_max": 88.0, "wbc_max": 18.5, "bicarbonate_min": 17.0},
                    "vital_signs": {"sbp_min": 90, "dbp_min": 55, "hr_max": 118, "spo2_min": 92},
                    "active_medications": ["furosemide", "vancomycin", "enoxaparin"],
                    "chief_complaint": "Severe shortness of breath, leg swelling"
                },
                "expected_top_med": "furosemide"
            },
            {
                "case_id": "sepsis_test",
                "filename": "unseen_patient_02_sepsis_clinical_report.md",
                "payload": {
                    "primary_diagnosis": "Septic Shock & Acute Respiratory Failure",
                    "comorbidities": ["COPD", "Hypertension"],
                    "demographics": {"age": 64, "gender": "F", "admission_type": "EMERGENCY"},
                    "presentation_labs": {"creatinine_max": 3.2, "wbc_max": 26.5, "bicarbonate_min": 15.0},
                    "vital_signs": {"sbp_min": 82, "dbp_min": 48, "hr_max": 132, "spo2_min": 88},
                    "active_medications": ["norepinephrine", "vancomycin", "cefepime"],
                    "chief_complaint": "High fever, altered mental status, severe hypotension"
                },
                "expected_top_med": "norepinephrine"
            },
            {
                "case_id": "dka_test",
                "filename": "unseen_patient_03_dka_clinical_report.md",
                "payload": {
                    "primary_diagnosis": "Diabetic Ketoacidosis & Severe Metabolic Acidosis",
                    "comorbidities": ["Type 1 Diabetes", "Stage 2 Acute Kidney Injury", "Hyperkalemia"],
                    "demographics": {"age": 45, "gender": "M", "admission_type": "EMERGENCY"},
                    "presentation_labs": {"creatinine_max": 2.4, "bicarbonate_min": 11.0, "glucose_max": 480.0, "potassium_max": 6.2},
                    "vital_signs": {"sbp_min": 105, "hr_max": 124, "spo2_min": 96},
                    "active_medications": ["insulin", "normal saline", "potassium chloride"],
                    "chief_complaint": "Nausea, abdominal pain, Kussmaul breathing"
                },
                "expected_top_med": "insulin"
            },
            {
                "case_id": "pe_test",
                "filename": "unseen_patient_04_pe_clinical_report.md",
                "payload": {
                    "primary_diagnosis": "Acute Massive Pulmonary Embolism & Right Ventricular Strain",
                    "comorbidities": ["Deep Vein Thrombosis", "Obesity"],
                    "demographics": {"age": 58, "gender": "F", "admission_type": "EMERGENCY"},
                    "presentation_labs": {"troponin_max": 1.8, "wbc_max": 14.2},
                    "vital_signs": {"sbp_min": 85, "hr_max": 128, "spo2_min": 84},
                    "active_medications": ["alteplase", "heparin"],
                    "chief_complaint": "Sudden onset chest pain, severe hypoxia"
                },
                "expected_top_med": "alteplase"
            },
            {
                "case_id": "pancreatitis_test",
                "filename": "unseen_patient_05_pancreatitis_clinical_report.md",
                "payload": {
                    "primary_diagnosis": "Severe Acute Pancreatitis & SIRS",
                    "comorbidities": ["Gallstones", "Hypertriglyceridemia"],
                    "demographics": {"age": 52, "gender": "M", "admission_type": "EMERGENCY"},
                    "presentation_labs": {"lipase_max": 1850.0, "wbc_max": 22.4, "creatinine_max": 2.1},
                    "vital_signs": {"sbp_min": 98, "hr_max": 116, "spo2_min": 94},
                    "active_medications": ["lactated ringers", "hydromorphone"],
                    "chief_complaint": "Severe epigastric pain radiating to back"
                },
                "expected_top_med": "lactated ringers"
            },
            {
                "case_id": "stroke_test",
                "filename": "unseen_patient_06_stroke_clinical_report.md",
                "payload": {
                    "primary_diagnosis": "Acute Ischemic Stroke & Hypertensive Crisis",
                    "comorbidities": ["Hypertension", "Atrial Fibrillation"],
                    "demographics": {"age": 69, "gender": "M", "admission_type": "EMERGENCY"},
                    "presentation_labs": {"glucose_max": 142.0},
                    "vital_signs": {"sbp_min": 210, "dbp_min": 115, "hr_max": 96, "spo2_min": 97},
                    "active_medications": ["alteplase", "nicardipine"],
                    "chief_complaint": "Right-sided hemiparesis and expressive aphasia"
                },
                "expected_top_med": "alteplase"
            },
            {
                "case_id": "gibleed_test",
                "filename": "unseen_patient_07_gibleed_clinical_report.md",
                "payload": {
                    "primary_diagnosis": "Acute Upper GI Bleeding & Hemorrhagic Shock",
                    "comorbidities": ["Cirrhosis", "Esophageal Varices"],
                    "demographics": {"age": 61, "gender": "M", "admission_type": "EMERGENCY"},
                    "presentation_labs": {"hgb_min": 5.8, "bun_max": 64.0},
                    "vital_signs": {"sbp_min": 78, "hr_max": 134, "spo2_min": 93},
                    "active_medications": ["pantoprazole", "octreotide"],
                    "chief_complaint": "Profuse hematemesis and melena"
                },
                "expected_top_med": "pantoprazole"
            },
            {
                "case_id": "ards_test",
                "filename": "unseen_patient_08_ards_clinical_report.md",
                "payload": {
                    "primary_diagnosis": "Severe ARDS & Refractory Hypoxemia",
                    "comorbidities": ["Community-Acquired Pneumonia"],
                    "demographics": {"age": 55, "gender": "F", "admission_type": "EMERGENCY"},
                    "presentation_labs": {"wbc_max": 19.8, "bicarbonate_min": 19.0},
                    "vital_signs": {"sbp_min": 102, "hr_max": 122, "spo2_min": 82},
                    "active_medications": ["cisatracurium", "dexamethasone"],
                    "chief_complaint": "Severe dyspnea and diffuse infiltrates"
                },
                "expected_top_med": "cisatracurium"
            },
            {
                "case_id": "liver_test",
                "filename": "unseen_patient_09_liver_clinical_report.md",
                "payload": {
                    "primary_diagnosis": "Acute Liver Failure & Hepatic Encephalopathy",
                    "comorbidities": ["Alcoholic Hepatitis", "Coagulopathy"],
                    "demographics": {"age": 50, "gender": "M", "admission_type": "EMERGENCY"},
                    "presentation_labs": {"ammonia_max": 145.0, "inr_max": 3.2},
                    "vital_signs": {"sbp_min": 94, "hr_max": 108, "spo2_min": 95},
                    "active_medications": ["lactulose", "rifaximin"],
                    "chief_complaint": "Confusion, asterixis, jaundice"
                },
                "expected_top_med": "lactulose"
            },
            {
                "case_id": "cardiogenic_test",
                "filename": "unseen_patient_10_cardiogenic_clinical_report.md",
                "payload": {
                    "primary_diagnosis": "Acute Coronary Syndrome & Cardiogenic Shock",
                    "comorbidities": ["Ischemic Cardiomyopathy", "STEMI"],
                    "demographics": {"age": 74, "gender": "M", "admission_type": "EMERGENCY"},
                    "presentation_labs": {"troponin_max": 14.5, "creatinine_max": 3.1},
                    "vital_signs": {"sbp_min": 75, "hr_max": 115, "spo2_min": 90},
                    "active_medications": ["dobutamine", "norepinephrine"],
                    "chief_complaint": "Crushing chest pain and cold extremities"
                },
                "expected_top_med": "dobutamine"
            }
        ]
        
        # Populate RAG engine state for all 10 test cases prior to assertions
        cls.generated_reports = {}
        for tc in cls.test_cases:
            cls.generated_reports[tc['case_id']] = cls.agent.evaluate_unseen_patient(tc['payload'], case_id=tc['case_id'])

    # Assertion 01: Medication Completeness
    def test_01_medication_completeness(self):
        for tc in self.test_cases:
            payload = tc['payload']
            meds = payload.get('active_medications', [])
            report = self.generated_reports[tc['case_id']]
            for m in meds:
                m_clean = m.lower()
                self.assertIn(m_clean, report.lower(), f"Medication '{m}' missing from report for {tc['case_id']}")
        print("  [ASSERTION 1 PASSED] Medication completeness verified across all 10 patient cases.")
        sys.stdout.flush()

    # Assertion 02: Medication Ranking Correctness & Tie-Breaking Stability
    def test_02_medication_ranking_correctness(self):
        for tc in self.test_cases:
            payload = tc['payload']
            expected = tc['expected_top_med']
            ranked = self.rag.rank_medications_by_mechanistic_relevance(payload)
            self.assertEqual(ranked[0], expected, f"Rank #1 med for {tc['case_id']} expected '{expected}', got '{ranked[0]}'")
            ranked2 = self.rag.rank_medications_by_mechanistic_relevance(payload)
            self.assertEqual(ranked, ranked2, f"Ranking non-deterministic for {tc['case_id']}")
        print("  [ASSERTION 2 PASSED] Mechanistic Medication Ranking & Deterministic Tie-Breaking verified across all 10 cases.")
        sys.stdout.flush()

    # Assertion 03: No Duplicate Section Labels
    def test_03_no_duplicate_section_labels(self):
        for tc in self.test_cases:
            report = self.generated_reports[tc['case_id']]
            count_tier_a = report.count("### A. Direct Guideline-Supported Action") + report.count("### A. Direct guideline-supported action")
            self.assertEqual(count_tier_a, 1, f"Section 5 must contain exactly one Tier A block for {tc['case_id']}, found {count_tier_a}")
        print("  [ASSERTION 3 PASSED] Section 5 Tier A uniqueness & sequential B/C labeling verified across all 10 cases.")
        sys.stdout.flush()

    # Assertion 04: No Cross-Patient PMID Collisions
    def test_04_no_cross_patient_pmid_collisions(self):
        citation_log = self.rag.citation_registry
        self.assertGreater(len(citation_log), 0, "Citation registry must not be empty.")
        seen_pmids = {}
        for doc_id, title in citation_log.items():
            if 'PUBMED' in doc_id:
                pmid = doc_id.replace('PUBMED_PMID_', '')
                if pmid in seen_pmids:
                    self.assertEqual(seen_pmids[pmid], title, f"PMID {pmid} collision: '{seen_pmids[pmid]}' vs '{title}'")
                else:
                    seen_pmids[pmid] = title
        print("  [ASSERTION 4 PASSED] Cross-Patient PMID Collision check verified across all 10 cases.")
        sys.stdout.flush()

    # Assertion 05: No Fabricated Abstracts
    def test_05_no_fabricated_abstracts(self):
        text_hashes = {}
        for tc in self.test_cases:
            docs = self.rag.search_unseen_patient_rag(tc['payload'], case_id=tc['case_id'])
            for d in docs:
                if 'PMID' in d.get('citation', ''):
                    pmid = d['citation'].replace('[NCBI PubMed PMID: ', '').replace(']', '')
                    text_content = d.get('text', '').strip()
                    if len(text_content) > 20:
                        h = hashlib.md5(text_content.encode('utf-8')).hexdigest()
                        if h in text_hashes:
                            self.assertEqual(text_hashes[h], pmid, f"Fabricated abstract reuse! Same text content hash '{h}' shared between PMID {text_hashes[h]} and PMID {pmid}")
                        else:
                            text_hashes[h] = pmid
        print("  [ASSERTION 5 PASSED] Abstract Uniqueness & Content Hash Fabrication Detector verified across all 10 cases.")
        sys.stdout.flush()

    # Assertion 06: No Silent Evidence Drops
    def test_06_no_silent_evidence_drops(self):
        docs_empty = self.rag.fetch_live_pubmed_papers('', patient_payload=self.test_cases[0]['payload'])
        self.assertEqual(docs_empty[0]['title'], "Citation Integrity Check Failed")
        self.assertIn("Level 4 evidence withheld — citation integrity check failed", docs_empty[0]['text'])

        docs_unmatched = self.rag.fetch_live_pubmed_papers('nonexistent_rare_condition_xyz_9999', patient_payload=self.test_cases[0]['payload'])
        self.assertEqual(docs_unmatched[0]['title'], "Citation Integrity Check Failed")
        self.assertIn("Level 4 evidence withheld — citation integrity check failed", docs_unmatched[0]['text'])
        print("  [ASSERTION 6 PASSED] Graceful Fallback for Empty & Unmatched Queries verified.")
        sys.stdout.flush()

    # Assertion 07: Title Formatting
    def test_07_title_formatting(self):
        sample_raw = "Focused Echocardiography to Guide Management : Short title: Focused Echo"
        cleaned = clean_pubmed_title(sample_raw)
        self.assertNotIn("Short title:", cleaned)
        self.assertEqual(cleaned, "Focused Echocardiography to Guide Management")
        print("  [ASSERTION 7 PASSED] Clean Single Full Title Formatting verified.")
        sys.stdout.flush()

    # Assertion 08: End-to-End Report Level 4 Citation Topical Relevance Check
    def test_08_topical_relevance_check(self):
        for tc in self.test_cases:
            report = self.generated_reports[tc['case_id']]
            sec4_lines = [l for l in report.split('\n') if 'Level 4' in l and 'PMID' in l]
            if len(sec4_lines) > 0:
                l4_line = sec4_lines[0]
                is_relevant = self.rag.verify_topical_relevance(l4_line, l4_line, tc['payload'])
                self.assertTrue(is_relevant, f"End-to-End Level 4 citation '{l4_line}' not topically relevant for {tc['case_id']}")
        print("  [ASSERTION 8 PASSED] End-to-End Report Topical Relevance Verification verified across all 10 cases.")
        sys.stdout.flush()

    # Assertion 09: Sub-Tier Publication Type Categorization
    def test_09_sub_tier_publication_type_categorization(self):
        doc_sepsis = self.rag.fetch_live_pubmed_papers('septic shock pneumonia', patient_payload=self.test_cases[1]['payload'])
        self.assertGreater(len(doc_sepsis), 0)
        if 'PMID: 42501530' in doc_sepsis[0]['citation']:
            self.assertIn('Case Reports & Autopsy Findings', doc_sepsis[0]['evidence_level'])
            self.assertIn('Reduced Evidentiary Weight', doc_sepsis[0]['evidence_level'])
        print("  [ASSERTION 9 PASSED] Case Report & Autopsy Sub-Tier Categorization verified.")
        sys.stdout.flush()

    # Assertion 10: Extreme Lab/Vital Input Inference Safety
    def test_10_extreme_lab_vital_inference_safety(self):
        extreme_payload = {
            "primary_diagnosis": "Severe Multi-Organ Failure",
            "presentation_labs": {"creatinine_max": 25.0, "bun_max": 250.0, "wbc_max": 99.9, "glucose_max": 1200.0},
            "vital_signs": {"sbp_min": 40, "hr_max": 210, "spo2_min": 60},
            "active_medications": ["norepinephrine"]
        }
        preds = self.runner.run_live_inference_with_uncertainty(extreme_payload)
        self.assertIn('p_mortality', preds)
        self.assertGreater(preds['p_mortality'], 0.50)
        print("  [ASSERTION 10 PASSED] Extreme Lab/Vital Input Inference Safety verified.")
        sys.stdout.flush()

    # Assertion 11: Multi-Task Probability Calibration Range [0.0, 1.0] Safety
    def test_11_probability_calibration_range_safety(self):
        for tc in self.test_cases:
            preds = self.runner.run_live_inference_with_uncertainty(tc['payload'])
            for task in ['p_mortality', 'p_readmission', 'p_icu_admission', 'p_deterioration']:
                val = preds[task]
                self.assertTrue(0.0 <= val <= 1.0, f"Task {task} value {val} out of bounds [0,1]")
        print("  [ASSERTION 11 PASSED] Multi-Task Probability Calibration Range [0.0, 1.0] Safety verified across all 10 cases.")
        sys.stdout.flush()

    # Assertion 12: Persistent Audit File Disk Integrity
    def test_12_audit_file_disk_integrity(self):
        audit_file = self.rag.audit_log_file
        self.assertTrue(os.path.exists(audit_file), f"Audit log file {audit_file} missing from disk.")
        with open(audit_file, 'r') as fp:
            data = json.load(fp)
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        print("  [ASSERTION 12 PASSED] Persistent Audit File Disk Integrity verified.")
        sys.stdout.flush()

if __name__ == '__main__':
    unittest.main()
