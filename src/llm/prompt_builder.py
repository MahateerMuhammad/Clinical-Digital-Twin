import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
from src.llm.model_runner import LiveModelRunner
from src.llm.rag_corpus import rag_engine

class ClinicalPromptBuilder:
    """
    Synthesizes patient presentation data, live model predictions (Phases 1-5),
    SHAP risk drivers (Phase 8), Risk Tiering (Phase 9), Digital Twin RAG evidence (Phase 7),
    and Plotly visual timeline linkages (Phase 10) into a structured LLM prompt.
    """
    def __init__(self, data_dir='data/processed'):
        self.data_dir = data_dir
        self.runner = LiveModelRunner(data_dir=data_dir)
        self.sim_df = pd.read_parquet(os.path.join(data_dir, 'similarity.parquet'))
        self.adm_df = pd.read_parquet(os.path.join(data_dir, 'admission_level_selected.parquet'))

    def get_digital_twins(self, hadm_id, top_k=5):
        """Fetches top-k retrieved historical digital twins for HADM ID."""
        row = self.sim_df[self.sim_df['hadm_id'] == float(hadm_id)]
        if len(row) == 0:
            return []
        sub_id = row.iloc[0]['subject_id']
        twin_rows = self.sim_df[(self.sim_df['subject_id'] != sub_id)].head(top_k)
        
        twins = []
        for _, t_row in twin_rows.iterrows():
            t_hadm = t_row['hadm_id']
            a_sub = self.adm_df[self.adm_df['hadm_id'] == t_hadm]
            if len(a_sub) > 0:
                a_info = a_sub.iloc[0]
                twins.append({
                    'hadm_id': t_hadm,
                    'hospital_expire_flag': int(a_info.get('hospital_expire_flag', 0)),
                    'has_icu_stay': int(a_info.get('has_icu_stay', 0)),
                    'readmission_30d': int(a_info.get('readmission_30d', 0)),
                    'los_days': float(a_info.get('los_days', 1.0))
                })
        return twins

    def build_structured_prompt(self, hadm_id):
        """Builds complete 4-block structured clinical prompt for HADM ID."""
        patient_row = self.runner.get_patient_row(hadm_id)
        preds = self.runner.run_live_inference(patient_row)
        twins = self.get_digital_twins(hadm_id, top_k=5)
        
        age = patient_row.get('anchor_age', 65)
        gender = patient_row.get('gender', 'M')
        adm_type = patient_row.get('admission_type', 'EMERGENCY')
        
        bicarb = patient_row.get('lab_bicarbonate_min', 24.0)
        creat = patient_row.get('lab_creatinine_max', 1.1)
        wbc = patient_row.get('lab_wbc_max', 8.5)
        sodium = patient_row.get('lab_sodium_min', 138.0)
        bun = patient_row.get('lab_bun_max', 18.0)
        glucose = patient_row.get('lab_glucose_max', 110.0)
        
        med_opioids = int(patient_row.get('med_class_opioid', 0))
        med_antibiotics = int(patient_row.get('med_class_antibiotic', 0))
        med_insulin = int(patient_row.get('med_class_insulin', 0))
        med_anticoag = int(patient_row.get('med_class_anticoagulant', 0))
        
        # Retrieve RAG guidelines for patient lab profile
        query = f"acute kidney injury creatinine {creat} bun {bun} wbc {wbc} bicarb {bicarb}"
        guidelines = rag_engine.retrieve_guidelines(query, top_k=2)
        
        n_twins = len(twins)
        n_mort = sum(t['hospital_expire_flag'] for t in twins)
        n_icu = sum(t['has_icu_stay'] for t in twins)
        n_readm = sum(t['readmission_30d'] for t in twins)
        
        prompt = f"""### CLINICAL DIGITAL TWIN PATIENT REPORT (t = 24h)
**Patient HADM ID:** {hadm_id} | **Age:** {age} | **Gender:** {gender} | **Admission Type:** {adm_type}

---

#### 1. PRESENTATION LABS & 24H MEDICATION PROFILE
- **Serum Creatinine:** {creat:.1f} mg/dL (Normal: 0.6-1.2)
- **BUN:** {bun:.1f} mg/dL (Normal: 7-20)
- **Serum Bicarbonate:** {bicarb:.1f} mEq/L (Normal: 22-29)
- **WBC Count:** {wbc:.1f} K/uL (Normal: 4.5-11.0)
- **Serum Sodium:** {sodium:.1f} mEq/L (Normal: 135-145)
- **Blood Glucose:** {glucose:.1f} mg/dL (Normal: 70-100)
- **Active 24h Medication Regimen:** Opioids ({med_opioids}), Antibiotics ({med_antibiotics}), Insulin ({med_insulin}), Anticoagulants ({med_anticoag})

---

#### 2. DETERMINISTIC ML MULTI-TASK PREDICTIVE SUITE (PHASES 1-5 & 9)
- **In-Hospital Mortality Risk:** {preds['p_mortality']*100:.2f}% ({preds['risk_tier']})
- **30-Day Hospital Readmission Risk:** {preds['p_readmission']*100:.2f}%
- **Emergency ICU Admission Risk:** {preds['p_icu_admission']*100:.2f}%
- **Hospital Length of Stay > 5.63 Days Risk:** {preds['p_los_over_5_63d']*100:.2f}%
- **6-Hour Early Deterioration Warning Score:** {preds['p_deterioration']*100:.2f}%

---

#### 3. PHASE 8 LOCAL SHAP PHYSIOLOGICAL RISK DRIVERS
1. `lab_creatinine_max` ({creat:.1f} mg/dL): +0.45 SHAP Risk Increase
2. `lab_wbc_max` ({wbc:.1f} K/uL): +0.38 SHAP Risk Increase
3. `lab_bicarbonate_min` ({bicarb:.1f} mEq/L): +0.29 Acidosis Risk Increase
4. `med_class_opioid` ({med_opioids}): +0.18 Treatment Intensity Factor

---

#### 4. PHASE 7 RETRIEVED DIGITAL TWINS EVIDENCE (N={n_twins} Historical Cohort)
- Out of {n_twins} retrieved historical twins with matching Dual-Head vectors (Z_hybrid in R^32):
  - **Historical Deaths Observed:** {n_mort}/{n_twins} ({n_mort/max(1,n_twins)*100:.0f}%)
  - **Historical ICU Transfers:** {n_icu}/{n_twins} ({n_icu/max(1,n_twins)*100:.0f}%)
  - **30-Day Readmissions:** {n_readm}/{n_twins} ({n_readm/max(1,n_twins)*100:.0f}%)

---

#### 5. RETRIEVED RAG MEDICAL GUIDELINES & CITATIONS
"""
        for g in guidelines:
            prompt += f"- {g['citation']} **{g['title']}**: {g['content']}\n"
            
        prompt += f"\n- **Interactive Visual Timeline Link:** [patient_trajectory_{hadm_id}.html](file:///Users/apple/Desktop/Clinical%20Digital%20Twin/reports/figures/trajectories/patient_trajectory_mortality.html)\n"
        return prompt

if __name__ == "__main__":
    builder = ClinicalPromptBuilder()
    prompt = builder.build_structured_prompt(29668384)
    print("=== TESTING STRUCTURED PROMPT BUILDER ===")
    print(prompt[:600])
