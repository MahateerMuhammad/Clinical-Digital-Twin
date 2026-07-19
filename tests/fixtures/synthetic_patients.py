"""
tests/fixtures/synthetic_patients.py
────────────────────────────────────
Generates a small, hand-crafted synthetic dataset (25 patients) covering all MIMIC-IV edge cases.
"""

import pandas as pd
import numpy as np

def create_synthetic_tables():
    # 1. Patients
    patients = pd.DataFrame([
        {"subject_id": 1001, "gender": "M", "anchor_age": 65, "anchor_year": 2150, "dod": None},
        {"subject_id": 1002, "gender": "F", "anchor_age": 72, "anchor_year": 2180, "dod": "2182-05-10"},
        {"subject_id": 1003, "gender": "M", "anchor_age": 50, "anchor_year": 2140, "dod": None},
        {"subject_id": 1004, "gender": "F", "anchor_age": 80, "anchor_year": 2130, "dod": None},
        {"subject_id": 1005, "gender": "M", "anchor_age": 30, "anchor_year": 2190, "dod": None},
    ])
    
    # Add subjects 1006..1025 to make ~25 patients total
    extra_pts = [
        {"subject_id": sid, "gender": "F" if sid % 2 == 0 else "M", "anchor_age": 40 + sid % 40, "anchor_year": 2150, "dod": None}
        for sid in range(1006, 1026)
    ]
    patients = pd.concat([patients, pd.DataFrame(extra_pts)], ignore_index=True)

    # 2. Admissions
    # subject 1001: 1 admission (hadm 2001) - 0 ICU stays
    # subject 1002: 5 admissions (hadm 2002..2006) - hadm 2002 has 1 stay, hadm 2003 has 3 ICU stays!
    # subject 1003: hadm 2007 - invalid time order (dischtime < admittime)
    # subject 1004: hadm 2008 - patient with zero matching lab/vital itemids
    # subject 1005: hadm 2009 - patient with multiple lab variants (serum creatinine 50912 + whole-blood creatinine 50806)
    admissions_data = [
        {"subject_id": 1001, "hadm_id": 2001, "admittime": "2150-01-01 10:00:00", "dischtime": "2150-01-05 12:00:00", "deathtime": None, "admission_type": "EMERGENCY", "insurance": "Medicare", "race": "WHITE"},
        {"subject_id": 1002, "hadm_id": 2002, "admittime": "2180-03-01 08:00:00", "dischtime": "2180-03-10 16:00:00", "deathtime": None, "admission_type": "URGENT", "insurance": "Other", "race": "BLACK"},
        {"subject_id": 1002, "hadm_id": 2003, "admittime": "2181-01-10 14:00:00", "dischtime": "2181-01-25 10:00:00", "deathtime": None, "admission_type": "EMERGENCY", "insurance": "Other", "race": "BLACK"},
        {"subject_id": 1002, "hadm_id": 2004, "admittime": "2181-06-01 10:00:00", "dischtime": "2181-06-05 11:00:00", "deathtime": None, "admission_type": "ELECTIVE", "insurance": "Other", "race": "BLACK"},
        {"subject_id": 1002, "hadm_id": 2005, "admittime": "2182-01-01 12:00:00", "dischtime": "2182-01-08 14:00:00", "deathtime": None, "admission_type": "EMERGENCY", "insurance": "Other", "race": "BLACK"},
        {"subject_id": 1002, "hadm_id": 2006, "admittime": "2182-05-01 09:00:00", "dischtime": "2182-05-10 10:00:00", "deathtime": "2182-05-10 10:00:00", "admission_type": "EMERGENCY", "insurance": "Other", "race": "BLACK"},
        {"subject_id": 1003, "hadm_id": 2007, "admittime": "2140-05-10 12:00:00", "dischtime": "2140-05-01 12:00:00", "deathtime": None, "admission_type": "EMERGENCY", "insurance": "Medicare", "race": "HISPANIC"}, # INVALID TIME ORDER
        {"subject_id": 1004, "hadm_id": 2008, "admittime": "2130-08-01 10:00:00", "dischtime": "2130-08-04 12:00:00", "deathtime": None, "admission_type": "ELECTIVE", "insurance": "Medicare", "race": "WHITE"},
        {"subject_id": 1005, "hadm_id": 2009, "admittime": "2190-11-01 10:00:00", "dischtime": "2190-11-10 12:00:00", "deathtime": None, "admission_type": "EMERGENCY", "insurance": "Private", "race": "ASIAN"},
    ]
    for sid in range(1006, 1026):
        admissions_data.append({
            "subject_id": sid, "hadm_id": 2000 + sid,
            "admittime": "2150-01-01 10:00:00", "dischtime": "2150-01-05 10:00:00",
            "deathtime": None, "admission_type": "ELECTIVE", "insurance": "Other", "race": "UNKNOWN"
        })
    admissions = pd.DataFrame(admissions_data)

    # 3. ICU Stays
    # hadm 2001: 0 stays
    # hadm 2002: 1 stay (stay 3001, duration 2.0 days)
    # hadm 2003: 3 ICU stays! (stay 3002 duration 1.5, stay 3003 duration 3.0, stay 3004 duration 2.5) -> total los = 7.0 days
    # hadm 2009: 1 stay (stay 3005)
    icustays = pd.DataFrame([
        {"subject_id": 1002, "hadm_id": 2002, "stay_id": 3001, "first_careunit": "MICU", "last_careunit": "MICU", "intime": "2180-03-01 10:00:00", "outtime": "2180-03-03 10:00:00", "los": 2.0},
        {"subject_id": 1002, "hadm_id": 2003, "stay_id": 3002, "first_careunit": "SICU", "last_careunit": "SICU", "intime": "2181-01-10 16:00:00", "outtime": "2181-01-12 04:00:00", "los": 1.5},
        {"subject_id": 1002, "hadm_id": 2003, "stay_id": 3003, "first_careunit": "MICU", "last_careunit": "MICU", "intime": "2181-01-15 10:00:00", "outtime": "2181-01-18 10:00:00", "los": 3.0},
        {"subject_id": 1002, "hadm_id": 2003, "stay_id": 3004, "first_careunit": "CVICU", "last_careunit": "CVICU", "intime": "2181-01-20 12:00:00", "outtime": "2181-01-23 00:00:00", "los": 2.5},
        {"subject_id": 1005, "hadm_id": 2009, "stay_id": 3005, "first_careunit": "MICU", "last_careunit": "MICU", "intime": "2190-11-01 12:00:00", "outtime": "2190-11-05 12:00:00", "los": 4.0},
    ])

    # 4. Lab events
    # Key lab itemids: 50912 (creatinine), 50931 (glucose), 50806 (creatinine_wb), 99999 (non-matching itemid)
    labevents = pd.DataFrame([
        # hadm 2001: 0 lab draws for key labs (only non-matching lab itemid 99999)
        {"subject_id": 1001, "hadm_id": 2001, "itemid": 99999, "charttime": "2150-01-01 11:00:00", "valuenum": 5.0, "flag": None},
        
        # hadm 2002: exactly 1 draw for creatinine 50912
        {"subject_id": 1002, "hadm_id": 2002, "itemid": 50912, "charttime": "2180-03-01 10:00:00", "valuenum": 1.2, "flag": "normal"},
        
        # hadm 2003: increasing trend for glucose 50931 (100 -> 150 -> 200), duplicate timestamp for creatinine 50912
        {"subject_id": 1002, "hadm_id": 2003, "itemid": 50931, "charttime": "2181-01-11 10:00:00", "valuenum": 100.0, "flag": "normal"},
        {"subject_id": 1002, "hadm_id": 2003, "itemid": 50931, "charttime": "2181-01-12 10:00:00", "valuenum": 150.0, "flag": "high"},
        {"subject_id": 1002, "hadm_id": 2003, "itemid": 50931, "charttime": "2181-01-13 10:00:00", "valuenum": 200.0, "flag": "high"},
        {"subject_id": 1002, "hadm_id": 2003, "itemid": 50912, "charttime": "2181-01-11 10:00:00", "valuenum": 1.4, "flag": "normal"},
        {"subject_id": 1002, "hadm_id": 2003, "itemid": 50912, "charttime": "2181-01-11 10:00:00", "valuenum": 1.5, "flag": "normal"}, # DUPLICATE TIMESTAMP
        
        # hadm 2008: patient sampled out of itemid filter entirely (itemid 88888)
        {"subject_id": 1004, "hadm_id": 2008, "itemid": 88888, "charttime": "2130-08-01 11:00:00", "valuenum": 42.0, "flag": None},
        
        # hadm 2009: multiple variants close together - serum creatinine 50912 + whole-blood creatinine 52546
        {"subject_id": 1005, "hadm_id": 2009, "itemid": 50912, "charttime": "2190-11-01 10:00:00", "valuenum": 1.8, "flag": "high"},
        {"subject_id": 1005, "hadm_id": 2009, "itemid": 52546, "charttime": "2190-11-01 10:30:00", "valuenum": 1.6, "flag": "high"},
    ])

    # 5. Chartevents (vitals)
    # stay 3001: 1 vital reading (heart rate 220045)
    # stay 3002: zero vital readings
    # stay 3003: 3 vital readings with increasing trend (heart rate 220045: 70 -> 85 -> 100) + duplicate timestamp
    # stay 3005: spo2 220277 (98 -> 95 -> 92)
    chartevents = pd.DataFrame([
        {"subject_id": 1002, "hadm_id": 2002, "stay_id": 3001, "itemid": 220045, "charttime": "2180-03-01 11:00:00", "valuenum": 75.0, "valueuom": "bpm"},
        {"subject_id": 1002, "hadm_id": 2003, "stay_id": 3003, "itemid": 220045, "charttime": "2181-01-15 11:00:00", "valuenum": 70.0, "valueuom": "bpm"},
        {"subject_id": 1002, "hadm_id": 2003, "stay_id": 3003, "itemid": 220045, "charttime": "2181-01-16 11:00:00", "valuenum": 85.0, "valueuom": "bpm"},
        {"subject_id": 1002, "hadm_id": 2003, "stay_id": 3003, "itemid": 220045, "charttime": "2181-01-16 11:00:00", "valuenum": 88.0, "valueuom": "bpm"}, # DUPLICATE TIMESTAMP
        {"subject_id": 1002, "hadm_id": 2003, "stay_id": 3003, "itemid": 220045, "charttime": "2181-01-17 11:00:00", "valuenum": 100.0, "valueuom": "bpm"},
        {"subject_id": 1005, "hadm_id": 2009, "stay_id": 3005, "itemid": 220277, "charttime": "2190-11-01 13:00:00", "valuenum": 98.0, "valueuom": "%"},
        {"subject_id": 1005, "hadm_id": 2009, "stay_id": 3005, "itemid": 220277, "charttime": "2190-11-02 13:00:00", "valuenum": 95.0, "valueuom": "%"},
        {"subject_id": 1005, "hadm_id": 2009, "stay_id": 3005, "itemid": 220277, "charttime": "2190-11-03 13:00:00", "valuenum": 92.0, "valueuom": "%"},
    ])

    # 6. Prescriptions (invalid time order: stoptime < starttime)
    prescriptions = pd.DataFrame([
        {"subject_id": 1001, "hadm_id": 2001, "pharmacy_id": 501, "drug": "Aspirin", "starttime": "2150-01-01 10:00:00", "stoptime": "2150-01-05 10:00:00", "dose_val_rx": "81", "dose_unit_rx": "mg"},
        {"subject_id": 1002, "hadm_id": 2002, "pharmacy_id": 502, "drug": "Heparin", "starttime": "2180-03-05 10:00:00", "stoptime": "2180-03-01 10:00:00", "dose_val_rx": "5000", "dose_unit_rx": "units"}, # INVALID TIME ORDER
    ])

    # 7. Radiology Detail (multiple field_ordinal values under same note_id)
    radiology_detail = pd.DataFrame([
        {"note_id": "10001-RR-1", "subject_id": 1001, "field_name": "exam_code", "field_value": "W82", "field_ordinal": 1},
        {"note_id": "10001-RR-1", "subject_id": 1001, "field_name": "exam_code", "field_value": "W82", "field_ordinal": 2}, # Multiple field_ordinal!
        {"note_id": "10001-RR-1", "subject_id": 1001, "field_name": "exam_code", "field_value": "W82", "field_ordinal": 3},
        {"note_id": "10002-RR-1", "subject_id": 1002, "field_name": "exam_name", "field_value": "CHEST PA & LAT", "field_ordinal": 1},
        {"note_id": "10002-RR-1", "subject_id": 1002, "field_name": "exam_name", "field_value": "CHEST PA & LAT", "field_ordinal": 1}, # EXACT DUPLICATE
    ])

    # 8. Diagnoses ICD & Procedures ICD
    diagnoses_icd = pd.DataFrame([
        {"subject_id": 1001, "hadm_id": 2001, "seq_num": 1, "icd_code": "I10", "icd_version": 10},
        {"subject_id": 1002, "hadm_id": 2002, "seq_num": 1, "icd_code": "E119", "icd_version": 10},
        {"subject_id": 1002, "hadm_id": 2003, "seq_num": 1, "icd_code": "I509", "icd_version": 10},
        {"subject_id": 1005, "hadm_id": 2009, "seq_num": 1, "icd_code": "N179", "icd_version": 10},
    ])
    procedures_icd = pd.DataFrame([
        {"subject_id": 1002, "hadm_id": 2003, "seq_num": 1, "icd_code": "5A1935Z", "icd_version": 10},
    ])

    # 9. Inputevents & Outputevents (fluids for stay 3002, 3003, 3004)
    inputevents = pd.DataFrame([
        {"subject_id": 1002, "hadm_id": 2003, "stay_id": 3002, "starttime": "2181-01-10 18:00:00", "amount": 500.0, "amountuom": "mL"},
        {"subject_id": 1002, "hadm_id": 2003, "stay_id": 3003, "starttime": "2181-01-15 12:00:00", "amount": 1000.0, "amountuom": "mL"},
        {"subject_id": 1002, "hadm_id": 2003, "stay_id": 3004, "starttime": "2181-01-20 14:00:00", "amount": 750.0, "amountuom": "mL"},
    ])
    outputevents = pd.DataFrame([
        {"subject_id": 1002, "hadm_id": 2003, "stay_id": 3002, "charttime": "2181-01-11 06:00:00", "value": 400.0, "valueuom": "mL"},
        {"subject_id": 1002, "hadm_id": 2003, "stay_id": 3003, "charttime": "2181-01-16 06:00:00", "value": 800.0, "valueuom": "mL"},
        {"subject_id": 1002, "hadm_id": 2003, "stay_id": 3004, "charttime": "2181-01-21 06:00:00", "value": 600.0, "valueuom": "mL"},
    ])

    # 10. Discharge notes
    discharge = pd.DataFrame([
        {"subject_id": 1001, "hadm_id": 2001, "note_id": "2001-DS", "charttime": "2150-01-05 12:00:00", "text": "Patient discharged in stable condition."},
        {"subject_id": 1002, "hadm_id": 2003, "note_id": "2003-DS", "charttime": "2181-01-25 10:00:00", "text": "Complex ICU course with multiple transfers."},
    ])

    return {
        "patients": patients,
        "admissions": admissions,
        "icustays": icustays,
        "labevents": labevents,
        "chartevents": chartevents,
        "prescriptions": prescriptions,
        "radiology_detail": radiology_detail,
        "diagnoses_icd": diagnoses_icd,
        "procedures_icd": procedures_icd,
        "inputevents": inputevents,
        "outputevents": outputevents,
        "discharge": discharge,
    }
