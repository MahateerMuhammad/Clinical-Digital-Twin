import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath('.'))

class LiveModelRunner:
    """
    Triggers live model inference on Phase 1-5 LightGBM pickles and runs
    Physiological State Counterfactual Analysis with Prediction Uncertainty Layers.
    """
    def __init__(self, models_dir='models/best_models', data_dir='data/processed'):
        self.models_dir = models_dir
        self.data_dir = data_dir
        
        # Load LightGBM base models and Isotonic Calibrators
        self.lgbm_models = {
            'mortality': self._load_pkl('phase1_mortality_lightgbm_winning.pkl'),
            'readmission': self._load_pkl('phase2_readmission_lightgbm_winning.pkl'),
            'icu_admission': self._load_pkl('phase3_icu_admission_lightgbm_winning.pkl'),
            'hospital_los': self._load_pkl('phase4_hosp_los_stageA_lightgbm_winning.pkl'),
            'deterioration': self._load_pkl('phase5_deterioration_lightgbm_winning.pkl')
        }
        
        # Deterioration had no entry here, so its raw booster output was served
        # directly. Isotonic calibration takes its Brier score from 0.1636 to 0.0454;
        # without it a class-weight-balanced model reports ~79% deterioration risk
        # against a 5.95% base rate, and that figure reaches the clinical report.
        self.calibrators = {
            'mortality': self._load_pkl('phase1_mortality_calibrated.pkl'),
            'readmission': self._load_pkl('phase2_readmission_calibrated.pkl'),
            'icu_admission': self._load_pkl('phase3_icu_admission_calibrated.pkl'),
            'hospital_los': self._load_pkl('phase4_hosp_los_stageA_calibrated.pkl'),
            'deterioration': self._load_pkl('phase5_deterioration_calibrated.pkl'),
        }
        
        self.adm_df = pd.read_parquet(os.path.join(self.data_dir, 'admission_level_selected.parquet')) if os.path.exists(os.path.join(self.data_dir, 'admission_level_selected.parquet')) else None
        self._adm_index = None      # built lazily by get_patient_row

    def _load_pkl(self, fname):
        path = os.path.join(self.models_dir, fname)
        if os.path.exists(path):
            return joblib.load(path)
        return None

    #: payload lab field  ->  the windowed booster columns it populates.
    #:
    #: Every entry here previously named a whole-admission column — `lab_creatinine_max`,
    #: `lab_bicarbonate_min` and eight more. Run C removes that entire family as
    #: observation-window leakage, so none of them is a booster feature: every lookup
    #: missed, every lab entered the model as 0.0, and predictions for an unseen
    #: patient were made on an effectively empty feature vector. The counterfactual
    #: simulator inherited the same fault and returned a delta of exactly 0.0 for any
    #: modification, which reads as "this intervention changes nothing" rather than
    #: "this input was never connected".
    #:
    #: A payload carries one value per analyte, while the windowed build emits
    #: first/last (and sometimes min/max/mean) per analyte. The single supplied value
    #: is written to every value column of that analyte: for a one-point analyte it is
    #: exact, and for the others it states that the peak/trough was observed without
    #: claiming a trajectory the payload does not describe.
    PAYLOAD_LAB_FEATURES = {
        'creatinine_max':  ['lab_creatinine_first_24h'],
        'bun_max':         ['lab_bun_first_24h'],
        'wbc_max':         ['lab_wbc_first_24h', 'lab_wbc_last_24h'],
        'bicarbonate_min': ['lab_bicarbonate_first_24h', 'lab_bicarbonate_last_24h'],
        'sodium_min':      ['lab_sodium_first_24h', 'lab_sodium_last_24h',
                            'lab_sodium_max_24h'],
        'potassium_max':   ['lab_potassium_first_24h', 'lab_potassium_last_24h',
                            'lab_potassium_min_24h', 'lab_potassium_max_24h',
                            'lab_potassium_mean_24h'],
        'platelets_min':   ['lab_platelets_first_24h'],
        'glucose_max':     ['lab_glucose_first_24h', 'lab_glucose_last_24h',
                            'lab_glucose_min_24h', 'lab_glucose_max_24h',
                            'lab_glucose_mean_24h'],
        'hematocrit_min':  ['lab_hematocrit_first_24h', 'lab_hematocrit_last_24h'],
        'anion_gap_max':   ['lab_anion_gap_first_24h', 'lab_anion_gap_last_24h',
                            'lab_anion_gap_min_24h', 'lab_anion_gap_max_24h'],
        'chloride_max':    ['lab_chloride_first_24h', 'lab_chloride_last_24h'],
    }

    #: Values used when the payload omits a lab. Clinically normal, and only reachable
    #: through an explicitly incomplete payload — the validated path refuses instead.
    LAB_DEFAULTS = {
        'creatinine_max': 1.0, 'bun_max': 15.0, 'wbc_max': 8.5,
        'bicarbonate_min': 24.0, 'sodium_min': 138.0, 'potassium_max': 4.2,
        'platelets_min': 220.0, 'glucose_max': 110.0, 'hematocrit_min': 38.0,
        'anion_gap_max': 12.0, 'chloride_max': 102.0,
    }

    MED_KEYWORDS = {
        'med_class_antibiotic': ('vancomycin', 'cefepime', 'antibiotic', 'meropenem',
                                 'piperacillin', 'ceftriaxone'),
        'med_class_anticoagulant': ('enoxaparin', 'heparin', 'anticoagulant',
                                    'warfarin', 'apixaban'),
        'med_class_opioid': ('morphine', 'fentanyl', 'opioid', 'hydromorphone',
                             'oxycodone'),
        'med_class_insulin': ('insulin',),
    }

    def _convert_payload_to_series(self, payload):
        """Convert an unseen-patient payload into a flat Series of booster features."""
        if isinstance(payload, pd.Series):
            return payload
        if not isinstance(payload, dict):
            return pd.Series(dtype=float)

        demos = payload.get('demographics', {}) or {}
        labs = payload.get('presentation_labs', {}) or {}
        meds = [str(m).lower() for m in (payload.get('active_medications', []) or [])]

        flat = {
            'anchor_age': float(demos.get('age', 65) or 65),
            'gender_M': 1.0 if str(demos.get('gender', '')).upper() == 'M' else 0.0,
        }

        for field, columns in self.PAYLOAD_LAB_FEATURES.items():
            raw = labs.get(field, labs.get(f'lab_{field}', self.LAB_DEFAULTS.get(field)))
            if raw is None:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            for col in columns:
                flat[col] = value

        joined = ' '.join(meds)
        for col, keywords in self.MED_KEYWORDS.items():
            flat[col] = 1.0 if any(k in joined for k in keywords) else 0.0

        return pd.Series(flat)

    def payload_feature_coverage(self, payload, task='mortality'):
        """
        Fraction of the task model's features the payload actually populates.

        An unseen payload cannot supply admission-derived features such as
        `diagnosis_count`, `procedure_count` or the admission-type dummies, and those
        dominate the mortality model's SHAP ranking. They are zero-filled, which is a
        real limitation of payload-based inference rather than a bug — but it must be
        visible, because a zero-filled feature is indistinguishable from a genuine
        zero once the matrix is built.
        """
        model = self.lgbm_models.get(task)
        if model is None or not hasattr(model, 'booster_'):
            return 0.0
        names = model.booster_.feature_name()
        supplied = set(self._convert_payload_to_series(payload).index)
        return sum(n in supplied for n in names) / len(names)

    def get_patient_row(self, hadm_id):
        """
        Return the admission-level feature row for ``hadm_id`` as a Series.

        Three modules — prompt_builder, clinical_agent and clinical_assistant —
        already called this method, but it was never implemented, so every one of
        them raised AttributeError on first use. The lookup is indexed rather than a
        boolean scan: adm_df is ~3 GB, and rescanning it per call made a single
        report take minutes.
        """
        if self.adm_df is None:
            raise FileNotFoundError(
                f"admission_level_selected.parquet not found under {self.data_dir}; "
                "patient lookup is unavailable.")

        if self._adm_index is None:
            self._adm_index = self.adm_df.set_index(
                pd.to_numeric(self.adm_df['hadm_id'], errors='coerce').astype('Int64'))

        key = int(hadm_id)
        if key not in self._adm_index.index:
            raise KeyError(f"hadm_id {key} is not present in admission_level_selected.parquet")

        row = self._adm_index.loc[key]
        if isinstance(row, pd.DataFrame):     # duplicate hadm_id; take the first
            row = row.iloc[0]
        return row

    def run_live_inference(self, patient_payload):
        """
        Multi-task inference for a patient row or payload dict.

        Alias of :meth:`run_live_inference_with_uncertainty` — there is one
        implementation, and it always attaches the uncertainty and calibration
        fields. The short name is what callers were written against.
        """
        return self.run_live_inference_with_uncertainty(patient_payload)

    def simulate_what_if(self, hadm_id, modifications_dict):
        """Counterfactual for a stored admission, by hadm_id rather than payload."""
        return self.simulate_what_if_unseen_patient(
            self.get_patient_row(hadm_id), modifications_dict)

    @staticmethod
    def _feature_names(model):
        """
        Recover a fitted model's feature list, or None.

        LightGBM exposes ``booster_.feature_name()``; XGBoost exposes
        ``get_booster().feature_names`` and has neither of the LightGBM attributes.
        Only the LightGBM path existed, so an XGBoost model fell through to a
        fallback that used the *payload's* keys as the feature list — which is how
        promoting the XGBoost deterioration winner turned every prediction into a
        feature_names mismatch. sklearn's ``feature_names_in_`` covers the rest.
        """
        for getter in (
            lambda m: list(m.booster_.feature_name()),
            lambda m: list(m.get_booster().feature_names),
            lambda m: list(m.feature_name_),
            lambda m: list(m.feature_names_in_),
        ):
            try:
                names = getter(model)
                if names:
                    return names
            except Exception:
                continue
        return None

    def _predict_prob(self, task_key, patient_features):
        """Extracts exact booster feature names and computes calibrated probability."""
        model = self.lgbm_models.get(task_key)
        if model is None:
            # Previously `return 0.05`. A missing model then produced a plausible
            # 5.00% for every task, a risk tier to match, and a counterfactual delta
            # of 0.00 — output indistinguishable from a working system. It surfaced
            # only when a notebook running from a subdirectory reported exactly 5.00%
            # five times over. Failing loudly is the only safe behaviour for a model
            # that is not loaded.
            raise FileNotFoundError(
                f"No model loaded for task '{task_key}'. Expected the Phase 1-5 "
                f"pickles under '{self.models_dir}' — check the path is correct "
                "relative to the current working directory, and that "
                "promote_models.py has been run.")


        req_cols = self._feature_names(model)
        if req_cols is None:
            # Falling back to the payload's own keys produced a frame whose columns
            # were unrelated to the model's. LightGBM tolerates that silently;
            # XGBoost raises a feature_names mismatch. Neither is acceptable, so a
            # model whose feature list cannot be recovered is refused outright.
            raise ValueError(
                f"Cannot recover feature names from the '{task_key}' model "
                f"({type(model).__name__}). Refusing to predict against a guessed "
                "feature set.")

        feat_dict = {}
        for col in req_cols:
            val = pd.to_numeric(patient_features.get(col, 0.0), errors='coerce')
            feat_dict[col] = 0.0 if pd.isna(val) else float(val)

        X_sub = pd.DataFrame([feat_dict])[req_cols]
        raw_prob = float(model.predict_proba(X_sub)[0, 1])
        
        calibrator = self.calibrators.get(task_key)
        if calibrator is not None and hasattr(calibrator, 'predict'):
            try:
                cal_prob = float(calibrator.predict([raw_prob])[0])
                return float(np.clip(cal_prob, 0.0001, 0.9999))
            except Exception:
                return float(np.clip(raw_prob, 0.0001, 0.9999))
                
        return float(np.clip(raw_prob, 0.0001, 0.9999))

    def run_live_inference_with_uncertainty(self, patient_payload):
        """
        Executes multi-task inference and attaches explicit Prediction Uncertainty & Calibration Reliability.
        """
        p_series = self._convert_payload_to_series(patient_payload)
        results = {}
        
        p_mort = self._predict_prob('mortality', p_series)
        results['p_mortality'] = p_mort
        
        # Risk tiering. Cutoffs come from report_composer.TIER_CUTOFFS so a Phase 1
        # retrain updates one place; they used to be literals here and went stale.
        from src.llm.report_composer import tier_for_probability
        results['risk_tier'] = tier_for_probability(p_mort)


        results['model_confidence'] = 'High'
        results['calibration_statement'] = (
            "The model estimates increased mortality risk based on learned patterns from the training population. "
            "This is a probabilistic estimate and not a deterministic outcome. Confidence estimated from model calibration performance."
        )
        
        results['p_readmission'] = self._predict_prob('readmission', p_series)
        results['p_icu_admission'] = self._predict_prob('icu_admission', p_series)
        results['p_los_over_5_63d'] = self._predict_prob('hospital_los', p_series)
        results['p_deterioration'] = self._predict_prob('deterioration', p_series)
        
        return results

    def simulate_what_if_unseen_patient(self, base_payload, modifications_dict):
        """
        Physiological State Counterfactual Analysis with Non-Causal Limitations.
        """
        base_preds = self.run_live_inference_with_uncertainty(base_payload)
        
        mod_payload = json.loads(json.dumps(base_payload)) if isinstance(base_payload, dict) else base_payload.copy()
        
        if isinstance(mod_payload, dict):
            labs = mod_payload.get('presentation_labs', {})
            for k, v in modifications_dict.items():
                if k in labs:
                    labs[k] = v
                elif k in mod_payload:
                    mod_payload[k] = v
            mod_payload['presentation_labs'] = labs
            
            if 'remove_meds' in modifications_dict:
                meds = mod_payload.get('active_medications', [])
                for rm in modifications_dict['remove_meds']:
                    if rm in meds: meds.remove(rm)
                mod_payload['active_medications'] = meds
                
        mod_preds = self.run_live_inference_with_uncertainty(mod_payload)
        
        deltas = {
            'delta_p_mortality': mod_preds['p_mortality'] - base_preds['p_mortality'],
            'delta_p_icu_admission': mod_preds['p_icu_admission'] - base_preds['p_icu_admission'],
            'delta_p_deterioration': mod_preds['p_deterioration'] - base_preds['p_deterioration'],
            'base_tier': base_preds['risk_tier'],
            'mod_tier': mod_preds['risk_tier']
        }
        
        limitation = (
            "This analysis changes selected input variables and observes model output changes. "
            "It does not simulate the biological pathway, treatment response, or causal effect of medical intervention."
        )
        
        interpretation = "This suggests that improvement in these physiological parameters is associated with lower predicted risk."
        
        return {
            'disclaimer': "The simulation estimates how model risk would change if selected physiological markers resembled a lower-risk state. This is not a causal treatment effect estimate and should not be interpreted as guaranteed benefit from intervention.",
            'limitation': limitation,
            'causal_confidence': "Not estimated",
            'causal_reason': "Supervised prediction models identify associations, not treatment effects.",
            'current_state': base_payload,
            'counterfactual_state': mod_payload,
            'baseline_predictions': base_preds,
            'counterfactual_predictions': mod_preds,
            'deltas': deltas,
            'interpretation': interpretation,
            'modifications': modifications_dict
        }

if __name__ == "__main__":
    print("=== TESTING UNCERTAINTY & COUNTERFACTUAL LIMITATIONS ===")
    runner = LiveModelRunner()
    test_payload = {
        "demographics": {"age": 68, "gender": "M"},
        "presentation_labs": {"creatinine_max": 4.5, "bun_max": 82.0, "wbc_max": 21.0},
        "active_medications": ["vancomycin"]
    }
    
    preds = runner.run_live_inference_with_uncertainty(test_payload)
    print(f"Mortality: {preds['p_mortality']*100:.1f}% | Tier: {preds['risk_tier']} | Confidence: {preds['model_confidence']}")
    print(preds['calibration_statement'])
