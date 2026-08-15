import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath('.'))

from src.llm.feature_space import (
    ONEHOT_KNOWN, align_to_model, as_frame, encode_admission_frame,
    feature_coverage, looks_like_admission_row, onehot_known,
)


def _dig_payload(payload, path):
    """
    Follow a dotted ``path`` into a nested payload, tolerating a flat one.

    Callers hand-build these dicts, so ``{"demographics": {"age": 70}}`` and
    ``{"age": 70}`` both occur in the wild and both must resolve. The leaf name is
    tried at the top level before giving up, which keeps every payload written
    against the older flat shape working unchanged.
    """
    node = payload
    for part in path.split('.'):
        if not isinstance(node, dict) or part not in node:
            leaf = path.rsplit('.', 1)[-1]
            value = payload.get(leaf) if isinstance(payload, dict) else None
            return value if value != '' else None
        node = node[part]
    return node if node != '' else None

#: Measured payload fidelity per task, from `scripts/evaluation/run_payload_fidelity_eval.py`
#: on the held-out test split. `reference` is AUROC with the full admission feature
#: row — it reproduces each phase's published figure, which is what validates the
#: harness. `payload` is AUROC when only the fields an unseen-patient payload can
#: carry are supplied and the rest are NaN.
#:
#: Regenerate with `--patch` after any Phase 1-5 retrain; `tests/test_payload_fidelity.py`
#: fails if these drift from reports/tables/payload_fidelity_evaluation.md.
PAYLOAD_FIDELITY = {
    'mortality':     {'reference': 0.9448, 'payload': 0.8809},
    'readmission':   {'reference': 0.7062, 'payload': 0.6684},
    'icu_admission': {'reference': 0.9209, 'payload': 0.8183},
    'hospital_los':  {'reference': 0.8997, 'payload': 0.7314},
    'deterioration': {'reference': 0.7858, 'payload': 0.7617},
}

#: A payload-derived prediction is served only if it keeps at least this fraction of
#: the validated model's discriminative lift, (AUROC − 0.5) / (AUROC_ref − 0.5).
#:
#: Retention rather than absolute AUROC, because the report quotes each model under
#: its *published* performance: an ICU-admission figure presented as coming from a
#: 0.92-AUROC model, computed from an input on which that model scores 0.60, is a
#: misrepresentation whatever the absolute number.
#:
#: Both decisions at this boundary are stable rather than coin flips — bootstrapped
#: over the test split, the served task's retention CI is [0.716, 0.835] and the
#: nearest withheld one's is [0.523, 0.662], neither crossing the floor. An absolute
#: AUROC threshold near 0.70 would have been the unstable choice: deterioration's
#: absolute CI straddles it.
PAYLOAD_RETENTION_FLOOR = 2.0 / 3.0


def payload_retention(task):
    """Fraction of the validated model's AUROC lift a payload preserves."""
    f = PAYLOAD_FIDELITY.get(task)
    if not f or f['reference'] <= 0.5:
        return 0.0
    return (f['payload'] - 0.5) / (f['reference'] - 0.5)


#: Derived, never written by hand, so the floor and the table cannot disagree.
PAYLOAD_SERVED_TASKS = frozenset(
    t for t in PAYLOAD_FIDELITY if payload_retention(t) >= PAYLOAD_RETENTION_FLOOR)


def payload_withheld_reason(task):
    """Why ``task`` is not served from a payload, or None if it is."""
    if task in PAYLOAD_SERVED_TASKS:
        return None
    f = PAYLOAD_FIDELITY[task]
    head = (f"a presentation payload supports AUROC {f['payload']:.3f} for this task "
            f"against {f['reference']:.3f} from the full admission record")
    retention = payload_retention(task)
    if retention <= 0:
        # Below 0.5 the model ranks patients backwards, and "-3% of the validated
        # discrimination" describes that badly — a reader takes it for a small
        # shortfall. It also breaks fail-closed verification, which reads the digits
        # of "-3%" as the number 3 and cannot match the negative constant.
        return (f"{head} — below chance, so the model ranks patients in the wrong "
                f"direction on this input")
    return (f"{head} — {retention:.0%} of the validated discrimination, below the "
            f"{PAYLOAD_RETENTION_FLOOR:.0%} floor")


class LiveModelRunner:
    """
    Triggers live model inference on Phase 1-5 LightGBM pickles and runs
    Physiological State Counterfactual Analysis with Prediction Uncertainty Layers.
    """
    def __init__(self, models_dir='models/best_models', data_dir='data/processed'):
        self.models_dir = models_dir
        self.data_dir = data_dir
        
        # Promoted per-task winners and their Isotonic Calibrators.
        #
        # The filenames are algorithm-neutral because the promoted model is whichever
        # algorithm won that task, not necessarily LightGBM. They used to be named
        # `..._lightgbm_winning.pkl`, and Phase 5's contained an XGBClassifier — the
        # promotion script had always copied the true winner into a name that said
        # otherwise. Anyone reading the filename to find out what was being served got
        # the wrong answer, which is how the deterioration model came to be served
        # uncalibrated. `_load_pkl` falls back to the legacy name so existing
        # checkouts keep working.
        self.lgbm_models = {
            'mortality': self._load_pkl('phase1_mortality_winning.pkl'),
            'readmission': self._load_pkl('phase2_readmission_winning.pkl'),
            'icu_admission': self._load_pkl('phase3_icu_admission_winning.pkl'),
            'hospital_los': self._load_pkl('phase4_hosp_los_stageA_winning.pkl'),
            'deterioration': self._load_pkl('phase5_deterioration_winning.pkl'),
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

    #: legacy artifact name -> current one, for checkouts promoted before the rename.
    LEGACY_ARTIFACT_NAMES = {
        'phase1_mortality_winning.pkl': 'phase1_mortality_lightgbm_winning.pkl',
        'phase2_readmission_winning.pkl': 'phase2_readmission_lightgbm_winning.pkl',
        'phase3_icu_admission_winning.pkl': 'phase3_icu_admission_lightgbm_winning.pkl',
        'phase4_hosp_los_stageA_winning.pkl': 'phase4_hosp_los_stageA_lightgbm_winning.pkl',
        'phase5_deterioration_winning.pkl': 'phase5_deterioration_lightgbm_winning.pkl',
    }

    def _load_pkl(self, fname):
        path = os.path.join(self.models_dir, fname)
        if os.path.exists(path):
            return joblib.load(path)
        legacy = self.LEGACY_ARTIFACT_NAMES.get(fname)
        if legacy:
            legacy_path = os.path.join(self.models_dir, legacy)
            if os.path.exists(legacy_path):
                return joblib.load(legacy_path)
        return None

    def describe_models(self):
        """
        What is actually loaded, per task: ``{task: (class name, n_features, calibrated)}``.

        Exists because the filename is not evidence. Phase 5's artifact was called
        `..._lightgbm_winning.pkl` and held an XGBClassifier; the mismatch went
        unnoticed until a hardcoded LightGBM assumption started serving raw,
        uncalibrated output as a clinical figure. Ask the object, not the path.
        """
        out = {}
        for task, model in self.lgbm_models.items():
            if model is None:
                out[task] = (None, 0, False)
                continue
            names = self._feature_names(model)
            calibrator = self.calibrators.get(task)
            out[task] = (type(model).__name__, len(names or []),
                         calibrator is not None and hasattr(calibrator, 'predict'))
        return out

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

    #: Payload path → the *source* categorical columns the boosters were fitted on.
    #:
    #: These are emitted as categoricals and expanded by `encode_admission_frame`,
    #: not written out as dummy names. Writing `gender_M` by hand was the old
    #: approach and it silently under-served every model: Phase 5 carries both
    #: `gender_F` and `gender_M`, so a female patient arrived with `gender_M = 0.0`
    #: and `gender_F` absent — the model saw "not male, sex unknown".
    PAYLOAD_CATEGORICALS = {
        'demographics.gender': 'gender',
        'demographics.race': 'race',
        'demographics.language': 'language',
        'demographics.insurance': 'insurance',
        'demographics.marital_status': 'marital_status',
        'admission_context.admission_type': 'admission_type',
        'admission_context.admission_location': 'admission_location',
    }

    #: Payload path → plain numeric features. `prior_*` is the Expansion A&D block
    #: that Phase 2 was built on: a readmission model deprived of the patient's own
    #: readmission history is being asked the question with the answer removed,
    #: which is the most likely single cause of its 32.6% payload retention.
    PAYLOAD_NUMERICS = {
        'demographics.age': 'anchor_age',
        'prior_utilisation.admissions_30d': 'prior_admissions_30d',
        'prior_utilisation.admissions_90d': 'prior_admissions_90d',
        'prior_utilisation.admissions_365d': 'prior_admissions_365d',
        'prior_utilisation.cumulative_los_days': 'prior_cumulative_los_days',
    }
    # `diagnosis_count` and `procedure_count` are deliberately absent. They dominate
    # the mortality SHAP ranking and mapping them would lift the coverage figure, but
    # they are counts of what the hospital has coded by hour 24 — not something a
    # clinician dictates at intake. Claiming them would make coverage measure the
    # schema's ambition rather than what an unseen patient can actually supply.

    MED_KEYWORDS = {
        'med_class_antibiotic': ('vancomycin', 'cefepime', 'antibiotic', 'meropenem',
                                 'piperacillin', 'ceftriaxone'),
        'med_class_anticoagulant': ('enoxaparin', 'heparin', 'anticoagulant',
                                    'warfarin', 'apixaban'),
        'med_class_opioid': ('morphine', 'fentanyl', 'opioid', 'hydromorphone',
                             'oxycodone'),
        'med_class_insulin': ('insulin',),
    }

    def _encode_row(self, row):
        """
        One-hot expand a stored admission row into the boosters' namespace.

        ``Series.to_frame().T`` casts every column to object, which would make
        ``encode_admission_frame`` treat the numeric features as categoricals and
        one-hot expand all 275 of them. The original dtypes are restored from
        ``adm_df`` first, and ``infer_objects`` covers anything not found there.
        """
        frame = row.to_frame().T
        if self.adm_df is not None:
            for c in frame.columns:
                if c in self.adm_df.columns:
                    try:
                        frame[c] = frame[c].astype(self.adm_df[c].dtype)
                    except (TypeError, ValueError):
                        pass
        return encode_admission_frame(frame.infer_objects())

    def _convert_payload_to_series(self, payload):
        """Convert an unseen-patient payload into a flat Series of booster features."""
        if isinstance(payload, pd.Series):
            # A stored admission row still carries `admission_type = "URGENT"` rather
            # than the `admission_type_URGENT` dummy the booster was fitted on, so
            # without this expansion the numeric coercion downstream turned every
            # categorical into NaN and the row path ran on 78 of 164 features.
            if looks_like_admission_row(payload):
                return self._encode_row(payload).iloc[0]
            return payload
        if not isinstance(payload, dict):
            return pd.Series(dtype=float)

        labs = payload.get('presentation_labs', {}) or {}
        meds = [str(m).lower() for m in (payload.get('active_medications', []) or [])]

        flat = {}
        for path, column in self.PAYLOAD_NUMERICS.items():
            value = _dig_payload(payload, path)
            if value is None:
                continue
            try:
                flat[column] = float(value)
            except (TypeError, ValueError):
                continue
        # Age is the one numeric with a default: it is a required field, so reaching
        # here without it means an explicitly incomplete payload, and 65 keeps the
        # older behaviour rather than dropping the feature entirely.
        flat.setdefault('anchor_age', 65.0)

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

        return self._expand_payload_categoricals(payload, flat)

    def _expand_payload_categoricals(self, payload, flat):
        """
        Attach the payload's categoricals as booster dummies and return the Series.

        The levels come from ``adm_df`` rather than a table in this file. A payload
        saying ``race = "WHITE"`` determines all thirty-two ``race_*`` features, but
        only if the encoder knows the other thirty-one exist — and the authority on
        that is the cohort the models were fitted on, which is the one source that
        cannot drift out of step with them.

        Where ``adm_df`` is absent the observed level is still emitted and the
        per-row provenance from ``encode_admission_frame`` travels on ``attrs``, so
        :func:`align_to_model` can license the same zeros downstream. That path is
        the fallback, not the design: it depends on the consumer honouring ``attrs``.
        """
        supplied = {}
        for path, column in self.PAYLOAD_CATEGORICALS.items():
            value = _dig_payload(payload, path)
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            supplied[column] = str(value).strip()

        if not supplied:
            return pd.Series(flat, dtype=float)

        frame = pd.DataFrame([{**flat, **supplied}])
        for column, value in supplied.items():
            levels = self._category_levels(column)
            if levels is None:
                continue
            # Normalised against the cohort's own spelling: a payload saying "white"
            # or "Male" must land on the level the booster was fitted with, and an
            # unrecognised value is left as-is so it expands to a column the reindex
            # discards — never onto the wrong level.
            match = {str(x).strip().casefold(): x for x in levels}.get(value.casefold())
            frame[column] = pd.Categorical(
                [match if match is not None else value],
                categories=list(levels) if match is not None else None,
            )

        encoded = encode_admission_frame(frame, drop_outcomes=False)
        series = encoded.iloc[0]
        series.attrs[ONEHOT_KNOWN] = onehot_known(encoded)
        return series

    def _category_levels(self, column):
        """The cohort's levels for ``column``, or None if they cannot be established."""
        if self.adm_df is None or column not in self.adm_df.columns:
            return None
        values = self.adm_df[column]
        if isinstance(values.dtype, pd.CategoricalDtype):
            return list(values.cat.categories)
        return None

    def payload_feature_coverage(self, payload, task='mortality'):
        """
        Fraction of the task model's features the payload actually populates.

        An unseen payload cannot supply admission-derived features such as
        `diagnosis_count` or `procedure_count`, and those dominate the mortality
        model's SHAP ranking. They are left missing, which is a real limitation of
        payload-based inference rather than a bug — but it must stay visible, because
        the number this returns is what the coverage guard acts on.

        The admission-type dummies used to be listed here as equally unreachable.
        They are not: they were merely never asked for. Asking moved a complete
        payload from ~18% to ~68% of the mortality feature space, and took three
        tasks over the retention floor. The distinction is worth preserving —
        genuinely unsuppliable is not the same as not-yet-requested, and the first
        was being used to excuse the second.
        """
        model = self.lgbm_models.get(task)
        names = None if model is None else self._feature_names(model)
        if not names:
            return 0.0
        supplied = self._convert_payload_to_series(payload)
        return feature_coverage(as_frame(supplied), names)

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
        return self.predict_prob(task_key, patient_features)[1]

    def predict_prob(self, task_key, patient_features):
        """
        Return ``(raw, calibrated)`` for one task.

        Both are needed to tell two different things apart. Isotonic calibration is
        piecewise constant, so a genuine change in the model's raw output can map to a
        bit-identical calibrated probability: a heart-failure counterfactual moves the
        booster from 0.14995 to 0.14166 and both land on 0.795985%. Reading that zero
        delta off the calibrated value alone says "this input is not wired to the
        model", which is false — the input is wired, and the calibrator cannot resolve
        the difference. The two failures need different fixes, so they are reported
        separately; see `scripts/evaluation/run_phase11_eval.py`.
        """
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
                "scripts/maintenance/promote_models.py has been run.")


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

        # Unsupplied features are NaN, not 0.0. A zero is a measurement — creatinine
        # of zero, admission in year zero, no bloods sent — and the boosters split on
        # it. NaN is the missing-value representation both LightGBM and XGBoost were
        # fitted with. On the test split this alone takes payload-based mortality
        # AUROC from 0.8180 to 0.8470; see src/llm/feature_space.py.
        # `as_frame`, not `to_frame().T`: the latter drops the one-hot provenance, and
        # with it the zeros a supplied category licenses — so the scored vector would
        # disagree with the coverage figure reported alongside it.
        X_sub = align_to_model(as_frame(patient_features), req_cols)
        raw_prob = float(model.predict_proba(X_sub)[0, 1])
        
        calibrator = self.calibrators.get(task_key)
        cal_prob = None
        if calibrator is not None and hasattr(calibrator, 'predict'):
            try:
                cal_prob = float(np.clip(float(calibrator.predict([raw_prob])[0]),
                                         0.0001, 0.9999))
            except Exception:
                cal_prob = None
        if cal_prob is None:
            cal_prob = float(np.clip(raw_prob, 0.0001, 0.9999))

        # Age-band refinement on top of the global value. `reports/slice_evaluation.md`
        # measured what the headline ECE of 0.0036 concealed: for patients aged 85+
        # the mortality model observed 5.13% and predicted 3.95%. The band
        # calibrators correct that residual bias without touching the booster.
        #
        # Fail-open on every path — absent artefact, unknown age, unfitted band,
        # transform error. A calibration refinement must not be able to take
        # prediction down; it improves a number that was already being served.
        band_prob = self.group_calibrators.calibrate(
            task_key, cal_prob, age=self._patient_age(patient_features))
        return raw_prob, band_prob if band_prob is not None else cal_prob

    @staticmethod
    def _patient_age(patient_features):
        """Age from the scored feature vector, or None if it is not there."""
        try:
            value = patient_features.get('anchor_age')
        except Exception:
            return None
        try:
            f = float(value)
        except (TypeError, ValueError):
            return None
        return None if f != f else f

    @property
    def group_calibrators(self):
        """Age-band calibrators, loaded once and never required."""
        if getattr(self, "_group_calibrators", None) is None:
            from src.models.group_calibration import GroupCalibrators
            self._group_calibrators = GroupCalibrators.load()
        return self._group_calibrators

    #: results key ← task key
    TASK_KEYS = {
        'p_mortality': 'mortality',
        'p_readmission': 'readmission',
        'p_icu_admission': 'icu_admission',
        'p_los_over_5_63d': 'hospital_los',
        'p_deterioration': 'deterioration',
    }

    def run_live_inference_with_uncertainty(self, patient_payload):
        """
        Executes multi-task inference and attaches explicit Prediction Uncertainty & Calibration Reliability.

        Tasks a presentation payload cannot support are withheld rather than
        reported. Every task used to be served from any input, so a payload carrying
        eleven labs produced an ICU-admission figure whose rank correlation with the
        same model's full-record prediction was 0.004 — a number with the form of a
        risk estimate and none of the content. A stored admission row populates the
        whole feature set, so nothing is withheld on that path.
        """
        is_row = isinstance(patient_payload, pd.Series) and looks_like_admission_row(patient_payload)
        p_series = self._convert_payload_to_series(patient_payload)
        results = {}

        raw_mort, p_mort = self.predict_prob('mortality', p_series)
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

        withheld = {}
        # Pre-calibration outputs, kept alongside. They are never quoted to a
        # clinician — an uncalibrated booster reports ~79% deterioration against a
        # 5.95% base rate — but they are the only way to distinguish "the input did
        # not reach the model" from "the isotonic step could not resolve the change".
        raw = {'p_mortality': raw_mort}
        for key, task in self.TASK_KEYS.items():
            if key == 'p_mortality':
                continue
            reason = None if is_row else payload_withheld_reason(task)
            if reason:
                results[key] = None
                withheld[key] = reason
            else:
                raw[key], results[key] = self.predict_prob(task, p_series)

        results['withheld_tasks'] = withheld
        results['raw_probabilities'] = raw
        results['input_kind'] = 'admission_row' if is_row else 'presentation_payload'
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
        
        # A withheld task has no probability, so it has no delta either. Subtracting
        # None raises; substituting 0.0 would report "this intervention changes
        # nothing", which is the reading the whole withholding mechanism exists to
        # prevent. The key is simply absent, and the reason travels with it.
        deltas = {'base_tier': base_preds['risk_tier'], 'mod_tier': mod_preds['risk_tier']}
        base_raw = base_preds.get('raw_probabilities') or {}
        mod_raw = mod_preds.get('raw_probabilities') or {}
        for key in ('p_mortality', 'p_icu_admission', 'p_deterioration'):
            base, mod = base_preds.get(key), mod_preds.get(key)
            if base is not None and mod is not None:
                deltas[f'delta_{key}'] = mod - base
            if key in base_raw and key in mod_raw:
                # The raw delta answers "did this input reach the model?". The
                # calibrated one answers "can the served number show it?". They differ
                # wherever the isotonic fit is flat, and conflating them reads a
                # resolution limit as a broken wire.
                deltas[f'delta_raw_{key}'] = mod_raw[key] - base_raw[key]
        deltas['withheld_tasks'] = dict(base_preds.get('withheld_tasks') or {})
        
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
