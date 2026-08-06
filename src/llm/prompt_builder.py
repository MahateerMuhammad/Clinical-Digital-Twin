import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
from src.llm.model_runner import LiveModelRunner
from src.llm.rag_corpus import rag_engine
from src.llm.report_composer import SYSTEM_CONSTANTS


def _risk_line(preds, key, label, suffix=''):
    """One risk bullet, or a statement of why there is no number for it."""
    withheld = (preds.get('withheld_tasks') or {}).get(key)
    if withheld:
        return f"- **{label}:** _withheld — {withheld}._"
    value = preds.get(key)
    if value is None:
        return f"- **{label}:** _not computed._"
    return f"- **{label}:** {value * 100:.2f}%{suffix}"


class ClinicalPromptBuilder:
    """
    Synthesizes patient presentation data, live model predictions (Phases 1-5),
    SHAP risk drivers (Phase 8), Risk Tiering (Phase 9), Digital Twin RAG evidence (Phase 7),
    and Plotly visual timeline linkages (Phase 10) into a structured LLM prompt.
    """
    #: Columns holding the Phase 7 patient embedding (dim_0 .. dim_31).
    EMBED_PREFIX = 'dim_'

    def __init__(self, data_dir='data/processed'):
        self.data_dir = data_dir
        self.runner = LiveModelRunner(data_dir=data_dir)
        self.sim_df = pd.read_parquet(os.path.join(data_dir, 'similarity.parquet'))

        # Outcome columns are the only thing needed from the admission frame. Reading
        # the whole 3 GB file and scanning it once per twin made a five-twin lookup
        # take minutes; this reads five columns and indexes them by hadm_id.
        outcome_cols = ['hadm_id', 'hospital_expire_flag', 'has_icu_stay',
                        'readmission_30d', 'los_days']
        adm_path = os.path.join(data_dir, 'admission_level_selected.parquet')
        available = pd.read_parquet(adm_path, columns=['hadm_id']).columns.tolist()
        try:
            self.adm_df = pd.read_parquet(adm_path, columns=outcome_cols)
        except Exception:                                   # column set varies by build
            self.adm_df = pd.read_parquet(adm_path)
        self._adm_index = self.adm_df.set_index(
            self.adm_df['hadm_id'].astype('int64'))

        self._embed_cols = sorted(
            (c for c in self.sim_df.columns if c.startswith(self.EMBED_PREFIX)),
            key=lambda c: int(c.split('_')[1]))
        self._nn = None
        self._nn_hadm = None
        self._nn_subject = None
        self._projector = None

    def _build_twin_index(self):
        """Fit the nearest-neighbour index over the Phase 7 embedding space."""
        from sklearn.neighbors import NearestNeighbors

        frame = self.sim_df.dropna(subset=self._embed_cols)
        if frame.empty:
            raise ValueError(
                "similarity.parquet has no populated dim_* columns. Run Phase 7 "
                "(notebooks/12_patient_embeddings_kaggle.ipynb) and install its "
                "similarity.parquet before requesting digital twins.")
        matrix = frame[self._embed_cols].to_numpy(dtype='float32')
        self._nn_hadm = frame['hadm_id'].astype('int64').to_numpy()
        self._nn_subject = frame['subject_id'].astype('int64').to_numpy()
        # Over-fetch so same-subject admissions can be dropped without a short result.
        self._nn = NearestNeighbors(n_neighbors=64, metric='euclidean',
                                    n_jobs=-1).fit(matrix)
        self._nn_matrix = matrix

    def get_digital_twins(self, hadm_id, top_k=5):
        """
        Retrieve the ``top_k`` nearest historical admissions in embedding space.

        Previously this returned ``sim_df[sim_df.subject_id != sub_id].head(top_k)``
        — the first rows of the file in storage order, identical for every query and
        entirely independent of the patient. The 32-dimensional Phase 7 embedding was
        loaded and never used, so "digital twin retrieval" returned the same five
        admissions to everyone.

        Twins are now true nearest neighbours by Euclidean distance in that space,
        excluding every admission belonging to the query patient (not just the query
        admission — a patient's own prior stays are not independent evidence).
        """
        if self._embed_cols and self._nn is None:
            self._build_twin_index()
        if not self._embed_cols:
            raise ValueError(
                "similarity.parquet contains no dim_* embedding columns; "
                "digital twin retrieval is unavailable until Phase 7 has been run.")

        hadm_id = int(hadm_id)
        pos = np.flatnonzero(self._nn_hadm == hadm_id)
        if len(pos) == 0:
            # Not in similarity.parquet — compute the embedding instead of giving up.
            # Previously this returned [], so every patient outside the Phase 7
            # snapshot silently got no twins at all.
            return self.get_twins_for_vector(*self._project(hadm_id), top_k=top_k)
        q_idx = int(pos[0])
        return self.get_twins_for_vector(
            self._nn_matrix[q_idx], self._nn_subject[q_idx], top_k=top_k)

    def get_twins_for_vector(self, vector, exclude_subject=None, top_k=5):
        """
        Retrieve twins for an arbitrary point in the 32-d embedding space.

        Shared by both entry points: a stored admission passes its own row of the
        index, a projected one passes a freshly computed vector. ``exclude_subject``
        drops every admission belonging to that patient — for a stored query this
        also removes the query itself, which is why the distance-0 self-match never
        appears in the results.
        """
        if self._nn is None:
            self._build_twin_index()

        vector = np.asarray(vector, dtype='float32').reshape(1, -1)
        n_probe = min(len(self._nn_hadm), max(64, top_k * 12))
        dists, inds = self._nn.kneighbors(vector, n_neighbors=n_probe)

        twins = []
        for dist, idx in zip(dists[0], inds[0]):
            if exclude_subject is not None and self._nn_subject[idx] == exclude_subject:
                continue
            t_hadm = int(self._nn_hadm[idx])
            if t_hadm not in self._adm_index.index:
                continue
            info = self._adm_index.loc[t_hadm]
            if isinstance(info, pd.DataFrame):
                info = info.iloc[0]
            twins.append({
                'hadm_id': t_hadm,
                'distance': float(dist),
                'hospital_expire_flag': int(info.get('hospital_expire_flag', 0) or 0),
                'has_icu_stay': int(info.get('has_icu_stay', 0) or 0),
                'readmission_30d': int(info.get('readmission_30d', 0) or 0),
                'los_days': float(info.get('los_days', 1.0) or 1.0),
            })
            if len(twins) == top_k:
                break
        return twins

    @property
    def projector(self):
        """Lazily constructed PatientProjector; only built if a query misses."""
        if self._projector is None:
            from src.llm.twin_projection import PatientProjector
            self._projector = PatientProjector(data_dir=self.data_dir)
        return self._projector

    def _project(self, hadm_id):
        """Embed an admission absent from similarity.parquet. Returns (vector, subject)."""
        z, frame = self.projector.project_hadm_ids([int(hadm_id)])
        if len(frame) == 0:
            raise KeyError(
                f"hadm_id {hadm_id} is in neither similarity.parquet nor "
                "admission_level_selected.parquet, so it cannot be embedded.")
        subject = (int(frame['subject_id'].iloc[0])
                   if 'subject_id' in frame.columns else None)
        return z[0], subject

    #: Labs rendered in the presentation block: candidate columns in preference
    #: order, then label, unit and adult reference range. Ranges are printed as
    #: reader context and are never used to compute anything.
    #:
    #: Only ``_24h`` columns are eligible. The block previously read
    #: ``lab_creatinine_max``, ``lab_wbc_max`` and friends — whole-admission extremes
    #: that the Run C leakage filter removes, so they were absent from the frame and
    #: every patient rendered as "not recorded" under a heading claiming t = 24h.
    #:
    #: The windowed rebuild did not produce the same aggregates for every analyte:
    #: sodium and glucose have min/max, bicarbonate and WBC have only first/last, and
    #: creatinine and BUN have no windowed *value* feature at all — just counts and
    #: missing-ratios. That is a gap in the feature build, not a gap in the patient's
    #: chart, so the two cases are reported differently below.
    PRESENTATION_LABS = (
        (('lab_creatinine_last_24h', 'lab_creatinine_first_24h'),
         'Serum Creatinine', 'mg/dL', '0.6-1.2'),
        (('lab_bun_last_24h', 'lab_bun_first_24h'),
         'BUN', 'mg/dL', '7-20'),
        (('lab_bicarbonate_last_24h', 'lab_bicarbonate_first_24h'),
         'Serum Bicarbonate', 'mEq/L', '22-29'),
        (('lab_wbc_last_24h', 'lab_wbc_first_24h'),
         'WBC Count', 'K/uL', '4.5-11.0'),
        (('lab_sodium_min_24h', 'lab_sodium_max_24h'),
         'Serum Sodium', 'mEq/L', '135-145'),
        (('lab_glucose_max_24h', 'lab_glucose_median_24h'),
         'Blood Glucose', 'mg/dL', '70-100'),
    )

    MED_CLASSES = (
        ('med_class_opioid', 'Opioids'),
        ('med_class_antibiotic', 'Antibiotics'),
        ('med_class_insulin', 'Insulin'),
        ('med_class_anticoagulant', 'Anticoagulants'),
    )

    def _shap_drivers(self, patient_row, task='mortality', top_k=5):
        """
        Local SHAP attributions for one admission, via ``shap.TreeExplainer``.

        These were previously four hardcoded literals — ``+0.45``, ``+0.38``,
        ``+0.29``, ``+0.18`` — printed under a "PHASE 8 LOCAL SHAP" heading for every
        patient regardless of their data. They were not computed from anything, and a
        clinician reading the report had no way to tell. Returns [] rather than
        inventing values when the model or shap is unavailable.
        """
        model = self.runner.lgbm_models.get(task)
        if model is None or not hasattr(model, 'booster_'):
            return []
        try:
            import shap
        except ImportError:
            return []

        feat_cols = list(model.booster_.feature_name())
        frame = pd.DataFrame([patient_row])
        cat_cols = [c for c in ('admission_type', 'admission_location', 'insurance',
                                'language', 'marital_status', 'race', 'gender',
                                'anchor_year_group') if c in frame.columns]
        if cat_cols:
            # No drop_first: on a single row it would drop that row's own category,
            # collapsing every patient onto the reference level. The reindex below
            # discards the reference column instead. Same trap as twin_projection.
            frame = pd.get_dummies(frame, columns=cat_cols)
        frame.columns = [str(c).replace(' ', '_') for c in frame.columns]

        X = (frame.reindex(columns=feat_cols, fill_value=0.0)
                  .apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(float))

        values = shap.TreeExplainer(model).shap_values(X)
        if isinstance(values, list):                 # binary classifier -> [neg, pos]
            values = values[1]
        values = np.asarray(values).reshape(len(feat_cols))

        order = np.argsort(np.abs(values))[::-1][:top_k]
        return [{
            'feature': feat_cols[i],
            'value': float(X.iloc[0, i]),
            'shap': float(values[i]),
            'direction': 'increases risk' if values[i] > 0 else 'lowers risk',
        } for i in order]

    def _trajectory_link(self, hadm_id):
        """
        Link to this patient's trajectory plot, or None.

        The link was hardcoded to ``file:///Users/apple/Desktop/...
        patient_trajectory_mortality.html`` — one developer's home directory, and the
        same archetype file for every patient. Phase 10 writes four archetype plots,
        not per-admission ones, so unless a file for this hadm_id exists there is
        nothing honest to link to.
        """
        rel = os.path.join('reports', 'figures', 'trajectories',
                           f'patient_trajectory_{int(hadm_id)}.html')
        path = os.path.join(os.path.abspath('.'), rel)
        return rel if os.path.exists(path) else None

    def build_structured_prompt(self, hadm_id):
        """
        Build the structured clinical prompt for an admission.

        Every number below is computed from this patient: predictions from the
        Phase 1-5 models, tier from the Phase 9 cutoffs, SHAP from TreeExplainer,
        twins from the Phase 7 embedding, guidelines from the RAG corpus. Blocks that
        cannot be produced are marked unavailable rather than filled with plausible
        values.
        """
        patient_row = self.runner.get_patient_row(hadm_id)
        preds = self.runner.run_live_inference(patient_row)
        drivers = self._shap_drivers(patient_row)

        # Twin evidence is the one optional block. Rebuilding the datasets rewrites
        # similarity.parquet without dim_* columns until Phase 7 is re-run, and a
        # missing evidence tier must not take the predictions and SHAP drivers down
        # with it — the report degrades by saying the block is unavailable, which is
        # the same contract rag_corpus follows for Level 5 evidence.
        try:
            twins = self.get_digital_twins(hadm_id, top_k=5)
            twin_error = ""
        except (ValueError, KeyError, FileNotFoundError) as exc:
            twins, twin_error = [], str(exc)

        def num(col, default=float('nan')):
            return pd.to_numeric(patient_row.get(col, default), errors='coerce')

        age = patient_row.get('anchor_age', 'unknown')
        gender = patient_row.get('gender', 'unknown')
        adm_type = patient_row.get('admission_type', 'unknown')

        lab_lines, lab_values = [], {}
        for cands, label, unit, ref in self.PRESENTATION_LABS:
            col = next((c for c in cands if c in patient_row.index), None)
            if col is None:
                # No windowed feature of this analyte exists in the build at all.
                shown = 'no 24h aggregate in this feature build'
            else:
                v = num(col)
                if pd.isna(v):
                    shown = 'not recorded in the first 24h'
                else:
                    shown = f'{v:.1f} {unit}'
                    lab_values[label] = v
            lab_lines.append(f'- **{label}:** {shown} (Normal: {ref} {unit})')

        meds = ', '.join(f'{label} ({int(num(col, 0) or 0)})'
                         for col, label in self.MED_CLASSES)

        # A retrieval query of bare "label value" pairs matched almost nothing. The
        # corpus is indexed on clinical language, so the query names the conditions
        # the values imply and falls back to the presentation itself when nothing is
        # deranged, rather than sending an empty string.
        concepts = []
        if lab_values.get('Serum Creatinine', 0) > 1.2 or lab_values.get('BUN', 0) > 20:
            concepts.append('acute kidney injury renal impairment')
        if lab_values.get('WBC Count', 0) > 11.0:
            concepts.append('leukocytosis sepsis infection')
        if lab_values.get('Serum Bicarbonate', 99) < 22:
            concepts.append('metabolic acidosis')
        if lab_values.get('Serum Sodium', 999) < 135:
            concepts.append('hyponatremia')
        if lab_values.get('Blood Glucose', 0) > 180:
            concepts.append('hyperglycemia diabetes management')
        query = ' '.join(concepts) or f'{adm_type} admission inpatient risk assessment'
        guidelines = rag_engine.retrieve_guidelines(query, top_k=2)

        n = len(twins)
        n_mort = sum(t['hospital_expire_flag'] for t in twins)
        n_icu = sum(t['has_icu_stay'] for t in twins)
        n_readm = sum(t['readmission_30d'] for t in twins)

        los_thr = SYSTEM_CONSTANTS['phase4_hosp_los_threshold_days']
        det_hrs = SYSTEM_CONSTANTS['phase5_deterioration_window_hours']

        parts = [
            f'### CLINICAL DIGITAL TWIN PATIENT REPORT (t = 24h)',
            f'**Patient HADM ID:** {int(hadm_id)} | **Age:** {age} | '
            f'**Gender:** {gender} | **Admission Type:** {adm_type}',
            '',
            '---',
            '',
            '#### 1. PRESENTATION LABS & 24H MEDICATION PROFILE',
            *lab_lines,
            f'- **Active 24h Medication Regimen:** {meds}',
            '',
            '---',
            '',
            '#### 2. MULTI-TASK PREDICTIVE SUITE (PHASES 1-5 & 9)',
            # This is the stored-admission path, so the full feature row is available
            # and every task is served. `_risk_line` still handles withholding: the
            # same helper is reachable from a payload, and printing None as 0.00%
            # would read as a confident negative.
            _risk_line(preds, 'p_mortality',
                       'In-Hospital Mortality Risk', f" ({preds['risk_tier']})"),
            _risk_line(preds, 'p_readmission', '30-Day Hospital Readmission Risk'),
            _risk_line(preds, 'p_icu_admission', 'Emergency ICU Admission Risk'),
            _risk_line(preds, 'p_los_over_5_63d',
                       f'Hospital Length of Stay > {los_thr} Days Risk'),
            _risk_line(preds, 'p_deterioration',
                       f'{det_hrs:.0f}-Hour Early Deterioration Warning Score'),
            '',
            '---',
            '',
            '#### 3. LOCAL SHAP RISK DRIVERS (PHASE 8, TreeExplainer)',
        ]

        if drivers:
            for i, d in enumerate(drivers, 1):
                parts.append(f"{i}. `{d['feature']}` = {d['value']:.4g}: "
                             f"SHAP {d['shap']:+.4f} ({d['direction']})")
            parts.append('')
            parts.append('_SHAP values are log-odds contributions for this admission '
                         'against the model\'s expected value; they are not '
                         'probabilities and do not sum to the risk above._')
        else:
            parts.append('_Unavailable: the mortality model or the `shap` package '
                         'could not be loaded. No values are substituted._')

        parts += ['', '---', '',
                  f'#### 4. RETRIEVED DIGITAL TWINS (PHASE 7, N={n})']
        if n:
            parts += [
                f'- Nearest {n} historical admissions in the 32-dimensional hybrid '
                f'embedding, excluding this patient\'s own stays:',
                f'  - **Deaths Observed:** {n_mort}/{n} ({n_mort/n*100:.0f}%)',
                f'  - **ICU Transfers:** {n_icu}/{n} ({n_icu/n*100:.0f}%)',
                f'  - **30-Day Readmissions:** {n_readm}/{n} ({n_readm/n*100:.0f}%)',
                f'  - **Mean embedding distance:** '
                f'{np.mean([t["distance"] for t in twins]):.3f}',
                '',
                # 0.7253 was the superseded unconditional figure; the conditional
                # metric in reports/tables/twin_retrieval_evaluation.md gives 0.8044
                # on the same embeddings.
                '_Twin outcomes are observed precedent, not a prediction. Retrieval '
                '(AUROC 0.8044 on 3,000 queries) is weaker than the tabular model._',
            ]
        else:
            parts.append(f'_Unavailable: {twin_error or "this admission could not be embedded"}. '
                         'No twin evidence is substituted._')

        parts += ['', '---', '', '#### 5. RETRIEVED GUIDELINES & CITATIONS']
        if guidelines:
            for g in guidelines:
                # 'text', not 'content' — the old key existed in no document the
                # engine returns, so this line raised KeyError on the first guideline
                # ever retrieved. Provenance is carried through: these are
                # paraphrased summaries, and the report must not imply otherwise.
                parts.append(
                    f"- {g['citation']} **{g['title']}**\n"
                    f"  {g.get('text', '').strip()}\n"
                    f"  _{g.get('evidence_level', 'unclassified')} · "
                    f"{g.get('provenance', 'provenance unrecorded')}_")
        else:
            parts.append('_No guideline evidence retrieved for this presentation._')

        link = self._trajectory_link(hadm_id)
        if link:
            parts += ['', f'- **Interactive Visual Timeline:** [{os.path.basename(link)}]({link})']

        return '\n'.join(parts) + '\n'


if __name__ == "__main__":
    builder = ClinicalPromptBuilder()
    target = int(sys.argv[1]) if len(sys.argv) > 1 else int(builder.sim_df['hadm_id'].iloc[0])
    print("=== STRUCTURED PROMPT BUILDER ===")
    print(builder.build_structured_prompt(target))
