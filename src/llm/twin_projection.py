"""
twin_projection.py
──────────────────
Project an arbitrary admission into the 32-dimensional Phase 7 embedding space.

Why this exists
───────────────
``similarity.parquet`` stores ``dim_0..dim_31`` for the 546 K admissions that were
present when Phase 7 last ran. ``ClinicalPromptBuilder.get_digital_twins`` looks a
query up **by hadm_id in that table**, so any admission not in it — a new patient, a
rebuilt cohort, a what-if — returns no twins at all. The embedding was reachable only
by table lookup, never by computation.

This module reconstructs the forward pass, so a patient can be embedded from their
features alone:

    adm row ──▶ booster design matrix ──▶ predict(pred_leaf=True) ──▶ scaler_leaf
                                                     └──▶ leaf AE encoder ──▶ 16d
            ──▶ encoder design matrix ──────────────────▶ scaler_static
                                                     └──▶ triplet AE encoder ──▶ 16d
                                                                       hstack ──▶ 32d

Two feature spaces, not one
───────────────────────────
The heads do **not** share inputs, and conflating them is the single most expensive
mistake available here:

* The **booster space** is the Phase 1 mortality model's own space — built from
  ``adm_df`` with ``MORTALITY_EXCLUDE_RUN_C`` applied, column names space-sanitised
  because LightGBM rewrites spaces to underscores. Feeding it the encoder frame
  zero-fills ~41% of its inputs and yields silently garbage leaf assignments.
* The **encoder space** is deliberately debiased: race, insurance, marital_status and
  language are dropped, and ``cci_*``/``dx_*``/``med_class_*`` are held out as
  triplet-mining supervision rather than used as inputs.

Both are reconstructed here from the saved artifacts, and both are reindexed against
the exact column list recorded at fit time — ``scaler_static.feature_names_in_`` for
the encoder, ``booster_.feature_name()`` for the trees. Nothing is positional.

The drop_first trap
───────────────────
Phase 7 built dummies with ``drop_first=True`` on the *full cohort*, where the dropped
reference level is implied by an all-zero row. Applying ``drop_first=True`` to a single
new row instead drops **that row's own** category, so a male patient would encode
identically to the female reference. Projection therefore builds dummies *without*
``drop_first`` and lets the reindex discard the reference column — which reproduces the
training encoding exactly, for any number of rows.

Usage
─────
    from src.llm.twin_projection import PatientProjector

    proj = PatientProjector()
    z = proj.project(frame)                  # (n, 32)
    z = proj.project_hadm_ids([20000001])    # loads the source rows for you
"""

from __future__ import annotations

import os
from functools import cached_property
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.features.leakage_filters import MORTALITY_EXCLUDE_RUN_C, match_column_patterns

#: Raw discharge-note text. Never a model input, and 96% of the parquet's bytes.
TEXT_COLS = ("text_clean", "text_tfidf_ready")

#: Dropped from the booster space by src/models/mortality.py. Mirrored from Cell 4.
BOOSTER_NON_FEATURES = (
    "subject_id", "hadm_id", "note_id", "admit_provider_id", "split",
    "hospital_expire_flag", "text_clean", "text_tfidf_ready",
)

#: Categoricals one-hot encoded in the encoder space (Cell 2).
ENCODER_CAT_COLS = ("gender", "admission_type", "admission_location")

#: Object columns with more levels than this were excluded at fit time (Cell 4).
MAX_CATEGORY_LEVELS = 100

LATENT_DIM = 16


class _NumpyEncoder:
    """
    The encoder half of a Phase 7 autoencoder, in NumPy.

    ``Linear(in, 128) → BatchNorm1d(128) → ReLU → Dropout(0.1) → Linear(128, 16)``,
    evaluated in inference mode: Dropout is the identity and BatchNorm uses its stored
    running statistics. Weights come from ``models/encoder_weights.npz``.

    Deliberately not torch. On this platform torch 2.2.2 (built for NumPy 1.x, the
    last macOS x86_64 wheel published) segfaults when a LightGBM booster is loaded in
    the same process, in either import order — and projection needs both libraries.
    See export_encoder_weights.py.
    """

    def __init__(self, npz, prefix: str):
        self.w1 = npz[f"{prefix}_w1"]          # (128, in)
        self.b1 = npz[f"{prefix}_b1"]
        self.gamma = npz[f"{prefix}_gamma"]
        self.beta = npz[f"{prefix}_beta"]
        self.mean = npz[f"{prefix}_mean"]
        self.var = npz[f"{prefix}_var"]
        self.w2 = npz[f"{prefix}_w2"]          # (16, 128)
        self.b2 = npz[f"{prefix}_b2"]
        self.eps = float(npz["bn_eps"])

    @property
    def in_features(self) -> int:
        return int(self.w1.shape[1])

    @property
    def out_features(self) -> int:
        return int(self.w2.shape[0])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = x.astype(np.float32) @ self.w1.T + self.b1
        h = (h - self.mean) / np.sqrt(self.var + self.eps) * self.gamma + self.beta
        np.maximum(h, 0.0, out=h)              # ReLU; Dropout is identity at eval
        return (h @ self.w2.T + self.b2).astype(np.float32)


class PatientProjector:
    """Embeds admissions into the Phase 7 32-d space from their features alone."""

    def __init__(self, models_dir: str = "models", data_dir: str = "data/processed"):
        self.models_dir = models_dir
        self.data_dir = data_dir

    # ── artifacts ───────────────────────────────────────────────────────────

    def _path(self, name: str) -> str:
        p = os.path.join(self.models_dir, name)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found. Projection needs scaler_static.pkl, scaler_leaf.pkl "
                "and encoder_weights.npz. The scalers are exported from "
                "notebooks/12_patient_embeddings_kaggle.ipynb (Cells 3 and 4); the "
                "npz is produced locally by `python export_encoder_weights.py` from "
                "the two .pt checkpoints.")
        return p

    @cached_property
    def scaler_static(self):
        import joblib
        return joblib.load(self._path("scaler_static.pkl"))

    @cached_property
    def scaler_leaf(self):
        import joblib
        return joblib.load(self._path("scaler_leaf.pkl"))

    @cached_property
    def booster(self):
        """The Phase 1 LightGBM estimator, unwrapped from its calibrator."""
        import joblib
        model = joblib.load(self._path("lightgbm_mortality.pkl"))
        return (model.calibrated_classifiers_[0].estimator
                if hasattr(model, "calibrated_classifiers_") else model)

    @cached_property
    def booster_features(self) -> list[str]:
        return list(self.booster.booster_.feature_name())

    @cached_property
    def encoder_features(self) -> list[str]:
        names = getattr(self.scaler_static, "feature_names_in_", None)
        if names is None:
            raise ValueError(
                "scaler_static.pkl carries no feature_names_in_, so the encoder "
                "column order cannot be recovered. Re-export it from a scikit-learn "
                "fit on a named DataFrame.")
        return list(names)

    @cached_property
    def _weights(self):
        path = self._path("encoder_weights.npz")
        return np.load(path)

    @cached_property
    def _triplet_ae(self) -> _NumpyEncoder:
        return _NumpyEncoder(self._weights, "triplet")

    @cached_property
    def _leaf_ae(self) -> _NumpyEncoder:
        return _NumpyEncoder(self._weights, "leaf")

    # ── design matrices ─────────────────────────────────────────────────────

    def encoder_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        """
        Build the 99-column debiased encoder input, scaled.

        Mirrors Cell 2, including its ``astype(str)`` on the categoricals — that turns
        genuine NaNs into the literal string ``"nan"``, which then becomes its own
        dummy level. Reproducing the quirk matters more than fixing it: the fitted
        weights encode it.
        """
        df = frame.copy()
        cats = [c for c in ENCODER_CAT_COLS if c in df.columns]
        for c in cats:
            df[c] = df[c].astype(str)

        if cats:
            df = pd.get_dummies(df, columns=cats, drop_first=False, dtype=float)

        df = df.reindex(columns=self.encoder_features, fill_value=0.0)
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return self.scaler_static.transform(df.to_numpy(dtype=np.float64))

    def booster_matrix(self, frame: pd.DataFrame, _reindex: bool = True) -> pd.DataFrame:
        """
        Rebuild the Phase 1 mortality model's own design matrix.

        Space-to-underscore sanitisation is not cosmetic: LightGBM rewrites feature
        names containing spaces at fit time, so ``admission_type_EW EMER.`` in the
        frame and ``admission_type_EW_EMER.`` in the booster are the same feature
        under two spellings. Skipping this step silently zero-fills every
        multi-word category — it is what drove booster coverage to 49.7%.
        """
        leaked = match_column_patterns(list(frame.columns), MORTALITY_EXCLUDE_RUN_C)
        df = frame.drop(columns=leaked, errors="ignore")

        keep = []
        for c in df.columns:
            if c in BOOSTER_NON_FEATURES or pd.api.types.is_datetime64_any_dtype(df[c]):
                continue
            is_cat = df[c].dtype == object or str(df[c].dtype) == "category"
            if is_cat and df[c].nunique(dropna=False) > MAX_CATEGORY_LEVELS:
                continue
            keep.append(c)

        X = pd.get_dummies(df[keep], drop_first=False)
        X.columns = [str(c).replace(" ", "_") for c in X.columns]
        X = X.loc[:, ~X.columns.duplicated()]
        if not _reindex:                      # coverage measurement; see booster_coverage
            return X
        return X.reindex(columns=self.booster_features, fill_value=0.0) \
                .astype(float).fillna(0.0)

    def booster_coverage(self, frame: pd.DataFrame) -> float:
        """
        Fraction of booster features genuinely present, measured *before* the
        reindex zero-fills the rest.

        This is the guard that would have caught the 49.7% and 58.9% coverage
        failures at the point of use rather than several plots downstream: a
        zero-filled feature is indistinguishable from a real zero once the matrix
        is built, so it has to be counted while the distinction still exists.
        """
        X = self.booster_matrix(frame, _reindex=False)
        have = set(X.columns)
        return sum(f in have for f in self.booster_features) / len(self.booster_features)

    # ── projection ──────────────────────────────────────────────────────────

    def _encode(self, model: _NumpyEncoder, X: np.ndarray,
                batch: int = 8192) -> np.ndarray:
        if len(X) == 0:
            return np.empty((0, model.out_features), dtype=np.float32)
        return np.vstack([model(X[i:i + batch]) for i in range(0, len(X), batch)])

    def project(self, frame: pd.DataFrame, check_coverage: bool = True) -> np.ndarray:
        """
        Embed ``frame`` into the 32-d space. Returns ``(len(frame), 32)``.

        ``dim_0..dim_15`` come from the triplet head, ``dim_16..dim_31`` from the
        tree-leaf head — the same ``hstack`` order Cell 4 wrote to
        ``similarity.parquet``. Getting that order backwards produces a valid-looking
        array that is silently in the wrong space, so it is asserted by the
        round-trip test rather than trusted.
        """
        if len(frame) == 0:
            return np.empty((0, 2 * LATENT_DIM), dtype=np.float32)

        if check_coverage:
            cov = self.booster_coverage(frame)
            if cov < 0.95:
                raise ValueError(
                    f"Booster feature coverage {cov:.1%} — the frame is missing "
                    "columns the Phase 1 model needs, so leaf assignments would be "
                    "meaningless. Supply the full admission-level feature set.")

        z_trip = self._encode(self._triplet_ae, self.encoder_matrix(frame))

        leaves = self.booster.predict(self.booster_matrix(frame), pred_leaf=True)
        leaves = np.asarray(leaves)
        if leaves.ndim == 1:                       # single row → LightGBM squeezes
            leaves = leaves.reshape(1, -1)
        z_leaf = self._encode(self._leaf_ae, self.scaler_leaf.transform(leaves))

        return np.hstack([z_trip, z_leaf])

    # ── convenience: load the source rows ───────────────────────────────────

    def load_source_frame(self, hadm_ids: Iterable[int] | None = None) -> pd.DataFrame:
        """
        Reconstruct the Cell 2 merge (``similarity ⋈ split ⋈ admission``) for the
        given admissions, skipping the two raw-text columns.
        """
        import pyarrow.parquet as pq

        sim = pd.read_parquet(os.path.join(self.data_dir, "similarity.parquet"))
        split = pd.read_parquet(os.path.join(self.data_dir, "patient_split.parquet"))

        if hadm_ids is not None:
            want = pd.Index(pd.Series(list(hadm_ids)).astype("int64"))
            sim = sim[sim["hadm_id"].astype("int64").isin(want)]

        adm_path = os.path.join(self.data_dir, "admission_level_selected.parquet")
        names = pq.ParquetFile(adm_path).schema_arrow.names
        adm = pd.read_parquet(adm_path,
                              columns=[c for c in names if c not in TEXT_COLS])
        if hadm_ids is not None:
            adm = adm[adm["hadm_id"].astype("int64").isin(want)]

        merged = sim.merge(split, on="subject_id", how="left")
        extra = [c for c in adm.columns if c not in merged.columns and c != "hadm_id"]
        return merged.merge(adm[["hadm_id"] + extra], on="hadm_id", how="left")

    def project_hadm_ids(self, hadm_ids: Sequence[int]) -> tuple[np.ndarray, pd.DataFrame]:
        """Load the source rows for ``hadm_ids`` and project them. Returns (Z, frame)."""
        frame = self.load_source_frame(hadm_ids)
        return self.project(frame), frame
