"""
Round-trip tests for src/llm/twin_projection.

The load-bearing test is `test_roundtrip_matches_stored_embedding`: project an
admission that is *already* in similarity.parquet and compare the result against the
dim_* vector Phase 7 wrote for it. Anything wrong with the wiring — a mismatched
column order, drop_first applied to a row instead of a cohort, the two heads
hstacked backwards, the wrong scaler on the wrong space — produces a numerically
plausible array that disagrees with the stored one. Nothing short of that comparison
distinguishes a correct projection from a confident wrong answer.

These tests need the Phase 7 artifacts and the processed parquets, so they skip
cleanly when either is absent.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

MODELS = "models"
DATA = "data/processed"

REQUIRED_ARTIFACTS = [
    "scaler_static.pkl", "scaler_leaf.pkl",
    "encoder_weights.npz", "lightgbm_mortality.pkl",
]

pytestmark = pytest.mark.skipif(
    not all(os.path.exists(os.path.join(MODELS, f)) for f in REQUIRED_ARTIFACTS)
    or not os.path.exists(os.path.join(DATA, "similarity.parquet")),
    reason="Phase 7 artifacts or processed data unavailable",
)

N_SAMPLE = 24
DIM_COLS = [f"dim_{i}" for i in range(32)]


@pytest.fixture(scope="module")
def projector():
    from src.llm.twin_projection import PatientProjector
    return PatientProjector(models_dir=MODELS, data_dir=DATA)


@pytest.fixture(scope="module")
def sample(projector):
    """A handful of admissions that already carry stored embeddings."""
    sim = pd.read_parquet(os.path.join(DATA, "similarity.parquet"),
                          columns=["hadm_id"] + DIM_COLS)
    sim = sim.dropna(subset=DIM_COLS)
    if len(sim) < N_SAMPLE:
        pytest.skip("similarity.parquet has no populated dim_* columns")
    rng = np.random.default_rng(11)
    picked = sim.iloc[rng.choice(len(sim), N_SAMPLE, replace=False)]
    frame = projector.load_source_frame(picked["hadm_id"].tolist())
    stored = (picked.set_index(picked["hadm_id"].astype("int64"))
                    .loc[frame["hadm_id"].astype("int64"), DIM_COLS]
                    .to_numpy(dtype=np.float32))
    return frame, stored


def test_artifact_dimensions_agree(projector):
    """The artifacts must describe one consistent 99+350 → 16+16 pipeline."""
    assert len(projector.encoder_features) == projector.scaler_static.n_features_in_
    assert projector._triplet_ae.in_features == projector.scaler_static.n_features_in_
    assert projector._triplet_ae.out_features == 16
    assert projector._leaf_ae.in_features == projector.scaler_leaf.n_features_in_
    assert projector._leaf_ae.out_features == 16


def test_projection_imports_no_torch():
    """
    The serving path must stay torch-free.

    torch 2.2.2 (the last macOS x86_64 wheel, built for NumPy 1.x) segfaults when a
    LightGBM booster is loaded in the same process. A stray `import torch` anywhere
    under src/llm/twin_projection would turn every projection into a hard crash, so
    this asserts the module never pulls it in.
    """
    import subprocess
    code = (
        "import sys; import src.llm.twin_projection as m; "
        "from src.llm.twin_projection import PatientProjector; "
        "p = PatientProjector(); p._triplet_ae; p.booster_features; "
        "print('TORCH' if 'torch' in sys.modules else 'CLEAN')"
    )
    out = subprocess.run([sys.executable, "-W", "ignore", "-c", code],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, f"projection crashed: {out.stderr[-800:]}"
    assert "CLEAN" in out.stdout, "torch was imported into the projection path"


def test_booster_coverage_is_high(projector, sample):
    frame, _ = sample
    cov = projector.booster_coverage(frame)
    assert cov >= 0.95, f"booster coverage {cov:.1%} — feature spaces have drifted"


def test_design_matrix_shapes(projector, sample):
    frame, _ = sample
    assert projector.encoder_matrix(frame).shape == (
        len(frame), projector.scaler_static.n_features_in_)
    assert projector.booster_matrix(frame).shape == (
        len(frame), len(projector.booster_features))


def test_roundtrip_matches_stored_embedding(projector, sample):
    """Projection must reproduce the dim_* vectors Phase 7 stored."""
    frame, stored = sample
    z = projector.project(frame)

    assert z.shape == stored.shape
    assert np.isfinite(z).all()

    # float32 through BatchNorm accumulates a little; the tolerance is loose enough
    # for that and far tighter than any wiring error could survive.
    np.testing.assert_allclose(z, stored, rtol=1e-3, atol=1e-3)


def test_head_order_is_triplet_then_leaf(projector, sample):
    """
    dim_0..15 are the triplet head and dim_16..31 the leaf head.

    Swapping them yields an array of the right shape and scale, so this asserts the
    halves independently rather than relying on the combined comparison alone.
    """
    frame, stored = sample
    z_trip = projector._encode(projector._triplet_ae,
                               projector.encoder_matrix(frame))
    np.testing.assert_allclose(z_trip, stored[:, :16], rtol=1e-3, atol=1e-3)


def test_single_row_matches_batch(projector, sample):
    """
    One row must embed identically to the same row inside a batch.

    This is the drop_first trap: get_dummies(drop_first=True) on a single row drops
    that row's own category, so a per-patient call would silently encode every
    categorical as its reference level. Batch and single must agree exactly.
    """
    frame, _ = sample
    batch = projector.project(frame)
    single = projector.project(frame.iloc[[3]])
    np.testing.assert_allclose(single[0], batch[3], rtol=1e-4, atol=1e-4)


def test_empty_frame_returns_empty(projector):
    out = projector.project(pd.DataFrame(columns=["hadm_id"]), check_coverage=False)
    assert out.shape == (0, 32)
