#!/usr/bin/env python3
"""
scripts/maintenance/export_encoder_weights.py
─────────────────────────
Convert the two Phase 7 autoencoder checkpoints into a NumPy archive so patient
projection never has to import torch.

Why this is necessary, not merely tidy
──────────────────────────────────────
On this platform torch 2.2.2 is compiled against NumPy 1.x while the environment runs
NumPy 2.3.5, and PyTorch published no macOS x86_64 wheel past 2.2.2 — so the version
skew cannot be closed by upgrading. The practical consequence is that **importing
torch and loading a LightGBM booster in the same process segfaults**, in either order:

    import torch; joblib.load('lightgbm_mortality.pkl')   -> SIGSEGV
    joblib.load('lightgbm_mortality.pkl'); import torch    -> SIGSEGV

Projection needs both — LightGBM for the leaf assignments, the autoencoders for the
two 16-d heads — so with torch in the serving path the pipeline cannot run at all.

Both encoders are two-layer MLPs, so their inference is a handful of matrix
operations. Exporting the weights removes torch from the runtime entirely rather than
splitting the work across processes, which would be slower and would leave the
segfault waiting for anyone who imported the module differently.

    Linear(in, 128) -> BatchNorm1d(128) -> ReLU -> Dropout(0.1) -> Linear(128, 16)

Dropout is identity at eval; BatchNorm in eval mode uses its stored running
statistics. Nothing about the forward pass needs a framework.

Run this in a torch-only process (it imports no LightGBM) after any Phase 7 rerun:

    .venv/bin/python scripts/maintenance/export_encoder_weights.py
"""


from __future__ import annotations


# ── repo-root bootstrap ──────────────────────────────────────────────────────
# These scripts live two levels below the project root. Python puts the *script's*
# directory on sys.path, not the working directory, so `import src...` would fail
# from here; and many of them address data with root-relative paths such as
# "models/" or "reports/tables/". Both are fixed by putting the root on the path
# and running from it, which makes execution identical from any directory.
import os as _os
import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
_os.chdir(_ROOT)
# ─────────────────────────────────────────────────────────────────────────────
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
OUT = MODELS / "encoder_weights.npz"

#: checkpoint -> prefix in the archive
HEADS = {
    "patient_autoencoder_triplet.pt": "triplet",   # dim_0..15
    "patient_autoencoder_lgb.pt": "leaf",          # dim_16..31
}

#: Sequential indices inside `encoder`. 2 is ReLU and 3 is Dropout — no parameters.
LINEAR_IN, BATCHNORM, LINEAR_OUT = "0", "1", "4"


def main() -> int:
    import torch

    arrays: dict[str, np.ndarray] = {}
    summary = []

    for filename, prefix in HEADS.items():
        path = MODELS / filename
        if not path.exists():
            print(f"Missing {path}", file=sys.stderr)
            return 1

        sd = torch.load(path, map_location="cpu")

        def take(key: str) -> np.ndarray:
            # .tolist() rather than .numpy(): the NumPy bridge is exactly what is
            # broken in this torch build, and it is the thing being routed around.
            return np.asarray(sd[f"encoder.{key}"].tolist(), dtype=np.float32)

        arrays[f"{prefix}_w1"] = take(f"{LINEAR_IN}.weight")
        arrays[f"{prefix}_b1"] = take(f"{LINEAR_IN}.bias")
        arrays[f"{prefix}_gamma"] = take(f"{BATCHNORM}.weight")
        arrays[f"{prefix}_beta"] = take(f"{BATCHNORM}.bias")
        arrays[f"{prefix}_mean"] = take(f"{BATCHNORM}.running_mean")
        arrays[f"{prefix}_var"] = take(f"{BATCHNORM}.running_var")
        arrays[f"{prefix}_w2"] = take(f"{LINEAR_OUT}.weight")
        arrays[f"{prefix}_b2"] = take(f"{LINEAR_OUT}.bias")

        summary.append((prefix, arrays[f"{prefix}_w1"].shape[1],
                        arrays[f"{prefix}_w2"].shape[0], filename))

    # BatchNorm1d default; stored so the forward pass never hardcodes it.
    arrays["bn_eps"] = np.asarray(1e-5, dtype=np.float32)

    np.savez(OUT, **arrays)

    print(f"{'head':<10}{'in':>6}{'out':>6}  source")
    print("-" * 52)
    for prefix, n_in, n_out, filename in summary:
        print(f"{prefix:<10}{n_in:>6}{n_out:>6}  {filename}")
    print("-" * 52)
    print(f"\nWrote {OUT} ({OUT.stat().st_size / 1e3:.1f} KB)")
    print("Projection now runs without torch; rerun this after any Phase 7 rerun.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
