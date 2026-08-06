"""
Process-wide import ordering, so the whole suite can run in one command.

The problem
───────────
On this platform torch 2.2.2 — the last macOS x86_64 wheel published, built against
NumPy 1.x — segfaults when a LightGBM booster is unpickled into the same process. The
crash is inside `lightgbm/basic.py:__setstate__` during `joblib.load`, and it happens in
*either* import order, so importing torch first does not avoid it.

Per-file runs have always worked because each file gets a fresh interpreter and only
some of them touch both libraries. `pytest tests/` collects everything into one process,
so any file that loads a promoted model crashes the run once torch is resident. The
symptom is a bare `Fatal Python error: Segmentation fault` naming no test, which reads
as a broken test rather than a broken environment.

The fix
───────
Keep the real torch out of the test process entirely, and register a stub under its
name. This is safe because nothing under test needs it:

* `src/llm/twin_projection.py` reimplements the Phase 7 encoder forward pass in NumPy
  specifically so that serving never imports torch;
* the only remaining consumer is `LocalLLMBackend` in `src/llm/backends.py`, which loads
  a `transformers` model — and every test runs with `use_llm=False` or `NullBackend`.

Two test modules were already installing this stub at the top of their own file, but
only as an ImportError fallback, so on a machine where torch *is* installed the real one
won. conftest.py is imported before any test module, which makes the choice for every
collection order rather than for whichever file happened to be imported first.

If you are adding a test that genuinely needs real torch, run that file on its own —
do not remove this, or the suite goes back to crashing without naming a cause.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Imported before the stub is installed so neither probes a fake torch and caches the
# result: scipy and sklearn both check for it at import time.
import scipy.stats            # noqa: E402,F401
import sklearn.feature_extraction.text  # noqa: E402,F401

class _NoGrad:
    """Stand-in for `torch.no_grad()`, which backends.py uses as a context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_stub = types.ModuleType("torch")
_stub.__doc__ = ("Test stub. Real torch segfaults this process when a LightGBM booster "
                 "is unpickled; see tests/conftest.py.")
_stub.Tensor = type("Tensor", (), {})
_stub.set_num_threads = lambda n: None
_stub.no_grad = _NoGrad
_stub.load = lambda *a, **k: (_ for _ in ()).throw(
    FileNotFoundError("torch is stubbed in tests; no checkpoint can be loaded"))

# Unconditional. The two modules that installed this themselves did so only as an
# ImportError fallback, so on a machine where torch *is* installed the real one loaded
# and the suite crashed — which is the case here.
sys.modules["torch"] = _stub
