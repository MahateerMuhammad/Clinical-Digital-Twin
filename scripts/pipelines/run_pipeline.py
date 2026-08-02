#!/usr/bin/env python3
"""
Clinical Digital Twin — Pipeline Entry Point

Usage
-----
    python scripts/pipelines/run_pipeline.py                    # standard mode (recommended first run)
    python scripts/pipelines/run_pipeline.py --full             # load entire large tables (hours)
    python scripts/pipelines/run_pipeline.py --skip-large       # small tables only (fast smoke test)
    python scripts/pipelines/run_pipeline.py --steps load clean # run specific steps
"""

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

from src.data.pipeline import main

if __name__ == "__main__":
    main()
