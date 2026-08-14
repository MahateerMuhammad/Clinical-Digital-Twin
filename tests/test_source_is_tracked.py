"""
Every source file under `src/` and `scripts/` must be visible to git.

`.gitignore` carried `data/`, `models/` and `logs/` without a leading slash. Those
patterns match a directory of that name at *any* depth, so `src/data/` and
`src/models/` were excluded — twelve files, including the whole cleaning pipeline and
every model-training module, were untracked and would not survive a fresh clone.

Nothing failed visibly. The code ran, the tests passed, the models trained. Only a
clone would have revealed it, and by then the working copy is the only copy.

The rule is narrow on purpose: source is tracked, data and binaries are not. Both
halves are asserted, because "un-ignore everything" would be a worse repository than
the one this fixes.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

SOURCE_DIRS = ("src", "scripts", "tests")

#: Must stay ignored. Committing these is the reason the patterns exist at all.
MUST_IGNORE = (
    "data/processed/admission_level.parquet",
    "models/best_models/phase1_mortality_winning.pkl",
)

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or not Path(".git").exists(),
    reason="not a git checkout",
)


def _ignored(path: str) -> bool:
    """True if git would ignore `path`. `check-ignore` exits 0 on a match."""
    return subprocess.run(
        ["git", "check-ignore", "-q", path],
        capture_output=True,
    ).returncode == 0


@pytest.mark.parametrize("directory", SOURCE_DIRS)
def test_no_python_source_is_ignored(directory, subtests):
    root = Path(directory)
    if not root.exists():
        pytest.skip(f"{directory}/ not present")

    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        with subtests.test(path=str(path)):
            assert not _ignored(str(path)), (
                f"{path} is ignored by .gitignore and would be lost on a fresh clone. "
                "Check for an unanchored directory pattern — `data/` matches at any "
                "depth, `/data/` matches only at the repository root.")


@pytest.mark.parametrize("path", MUST_IGNORE)
def test_data_and_model_binaries_stay_ignored(path):
    """The other half of the rule. Anchoring must not have un-ignored the artifacts."""
    if not Path(path).exists():
        pytest.skip(f"{path} not present in this checkout")
    assert _ignored(path), (
        f"{path} is no longer ignored. Data and model binaries must never enter the "
        "repository — MIMIC-IV is credentialed data.")


def test_top_level_artifact_directories_stay_ignored(subtests):
    for directory in ("data", "models", "logs"):
        if not Path(directory).exists():
            continue
        with subtests.test(directory=directory):
            assert _ignored(directory), (
                f"top-level {directory}/ is no longer ignored")
