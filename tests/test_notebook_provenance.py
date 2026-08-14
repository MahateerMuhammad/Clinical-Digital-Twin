"""
A notebook carrying stored outputs must say where those outputs came from.

Notebook 11 published a superseded 6-hour design for weeks — including an invented
">0.99 AUROC" attributed to features the matrix does not contain — and nothing in the
file said which run produced it. The fix there was to make the notebook *derive* its
figures. That works for a reporting notebook and not for a training one, which
legitimately produces numbers of its own that differ from the promoted model's.

So the rule enforced here is the weaker but universal one: if a notebook stores
outputs, it must state their provenance in its first cell. Notebook 10's stored Stage
A AUROC is 0.8114 against the served model's 0.8997 — a nine-point gap that is
entirely legitimate (it predates the laboratory-join repair) and entirely misleading
without a sentence saying so.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

NOTEBOOKS = Path("notebooks")

#: Notebooks whose stored outputs come from a real execution against project data.
#: Notebooks 01-05 are unexecuted single-cell wrappers and carry nothing to mislabel.
TRACKED = (
    "09_icu_admission_baseline.ipynb",
    "10_los_two_stage.ipynb",
    "11_deterioration_baseline.ipynb",
)

pytestmark = pytest.mark.skipif(
    not NOTEBOOKS.exists(), reason="notebooks/ not present in this checkout")


def _cells(name: str):
    path = NOTEBOOKS / name
    if not path.exists():
        pytest.skip(f"{name} not present")
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


def _has_outputs(cells) -> bool:
    return any(c.get("outputs") for c in cells)


@pytest.mark.parametrize("name", TRACKED)
def test_stored_outputs_carry_a_provenance_statement(name):
    cells = _cells(name)
    if not _has_outputs(cells):
        pytest.skip(f"{name} stores no outputs")

    head = "".join(
        "".join(c.get("source", [])) for c in cells[:2] if c.get("cell_type") == "markdown")
    assert "Provenance" in head or "supersedes" in head, (
        f"{name} stores executed outputs but its opening cells do not say where they "
        "came from. A reader cannot tell a current figure from a superseded one.")


@pytest.mark.parametrize("name", TRACKED)
def test_provenance_names_the_authoritative_report(name):
    cells = _cells(name)
    if not _has_outputs(cells):
        pytest.skip(f"{name} stores no outputs")

    head = "".join("".join(c.get("source", [])) for c in cells[:2])
    assert "reports/" in head, (
        f"{name} states provenance but does not point at the report that is "
        "authoritative. Provenance without a forwarding address still leaves the "
        "reader holding the notebook's number.")


def test_training_notebook_warns_that_rerunning_does_not_promote():
    """
    The specific trap in notebook 10: it calls `save_artifacts`.

    That writes to `models/`, not `models/best_models/`, so a well-meaning re-run
    produces artifacts that look promoted and are not — and silently invalidates the
    Phase 9 tier cutoffs at the same time.
    """
    cells = _cells("10_los_two_stage.ipynb")
    source = "".join("".join(c.get("source", [])) for c in cells)
    if "save_artifacts" not in source:
        pytest.skip("notebook 10 no longer saves artifacts")

    head = "".join("".join(c.get("source", [])) for c in cells[:2])
    assert "promote_models" in head, (
        "notebook 10 trains and saves artifacts but does not warn that promotion is "
        "a separate step. Re-running it changes nothing that is served, which is not "
        "what the output log implies.")
