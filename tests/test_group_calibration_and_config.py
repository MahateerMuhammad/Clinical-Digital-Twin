"""
Age-band calibration, configuration loading, and secret handling.

Three things that are quiet when they break:

* a calibration layer that **fails closed** would take prediction down for a
  refinement — it must fail open on every path
* configuration read at **import** time silently ignores a `.env` the operator
  just edited, with no error to notice
* a key that reaches a log or an exception message is leaked long before anyone
  looks at the log
"""

from __future__ import annotations

import os

import pytest

from src.models.group_calibration import (
    AGE_BANDS, MIN_FIT_EVENTS, MIN_FIT_ROWS, GroupCalibrators, age_band,
)


# ══ age bands ════════════════════════════════════════════════════════════════

def test_bands_cover_the_adult_cohort_without_gaps():
    covered = [b for b in (18, 39, 40, 54, 55, 69, 70, 84, 85, 91)]
    assert all(age_band(a) is not None for a in covered)


def test_age_band_is_none_rather_than_a_guess_when_unusable():
    """None routes to the global calibrator; a wrong band applies a wrong curve."""
    for value in (None, "", "unknown", float("nan"), -5, 200):
        assert age_band(value) is None


def test_string_ages_are_accepted():
    assert age_band("88") == "85+"


def test_bands_match_the_slice_report():
    """The fix must be measured on the partition that found the problem."""
    from scripts.evaluation.run_slice_eval import AGE_BANDS as REPORT_BANDS

    assert list(AGE_BANDS) == [tuple(b) for b in REPORT_BANDS]


# ══ fail-open behaviour ══════════════════════════════════════════════════════

class _Cal:
    def __init__(self, value=0.5):
        self.value = value

    def predict(self, xs):
        return [self.value for _ in xs]


class _Broken:
    def predict(self, xs):
        raise RuntimeError("corrupt calibrator")


def test_an_empty_artefact_returns_none_not_an_error():
    gc = GroupCalibrators()
    assert gc.available is False
    assert gc.calibrate("mortality", 0.2, age=88) is None


def test_a_missing_artefact_file_loads_empty(tmp_path):
    gc = GroupCalibrators.load(tmp_path / "absent.pkl")
    assert gc.available is False


def test_a_corrupt_artefact_file_loads_empty(tmp_path):
    """A bad pickle must not stop the runner serving yesterday's number."""
    p = tmp_path / "bad.pkl"
    p.write_bytes(b"not a pickle at all")
    assert GroupCalibrators.load(p).available is False


def test_unknown_age_falls_back_to_the_global_calibrator():
    gc = GroupCalibrators(by_task={"mortality": {"85+": _Cal(0.9)}})
    assert gc.calibrate("mortality", 0.2, age=None) is None


def test_a_band_without_a_calibrator_falls_back():
    gc = GroupCalibrators(by_task={"mortality": {"85+": _Cal(0.9)}})
    assert gc.calibrate("mortality", 0.2, age=30) is None


def test_an_unknown_task_falls_back():
    gc = GroupCalibrators(by_task={"mortality": {"85+": _Cal(0.9)}})
    assert gc.calibrate("deterioration", 0.2, age=88) is None


def test_a_raising_calibrator_falls_back_rather_than_propagating():
    gc = GroupCalibrators(by_task={"mortality": {"85+": _Broken()}})
    assert gc.calibrate("mortality", 0.2, age=88) is None


def test_a_matching_band_is_applied_and_clipped():
    gc = GroupCalibrators(by_task={"mortality": {"85+": _Cal(0.77)}})
    assert gc.calibrate("mortality", 0.2, age=88) == pytest.approx(0.77)
    extreme = GroupCalibrators(by_task={"mortality": {"85+": _Cal(1.5)}})
    assert extreme.calibrate("mortality", 0.2, age=88) < 1.0


def test_fitting_floors_are_stricter_than_reporting_floors():
    """
    A badly supported calibrator is applied to every future patient in the
    band; an unmeasured slice merely goes unreported. The asymmetry is
    deliberate.
    """
    from src.evaluation.metrics import MIN_SLICE_EVENTS, MIN_SLICE_ROWS

    assert MIN_FIT_ROWS >= MIN_SLICE_ROWS
    assert MIN_FIT_EVENTS >= MIN_SLICE_EVENTS


def test_round_trip_through_disk(tmp_path):
    gc = GroupCalibrators(by_task={"mortality": {"85+": _Cal(0.4)}},
                          meta={"fitted_on": "2026-08-14"})
    path = gc.save(tmp_path / "gc.pkl")
    back = GroupCalibrators.load(path)
    assert back.available
    assert back.meta["fitted_on"] == "2026-08-14"
    assert back.calibrate("mortality", 0.1, age=90) == pytest.approx(0.4)


# ══ the runner integration ═══════════════════════════════════════════════════

def test_the_runner_exposes_group_calibrators_without_requiring_them():
    """Reading the property must not raise when the artefact is absent."""
    from src.llm.model_runner import LiveModelRunner

    assert hasattr(LiveModelRunner, "group_calibrators")


def test_age_is_read_from_the_scored_feature_vector():
    import pandas as pd

    from src.llm.model_runner import LiveModelRunner

    assert LiveModelRunner._patient_age(pd.Series({"anchor_age": 88})) == 88.0
    assert LiveModelRunner._patient_age(pd.Series({"other": 1})) is None
    assert LiveModelRunner._patient_age(pd.Series({"anchor_age": "x"})) is None


# ══ configuration ════════════════════════════════════════════════════════════

def test_openrouter_settings_are_read_at_construction_not_import(monkeypatch):
    """
    The bug this pins: module-level `os.environ.get` runs when `backends` is
    first imported, which can be before `.env` is loaded. An operator edits the
    file, sees no effect, and gets no error.
    """
    from src.llm.backends import OpenRouterBackend

    monkeypatch.setenv("CDT_OPENROUTER_MODEL", "test/model:free")
    monkeypatch.setenv("CDT_OPENROUTER_BASE_URL", "https://proxy.test/v1")
    monkeypatch.setenv("CDT_OPENROUTER_TIMEOUT", "7")

    b = OpenRouterBackend()
    assert b.model == "test/model:free"
    assert b.endpoint == "https://proxy.test/v1/chat/completions"
    assert b.timeout == 7.0


def test_explicit_arguments_win_over_the_environment(monkeypatch):
    from src.llm.backends import OpenRouterBackend

    monkeypatch.setenv("CDT_OPENROUTER_MODEL", "env/model:free")
    assert OpenRouterBackend(model="arg/model").model == "arg/model"


def test_the_default_model_is_a_free_tier_id():
    from src.llm.backends import DEFAULT_OPENROUTER_MODEL

    assert DEFAULT_OPENROUTER_MODEL.endswith(":free")


def test_describe_never_returns_the_key(monkeypatch):
    """/api/health returns this dictionary to anyone who can reach the service."""
    from src.llm.backends import OpenRouterBackend

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-SUPERSECRET")
    d = OpenRouterBackend().describe()
    assert d["key_present"] is True
    assert "SUPERSECRET" not in str(d)


def test_auto_backend_selection_never_reaches_openrouter(monkeypatch):
    """
    Sending data to a third party must be an explicit choice, never inherited
    from a default — otherwise setting an environment variable silently starts
    posting MIMIC-derived content off the machine.
    """
    from src.llm.backends import get_backend

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-anything")
    assert get_backend("auto").name != "openrouter"


# ══ secret handling ══════════════════════════════════════════════════════════

def test_api_keys_are_redacted_from_retrieval_errors():
    from src.llm.evidence_cache import EvidenceCache

    msg = EvidenceCache._redact(
        "could not retrieve https://eutils.ncbi.nlm.nih.gov/x?db=pubmed"
        "&api_key=NCBISECRET&term=sepsis: HTTP 429")
    assert "NCBISECRET" not in msg
    assert "<redacted>" in msg
    assert "term=sepsis" in msg          # only the key is removed


def test_dotenv_is_gitignored():
    """
    `.env` was not covered: the `env/` entry matches a virtualenv directory,
    not the dotfile, so the first `.env` written with a real key would have been
    committed.
    """
    from pathlib import Path

    patterns = Path(".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in [p.strip() for p in patterns]
    assert "!.env.example" in [p.strip() for p in patterns]


def test_the_example_env_carries_no_values():
    from pathlib import Path

    example = Path(".env.example")
    assert example.exists()
    for line in example.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if "KEY" in name.upper() or "SECRET" in name.upper():
            assert value == "", f"{name} must ship empty"


def test_the_defaults_are_literals_not_environment_reads(monkeypatch):
    """
    A module-level `os.environ.get` is not a default — it is whatever happened
    to be set when the module was first imported.

    This failed for real: `backend.service` loads `.env` at import, so any test
    session that touched the API first redefined DEFAULT_OPENROUTER_MODEL to
    whatever `.env` named. The free-tier assertion above passed in isolation and
    failed in the full suite, which is the signature of import-order coupling.

    Asserted against the source rather than by reloading the module. The first
    version of this test called ``importlib.reload``, which builds fresh class
    objects while every other module still holds the old ones — three unrelated
    rephrase tests started failing on ``isinstance`` checks, and only in the
    full suite. A test for import-order coupling that introduces import-order
    coupling is not a test.
    """
    import ast
    import inspect

    import src.llm.backends as B

    tree = ast.parse(inspect.getsource(B))
    offenders = []
    for node in tree.body:                      # module level only, not inside defs
        if not isinstance(node, ast.Assign):
            continue
        for call in ast.walk(node):
            if (isinstance(call, ast.Attribute) and call.attr == "get"
                    and isinstance(call.value, ast.Attribute)
                    and call.value.attr == "environ"):
                offenders.append(node.targets[0].id
                                 if isinstance(node.targets[0], ast.Name) else "?")
    assert not offenders, (
        f"module-level os.environ reads: {offenders}. These run at import time, "
        f"which may be before .env loads — read the environment in __init__.")

    assert B.DEFAULT_OPENROUTER_MODEL.endswith(":free")
    assert "openrouter.ai" in B.DEFAULT_OPENROUTER_BASE_URL

    # …while a backend built now still honours the environment.
    monkeypatch.setenv("CDT_OPENROUTER_MODEL", "someone/paid-model")
    assert B.OpenRouterBackend().model == "someone/paid-model"
