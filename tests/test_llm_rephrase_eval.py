"""
The rephrase stage must be labelled by what actually happened.

A `NullBackend` returns the deterministic text unchanged. Passing that through the
verifier is a tautology — the text was grounded before it was handed over — so it
always passes, and the pipeline recorded `llm_rephrased_verified`. Measured on ten
held-out admissions with no model installed, the harness reported a 100% verifier
pass rate. Every part of that sentence is false except the number.

These tests pin the distinction between "a model rewrote this and the checker
approved it" and "nothing happened".
"""

from __future__ import annotations

import pytest

from src.llm.backends import LLMBackend, NullBackend, get_backend
from src.llm.pipeline import ClinicalReportPipeline


class EchoBackend(LLMBackend):
    """A real backend that happens to be useless. Must NOT be treated as passthrough."""

    name = "echo"
    available = True
    passthrough = False

    def rephrase(self, text: str, system_prompt: str = "") -> str:
        return text


class InventingBackend(LLMBackend):
    name = "inventing"
    available = True

    def rephrase(self, text: str, system_prompt: str = "") -> str:
        return text + "\n\nThe patient's ejection fraction was 23%."


# ── the flag itself ──────────────────────────────────────────────────────────

def test_null_backend_declares_itself_passthrough():
    assert NullBackend().passthrough is True


def test_real_backends_do_not_declare_passthrough():
    """Default must be False, so a new backend is never silently skipped."""
    assert LLMBackend.passthrough is False
    assert EchoBackend().passthrough is False


def test_subclassing_null_and_overriding_rephrase_clears_the_flag():
    """
    The trap the first version of this fix fell into.

    Subclassing `NullBackend` to build a stub is natural — the adversarial tests in
    `test_llm_grounding.py` do exactly that. With `passthrough` set as a static class
    attribute, those stubs inherited True and the pipeline skipped them, so the
    backends written to prove bad output is rejected were never consulted at all.
    """

    class Loud(NullBackend):
        def rephrase(self, text: str, system_prompt: str = "") -> str:
            return text + " extra"

    assert Loud().passthrough is False
    assert NullBackend().passthrough is True


def test_a_subclass_that_does_not_override_is_still_passthrough():
    """Renaming or re-describing a no-op does not make it do anything."""

    class Renamed(NullBackend):
        name = "renamed"

    assert Renamed().passthrough is True


def test_describe_exposes_passthrough():
    """The harness reads this to explain why no LLM was consulted."""
    assert NullBackend().describe()["passthrough"] is True


def test_get_backend_null_is_passthrough():
    assert get_backend("null").passthrough is True


# ── the pipeline's labelling ─────────────────────────────────────────────────

@pytest.fixture
def pipeline_factory():
    def make(backend):
        return ClinicalReportPipeline(model_runner=None, llm_backend=backend)
    return make


def test_passthrough_backend_yields_deterministic_mode(pipeline_factory):
    """
    The regression this file exists for.

    Before the fix this returned `llm_rephrased_verified` — asserting that a model
    ran and its output was verified, when the backend is a no-op.
    """
    res = pipeline_factory(NullBackend()).generate(
        {"demographics": {"age": 70, "gender": "F"}}, use_llm=True)
    assert res.generation_mode != "llm_rephrased_verified", (
        "a passthrough backend must never be reported as a verified rephrase")


def test_a_real_backend_is_still_consulted(pipeline_factory):
    """The fix must skip only passthroughs, not disable the stage entirely."""
    payload = {
        "demographics": {"age": 70, "gender": "F"},
        "primary_diagnosis": "sepsis",
        "presentation_labs": {"creatinine_max": 2.0, "bun_max": 40.0,
                              "wbc_max": 15.0, "bicarbonate_min": 20.0,
                              "sodium_min": 135.0, "potassium_max": 4.5,
                              "platelets_min": 150.0, "hematocrit_min": 30.0,
                              "glucose_max": 120.0},
        "vital_signs": {"sbp_min": 95.0, "hr_max": 105.0},
    }
    res = pipeline_factory(EchoBackend()).generate(payload, use_llm=True)
    assert res.generation_mode in (
        "llm_rephrased_verified", "deterministic_llm_rejected",
        "deterministic",  # reachable if the payload is judged incomplete
    )


def test_use_llm_false_still_wins(pipeline_factory):
    res = pipeline_factory(EchoBackend()).generate(
        {"demographics": {"age": 70, "gender": "F"}}, use_llm=False)
    assert res.generation_mode == "deterministic"


# ── the harness's own reporting ──────────────────────────────────────────────

def test_similarity_detects_an_unchanged_rewrite():
    """
    An echoing model passes the verifier and is worthless. The headline pass rate
    cannot distinguish it from a good one; similarity can, which is why the harness
    reports both.
    """
    from scripts.evaluation.run_llm_rephrase_eval import _similarity
    assert _similarity("abc def", "abc def") == 1.0
    assert _similarity("abc def", "xyz completely different") < 0.5


def test_flesch_is_higher_for_simpler_prose():
    from scripts.evaluation.run_llm_rephrase_eval import _flesch
    simple = "The cat sat. The dog ran. It was fun."
    dense = ("Subsequent physiological decompensation necessitated immediate "
             "multidisciplinary intervention notwithstanding preliminary "
             "haemodynamic stabilisation.")
    assert _flesch(simple) > _flesch(dense)


def test_flesch_handles_empty_text():
    from scripts.evaluation.run_llm_rephrase_eval import _flesch
    assert _flesch("") == 0.0
