"""
tests/test_llm_grounding.py
───────────────────────────
Tests for the grounded generation layer.

The central test is :class:`AdversarialLLMTests`: a backend that deliberately
fabricates must never have its output returned to the caller.
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scipy.stats  # noqa: E402,F401
import sklearn.feature_extraction.text  # noqa: E402,F401

if "torch" not in sys.modules:
    try:
        import torch  # noqa: F401
    except ImportError:
        _t = types.ModuleType("torch")
        _t.Tensor = type("Tensor", (), {})
        _t.set_num_threads = lambda n: None
        _t.load = lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no ckpt"))
        sys.modules["torch"] = _t

from src.llm.backends import NullBackend                      # noqa: E402
from src.llm.grounding import build_fact_store, verify_text   # noqa: E402
from src.llm.pipeline import ClinicalReportPipeline           # noqa: E402
from src.llm.report_composer import (                        # noqa: E402
    SYSTEM_CONSTANTS,
    compose_report,
)

PAYLOAD = {
    "demographics": {"age": 68, "gender": "M"},
    "primary_diagnosis": "Septic Shock",
    "comorbidities": ["AKI"],
    "presentation_labs": {
        "creatinine_max": 3.2, "bun_max": 54.0, "wbc_max": 26.5,
        "bicarbonate_min": 16.0, "sodium_min": 132.0, "potassium_max": 5.1,
        "platelets_min": 90.0, "hematocrit_min": 28.0, "glucose_max": 180.0,
    },
    "vital_signs": {"sbp_min": 82, "hr_max": 132},
    "active_medications": ["Levophed 4mg/250mL", "vanc"],
}

DOCS = [{
    "doc_id": "SSC2021-ABX",
    "citation": "[SCCM/ESICM Surviving Sepsis Campaign 2021, Antimicrobials]",
    "title": "Surviving Sepsis Campaign — Antimicrobials",
    "evidence_level": "Level 1: Clinical Practice Guidelines",
    "text": "Administer antimicrobials within 1 hour of recognition of septic shock.",
    "url": "https://example.org/ssc",
}]

PREDICTIONS = {"p_mortality": 0.0286, "p_deterioration": 0.4132,
               "risk_tier": "Tier 2: Moderate Risk"}


def _fs():
    return build_fact_store(payload=PAYLOAD, predictions=PREDICTIONS, documents=DOCS,
                            extra_numbers=SYSTEM_CONSTANTS)


class FactStoreTests(unittest.TestCase):
    def test_payload_numbers_are_known(self):
        fs = _fs()
        for v in (3.2, 54.0, 26.5, 82, 132, 68):
            self.assertIsNotNone(fs.knows_number(float(v)), f"{v} should be grounded")

    def test_probability_renderings_accepted(self):
        """0.0286 may legitimately be written as 2.86% or 2.9%."""
        fs = _fs()
        for v in (0.0286, 2.86, 2.9):
            self.assertIsNotNone(fs.knows_number(v))

    def test_unknown_number_rejected(self):
        self.assertIsNone(_fs().knows_number(0.45))

    def test_normalised_drug_identity_registered(self):
        """Charted as 'vanc' and 'Levophed'; both ingredients must be admissible."""
        fs = _fs()
        self.assertIn("vancomycin", fs.entities)
        self.assertIn("norepinephrine", fs.entities)


class VerifierTests(unittest.TestCase):
    def test_clean_text_passes(self):
        txt = ("Peak creatinine 3.2 mg/dL. Estimated mortality 2.9%. "
               "[SCCM/ESICM Surviving Sepsis Campaign 2021, Antimicrobials]")
        self.assertTrue(verify_text(txt, _fs()).ok)

    def test_invented_shap_value_rejected(self):
        txt = "Serum creatinine elevations (+0.45 SHAP) drive the mortality risk."
        r = verify_text(txt, _fs())
        self.assertFalse(r.ok)
        kinds = {v.kind for v in r.violations}
        self.assertIn("ungrounded_number", kinds)
        self.assertIn("shap_claim", kinds)

    def test_invented_citation_rejected(self):
        txt = "Hold nephrotoxins per [KDIGO 2023 AKI Guideline 3.1]."
        r = verify_text(txt, _fs())
        self.assertFalse(r.ok)
        self.assertIn("unknown_citation", {v.kind for v in r.violations})

    def test_invented_drug_rejected(self):
        txt = "Consider starting dobutamine for inotropic support."
        r = verify_text(txt, _fs())
        self.assertFalse(r.ok)
        self.assertIn("ungrounded_medication", {v.kind for v in r.violations})

    def test_patient_drug_accepted(self):
        txt = "The patient is receiving norepinephrine and vancomycin."
        self.assertTrue(verify_text(txt, _fs()).ok)

    def test_statistical_overclaim_rejected(self):
        txt = "Mortality was reduced (p < 0.05, 95% CI 1.2-3.4)."
        self.assertFalse(verify_text(txt, _fs()).ok)

    def test_certainty_overclaim_rejected(self):
        txt = "This intervention will prevent deterioration and is guaranteed to help."
        r = verify_text(txt, _fs())
        self.assertFalse(r.ok)
        self.assertIn("causal_overclaim", {v.kind for v in r.violations})

    def test_pmid_inside_citation_is_not_a_clinical_number(self):
        txt = "See [NCBI PubMed PMID: 28101605] for the consensus statement."
        r = verify_text(txt, build_fact_store(
            payload=PAYLOAD, predictions=PREDICTIONS,
            documents=DOCS + [{"doc_id": "PUBMED_PMID_28101605",
                               "citation": "[NCBI PubMed PMID: 28101605]",
                               "title": "consensus", "text": "consensus statement"}]))
        self.assertTrue(r.ok, r.render())

    def test_empty_generation_rejected(self):
        self.assertFalse(verify_text("", _fs()).ok)


class ComposerTests(unittest.TestCase):
    def test_composed_report_is_grounded_by_construction(self):
        rep = compose_report(PAYLOAD, PREDICTIONS, DOCS)
        check = verify_text(rep.to_markdown(), _fs())
        self.assertTrue(check.ok, check.render())

    def test_report_contains_required_sections(self):
        md = compose_report(PAYLOAD, PREDICTIONS, DOCS).to_markdown()
        for heading in ["Observed values", "Model risk estimates", "Retrieved evidence",
                        "Uncertainty", "Limitations", "Provenance"]:
            self.assertIn(heading, md)

    def test_no_predictions_is_stated_not_invented(self):
        md = compose_report(PAYLOAD, {}, DOCS).to_markdown()
        self.assertIn("No model predictions", md)

    def test_unrecognised_medication_surfaced(self):
        rep = compose_report(PAYLOAD, PREDICTIONS, DOCS,
                             ranked_medications=[{"raw": "zzz-drug", "recognised": False}])
        self.assertIn("not recognised", rep.to_markdown())
        self.assertTrue(rep.warnings)


class _HallucinatingBackend(NullBackend):
    """A backend that behaves like a badly-prompted LLM."""

    name = "hallucinating"

    def rephrase(self, text: str, system_prompt: str = "") -> str:
        return (text + "\n\n## Assessment\n\n"
                "Serum creatinine elevations (+0.45 SHAP) and leukocytosis (+0.38 SHAP) "
                "drive risk. Per [KDIGO 2023 AKI Guideline 3.1], start dobutamine. "
                "This will prevent deterioration (p < 0.01).")


class _TruncatingBackend(NullBackend):
    name = "truncating"

    def rephrase(self, text: str, system_prompt: str = "") -> str:
        return ""


class AdversarialLLMTests(unittest.TestCase):
    """A fabricating model must never reach the caller."""

    class _StubRag:
        def search_unseen_patient_rag(self, payload, case_id="c", require_complete=True):
            return {"status": "ok", "documents": DOCS, "ranked_medications": [],
                    "twin_status": "cohort_embeddings_missing", "retrieval_errors": []}

    class _StubRunner:
        lgbm_models: dict = {}

        def run_live_inference_with_uncertainty(self, payload):
            return dict(PREDICTIONS)

    def _pipeline(self, backend):
        return ClinicalReportPipeline(
            model_runner=self._StubRunner(), rag_store=self._StubRag(),
            llm_backend=backend,
        )

    def test_hallucinated_rephrase_is_rejected(self):
        r = self._pipeline(_HallucinatingBackend()).generate(PAYLOAD)
        self.assertEqual(r.generation_mode, "deterministic_llm_rejected")
        self.assertNotIn("0.45 SHAP", r.report_markdown)
        self.assertNotIn("KDIGO 2023", r.report_markdown)
        self.assertTrue(any("rejected by grounding" in w for w in r.warnings))

    def test_rejected_output_still_returns_usable_report(self):
        r = self._pipeline(_HallucinatingBackend()).generate(PAYLOAD)
        self.assertEqual(r.status, "ok")
        self.assertIn("Retrieved evidence", r.report_markdown)

    def test_empty_rephrase_is_rejected(self):
        r = self._pipeline(_TruncatingBackend()).generate(PAYLOAD)
        self.assertEqual(r.generation_mode, "deterministic_llm_rejected")
        self.assertIn("Observed values", r.report_markdown)

    def test_null_backend_passes_through_verified(self):
        r = self._pipeline(NullBackend()).generate(PAYLOAD)
        self.assertEqual(r.generation_mode, "llm_rephrased_verified")
        self.assertTrue(r.grounding["ok"])


class PipelineGateTests(unittest.TestCase):
    class _StubRag:
        def search_unseen_patient_rag(self, payload, case_id="c", require_complete=True):
            return {"status": "ok", "documents": DOCS, "ranked_medications": []}

    def test_incomplete_payload_asks_and_does_not_generate(self):
        p = ClinicalReportPipeline(rag_store=self._StubRag())
        r = p.generate({"primary_diagnosis": "Septic Shock"})
        self.assertEqual(r.status, "incomplete_input")
        self.assertTrue(r.question_for_user)
        self.assertEqual(r.report_markdown, "")

    def test_low_feature_coverage_withholds_predictions(self):
        class _Runner:
            class _M:
                class _B:
                    @staticmethod
                    def feature_name():
                        return [f"f{i}" for i in range(100)]
                booster_ = _B()
            lgbm_models = {"mortality": _M()}

            def run_live_inference_with_uncertainty(self, payload):
                return {"p_mortality": 0.9, "risk_tier": "Tier 4: Extreme Risk"}

            def _convert_payload_to_series(self, payload):
                import pandas as pd
                return pd.Series({"f0": 1.0, "f1": 2.0})    # 2% coverage

        p = ClinicalReportPipeline(model_runner=_Runner(), rag_store=self._StubRag())
        r = p.generate(PAYLOAD)
        self.assertNotIn("p_mortality", r.predictions)
        self.assertIn("withheld_reason", r.predictions)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class ExplainabilityScreenTests(unittest.TestCase):
    """The Phase 8 leakage screen must fire on removed families and not on _24h."""

    def test_removed_families_are_flagged(self):
        from scripts.evaluation.run_explainability_audit import screen_feature
        for feat in ["med_class_opioid", "sentence_count", "medical_keyword_count",
                     "negation_count", "lab_bicarbonate_min", "lab_wbc_max",
                     "lab_creatinine_first", "lab_unique_items", "lab_total_count",
                     "cci_malignancy", "dx_sepsis", "icu_los_days", "los_days"]:
            with self.subTest(feat=feat):
                self.assertIsNotNone(screen_feature(feat),
                                     f"{feat} should be flagged as a removed leak family")

    def test_windowed_and_benign_features_pass(self):
        from scripts.evaluation.run_explainability_audit import screen_feature
        for feat in ["lab_wbc_max_24h", "lab_bicarbonate_min_24h", "lab_total_count_24h",
                     "lab_unique_items_24h", "anchor_age", "gender_M",
                     "admission_type_EW EMER.", "admission_location_PHYSICIAN REFERRAL"]:
            with self.subTest(feat=feat):
                self.assertIsNone(screen_feature(feat),
                                  f"{feat} is legitimate and must not be flagged")


class CitationNumberTests(unittest.TestCase):
    """Numbers inside URLs and DOIs are bibliographic, not clinical claims."""

    def _checked(self, text):
        from src.llm.grounding import _NUMBER_RE, _URL_RE, _CITATION_RE
        spans = [(m.start(), m.end()) for m in _CITATION_RE.finditer(text)]
        spans += [(m.start(), m.end()) for m in _URL_RE.finditer(text)]
        return [m.group(1) for m in _NUMBER_RE.finditer(text)
                if not any(a <= m.start() < b for a, b in spans)]

    def test_doi_and_url_digits_are_not_treated_as_clinical_values(self):
        text = ("Reference https://www.ahajournals.org/doi/10.1161/CIR.0000000000001038 "
                "and doi 10.1001/jama.2016.0287.")
        self.assertEqual(self._checked(text), [],
                         "DOI/URL identifiers must not be checked as clinical numbers")

    def test_real_clinical_values_are_still_checked(self):
        text = ("See https://example.org/doi/10.1161/CIR.0000000000001038 — "
                "peak creatinine 9.9 mg/dL and lowest bicarbonate 12 mEq/L.")
        self.assertEqual(self._checked(text), ["9.9", "12"],
                         "clinical values outside URLs must still be verified")


def test_outcome_window_labels_are_not_ungrounded_numbers():
    """
    "30-day readmission" names an outcome; its 30 is not a patient value.

    This was latent for as long as predictions were withheld from every report by the
    payload coverage floor — the risk table is the only place these labels appear. As
    soon as a report displayed its own predictions, the verifier rejected it for the
    numeral in its column heading.
    """
    from src.llm.grounding import build_fact_store, verify_text

    store = build_fact_store(payload={}, predictions={"p_readmission": 0.17},
                             documents=[])
    text = ("| Task | Calibrated probability |\n"
            "| 30-day readmission | 17.0% |\n"
            "| Clinical deterioration within 6 hours | 17.0% |\n")
    result = verify_text(text, store)
    bad = [v for v in result.violations
           if v.kind == "ungrounded_number" and v.detail.split()[1] in ("30", "6")]
    assert not bad, f"outcome-window labels flagged as claims: {bad}"


def test_a_genuinely_invented_number_is_still_caught():
    """The label exemption must not open a hole: bare numbers still fail."""
    from src.llm.grounding import build_fact_store, verify_text

    store = build_fact_store(payload={}, predictions={"p_readmission": 0.17},
                             documents=[])
    result = verify_text("The patient's creatinine was 7.4 mg/dL.", store)
    assert any(v.kind == "ungrounded_number" for v in result.violations)
