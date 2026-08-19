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
from src.llm.grounding import (                               # noqa: E402
    build_fact_store, verify_rephrase, verify_text)
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
        for heading in ["What the models estimate", "What the guidelines say",
                        "What this cannot tell you", "Appendix"]:
            self.assertIn(heading, md)

    def test_no_predictions_is_stated_not_invented(self):
        md = compose_report(PAYLOAD, {}, DOCS).to_markdown()
        self.assertIn("No model predictions", md)

    def test_unrecognised_medication_surfaced(self):
        rep = compose_report(PAYLOAD, PREDICTIONS, DOCS,
                             ranked_medications=[{"raw": "zzz-drug", "recognised": False}])
        self.assertIn("not recognised", rep.to_markdown())
        self.assertTrue(rep.warnings)


_DRIVERS = [
    {"feature": "anchor_age", "label": "Age",
     "contribution": 1.445, "value": 68.0, "supplied": True},
    {"feature": "diagnosis_count", "label": "Number of coded diagnoses",
     "contribution": 0.770, "value": None, "supplied": False},
    {"feature": "lab_wbc_last_24h", "label": "White cells, last 24h",
     "contribution": -0.225, "value": 26.5, "supplied": True},
]


class ClinicianReadabilityTests(unittest.TestCase):
    """
    The report is read by a clinician, not by the person who built the system.

    Every assertion here corresponds to something a real rendering actually did:
    log-odds printed as the explanation, a probability of 0.0% for an outcome
    with a 2.19% base rate, an internal error record rendered as a guideline, and
    a raw database code where a word belonged.
    """

    def _md(self, predictions=None, **kw):
        return compose_report(PAYLOAD, predictions if predictions is not None
                              else {**PREDICTIONS, "drivers": _DRIVERS},
                              DOCS, **kw).to_markdown()

    def _clinical_part(self, md: str) -> str:
        return md.split("## Appendix")[0]

    def test_the_answer_comes_before_the_restatement_of_the_input(self):
        md = self._md()
        self.assertLess(md.index("Estimated risk of dying"), md.index("Values used"),
                        "the estimate must precede the echo of the supplied values")

    def test_log_odds_never_appear_in_the_clinical_body(self):
        body = self._clinical_part(self._md())
        self.assertNotIn("log-odds", body)
        self.assertNotIn("SHAP", body)
        self.assertNotIn("+1.445", body)
        # but the arithmetic is still auditable
        self.assertIn("+1.445", self._md())

    def test_an_unsupplied_driver_is_not_shown_as_a_patient_finding(self):
        body = self._clinical_part(self._md())
        absent_heading = "Counted, but not measured in this patient"
        self.assertIn(absent_heading, body)
        self.assertGreater(body.index("Number of coded diagnoses"),
                           body.index(absent_heading),
                           "an unsupplied feature must sit under the absence heading")

    def test_a_vanishing_probability_is_not_rendered_as_zero(self):
        """
        7.2e-07 printed as "0.0%" reads as "this will not happen".

        The deterioration model returns exactly this on a presentation payload,
        against a 2.19% base rate in its own training cohort.
        """
        md = self._md({**PREDICTIONS, "p_deterioration": 7.2e-07})
        self.assertNotIn("0.0%", md)
        self.assertIn("near zero", md)
        self.assertIn("not as an assurance the outcome will not happen", md)

    def test_a_record_that_failed_its_integrity_check_is_not_shown_as_guidance(self):
        failed = {"doc_id": "FDA_DAILYMED_vancomycin",
                  "citation": "[NIH DailyMed FDA Label: VANCOMYCIN]",
                  "title": "Citation Integrity Check Failed",
                  "evidence_level": "Level 2: FDA Medication Labels",
                  "text": "Level 2 evidence withheld, citation integrity check failed"}
        md = compose_report(PAYLOAD, PREDICTIONS, DOCS + [failed]).to_markdown()
        guidance = md.split("## Appendix")[0]
        self.assertNotIn("Citation Integrity Check Failed", guidance)
        self.assertIn("Records retrieved but not shown", md)

    def test_a_withheld_task_says_what_that_means_before_it_says_why(self):
        md = self._md({**PREDICTIONS,
                       "withheld_tasks": {"p_los_over_5_63d": "AUROC 0.731 against 0.900"}})
        body = self._clinical_part(md)
        self.assertIn("not reported", body)
        self.assertIn("It does not mean the risk is low", body)
        self.assertNotIn("AUROC", body, "the audit belongs in the appendix")
        self.assertIn("AUROC", md)

    def test_sex_is_a_word_not_a_database_code(self):
        md = compose_report({**PAYLOAD, "demographics": {"age": 68, "gender": "F"}},
                            PREDICTIONS, DOCS).to_markdown()
        self.assertIn("68-year-old female", md)

    def test_an_empty_section_is_omitted_rather_than_rendered_empty(self):
        md = self._md()
        self.assertNotIn("Medications on the list", md)

    def test_a_twin_failure_never_prints_an_exception_string(self):
        md = self._md(twin_status="projection_unavailable: No module named 'src.models.x'")
        self.assertNotIn("No module named", md)
        self.assertIn("unavailable in this deployment", md)


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
        self.assertIn("What the guidelines say", r.report_markdown)

    def test_empty_rephrase_is_rejected(self):
        r = self._pipeline(_TruncatingBackend()).generate(PAYLOAD)
        self.assertEqual(r.generation_mode, "deterministic_llm_rejected")
        self.assertIn("Values used", r.report_markdown)

    def test_null_backend_reports_deterministic_not_verified(self):
        """
        A no-op backend must not be reported as a verified LLM rephrase.

        This previously asserted `llm_rephrased_verified`, and the assertion was
        wrong rather than the code. NullBackend returns the deterministic text
        unchanged; running that through the verifier is a tautology, because the
        text was grounded before it was handed over. The mode is read as evidence
        that a model produced the prose and a checker approved it, and neither
        happened.

        It was not academic: `run_llm_rephrase_eval.py --backend null` reported a
        100% verifier pass rate over ten held-out admissions with no model
        installed. The pipeline now skips a passthrough backend entirely.
        """
        r = self._pipeline(NullBackend()).generate(PAYLOAD)
        self.assertEqual(r.generation_mode, "deterministic")
        self.assertIn("Values used", r.report_markdown)


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


class VerifierBlindSpotTests(unittest.TestCase):
    """
    Failures a fact-store check cannot see, and the narrowest fixes for them.

    Everything above tests the verifier against text that *invents* something.
    These test it against text that keeps every number and every citation
    intact and still changes what the document says — the class of failure a
    rephrasing model is most likely to produce, because it is the class its
    instructions do not obviously forbid.
    """

    #: A guideline passage carrying a sub-1.0 threshold, a causal clause and a
    #: confidence interval — all three legitimate in a source, none of them
    #: things the system may assert in its own voice.
    QUOTE = ("In patients with stage 2 or 3 AKI, discontinue nephrotoxic agents. "
             "Acute kidney injury caused by sepsis carries higher mortality "
             "(95% CI 1.2-2.4). A creatinine rise of 0.3 mg/dL within 48 hours "
             "defines stage 1.")

    def _store(self):
        return build_fact_store(
            payload={"creatinine_max": 3.2},
            predictions={"p_mortality": 0.34},
            documents=[{"doc_id": "KDIGO-AKI", "citation": "KDIGO AKI 2012 2.1",
                        "title": "KDIGO AKI", "text": self.QUOTE}],
            extra_numbers={"phase9_tier3_observed_mortality_pct": 3.79},
        )

    def test_a_guideline_threshold_does_not_license_its_hundredfold(self):
        """
        0.3 mg/dL in a retrieved document must not ground "30%".

        The percentage expansion was applied to every value in [0, 1] whatever
        it meant, so KDIGO's stage-1 creatinine threshold registered 30.0 as a
        permissible number. Evidence numbers are stored un-scoped to their
        document, so any report could then state 30% of anything and verify
        clean, with provenance pointing at a renal guideline.
        """
        r = verify_text("Estimated mortality 30%.", self._store())
        self.assertFalse(r.ok)
        self.assertIn("ungrounded_number", {v.kind for v in r.violations})

    def test_a_probability_is_still_rendered_as_a_percentage(self):
        """The expansion is scoped, not removed: predictions keep both forms."""
        store = self._store()
        for text in ("Estimated mortality 34.0%.", "Mortality probability 0.34.",
                     "3.79% of patients in this band died in hospital."):
            self.assertTrue(verify_text(text, store).ok, text)

    def test_an_evidence_number_is_quotable_as_written(self):
        r = verify_text("A rise of 0.3 mg/dL defines stage 1.", self._store())
        self.assertTrue(r.ok)

    def test_an_attribution_restated_as_a_cause_is_rejected(self):
        """
        SHAP says what moved the model, not what moved the patient.

        No number changes between "creatinine contributed most" and "creatinine
        drove the mortality risk", and no citation is invented, so every other
        check in this module passes both.
        """
        for text in ("Elevated creatinine of 3.2 drove the mortality risk to 34.0%.",
                     "Renal failure caused the estimate of 34.0%.",
                     "The creatinine of 3.2 resulted in a higher estimate."):
            r = verify_text(text, self._store())
            self.assertFalse(r.ok, text)
            self.assertIn("causal_overclaim", {v.kind for v in r.violations})

    def test_attribution_and_association_language_survives(self):
        """
        The correct wording must not be collateral damage.

        A verifier that rejects the language the composer actually emits does
        not make the system stricter, it makes the check something its owner
        turns off.
        """
        for text in ("Creatinine, peak 3.2 made the largest contribution.",
                     "A creatinine of 3.2 is associated with a higher estimate.",
                     "Estimated mortality 34.0%, seen alongside a creatinine of 3.2.",
                     # The composer's own driver disclaimer. A rule that flags
                     # this withholds every scored report for carrying the
                     # warning against the thing the rule looks for.
                     "These are associations learned from past admissions, not "
                     "causes, and changing one would not change the outcome."):
            self.assertTrue(verify_text(text, self._store()).ok, text)

    def test_a_verbatim_guideline_quotation_is_the_source_speaking(self):
        """
        Guideline authors write "caused by" and quote intervals; reproducing
        them faithfully is not the system making the claim.
        """
        self.assertTrue(verify_text("> " + self.QUOTE, self._store()).ok)

    def test_a_claim_cannot_hide_inside_a_fabricated_quotation(self):
        """The exemption is earned by matching the corpus, not by a '>'."""
        r = verify_text("> Creatinine drove the mortality risk in this patient.",
                        self._store())
        self.assertFalse(r.ok)
        self.assertIn("causal_overclaim", {v.kind for v in r.violations})


class RephraseOmissionTests(unittest.TestCase):
    """
    ``verify_text`` inspects only what it finds, so it can only ever catch what
    a generator added. These cover what a generator removed.
    """

    SOURCE = ("Estimated mortality 34.0%. Readmission 21.0%. "
              "Hold nephrotoxic agents [KDIGO AKI 2012 2.1].")

    def _store(self):
        return build_fact_store(
            payload={}, predictions={"p_mortality": 0.34, "p_readmission": 0.21},
            documents=[{"doc_id": "KDIGO-AKI", "citation": "KDIGO AKI 2012 2.1",
                        "title": "KDIGO AKI", "text": "Hold nephrotoxic agents."}])

    def test_a_faithful_rewording_passes(self):
        cand = ("Estimated mortality is 34.0% and readmission 21.0%. "
                "Nephrotoxic agents should be held [KDIGO AKI 2012 2.1].")
        self.assertTrue(verify_rephrase(cand, self.SOURCE, self._store()).ok)

    def test_a_number_written_out_in_words_is_rejected(self):
        """
        "0.34" rephrased as "roughly a third" is not a mismatch — it is not a
        number, so nothing is checked and the report passes on a claim the
        verifier never read.
        """
        cand = ("Mortality is roughly a third. Readmission 21.0%. "
                "Hold nephrotoxic agents [KDIGO AKI 2012 2.1].")
        r = verify_rephrase(cand, self.SOURCE, self._store())
        self.assertFalse(r.ok)
        self.assertIn("numbers_dropped", {v.kind for v in r.violations})

    def test_a_dropped_citation_is_rejected(self):
        """Every surviving citation still verifies; the recommendation is now
        detached from the guideline that carries it."""
        cand = "Estimated mortality 34.0%. Readmission 21.0%. Hold nephrotoxic agents."
        r = verify_rephrase(cand, self.SOURCE, self._store())
        self.assertFalse(r.ok)
        self.assertIn("citations_dropped", {v.kind for v in r.violations})

    def test_a_plain_fact_store_check_accepts_all_of_these(self):
        """
        The point of the differential: on the fact store alone every one of
        these is clean, which is why the comparison exists.
        """
        store = self._store()
        for cand in ("Mortality is roughly a third.",
                     "Estimated mortality 34.0%. Hold nephrotoxic agents."):
            self.assertTrue(verify_text(cand, store).ok, cand)


def test_an_age_is_written_the_way_it_is_spoken():
    """Extraction yields a float; "88.0-year-old" is spreadsheet, not speech."""
    md = compose_report({**PAYLOAD, "demographics": {"age": 88.0, "gender": "F"}},
                        PREDICTIONS, DOCS).to_markdown()
    assert "88-year-old" in md
    assert "88.0-year-old" not in md
