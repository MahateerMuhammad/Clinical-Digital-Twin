"""
src/llm/pipeline.py
───────────────────
End-to-end grounded clinical decision-support pipeline.

    payload
      → completeness gate      (refuse + ask if insufficient)
      → model inference        (calibrated, with feature-coverage guard)
      → evidence retrieval     (tiered, cited)
      → deterministic composer (fully grounded by construction)
      → optional LLM rephrase  (readability only)
      → grounding verifier     (reject rephrase if it invents anything)

The LLM is the only optional stage, and it is the only stage that can fail closed:
if its output does not verify, the deterministic text is returned instead. There
is no path in which a language model's unverified prose reaches the caller.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.llm.grounding import build_fact_store, verify_text
from src.llm.payload_validation import validate_payload
from src.llm.report_composer import SYSTEM_CONSTANTS, compose_report

__all__ = ["PipelineResult", "ClinicalReportPipeline"]

#: Below this fraction of a model's trained features, a prediction is withheld.
#: Zero-filling unsupplied features drove ICU-admission estimates to 0.89 against
#: a 0.16 base rate, so a low-coverage prediction is worse than none.
MIN_FEATURE_COVERAGE = 0.30


@dataclass
class PipelineResult:
    status: str                       # ok | incomplete_input | no_evidence | refused
    report_markdown: str = ""
    question_for_user: str = ""
    predictions: Dict[str, Any] = field(default_factory=dict)
    documents: List[dict] = field(default_factory=list)
    validation: Dict[str, Any] = field(default_factory=dict)
    grounding: Dict[str, Any] = field(default_factory=dict)
    generation_mode: str = "deterministic"
    warnings: List[str] = field(default_factory=list)
    timings_ms: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "generation_mode": self.generation_mode,
            "question_for_user": self.question_for_user,
            "predictions": self.predictions,
            "n_documents": len(self.documents),
            "validation": self.validation,
            "grounding": self.grounding,
            "warnings": self.warnings,
            "timings_ms": {k: round(v, 1) for k, v in self.timings_ms.items()},
            "report_markdown": self.report_markdown,
        }


class ClinicalReportPipeline:
    """Grounded report generation over the Phase 1–5 models and the RAG layer."""

    def __init__(
        self,
        model_runner: Any = None,
        rag_store: Any = None,
        llm_backend: Any = None,
        min_feature_coverage: float = MIN_FEATURE_COVERAGE,
    ) -> None:
        self.model_runner = model_runner
        self._rag_store = rag_store
        self.llm_backend = llm_backend
        self.min_feature_coverage = min_feature_coverage

    # ── lazy dependencies ────────────────────────────────────────────────
    @property
    def rag_store(self):
        if self._rag_store is None:
            from src.llm.rag_corpus import get_rag_store
            self._rag_store = get_rag_store()
        return self._rag_store

    # ── prediction with a coverage guard ─────────────────────────────────
    def _predict(self, payload: dict) -> Dict[str, Any]:
        if self.model_runner is None:
            return {}
        preds = self.model_runner.run_live_inference_with_uncertainty(payload)

        coverage = self._feature_coverage(payload)
        if coverage is not None:
            preds["feature_coverage"] = coverage
            if coverage < self.min_feature_coverage:
                withheld = [k for k in list(preds) if k.startswith("p_")]
                for k in withheld:
                    preds.pop(k, None)
                preds.pop("risk_tier", None)
                preds["withheld_reason"] = (
                    f"only {100 * coverage:.0f}% of trained features supplied "
                    f"(minimum {100 * self.min_feature_coverage:.0f}%)"
                )
        return preds

    def _feature_coverage(self, payload: dict) -> Optional[float]:
        """Fraction of each model's trained features actually supplied."""
        runner = self.model_runner
        if runner is None or not getattr(runner, "lgbm_models", None):
            return None
        try:
            series = runner._convert_payload_to_series(payload)
        except Exception:
            return None
        supplied = set(getattr(series, "index", []))
        ratios: List[float] = []
        for model in runner.lgbm_models.values():
            if model is None:
                continue
            if hasattr(model, "booster_"):
                req = model.booster_.feature_name()
            elif hasattr(model, "feature_name_"):
                req = list(model.feature_name_)
            else:
                continue
            if req:
                ratios.append(len(supplied & set(req)) / len(req))
        return (sum(ratios) / len(ratios)) if ratios else None

    # ── main entry point ─────────────────────────────────────────────────
    def generate(
        self,
        payload: dict,
        *,
        case_id: str = "case_1",
        use_llm: bool = True,
        require_complete: bool = True,
        predictions: Optional[Dict[str, Any]] = None,
    ) -> PipelineResult:
        """
        ``predictions`` lets a caller supply model output computed from a richer
        source than the payload. A cohort admission has its full feature row, so
        inference from it is not subject to the payload coverage floor that
        legitimately withholds predictions for an unseen patient. Passing partial or
        payload-derived values here would defeat that guard, so it is only for
        callers holding the complete feature set.
        """
        t0 = time.perf_counter()
        res = PipelineResult(status="ok")

        # 1. completeness gate — never guess at missing clinical input
        report = validate_payload(payload)
        res.validation = report.to_dict()
        if require_complete and not report.ok:
            res.status = "incomplete_input"
            res.question_for_user = report.question_for_user()
            res.timings_ms["total"] = (time.perf_counter() - t0) * 1000
            return res

        # 2. predictions
        t = time.perf_counter()
        if predictions is not None:
            res.predictions = dict(predictions)
        else:
            try:
                res.predictions = self._predict(payload)
            except Exception as e:
                res.warnings.append(f"model inference failed: {e}")
                res.predictions = {}
        res.timings_ms["predict"] = (time.perf_counter() - t) * 1000

        # 3. evidence retrieval
        t = time.perf_counter()
        retrieval: Dict[str, Any] = {}
        try:
            retrieval = self.rag_store.search_unseen_patient_rag(
                payload, case_id=case_id, require_complete=False
            ) or {}
        except Exception as e:
            res.warnings.append(f"evidence retrieval failed: {e}")
        res.timings_ms["retrieve"] = (time.perf_counter() - t) * 1000

        res.documents = list(retrieval.get("documents", []))
        ranked_meds = retrieval.get("ranked_medications", [])
        retrieval_status = retrieval.get("status", "unavailable")
        if retrieval_status not in ("ok",):
            res.warnings.append(f"retrieval status: {retrieval_status}")

        # 4. deterministic composition — grounded by construction
        composed = compose_report(
            payload=payload,
            predictions=res.predictions,
            documents=res.documents,
            validation=res.validation,
            ranked_medications=ranked_meds,
            retrieval_status=retrieval_status,
            twin_status=retrieval.get("twin_status", ""),
            retrieval_errors=retrieval.get("retrieval_errors", []),
        )
        deterministic_md = composed.to_markdown()
        res.warnings.extend(composed.warnings)

        # 5. optional LLM rephrase, gated by the verifier
        # SYSTEM_CONSTANTS are published Phase 4/5/9 figures the composer may quote;
        # they are grounded in the audit reports rather than in this patient's data.
        #
        # Medication relevance scores are computed by this system from the payload,
        # exactly like a model prediction, and the composer renders them
        # ("relevance 9.5"). They were absent from the fact store, so the verifier
        # correctly refused any report containing a ranked medication — every DKA and
        # GI-bleed case in the Phase 11 set was withheld for quoting a number the
        # system itself produced. Registering them keeps the verifier fail-closed
        # while letting it recognise the system's own arithmetic.
        med_scores = {
            f"medication_relevance_{m.get('ingredient', i)}": m["score"]
            for i, m in enumerate(ranked_meds)
            if isinstance(m, dict) and m.get("score") is not None
        }
        fact_store = build_fact_store(
            payload=payload, predictions=res.predictions, documents=res.documents,
            extra_numbers={**SYSTEM_CONSTANTS, **med_scores},
        )
        res.report_markdown = deterministic_md
        res.generation_mode = "deterministic"

        if use_llm and self.llm_backend is not None:
            t = time.perf_counter()
            try:
                candidate = self.llm_backend.rephrase(
                    deterministic_md,
                    system_prompt=self.system_prompt(),
                )
                check = verify_text(candidate, fact_store)
                res.grounding = check.to_dict()
                if check.ok:
                    res.report_markdown = candidate
                    res.generation_mode = "llm_rephrased_verified"
                else:
                    res.generation_mode = "deterministic_llm_rejected"
                    res.warnings.append(
                        f"LLM output rejected by grounding check "
                        f"({len(check.critical)} critical violation(s)); "
                        "deterministic text returned instead"
                    )
            except Exception as e:
                res.warnings.append(f"LLM backend failed: {e}")
            res.timings_ms["llm"] = (time.perf_counter() - t) * 1000

        # 6. always verify what we are about to return
        final_check = verify_text(res.report_markdown, fact_store)
        res.grounding = final_check.to_dict()
        if not final_check.ok:
            res.status = "refused"
            res.report_markdown = (
                "# Report withheld\n\n"
                "The generated summary did not pass grounding verification, meaning it "
                "contained a statement not traceable to the supplied data, the model "
                "outputs, or a retrieved document. It has been withheld rather than "
                "shown.\n\n"
                + final_check.render()
            )

        if not res.documents and res.status == "ok":
            res.status = "ok_no_evidence"

        res.timings_ms["total"] = (time.perf_counter() - t0) * 1000
        return res

    # ── prompt ───────────────────────────────────────────────────────────
    @staticmethod
    def system_prompt() -> str:
        return (
            "You are a clinical documentation assistant. You will be given a factual "
            "report that has already been assembled from validated data.\n\n"
            "Your ONLY task is to improve readability and flow.\n\n"
            "Absolute constraints:\n"
            "1. Do NOT introduce any number that is not already in the text.\n"
            "2. Do NOT introduce any citation, guideline, drug or study not already present.\n"
            "3. Do NOT add clinical interpretation, diagnosis, or recommendation.\n"
            "4. Do NOT remove any uncertainty statement, limitation, or citation.\n"
            "5. If you are unsure, copy the original sentence verbatim.\n\n"
            "Output the rewritten report only, in Markdown, with all sections preserved."
        )
