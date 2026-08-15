"""
backend/service.py
──────────────────
Process-wide state: the loaded models and the assistant that uses them.

Models are loaded once, at startup, because unpickling the boosters takes
seconds and most turns never touch them. Loading is allowed to *fail* without
taking the server down: a guideline lookup needs no model, and refusing to
start because `models/best_models` is absent would make the whole API
unavailable for the majority of requests that do not need it. `/api/health`
reports what actually loaded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from src.assistant.audit import AuditLog
from src.assistant.orchestrator import Assistant

# Load `.env` before any module-level `os.environ.get` runs. Import order is the
# whole point: a settings read that happens first sees the unloaded default and
# silently ignores the file the operator just edited.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
except ImportError:                                   # dotenv is optional
    pass

#: Set CDT_ASSISTANT_DEBUG=1 to attach the gate decision, extraction proposals
#: and faithfulness checks to each turn. Off by default: spec 28 keeps the
#: reasoning trace out of what the person being helped receives.
DEBUG = os.environ.get("CDT_ASSISTANT_DEBUG", "").strip() in ("1", "true", "yes")


@dataclass
class Service:
    models_dir: str = "models/best_models"
    data_dir: str = "data/processed"

    runner: Any = None
    pipeline: Any = None
    assistant: Optional[Assistant] = None
    load_error: str = ""
    load_error_detail: str = ""
    audit: AuditLog = field(default_factory=lambda: AuditLog(
        path=Path("logs/assistant_audit.jsonl")))

    # ── startup ─────────────────────────────────────────────────────────────
    def start(self) -> None:
        try:
            from src.llm.model_runner import LiveModelRunner
            from src.llm.pipeline import ClinicalReportPipeline

            self.runner = LiveModelRunner(models_dir=self.models_dir,
                                          data_dir=self.data_dir)
            self.pipeline = ClinicalReportPipeline(model_runner=self.runner)
        except Exception as exc:                       # missing artefacts, bad pickle
            # The full message can carry absolute filesystem paths. It is kept
            # for the operator and a short form is what reaches a client — a
            # 503 should say the models are unavailable, not describe the disk.
            self.load_error_detail = f"{type(exc).__name__}: {exc}"
            self.load_error = type(exc).__name__

        self.assistant = Assistant.clinician(
            model_runner=self.runner,
            pipeline=self.pipeline,
            audit_log=self.audit,
            backend=_extraction_backend(),
        )

    @property
    def models_loaded(self) -> bool:
        return self.runner is not None

    def require_models(self) -> Any:
        from fastapi import HTTPException

        if self.runner is None:
            raise HTTPException(
                status_code=503,
                detail=f"risk models are not loaded: {self.load_error or 'unknown'}")
        return self.runner

    def describe(self) -> Dict[str, Any]:
        from src.assistant.evidence import corpus_stats
        from src.llm.guidelines import corpus_stats as guideline_stats

        backend = getattr(self.assistant, "backend", None)
        return {
            "status": "ok",
            "audience": "clinician",
            "models_loaded": self.models_loaded,
            "models_error": self.load_error or None,
            "guidelines": guideline_stats(),
            "patient_corpus": corpus_stats(),
            "extraction_backend": backend.describe() if backend else None,
            "debug_mode": DEBUG,
            "sessions": len(self.assistant.sessions) if self.assistant else 0,
        }


def _extraction_backend():
    """
    OpenRouter for fact extraction, when a key is present.

    Asked for by name rather than via ``get_backend("auto")``, which
    deliberately never returns it. Without a key this is None and extraction
    falls back to its deterministic patterns — fewer fields per turn, more
    questions asked, nothing guessed.
    """
    if not os.environ.get("OPENROUTER_API_KEY"):
        return None
    from src.llm.backends import OpenRouterBackend

    backend = OpenRouterBackend()      # reads model, base URL and timeout from env
    return backend if backend.available else None


service = Service()
