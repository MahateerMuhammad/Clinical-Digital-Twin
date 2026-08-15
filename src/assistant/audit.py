"""
src/assistant/audit.py
──────────────────────
The traceability record.  Spec 28, 33.16.

Every turn writes one JSON line: what came in, what was understood, what was
retrieved, what the gates decided, and what went out. Append-only, so a record
is evidence rather than a mutable summary of the current belief.

What is deliberately absent
───────────────────────────
**Hidden reasoning.** Spec 24 and 33.16 both draw the line in the same place:
the record holds the *inputs and outcomes* of each stage, not a chain of
thought. A stored rationale is a stored fabrication risk — it reads as
justification and nothing checks it.

**The patient, by default.** ``redact=True`` stores the audit without free-text
patient content: field names and decisions, not the values or the quotes. That
is enough to answer "why did it refuse?", "did triage fire?", "was a
fabrication rejected?" — the questions an audit exists for — without a log file
becoming a second, unsecured copy of everything the patient typed. Turn it off
deliberately, for a debugging session, not as the default.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["AuditRecord", "AuditLog", "DEFAULT_AUDIT_PATH"]

DEFAULT_AUDIT_PATH = Path("logs/assistant_audit.jsonl")

#: Keys whose values are free-text patient content and are dropped when
#: redacting. Everything else in a record is a decision, a field name, a status
#: or a count — none of which identifies anyone.
_PATIENT_TEXT_KEYS = frozenset({
    "user_message", "reply", "quote", "source_quote", "value", "values",
    "current", "history", "messages", "matched", "span",
})


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _redact(node: Any, *, depth: int = 0) -> Any:
    """Strip free-text patient content, keeping structure and decisions."""
    if depth > 12:
        return "…"
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k in _PATIENT_TEXT_KEYS:
                if isinstance(v, str):
                    out[k] = f"<redacted {len(v)} chars>"
                elif isinstance(v, (list, tuple, dict)):
                    out[k] = f"<redacted {len(v)} items>"
                else:
                    out[k] = "<redacted>"
                continue
            out[k] = _redact(v, depth=depth + 1)
        return out
    if isinstance(node, (list, tuple)):
        return [_redact(v, depth=depth + 1) for v in node]
    return node


@dataclass
class AuditRecord:
    """One turn, end to end. Spec 28's field list, plus what it took to get it."""

    session_id: str
    turn: int
    user_message: str = ""
    intent: Optional[str] = None
    intent_confidence: Optional[float] = None
    triage: Dict[str, Any] = field(default_factory=dict)
    extraction: Dict[str, Any] = field(default_factory=dict)
    required_information: List[str] = field(default_factory=list)
    missing_information: List[str] = field(default_factory=list)
    safety_flags: List[str] = field(default_factory=list)
    gate: Dict[str, Any] = field(default_factory=dict)
    retrieved_sources: List[str] = field(default_factory=list)
    validation: Dict[str, Any] = field(default_factory=dict)
    status: str = ""
    reply: str = ""
    #: Wall-clock time for the whole turn. Serve-time telemetry needs a
    #: latency distribution, and a p95 cannot be reconstructed after the fact
    #: from records that never carried a duration.
    latency_ms: Optional[float] = None
    timestamp: str = field(default_factory=_utcnow)

    def to_dict(self, *, redact: bool = True) -> dict:
        raw = {
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "turn": self.turn,
            "user_message": self.user_message,
            "intent": self.intent,
            "intent_confidence": self.intent_confidence,
            "triage": self.triage,
            "extraction": self.extraction,
            "required_information": self.required_information,
            "missing_information": self.missing_information,
            "safety_flags": self.safety_flags,
            "gate": self.gate,
            "retrieved_sources": self.retrieved_sources,
            "validation": self.validation,
            "status": self.status,
            "reply": self.reply,
            "latency_ms": self.latency_ms,
        }
        return _redact(raw) if redact else raw


class AuditLog:
    """
    Append-only JSONL sink.

    ``path=None`` keeps records in memory only, which is what the tests and any
    caller that has its own persistence want. A failure to write is logged and
    swallowed: an audit sink that can take the assistant down converts a disk
    problem into a clinical one.
    """

    def __init__(self, path: Optional[Path] = DEFAULT_AUDIT_PATH,
                 *, redact: bool = True) -> None:
        self.path = Path(path) if path else None
        self.redact = redact
        self.records: List[AuditRecord] = []
        self.write_errors: List[str] = []

    def append(self, record: AuditRecord) -> AuditRecord:
        self.records.append(record)
        if self.path is None:
            return record
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record.to_dict(redact=self.redact),
                                    default=str) + "\n")
        except OSError as exc:
            self.write_errors.append(str(exc))
        return record

    def __len__(self) -> int:
        return len(self.records)

    def last(self) -> Optional[AuditRecord]:
        return self.records[-1] if self.records else None
