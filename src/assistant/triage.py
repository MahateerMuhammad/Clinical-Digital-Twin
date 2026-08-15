"""
src/assistant/triage.py
───────────────────────
Emergency detection.  Spec section 16.

This runs **first** — before extraction, before intent classification, before any
model call — and it is the only stage permitted to answer without collecting
information. Spec 16 is explicit that a possible emergency must not be met with
a long questionnaire, and every other module in this package is built to refuse
until it has enough facts. Triage is the deliberate exception, so it is kept
small, deterministic and separate.

Why regex and not a classifier
──────────────────────────────
This is the stage that must work when everything else is broken: no network, no
API key, no model, no embeddings. A missed myocardial infarction because the
classifier was cold-starting is not a degraded experience, it is the worst
outcome the system can produce. Rules in a YAML file have no failure mode beyond
being wrong in a way a clinician can read and correct.

Calibration
───────────
Ambiguity resolves *upward*. Three specific consequences:

**Third-person mentions still fire.** "My father is having crushing chest pain"
is someone asking on behalf of a person who needs an ambulance. Suppressing it
as "not about the user" would withhold the one response that matters.

**Only explicit, adjacent negation suppresses.** "I have no chest pain" is
suppressed. "I have chest pain but no shortness of breath" is not, because the
negation attaches to the second clause. The window is deliberately short, and
where the parse is unclear the rule fires.

**Suppression never applies to `emergency`-severity rules by default.** A
negation scope error that hides a stroke is unrecoverable; one that produces an
unnecessary warning is not. Only `urgent_assess` rules can be talked down.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

__all__ = [
    "Severity", "RedFlag", "TriageResult", "screen", "load_rules",
    "EMERGENCY", "URGENT_ASSESS", "NONE", "RED_FLAGS_PATH",
]

EMERGENCY = "emergency"
URGENT_ASSESS = "urgent_assess"
NONE = "none"

Severity = str

RED_FLAGS_PATH = Path(__file__).resolve().parent / "config" / "red_flags.yaml"

#: How much text before a trigger is searched for a negation. Long enough for
#: "I do not have any chest pain", short enough that a negation in the previous
#: clause does not reach across and cancel an unrelated symptom.
_NEGATION_WINDOW = 40

_CACHE: Dict[str, dict] = {}


class TriageConfigError(ValueError):
    """The red-flag rule set is malformed."""


@dataclass(frozen=True)
class RedFlag:
    """One rule that fired, and the evidence for it."""

    rule_id: str
    severity: Severity
    label: str
    matched: str
    advice: str
    warning_signs: Tuple[str, ...] = ()
    escalated: bool = False
    suppress_default_disclaimer: bool = False

    def to_dict(self) -> dict:
        return {"rule_id": self.rule_id, "severity": self.severity,
                "label": self.label, "matched": self.matched,
                "escalated": self.escalated,
                "warning_signs": list(self.warning_signs)}


@dataclass
class TriageResult:
    """The verdict, plus everything needed to render it and to audit it."""

    severity: Severity = NONE
    flags: List[RedFlag] = field(default_factory=list)
    #: Rules that matched but were suppressed, with the reason. Kept because a
    #: suppression is a decision not to warn, and that must be reviewable.
    suppressed: List[Dict[str, str]] = field(default_factory=list)

    @property
    def is_emergency(self) -> bool:
        return self.severity == EMERGENCY

    @property
    def bypasses_questioning(self) -> bool:
        """Spec 16: an emergency answers immediately and collects nothing."""
        return self.severity == EMERGENCY

    @property
    def labels(self) -> List[str]:
        return [f.label for f in self.flags]

    def to_dict(self) -> dict:
        return {"severity": self.severity,
                "flags": [f.to_dict() for f in self.flags],
                "suppressed": self.suppressed}


# ── loading ──────────────────────────────────────────────────────────────────

def _compile(patterns: Any, where: str) -> Tuple[re.Pattern, ...]:
    if patterns is None:
        return ()
    if not isinstance(patterns, (list, tuple)):
        raise TriageConfigError(f"{where}: expected a list of patterns")
    out = []
    for p in patterns:
        try:
            out.append(re.compile(str(p), re.I))
        except re.error as exc:
            raise TriageConfigError(f"{where}: bad pattern {p!r}: {exc}") from None
    return tuple(out)


def load_rules(path: Optional[Path] = None, *, refresh: bool = False) -> dict:
    """Load, compile and validate the red-flag rules."""
    p = Path(path) if path else RED_FLAGS_PATH
    key = str(p)
    if refresh:
        _CACHE.pop(key, None)
    if key in _CACHE:
        return _CACHE[key]

    if not p.exists():
        raise TriageConfigError(f"red-flag rules not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    rules = raw.get("rules")
    if not isinstance(rules, list) or not rules:
        raise TriageConfigError(f"{p}: no `rules:` list")

    seen = set()
    compiled = []
    for r in rules:
        if not isinstance(r, dict):
            raise TriageConfigError(f"{p}: each rule must be a mapping")
        rid = r.get("id")
        if not rid:
            raise TriageConfigError(f"{p}: a rule has no id")
        if rid in seen:
            raise TriageConfigError(f"{p}: duplicate rule id {rid!r}")
        seen.add(rid)
        sev = r.get("severity")
        if sev not in (EMERGENCY, URGENT_ASSESS):
            raise TriageConfigError(
                f"{p}: rule {rid!r} has severity {sev!r}, expected "
                f"{EMERGENCY!r} or {URGENT_ASSESS!r}")
        if not r.get("patterns"):
            raise TriageConfigError(f"{p}: rule {rid!r} has no patterns")
        if sev == EMERGENCY and not str(r.get("emergency_advice") or "").strip():
            raise TriageConfigError(
                f"{p}: emergency rule {rid!r} has no emergency_advice; a rule that "
                f"fires with nothing to say is worse than no rule")
        if sev == URGENT_ASSESS and not str(r.get("urgent_advice") or "").strip():
            raise TriageConfigError(
                f"{p}: urgent rule {rid!r} has no urgent_advice")

        compiled.append({
            "id": rid,
            "severity": sev,
            "label": r.get("label") or rid.replace("_", " ").title(),
            "patterns": _compile(r.get("patterns"), f"{rid}.patterns"),
            "escalate_with": _compile(r.get("escalate_with"), f"{rid}.escalate_with"),
            "urgent_advice": str(r.get("urgent_advice") or "").strip(),
            "emergency_advice": str(r.get("emergency_advice") or "").strip(),
            "warning_signs": tuple(r.get("warning_signs") or ()),
            "suppress_default_disclaimer": bool(r.get("suppress_default_disclaimer")),
        })

    out = {
        "version": raw.get("version", "unknown"),
        "rules": compiled,
        "negations": _compile(raw.get("negations"), "negations"),
        "historical": _compile(raw.get("historical"), "historical"),
    }
    _CACHE[key] = out
    return out


# ── suppression ──────────────────────────────────────────────────────────────

def _preceding(text: str, start: int) -> str:
    return text[max(0, start - _NEGATION_WINDOW):start]


def _suppression_reason(text: str, match: re.Match,
                        cfg: dict) -> Optional[str]:
    """
    Whether an `urgent_assess` match should be dropped, and why.

    Reads only the short span immediately before the trigger. A negation in an
    earlier clause must not reach forward and cancel a symptom the patient did
    report — "I have no fever but I do have chest pain" has to keep the chest
    pain.
    """
    before = _preceding(text, match.start())
    # A clause boundary ends a negation's reach.
    before = re.split(r"[.;,]|\b(?:but|however|although|though)\b", before)[-1]

    for pat in cfg["negations"]:
        if pat.search(before):
            return f"negated by {pat.pattern!r}"
    for pat in cfg["historical"]:
        if pat.search(before) or pat.search(
                text[match.end():match.end() + _NEGATION_WINDOW]):
            return f"placed in the past by {pat.pattern!r}"
    return None


# ── screening ────────────────────────────────────────────────────────────────

def screen(message: str, *, path: Optional[Path] = None,
           history: Optional[List[str]] = None) -> TriageResult:
    """
    Screen a message for red flags.

    ``history`` may carry earlier patient messages. Escalation reads them, so a
    patient who says "I have chest pain" and then, two turns later, "now it's
    spreading to my jaw" is escalated on the combination — the two facts arrived
    in different turns but describe one presentation.

    Trigger detection itself deliberately does *not* read history: a red flag
    that fired and was answered three turns ago should not re-fire on every
    subsequent message.
    """
    text = (message or "").strip()
    result = TriageResult()
    if not text:
        return result

    cfg = load_rules(path)
    prior = " ".join(history or [])
    escalation_corpus = " ".join([text, prior]).strip()

    for rule in cfg["rules"]:
        match = None
        for pat in rule["patterns"]:
            match = pat.search(text)
            if match:
                break

        # The trigger may have been stated in an earlier turn. Escalating
        # features arriving now still describe that presentation: "I have chest
        # pain" followed two turns later by "now it's spreading to my jaw" is
        # one patient, and the second message is the one that matters.
        from_history = False
        if match is None and prior:
            for pat in rule["patterns"]:
                match = pat.search(prior)
                if match:
                    from_history = True
                    break
        if match is None:
            continue

        severity = rule["severity"]
        # Suppression was already evaluated in the turn the trigger arrived.
        haystack = text if not from_history else prior

        # Suppression applies only to urgent rules. Hiding a stroke because a
        # negation window was misjudged is not a recoverable error; an
        # unnecessary warning is.
        if severity == URGENT_ASSESS and not from_history:
            reason = _suppression_reason(haystack, match, cfg)
            if reason:
                result.suppressed.append(
                    {"rule_id": rule["id"], "matched": match.group(0),
                     "reason": reason})
                continue

        escalated = False
        if severity == URGENT_ASSESS and rule["escalate_with"]:
            # When the trigger came from an earlier turn, only a feature in the
            # *current* message counts. Re-reading the old turn would escalate on
            # the same words every turn thereafter.
            corpus = text if from_history else escalation_corpus
            for pat in rule["escalate_with"]:
                found = pat.search(corpus)
                # The trigger phrase must not escalate itself: "severe chest
                # pain" matches `severe` inside its own span.
                if found and not (not from_history
                                  and match.start() <= found.start() < match.end()):
                    severity = EMERGENCY
                    escalated = True
                    break

        # A trigger recalled from history is reported only when this turn
        # escalates it. Otherwise an unchanged flag would re-fire on every
        # subsequent message for the rest of the conversation.
        if from_history and not escalated:
            continue

        advice = (rule["emergency_advice"] if severity == EMERGENCY
                  else rule["urgent_advice"])
        if not advice:                     # an escalated urgent rule may lack one
            advice = rule["urgent_advice"] or rule["emergency_advice"]

        result.flags.append(RedFlag(
            rule_id=rule["id"], severity=severity, label=rule["label"],
            matched=match.group(0), advice=advice,
            warning_signs=rule["warning_signs"], escalated=escalated,
            suppress_default_disclaimer=rule["suppress_default_disclaimer"],
        ))

    if any(f.severity == EMERGENCY for f in result.flags):
        result.severity = EMERGENCY
    elif result.flags:
        result.severity = URGENT_ASSESS

    # Emergencies first, then the order the rules are written in.
    result.flags.sort(key=lambda f: 0 if f.severity == EMERGENCY else 1)
    return result
