"""
src/assistant/requirements.py
─────────────────────────────
Per-intent information policy.  Spec sections 5, 7, 15.

Loads ``config/requirements.yaml`` and answers one question: for this intent and
this patient context, which fields are needed and at what level.

The policy is data, not code, for two reasons. It can be reviewed by a clinician
without reading Python — the one open safety item this project already carries.
And it cannot be renegotiated at runtime: spec 33.9 forbids the language model
from bypassing an application-level gate, and a model cannot edit a file it is
never shown.

Field names are checked against ``state.FIELDS`` at load time. The alternative —
discovering a typo when the gate blocks forever on a field nothing can fill — is
the same class of defect as a guard whose pattern never matches, which this
project has already shipped once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from src.assistant.state import FIELDS, PatientContext, field_spec

__all__ = [
    "Requirement", "RequirementSet", "for_intent", "load_policy",
    "SAFETY_CRITICAL", "REQUIRED", "OPTIONAL", "POLICY_PATH",
]

SAFETY_CRITICAL = "safety_critical"
REQUIRED = "required"
OPTIONAL = "optional"

_LEVELS = (SAFETY_CRITICAL, REQUIRED, OPTIONAL)

POLICY_PATH = Path(__file__).resolve().parent / "config" / "requirements.yaml"

_CACHE: Dict[str, dict] = {}


@dataclass(frozen=True)
class Requirement:
    """One field the current intent needs, and why."""

    field: str
    level: str
    reason: str = ""
    #: True when this came from a `conditional:` block rather than the base list,
    #: so the audit trail can show which patient facts caused it to apply.
    conditional: bool = False
    #: Audience-specific wording from the policy's `prompts:` block. The field
    #: registry's prompt addresses a patient ("How old are you?"); asked of a
    #: clinician presenting a case that is simply the wrong question.
    prompt_override: str = ""

    @property
    def prompt(self) -> str:
        return self.prompt_override or field_spec(self.field).prompt

    def to_dict(self) -> dict:
        return {"field": self.field, "level": self.level, "reason": self.reason,
                "conditional": self.conditional, "prompt": self.prompt}


class PolicyError(ValueError):
    """The requirement policy is malformed."""


# ── loading ──────────────────────────────────────────────────────────────────

def load_policy(path: Optional[Path] = None, *, refresh: bool = False) -> dict:
    """Load and validate the policy, caching by path."""
    p = Path(path) if path else POLICY_PATH
    key = str(p)
    if refresh:
        _CACHE.pop(key, None)
    if key in _CACHE:
        return _CACHE[key]

    if not p.exists():
        raise PolicyError(f"requirement policy not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    intents = raw.get("intents")
    if not isinstance(intents, dict) or not intents:
        raise PolicyError(f"{p}: no `intents:` mapping")

    for name, spec in intents.items():
        if not isinstance(spec, dict):
            raise PolicyError(f"{p}: intent {name!r} is not a mapping")
        for level in _LEVELS:
            names = spec.get(level) or []
            if not isinstance(names, (list, tuple)):
                raise PolicyError(f"{p}: {name}.{level} must be a list")
            for fname in names:
                if fname not in FIELDS:
                    raise PolicyError(
                        f"{p}: intent {name!r} lists unknown field {fname!r}; "
                        f"add it to src.assistant.state.FIELDS or fix the typo")
        seen: Dict[str, str] = {}
        for level in _LEVELS:
            for fname in spec.get(level) or []:
                if fname in seen:
                    raise PolicyError(
                        f"{p}: intent {name!r} lists {fname!r} at both "
                        f"{seen[fname]} and {level}")
                seen[fname] = level

        for cond in spec.get("conditional") or []:
            if not isinstance(cond, dict):
                raise PolicyError(f"{p}: {name}.conditional entries must be mappings")
            cfield = cond.get("field")
            if cfield not in FIELDS:
                raise PolicyError(
                    f"{p}: intent {name!r} conditional names unknown field {cfield!r}")
            if cond.get("level") not in _LEVELS:
                raise PolicyError(
                    f"{p}: intent {name!r} conditional on {cfield!r} has level "
                    f"{cond.get('level')!r}, expected one of {_LEVELS}")
            if not isinstance(cond.get("when") or {}, dict):
                raise PolicyError(f"{p}: {name}.conditional.when must be a mapping")

    _CACHE[key] = raw
    return raw


# ── conditional predicates ───────────────────────────────────────────────────

def _predicate(name: str, arg: Any, context: PatientContext) -> Optional[bool]:
    """
    Evaluate one `when:` predicate.

    Returns ``None`` for "cannot tell yet" — the field the predicate reads is
    itself unknown. That is deliberately not ``False``: a conditional that reads
    ``sex`` must not quietly decide the patient is not pregnant merely because
    nobody has asked their sex. Unknown propagates, and the conditional does not
    fire until the fact exists.
    """
    if name == "known":
        names = arg if isinstance(arg, (list, tuple)) else [arg]
        return all(context.is_known(str(n)) for n in names)
    if name == "unknown":
        names = arg if isinstance(arg, (list, tuple)) else [arg]
        return all(not context.is_known(str(n)) for n in names)

    if name.endswith("_in"):
        fname = name[:-3]
        field_spec(fname)
        value = context.get(fname)
        if value is None:
            return None
        allowed = {str(a).strip().lower() for a in (arg or [])}
        if isinstance(value, list):
            return any(str(v).strip().lower() in allowed for v in value)
        return str(value).strip().lower() in allowed

    if name.endswith("_between"):
        fname = name[:-8]
        field_spec(fname)
        value = context.get(fname)
        if value is None:
            return None
        try:
            num = float(value)
            low, high = float(arg[0]), float(arg[1])
        except (TypeError, ValueError, IndexError):
            raise PolicyError(f"predicate {name}: expected [low, high], got {arg!r}")
        return low <= num <= high

    raise PolicyError(
        f"unknown predicate {name!r}; supported: known, unknown, <field>_in, "
        f"<field>_between")


def _conditional_applies(when: Dict[str, Any],
                         context: PatientContext) -> Optional[bool]:
    """All predicates must hold. Unknown anywhere makes the whole clause unknown."""
    if not when:
        return True
    saw_unknown = False
    for name, arg in when.items():
        result = _predicate(name, arg, context)
        if result is False:
            return False
        if result is None:
            saw_unknown = True
    return None if saw_unknown else True


# ── the resolved set ─────────────────────────────────────────────────────────

@dataclass
class RequirementSet:
    """What this intent needs from this patient, right now."""

    intent: str
    requirements: List[Requirement] = field(default_factory=list)
    #: Conditionals whose predicates could not be evaluated because the fields
    #: they read are themselves unknown. Surfaced so the clarification engine can
    #: prioritise the *enabling* question — asking sex before pregnancy status —
    #: rather than treating the conditional as settled.
    pending_conditionals: List[Requirement] = field(default_factory=list)
    #: field → audience-specific question wording, from the policy file.
    prompts: Dict[str, str] = field(default_factory=dict)

    def at(self, level: str) -> List[str]:
        return [r.field for r in self.requirements if r.level == level]

    @property
    def safety_critical(self) -> List[str]:
        return self.at(SAFETY_CRITICAL)

    @property
    def required(self) -> List[str]:
        return self.at(REQUIRED)

    @property
    def optional(self) -> List[str]:
        return self.at(OPTIONAL)

    @property
    def all_fields(self) -> List[str]:
        return [r.field for r in self.requirements]

    def missing(self, context: PatientContext, level: str) -> List[str]:
        return [f for f in self.at(level) if not context.is_known(f)]

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "safety_critical": self.safety_critical,
            "required": self.required,
            "optional": self.optional,
            "pending_conditionals": [r.field for r in self.pending_conditionals],
        }


def for_intent(intent: Any, context: Optional[PatientContext] = None,
               *, path: Optional[Path] = None) -> RequirementSet:
    """
    Resolve the information policy for an intent against a patient context.

    Passing no context returns the unconditional base policy, which is what a
    reviewer wants to read. Passing a context additionally evaluates the
    `conditional:` blocks — pregnancy status becomes required once the patient's
    sex and age make it applicable, and not before.
    """
    name = getattr(intent, "value", intent)
    name = str(name)
    policy = load_policy(path)
    spec = policy["intents"].get(name)
    if spec is None:
        raise PolicyError(
            f"no requirement policy for intent {name!r}; every intent in "
            f"src.assistant.intents.Intent needs an entry in {POLICY_PATH.name}")

    prompts = {str(k): str(v) for k, v in (policy.get("prompts") or {}).items()}
    out = RequirementSet(intent=name, prompts=prompts)
    for level in _LEVELS:
        for fname in spec.get(level) or []:
            out.requirements.append(
                Requirement(fname, level, prompt_override=prompts.get(fname, "")))

    ctx = context if context is not None else PatientContext()
    for cond in spec.get("conditional") or []:
        applies = _conditional_applies(cond.get("when") or {}, ctx)
        req = Requirement(cond["field"], cond["level"],
                          reason=str(cond.get("reason") or "").strip(),
                          conditional=True,
                          prompt_override=prompts.get(cond["field"], ""))
        if applies is True:
            # A conditional promotion outranks a base listing of the same field.
            out.requirements = [r for r in out.requirements if r.field != req.field]
            out.requirements.append(req)
        elif applies is None:
            out.pending_conditionals.append(req)

    return out


def all_intents(*, path: Optional[Path] = None) -> List[str]:
    return sorted(load_policy(path)["intents"])
