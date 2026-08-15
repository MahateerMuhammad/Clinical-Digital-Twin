"""
src/llm/backends.py
───────────────────
Language-model backends for the rephrasing stage.

Scope discipline: a backend's only job is to restate already-assembled, already-
grounded text more readably. It is never asked to reason clinically, never given
raw patient data without the composed report, and its output is always checked by
:func:`src.llm.grounding.verify_text` before it can reach a caller.

This replaces ``RealLLMEngine._generate_grounded_fallback``, which ignored its
prompt entirely and returned a fixed string containing invented SHAP values
("+0.45", "+0.38") and a fabricated guideline citation for every patient.

Backends
--------
``NullBackend``          returns the input unchanged (the honest default)
``OllamaBackend``        local Ollama daemon
``TransformersBackend``  local HuggingFace weights (Kaggle / offline GPU)
"""

from __future__ import annotations

import os
from typing import Any, Optional

__all__ = ["LLMBackend", "NullBackend", "OllamaBackend", "TransformersBackend",
           "OpenRouterBackend", "get_backend"]

#: Literal fallback, read at construction — see DEFAULT_OPENROUTER_MODEL below
#: for why a module-level environment read is not a default.
DEFAULT_MODEL = "qwen2.5:3b-instruct"

#: Fallback model for the assistant's structured stages, used when nothing is
#: configured. A 120B mixture-of-experts with ~12B active parameters: the task
#: sends ~600 tokens and expects a small JSON object, so context length is
#: irrelevant and latency is everything.
#:
#: A literal, not an environment read — as are the two below.
#: ``OpenRouterBackend.__init__`` consults the environment at construction,
#: which is the only correct place: a module-level read happens whenever
#: `backends` is first imported, and that may be before `.env` has loaded.
#:
#: These three did read the environment, and it caused exactly that confusion.
#: Importing `backend.service` (which loads `.env`) earlier in a test session
#: changed what "default" meant, so the test asserting the default is a
#: free-tier id passed in isolation and failed in the full suite.
DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

#: API base. Configurable so a proxy or compatible gateway can be substituted
#: without editing source — a hardcoded endpoint is the kind of thing that gets
#: patched locally and then committed by accident.
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

DEFAULT_OPENROUTER_TIMEOUT = 30.0


class LLMBackend:
    """Interface for the rephrasing stage."""

    name = "base"
    available = False

    #: True when `rephrase` returns its input unchanged, i.e. no model runs.
    #:
    #: The pipeline needs this to label its output honestly. Without it a
    #: NullBackend's echo goes through the verifier, trivially passes — the text was
    #: grounded before it was handed over — and is recorded as
    #: `llm_rephrased_verified`. That field is read as evidence a model ran and its
    #: output was checked. For a passthrough neither is true.
    passthrough = False

    def rephrase(self, text: str, system_prompt: str = "") -> str:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"backend": self.name, "available": self.available,
                "passthrough": self.passthrough}


class NullBackend(LLMBackend):
    """No model: return the deterministic text unchanged.

    This is the correct default. A system with no LLM available should return
    grounded text verbatim, not synthesise prose to fill the gap.
    """

    name = "null"
    available = True

    def rephrase(self, text: str, system_prompt: str = "") -> str:
        return text

    @property
    def passthrough(self) -> bool:
        """
        True only while `rephrase` is still this class's own no-op.

        Derived rather than set, because subclassing `NullBackend` to build a stub is
        a natural thing to do — `tests/test_llm_grounding.py` defines a hallucinating
        and a truncating backend that way. A static `passthrough = True` was
        inherited by both, so the pipeline skipped the very backends written to prove
        it rejects bad output. Overriding `rephrase` now clears the flag by
        construction, and no stub author has to remember to.
        """
        return type(self).rephrase is NullBackend.rephrase


class OllamaBackend(LLMBackend):
    """Local Ollama daemon (llama3.x, qwen2.5, mistral, …)."""

    name = "ollama"

    def __init__(self, model: Optional[str] = None,
                 timeout: float = 120.0) -> None:
        # A default *argument* is evaluated at import too, so reading the
        # environment here rather than in the signature is the same fix.
        self.model = model or os.environ.get("CDT_LLM_MODEL") or DEFAULT_MODEL
        self.timeout = timeout
        self._client = None
        try:
            import ollama
            ollama.list()                      # fails fast if the daemon is down
            self._client = ollama
            self.available = True
        except Exception:
            self.available = False

    def rephrase(self, text: str, system_prompt: str = "") -> str:
        if not self.available or self._client is None:
            raise RuntimeError("Ollama backend unavailable")
        resp = self._client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            options={"temperature": 0.1, "num_predict": 2048},
        )
        return resp["message"]["content"]

    def describe(self) -> dict:
        return {"backend": self.name, "available": self.available, "model": self.model}


class TransformersBackend(LLMBackend):
    """Local HuggingFace weights, optionally with a LoRA adapter.

    Sized for a Kaggle T4: a 3B instruct model in 4-bit fits comfortably and
    leaves headroom. Greedy decoding at low temperature — this stage must not be
    creative.
    """

    name = "transformers"

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-3B-Instruct",
        adapter_path: Optional[str] = None,
        load_in_4bit: bool = True,
        max_new_tokens: int = 2048,
        device_map: str = "auto",
    ) -> None:
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self._tok = None
        self._model = None
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer

            kwargs: dict[str, Any] = {"device_map": device_map}
            if load_in_4bit:
                try:
                    from transformers import BitsAndBytesConfig
                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype="float16",
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                except Exception:
                    pass

            self._tok = AutoTokenizer.from_pretrained(model_id)
            self._model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

            if adapter_path:
                from peft import PeftModel
                self._model = PeftModel.from_pretrained(self._model, adapter_path)

            self._model.eval()
            self.available = True
        except Exception:
            self.available = False

    def rephrase(self, text: str, system_prompt: str = "") -> str:
        if not self.available:
            raise RuntimeError("Transformers backend unavailable")
        import torch

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]
        prompt = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tok(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,              # deterministic: no creativity here
                temperature=None,
                top_p=None,
                pad_token_id=self._tok.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[-1]:]
        return self._tok.decode(gen, skip_special_tokens=True).strip()

    def describe(self) -> dict:
        return {"backend": self.name, "available": self.available,
                "model": self.model_id, "adapter": self.adapter_path}


class OpenRouterBackend(LLMBackend):
    """
    Hosted inference via OpenRouter.

    Data boundary — read before using this for anything else
    ───────────────────────────────────────────────────────
    This backend sends its input to a third party, and OpenRouter *routes* to
    whichever upstream provider is serving the model at that moment. That makes
    it unsuitable for MIMIC-derived content: PhysioNet's data use agreement
    restricts redistribution of credentialed data, and "I cannot say which
    company received it" is not a position to be in under a DUA.

    It is appropriate for ``src/assistant/``, whose inputs are what a patient
    typed about themselves, and for evaluation runs over synthetic payloads. It
    is not appropriate for the clinician pipeline in ``src/llm/pipeline.py``
    when that pipeline is handling real admissions. Nothing here enforces that
    distinction — it is a judgement about the caller, recorded so the judgement
    is visible at the point of use.

    ``json_mode`` requests a JSON object rather than prose. The assistant's
    extraction and classification stages depend on parseable output, and spec 26
    puts the parse in application code: a response that does not parse is
    discarded, never repaired by guessing.
    """

    name = "openrouter"

    #: Read at call time rather than import time so tests can set it.
    ENV_KEY = "OPENROUTER_API_KEY"

    def __init__(self, model: Optional[str] = None,
                 timeout: Optional[float] = None, api_key: Optional[str] = None,
                 max_tokens: int = 2048, temperature: float = 0.1,
                 base_url: Optional[str] = None,
                 reasoning: Optional[str] = None) -> None:
        # Settings are read here, at construction, not at module import.
        #
        # The module constants are evaluated when `backends` is first imported,
        # which may be before `.env` has been loaded — and then an operator who
        # edits `.env` sees no effect and no error, which is the worst kind of
        # configuration bug. Reading at construction means the backend picks up
        # whatever the environment holds by the time someone actually builds one.
        self.model = model or os.environ.get(
            "CDT_OPENROUTER_MODEL") or DEFAULT_OPENROUTER_MODEL
        self.base_url = (base_url or os.environ.get("CDT_OPENROUTER_BASE_URL")
                         or DEFAULT_OPENROUTER_BASE_URL).rstrip("/")
        self.endpoint = f"{self.base_url}/chat/completions"
        if timeout is None:
            timeout = float(os.environ.get("CDT_OPENROUTER_TIMEOUT")
                            or DEFAULT_OPENROUTER_TIMEOUT)
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        # Reasoning effort. Off by default, and that is a correctness decision
        # rather than a cost one: a reasoning model spends its *completion*
        # budget on thinking, so the default 120B model burned 1841 of 2048
        # tokens reasoning about a fact-extraction prompt and had ~200 left to
        # answer in. The JSON came back cut mid-field, parsed as a failure, and
        # the system fell back to its deterministic floor — silently, on every
        # turn. Neither stage here benefits from chain-of-thought: extraction
        # copies spans out of a sentence, and the judge scores against fixed
        # anchors. Set CDT_OPENROUTER_REASONING to low/medium/high to enable.
        self.reasoning = (reasoning if reasoning is not None
                          else os.environ.get("CDT_OPENROUTER_REASONING", "off")
                          ).strip().lower()
        # Any OpenAI-compatible host, not only OpenRouter — the class keeps its
        # name because that is what it is usually pointed at, but the endpoint
        # is configuration and Google, Groq and xAI all speak the same protocol.
        #
        # What is *not* shared is the extensions. `reasoning` is OpenRouter's,
        # and Gemini rejects the whole request with 400 "Unknown name
        # 'reasoning'" rather than ignoring it. So provider-specific fields are
        # sent only to the provider that defined them; anything else gets plain
        # OpenAI-shaped JSON.
        self.is_openrouter = "openrouter.ai" in self.base_url
        self._api_key = (api_key or os.environ.get("CDT_LLM_API_KEY")
                         or os.environ.get(self.ENV_KEY))
        try:
            import requests  # noqa: F401
            self._requests = requests
        except ImportError:
            self._requests = None
        # Availability is "could this call succeed", not "has it succeeded".
        # A missing key is the common case and must not raise on construction —
        # `get_backend("auto")` has to be able to skip past it.
        self.available = bool(self._api_key) and self._requests is not None

    # ── low-level call ──
    def _chat(self, messages: list, *, json_mode: bool = False) -> str:
        if not self.available:
            raise RuntimeError(
                f"OpenRouter backend unavailable: "
                + ("`requests` is not installed" if self._requests is None
                   else f"set {self.ENV_KEY} in the environment"))
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        # Same intent, two spellings. OpenRouter takes a `reasoning` object;
        # the OpenAI-compatible spelling is `reasoning_effort`, and each host
        # rejects the other outright rather than ignoring it.
        if self.is_openrouter:
            body["reasoning"] = ({"enabled": False} if self.reasoning in ("off", "")
                                 else {"effort": self.reasoning})
        else:
            body["reasoning_effort"] = ("none" if self.reasoning in ("off", "")
                                        else self.reasoning)

        resp = self._requests.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {self._api_key}",
                     "Content-Type": "application/json"},
            json=body, timeout=self.timeout)
        resp.raise_for_status()
        payload = resp.json()
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError(f"OpenRouter returned no choices: {payload}")
        choice = choices[0]
        # A response cut off at the token limit is not a response. Returned as
        # text it becomes half a JSON object, which the caller reports as a
        # parse failure — indistinguishable from a model that answered badly,
        # when in fact it was never allowed to finish. Say which it was; the
        # caller still falls back, but the audit records a truthful reason.
        if choice.get("finish_reason") == "length":
            used = (payload.get("usage") or {}).get(
                "completion_tokens_details", {}).get("reasoning_tokens")
            raise RuntimeError(
                f"OpenRouter response truncated at max_tokens={self.max_tokens}"
                + (f" ({used} of them spent on reasoning — set "
                   f"CDT_OPENROUTER_REASONING=off)" if used else ""))
        return (choice.get("message") or {}).get("content") or ""

    def rephrase(self, text: str, system_prompt: str = "") -> str:
        return self._chat([{"role": "system", "content": system_prompt},
                           {"role": "user", "content": text}]).strip()

    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        """Raw JSON text. Parsing and validation belong to the caller (spec 26)."""
        return self._chat([{"role": "system", "content": system_prompt},
                           {"role": "user", "content": user_prompt}],
                          json_mode=True).strip()

    def describe(self) -> dict:
        # `key_present`, never the key. This dictionary is returned by
        # /api/health, so anything in it is public to whoever can reach the
        # service.
        return {"backend": self.name, "available": self.available,
                "model": self.model, "hosted": True,
                "base_url": self.base_url,
                "provider": "openrouter" if self.is_openrouter else "openai-compatible",
                "key_present": bool(self._api_key)}


def get_backend(prefer: str = "auto", **kwargs) -> LLMBackend:
    """
    Return the best available backend.

    ``auto`` tries Ollama, then Transformers, then falls back to
    :class:`NullBackend`. The fallback returns grounded text unchanged — it never
    generates substitute prose.

    ``auto`` deliberately does **not** reach for OpenRouter. Sending data to a
    third party is a decision a caller makes explicitly, not one it inherits from
    a default: the clinician pipeline handles MIMIC-derived content, and a
    fallback chain that quietly posted it off the machine would be a data-use
    breach caused by an environment variable being set. Ask for it by name.
    """
    if prefer in ("null", "none"):
        return NullBackend()
    if prefer == "openrouter":
        return OpenRouterBackend(**{k: v for k, v in kwargs.items()
                                    if k in ("model", "timeout", "api_key",
                                             "max_tokens", "temperature")})
    if prefer in ("ollama", "auto"):
        b = OllamaBackend(**{k: v for k, v in kwargs.items() if k in ("model", "timeout")})
        if b.available:
            return b
        if prefer == "ollama":
            return NullBackend()
    if prefer in ("transformers", "auto"):
        b = TransformersBackend(**{k: v for k, v in kwargs.items()
                                   if k in ("model_id", "adapter_path", "load_in_4bit")})
        if b.available:
            return b
    return NullBackend()
