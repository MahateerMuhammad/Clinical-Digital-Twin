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
           "get_backend"]

DEFAULT_MODEL = os.environ.get("CDT_LLM_MODEL", "qwen2.5:3b-instruct")


class LLMBackend:
    """Interface for the rephrasing stage."""

    name = "base"
    available = False

    def rephrase(self, text: str, system_prompt: str = "") -> str:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"backend": self.name, "available": self.available}


class NullBackend(LLMBackend):
    """No model: return the deterministic text unchanged.

    This is the correct default. A system with no LLM available should return
    grounded text verbatim, not synthesise prose to fill the gap.
    """

    name = "null"
    available = True

    def rephrase(self, text: str, system_prompt: str = "") -> str:
        return text


class OllamaBackend(LLMBackend):
    """Local Ollama daemon (llama3.x, qwen2.5, mistral, …)."""

    name = "ollama"

    def __init__(self, model: str = DEFAULT_MODEL, timeout: float = 120.0) -> None:
        self.model = model
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


def get_backend(prefer: str = "auto", **kwargs) -> LLMBackend:
    """
    Return the best available backend.

    ``auto`` tries Ollama, then Transformers, then falls back to
    :class:`NullBackend`. The fallback returns grounded text unchanged — it never
    generates substitute prose.
    """
    if prefer in ("null", "none"):
        return NullBackend()
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
