"""HuggingFace causal LM adapter.

Wraps ``transformers.AutoModelForCausalLM.from_pretrained`` and normalizes
its output to ``ModelOutput``. Endpoint / token plumbing is environment-
based: ``HF_TOKEN``, ``HF_ENDPOINT`` (and the ``huggingface_hub``-aware
``HF_HUB_ENDPOINT``) are read from ``os.environ`` — populate them via your
shell or via a project-root ``.env`` (loaded by the CLI on startup).
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn

from lighttrain.protocols import ModelOutput
from lighttrain.registry import register

_DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


@register("model", "hf_causal")
class HFCausalLM(nn.Module):
    """Adapter around an HF causal LM."""

    def __init__(
        self,
        pretrained: str,
        *,
        dtype: str | None = "bfloat16",
        trust_remote_code: bool = False,
        revision: str | None = None,
        use_auth_token: bool | str | None = None,
        from_pretrained_kwargs: dict[str, Any] | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> None:
        super().__init__()
        self.pretrained = pretrained
        self._default_output_hidden_states = bool(output_hidden_states)
        self._default_output_attentions = bool(output_attentions)
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "hf_causal requires `transformers`; install with `pip install -e .[dev]`."
            ) from e

        kwargs: dict[str, Any] = dict(from_pretrained_kwargs or {})
        if dtype is not None:
            torch_dtype = _DTYPE_MAP.get(str(dtype).lower())
            if torch_dtype is None:
                raise ValueError(f"Unknown dtype {dtype!r}.")
            kwargs.setdefault("torch_dtype", torch_dtype)
        if revision is not None:
            kwargs.setdefault("revision", revision)
        if trust_remote_code:
            kwargs.setdefault("trust_remote_code", True)

        token = os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGING_FACE_HUB_TOKEN"
        )
        if isinstance(use_auth_token, str):
            kwargs.setdefault("token", use_auth_token)
        elif use_auth_token is True or (use_auth_token is None and token):
            kwargs.setdefault("token", token if token else True)

        self.inner = AutoModelForCausalLM.from_pretrained(pretrained, **kwargs)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,  # noqa: ARG002 — kept for protocol parity
        *,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        **kwargs: Any,
    ) -> ModelOutput:
        if output_hidden_states is None:
            output_hidden_states = self._default_output_hidden_states
        if output_attentions is None:
            output_attentions = self._default_output_attentions
        # Intentionally do NOT forward `labels` to the HF inner: HF would compute
        # its own shifted CE and return `out.loss`, but lighttrain runs an
        # external LossFn (which does the shift). Forwarding labels would either
        # waste compute or risk double-shift if a caller ever consumed out.loss.
        out = self.inner(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            **kwargs,
        )
        logits = getattr(out, "logits", None)
        if logits is None and isinstance(out, dict):
            logits = out.get("logits")
        if logits is None:
            raise RuntimeError(
                f"HF model {type(self.inner).__name__} returned no logits."
            )
        hidden_states = getattr(out, "hidden_states", None)
        attentions = getattr(out, "attentions", None)
        return ModelOutput(
            outputs={"logits": logits},
            loss=None,
            hidden_states=tuple(hidden_states) if hidden_states is not None else None,
            attentions=tuple(attentions) if attentions is not None else None,
        )


__all__ = ["HFCausalLM"]
