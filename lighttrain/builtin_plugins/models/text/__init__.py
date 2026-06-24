"""Model adapters."""

from __future__ import annotations

from .hf_causal import HFCausalLM
from .tiny_lm import TinyCausalLM

__all__ = ["HFCausalLM", "TinyCausalLM"]
