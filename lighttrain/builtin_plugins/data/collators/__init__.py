"""Collators (modality-grouped).

Text collators (causal-LM / preference pairs) live in ``text``; ``multimodal``
stacks per-modality tensors. Registration is via auto-discovery.
"""

from .multimodal import MultiModalCollator
from .text import CausalLMCollator, PreferenceCollator

__all__ = ["CausalLMCollator", "MultiModalCollator", "PreferenceCollator"]
