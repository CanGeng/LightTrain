"""MiniMind model adapter for lighttrain.

Registers MiniMindForCausalLM as ``@register("model", "minimind")`` so it can
be used in a lighttrain recipe via::

    model:
      name: minimind
      hidden_size: 512
      num_hidden_layers: 8

The forward pass returns only logits; loss is computed by lighttrain's
``next_token`` objective (default for the ``pretrain`` trainer).

Note on MoE: when ``use_moe=True`` the model produces an ``aux_loss`` in
addition to the CE loss. lighttrain's standard objective ignores it; add a
custom callback or objective to include it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch.nn as nn

# Allow ``from model.model_minimind import ...`` to resolve.
_mm_root = str(Path(__file__).resolve().parents[1])
if _mm_root not in sys.path:
    sys.path.insert(0, _mm_root)

from lighttrain.protocols import ModelOutput  # noqa: E402
from lighttrain.registry import register  # noqa: E402
from model.model_minimind import (  # noqa: E402  # noqa: E402
    MiniMindConfig,
    MiniMindForCausalLM,
)


@register("model", "minimind")
class MiniMindLightTrain(nn.Module):
    """Thin lighttrain wrapper around MiniMindForCausalLM.

    All MiniMindConfig kwargs are forwarded by name, e.g.::

        model:
          name: minimind
          hidden_size: 512
          num_hidden_layers: 8
          use_moe: false
    """

    def __init__(
        self,
        hidden_size: int = 512,
        num_hidden_layers: int = 8,
        **config_kwargs: Any,
    ) -> None:
        super().__init__()
        cfg = MiniMindConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            **config_kwargs,
        )
        self.model = MiniMindForCausalLM(cfg)
        print(
            f"MiniMind: hidden={hidden_size} layers={num_hidden_layers}"
            f" moe={cfg.use_moe}"
            f" params={sum(p.numel() for p in self.parameters()) / 1e6:.1f}M"
        )

    def forward(
        self,
        input_ids: Any,
        attention_mask: Any = None,
        **_: Any,
    ) -> ModelOutput:
        # Labels are NOT passed — lighttrain's next_token objective handles loss.
        out = self.model(input_ids, attention_mask=attention_mask)
        return ModelOutput(outputs={"logits": out.logits})

    @property
    def config(self) -> MiniMindConfig:
        return self.model.config


__all__ = ["MiniMindLightTrain"]
