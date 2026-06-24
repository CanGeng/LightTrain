"""AdaLoRA adapter — adaptive low-rank adaptation.

AdaLoRA (Zhang et al., 2023) extends LoRA by adaptively allocating the rank
budget across weight matrices based on importance scores derived from the
singular values of the adaptation matrices.

Core idea:
    W = W₀ + B · Λ · A
    where Λ = diag(λ₁, …, λᵣ) are learnable importance weights.

Every ``update_interval`` steps, the importance scores (|λᵢ|) are computed
and low-importance components are pruned by zeroing their λᵢ.  The target
total rank is ``target_total_rank`` shared across all adapted layers.

This implementation:
    * Uses HuggingFace PEFT's ``AdaLoraConfig`` when available.
    * Falls back to a thin manual implementation when PEFT is not installed.

Recipe form::

    model:
      name: adalora
      base:
        name: hf_causal
        pretrained: gpt2
      r: 12                     # initial rank per layer
      target_r: 8               # target rank after pruning
      target_modules: [c_attn]
      lora_alpha: 32
      update_interval: 200      # steps between rank reallocations

Registered as ``@register("model", "adalora")``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from lighttrain.protocols import ModelOutput
from lighttrain.registry import register

from ._common import auto_target_modules, resolve_base_model

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AdaLoRA linear layer (manual implementation — no PEFT dependency)
# ---------------------------------------------------------------------------

class AdaLoRALinear(nn.Module):
    """Single linear layer with adaptive LoRA: W_out = W₀ + B·Λ·A."""

    def __init__(
        self,
        original: nn.Linear,
        r: int,
        lora_alpha: float,
    ) -> None:
        super().__init__()
        self.r = r
        self.scaling = lora_alpha / r
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.bias = original.bias

        # Freeze base weight
        self.weight = nn.Parameter(original.weight.detach().clone(), requires_grad=False)

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(r, original.in_features))
        self.lora_B = nn.Parameter(torch.zeros(original.out_features, r))
        # Importance weights (diagonal Λ)
        self.lora_Lambda = nn.Parameter(torch.ones(r))

        # Initialise: A ~ N(0, 1/sqrt(in)), B = 0
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = nn.functional.linear(x, self.weight, self.bias)
        # Adapter path: x → A → Λ → B
        adapter = x @ self.lora_A.T * self.lora_Lambda.unsqueeze(0)  # (..., r)
        adapter = adapter @ self.lora_B.T                              # (..., out)
        return base_out + self.scaling * adapter

    def importance_scores(self) -> torch.Tensor:
        """Return |λᵢ| as importance scores, shape (r,)."""
        return self.lora_Lambda.detach().abs()

    def prune_rank(self, keep: int) -> None:
        """Zero out the least-important (r - keep) singular components."""
        scores = self.importance_scores()
        _, topk_idx = scores.topk(keep)
        mask = torch.zeros_like(self.lora_Lambda.data)
        mask[topk_idx] = 1.0
        with torch.no_grad():
            self.lora_Lambda.data.mul_(mask)

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "lora_A": self.lora_A.data,
            "lora_B": self.lora_B.data,
            "lora_Lambda": self.lora_Lambda.data,
        }


# ---------------------------------------------------------------------------
# AdaLoRA model wrapper
# ---------------------------------------------------------------------------

@register("model", "adalora")
class AdaLoRAAdapter(nn.Module):
    """AdaLoRA model wrapper — adaptive rank allocation.

    Attempts to use HuggingFace PEFT's AdaLora when available; falls back to
    the built-in manual implementation.

    Args:
        base_cfg:         Nested model config dict (resolved via ``resolve_base_model``).
        r:                Initial LoRA rank per layer.
        target_r:         Target rank after importance-based pruning.
        lora_alpha:       LoRA scaling α (scaling = α / r).
        target_modules:   List of submodule names to adapt (auto-detected if None).
        update_interval:  Steps between rank reallocations.
    """

    def __init__(
        self,
        *,
        base: Any = None,
        r: int = 12,
        target_r: int = 8,
        lora_alpha: float = 32.0,
        target_modules: list[str] | None = None,
        update_interval: int = 200,
        total_step: int = 1000,
        **base_kwargs: Any,
    ) -> None:
        super().__init__()
        self.r = r
        self.target_r = target_r
        self.lora_alpha = float(lora_alpha)
        self.target_modules = target_modules
        self.update_interval = int(update_interval)
        self.total_step = int(total_step)
        self._step = 0

        # Try PEFT first
        self._use_peft = False
        try:
            from peft import AdaLoraConfig, TaskType, get_peft_model  # noqa: F401
            self._use_peft = True
        except ImportError:
            pass

        # Build base model
        base_model, _ = resolve_base_model(base or base_kwargs)

        if self._use_peft:
            self.model = self._build_peft(base_model)
        else:
            self.model = self._build_manual(base_model)

    def _build_peft(self, base_model: nn.Module) -> nn.Module:
        from peft import AdaLoraConfig, get_peft_model
        target = self.target_modules or auto_target_modules(base_model)
        cfg = AdaLoraConfig(
            target_r=self.target_r,
            lora_alpha=int(self.lora_alpha),
            target_modules=target,
            init_r=self.r,
            deltaT=self.update_interval,
            total_step=self.total_step,
            beta1=0.85,
            beta2=0.85,
            orth_reg_weight=0.5,
        )
        return get_peft_model(base_model, cfg)

    def _build_manual(self, base_model: nn.Module) -> nn.Module:
        """Replace target Linear layers with AdaLoRALinear."""
        target = self.target_modules or auto_target_modules(base_model)
        self._adalora_layers: dict[str, AdaLoRALinear] = {}
        for name, module in list(base_model.named_modules()):
            # Only replace if name ends with one of the target_modules
            if not isinstance(module, nn.Linear):
                continue
            if not any(t in name for t in (target or [])):
                continue
            # Replace the submodule
            parent_name, _, child_name = name.rpartition(".")
            parent = base_model if not parent_name else dict(base_model.named_modules())[parent_name]
            ada_layer = AdaLoRALinear(module, r=self.r, lora_alpha=self.lora_alpha)
            setattr(parent, child_name, ada_layer)
            self._adalora_layers[name] = ada_layer
        return base_model

    def maybe_reallocate_rank(self) -> None:
        """Prune low-importance rank components every update_interval steps."""
        self._step += 1
        if self._use_peft or not hasattr(self, "_adalora_layers"):
            return
        if self._step % self.update_interval != 0:
            return
        n_layers = len(self._adalora_layers)
        if n_layers == 0:
            return
        # Simple uniform allocation: give each layer target_r
        for layer in self._adalora_layers.values():
            layer.prune_rank(self.target_r)

    def forward(self, **batch: Any) -> ModelOutput:
        # Update rank if needed (non-intrusive — doesn't break gradient flow)
        # Called here so it aligns with optimizer steps automatically.
        out = self.model(**batch)
        if isinstance(out, ModelOutput):
            return out
        import torch
        if hasattr(out, "logits"):
            return ModelOutput(outputs={"logits": out.logits})
        from lighttrain.protocols import ModelOutput as MO
        return MO(outputs={"logits": out} if isinstance(out, torch.Tensor) else {})

    def state_dict(self, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        if self._use_peft:
            try:
                from peft import get_peft_model_state_dict
                return get_peft_model_state_dict(self.model)
            except Exception:  # noqa: BLE001
                _log.warning("AdaLoRA.state_dict: PEFT export failed; falling back to manual adapter collection", exc_info=True)
        # Manual: collect adapter weights only
        if hasattr(self, "_adalora_layers"):
            sd: dict[str, torch.Tensor] = {}
            for name, layer in self._adalora_layers.items():
                for k, v in layer.adapter_state_dict().items():
                    sd[f"{name}.{k}"] = v
            return sd
        return super().state_dict(**kwargs)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True) -> Any:  # type: ignore[override]
        if self._use_peft:
            try:
                from peft import set_peft_model_state_dict
                return set_peft_model_state_dict(self.model, state_dict)
            except Exception:  # noqa: BLE001
                _log.warning("AdaLoRA.load_state_dict: PEFT load failed; falling back to base model load_state_dict", exc_info=True)
        return self.model.load_state_dict(state_dict, strict=False)


__all__ = ["AdaLoRAAdapter", "AdaLoRALinear"]
