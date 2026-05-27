"""PCNUpdateRule — Predictive Coding Networks.

Predictive Coding (Rao & Ballard 1999; Millidge et al. 2022) is a biologically
plausible learning algorithm where each layer predicts the activity of the layer
below it.  Weight updates are local (Hebbian-like) and don't require
backpropagation across layers.

Algorithm:
    For a network with L linear layers W₁ … Wₗ and activations x₀ … xₗ:

    Inference phase (N_infer steps):
        e_l = x_l − f(W_l · x_{l-1})          prediction error at layer l
        x_l ← x_l − lr_infer · (e_l − W_{l+1}^T · e_{l+1})   (middle layers)
        x_l ← x_l − lr_infer · e_l            (top layer — no higher correction)

    Weight update:
        ΔW_l = lr_weight · e_l · x_{l-1}^T

This implementation:
    * Operates on an ``nn.Sequential`` of ``nn.Linear`` layers.
    * ``batch["x"]`` — input (B, D_in); ``batch.get("labels")`` for supervised top layer.
    * Supervised top layer: replaces top activation with one-hot label.

Registered as ``@register("update_rule", "pcn")``.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

from lighttrain.registry import register


def _current_lr(optimizer: Any) -> float:
    inner = getattr(optimizer, "optimizer", optimizer)
    groups = getattr(inner, "param_groups", None)
    if not groups:
        return 0.0
    return float(groups[0].get("lr", 0.0))


def _get_linear_layers(model: nn.Module) -> list[nn.Linear]:
    layers = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            layers.append(m)
    return layers


@register("update_rule", "pcn")
class PCNUpdateRule:
    """Predictive Coding Network update rule.

    Args:
        n_infer:       Number of inference (relaxation) steps.
        lr_infer:      Learning rate for activation inference.
        lr_weight:     Learning rate for Hebbian weight updates.
        activation:    Nonlinearity name: "tanh" | "relu" | "none".
    """

    def __init__(
        self,
        n_infer: int = 20,
        lr_infer: float = 0.1,
        lr_weight: float = 0.01,
        activation: str = "tanh",
    ) -> None:
        self.n_infer = int(n_infer)
        self.lr_infer = float(lr_infer)
        self.lr_weight = float(lr_weight)
        act_map = {"tanh": torch.tanh, "relu": torch.relu, "none": lambda x: x}
        if activation not in act_map:
            raise ValueError(f"Unknown activation '{activation}'. Use tanh/relu/none.")
        self._act = act_map[activation]

    def setup(self, model: Any, sample: Any) -> None:  # noqa: ARG002
        return None

    def state_dict(self) -> dict[str, Any]:
        return {"n_infer": self.n_infer, "lr_infer": self.lr_infer, "lr_weight": self.lr_weight}

    def load_state_dict(self, sd: Mapping[str, Any]) -> None:
        self.n_infer = int(sd.get("n_infer", self.n_infer))
        self.lr_infer = float(sd.get("lr_infer", self.lr_infer))
        self.lr_weight = float(sd.get("lr_weight", self.lr_weight))

    def step(
        self,
        model: Any,
        batch: Mapping[str, Any],
        ctx: Any,
    ) -> dict[str, Any]:
        bus = ctx.bus
        if bus is not None:
            bus.dispatch("on_step_begin", step=ctx.step, ctx=ctx, batch=batch)

        ctx.extras["model"] = model

        x_input = batch.get("x", batch.get("input_ids"))
        if x_input is None:
            raise KeyError("PCNUpdateRule: expected 'x' or 'input_ids' in batch.")
        x_input = x_input.float()
        labels = batch.get("labels")

        layers = _get_linear_layers(model)
        if not layers:
            raise RuntimeError("PCNUpdateRule: no nn.Linear layers found in model.")

        L = len(layers)
        # Initialise activations with forward pass
        acts = [x_input.detach()]
        h = x_input.detach()
        for layer in layers:
            h = self._act(layer(h))
            acts.append(h.clone())

        # Supervised: clamp top layer to target
        if labels is not None and labels.shape == acts[-1].shape:
            acts[-1] = labels.float()

        # Inference phase — relax activations to minimise prediction errors
        for _ in range(self.n_infer):
            errors = []
            for l_idx, layer in enumerate(layers):
                pred = self._act(layer(acts[l_idx]))
                e = acts[l_idx + 1] - pred
                errors.append(e)

            # Update middle activations (gradient of variational free energy)
            for l_idx in range(1, L):  # x_1 … x_{L-1}; x_0 is clamped input
                correction = errors[l_idx - 1]
                if l_idx < L:
                    # propagate error from above: W_l+1^T · e_{l+1}
                    e_up = errors[l_idx]
                    W_up = layers[l_idx].weight.detach()
                    correction = correction - e_up @ W_up
                acts[l_idx] = acts[l_idx] - self.lr_infer * correction

        # Weight update (Hebbian, no autograd)
        total_error_sq = 0.0
        for l_idx, layer in enumerate(layers):
            pred = self._act(layer(acts[l_idx]))
            e = acts[l_idx + 1] - pred
            total_error_sq += float(e.pow(2).mean().item())
            # ΔW = lr * e · x_{l-1}^T
            delta_W = self.lr_weight * (e.T @ acts[l_idx]) / acts[l_idx].shape[0]
            with torch.no_grad():
                layer.weight.add_(delta_W)
                if layer.bias is not None:
                    layer.bias.add_(self.lr_weight * e.mean(0))

        ctx.metrics["loss"] = total_error_sq / max(L, 1)
        ctx.metrics["grad_norm"] = 0.0
        ctx.metrics["lr"] = self.lr_weight
        ctx.metrics["skipped"] = 0.0

        if bus is not None:
            bus.dispatch("on_step_end", step=ctx.step, metrics=ctx.metrics, batch=batch, model=model)

        return dict(ctx.metrics)


__all__ = ["PCNUpdateRule"]
