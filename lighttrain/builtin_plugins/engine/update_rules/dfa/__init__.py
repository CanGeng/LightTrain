"""DFAUpdateRule — Direct Feedback Alignment.

Direct Feedback Alignment (Nøkland 2016) replaces the backpropagation weight
transport with fixed random feedback matrices:

    δ_l = f'(z_l) ⊙ (B_l · e_out)

where ``B_l`` are fixed random matrices and ``e_out = dL/d_y_out`` is the
output layer error.

This makes the backward pass biologically plausible (no weight symmetry needed)
and is an alternative to backprop for shallow networks.

Implementation details:
    * Feedback matrices B_l are registered as non-trainable buffers.
    * The model must expose a list of ``nn.Linear`` layers.
    * The forward pass records pre-nonlinearity activations z_l via hooks.
    * After the forward pass the DFA update rule:
        1. computes output error e_out = ∂L/∂y via autograd of the output layer only
        2. computes δ_l = f'(z_l) ⊙ (B_l · e_out)
        3. applies weight update ΔW_l = -lr · δ_l · x_{l-1}^T

Registered as ``@register("update_rule", "dfa")``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from lighttrain.registry import register


def _current_lr(optimizer: Any) -> float:
    inner = getattr(optimizer, "optimizer", optimizer)
    groups = getattr(inner, "param_groups", None)
    if not groups:
        return 0.0
    return float(groups[0].get("lr", 0.0))


def _get_linear_layers(model: nn.Module) -> list[nn.Linear]:
    return [m for m in model.modules() if isinstance(m, nn.Linear)]


@register("update_rule", "dfa")
class DFAUpdateRule:
    """Direct Feedback Alignment update rule.

    Args:
        feedback_scale: Scale factor for random feedback matrices initialisation.
        activation:     Nonlinearity for derivative computation: "relu" | "tanh" | "none".
        lr:             Per-step learning rate (also uses ctx.optimizer lr if available).
    """

    def __init__(
        self,
        feedback_scale: float = 0.01,
        activation: str = "relu",
        lr: float = 1e-3,
    ) -> None:
        self.feedback_scale = float(feedback_scale)
        self.lr = float(lr)
        act_map = {"relu": "relu", "tanh": "tanh", "none": "none"}
        if activation not in act_map:
            raise ValueError(f"Unknown activation '{activation}'.")
        self.activation = activation
        # B_l matrices: {layer_id -> Tensor(out_l, out_last)}
        self._feedback: dict[int, torch.Tensor] = {}

    def _act_deriv(self, z: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            return (z > 0).float()
        if self.activation == "tanh":
            return 1.0 - torch.tanh(z) ** 2
        return torch.ones_like(z)

    def _ensure_feedback(self, layers: list[nn.Linear], out_size: int, device: torch.device) -> None:
        for _i, layer in enumerate(layers[:-1]):  # no feedback matrix for last layer
            key = id(layer)
            if key not in self._feedback or self._feedback[key].shape != (layer.out_features, out_size):
                self._feedback[key] = torch.randn(
                    layer.out_features, out_size, device=device
                ) * self.feedback_scale

    def setup(self, model: Any, sample: Any) -> None:  # noqa: ARG002
        return None

    def state_dict(self) -> dict[str, Any]:
        return {"feedback_scale": self.feedback_scale, "lr": self.lr, "activation": self.activation}

    def load_state_dict(self, sd: Mapping[str, Any]) -> None:
        self.feedback_scale = float(sd.get("feedback_scale", self.feedback_scale))
        self.lr = float(sd.get("lr", self.lr))
        self.activation = sd.get("activation", self.activation)

    def step(
        self,
        model: Any,
        batch: Mapping[str, Any],
        ctx: Any,
    ) -> dict[str, Any]:
        bus = ctx.bus
        optimizer = ctx.optimizer
        scheduler = ctx.scheduler

        if bus is not None:
            bus.dispatch("on_step_begin", step=ctx.step, ctx=ctx, batch=batch)

        ctx.extras["model"] = model

        x_input = batch.get("x", batch.get("input_ids"))
        if x_input is None:
            raise KeyError("DFAUpdateRule: expected 'x' or 'input_ids' in batch.")
        x_input = x_input.float()
        labels = batch.get("labels")

        layers = _get_linear_layers(model)
        if len(layers) < 2:
            raise RuntimeError("DFAUpdateRule: need at least 2 nn.Linear layers.")

        out_size = layers[-1].out_features
        device = x_input.device
        self._ensure_feedback(layers, out_size, device)

        # --- Forward pass (record pre-activations z_l and inputs x_l) ----
        acts = [x_input]   # x_0 = input
        pre_acts = []      # z_l = W_l · x_{l-1} + b_l
        h = x_input
        for layer in layers:
            z = F.linear(h, layer.weight, layer.bias)
            pre_acts.append(z.detach())
            if self.activation == "relu":
                h = F.relu(z)
            elif self.activation == "tanh":
                h = torch.tanh(z)
            else:
                h = z
            acts.append(h.detach())

        # --- Output error (use autograd for last layer only) ---------------
        # Recompute final layer with grad
        z_last = F.linear(acts[-2].detach(), layers[-1].weight, layers[-1].bias)
        if labels is not None:
            if labels.dim() == 1:
                loss = F.cross_entropy(z_last, labels.long())
            else:
                loss = F.mse_loss(z_last, labels.float())
        else:
            # unsupervised: minimise reconstruction
            loss = F.mse_loss(z_last, acts[-2].detach()[:, :out_size])

        # e_out: gradient w.r.t. z_last (B, out_last)
        e_out = torch.autograd.grad(loss, z_last, retain_graph=False)[0].detach()

        # --- DFA weight updates for hidden layers -------------------------
        eff_lr = _current_lr(optimizer) or self.lr
        with torch.no_grad():
            for i, layer in enumerate(layers[:-1]):
                B_l = self._feedback[id(layer)]  # (out_l, out_last)
                # δ_l = f'(z_l) ⊙ (B_l · e_out^T)^T
                feedback = (B_l @ e_out.T).T   # (B, out_l)
                delta = self._act_deriv(pre_acts[i]) * feedback
                dW = -(eff_lr * delta.T @ acts[i]) / acts[i].shape[0]
                layer.weight.add_(dW)
                if layer.bias is not None:
                    layer.bias.add_(-eff_lr * delta.mean(0))

        # Last layer: standard gradient step (outside no_grad so loss2 has grad_fn)
        z_last2 = F.linear(acts[-2], layers[-1].weight, layers[-1].bias)
        if labels is not None:
            if labels.dim() == 1:
                loss2 = F.cross_entropy(z_last2, labels.long())
            else:
                loss2 = F.mse_loss(z_last2, labels.float())
        else:
            loss2 = F.mse_loss(z_last2, acts[-2][:, :out_size])

        if layers[-1].weight.grad is not None:
            layers[-1].weight.grad.zero_()
        loss2.backward()
        with torch.no_grad():
            if layers[-1].weight.grad is not None:
                layers[-1].weight.add_(-eff_lr * layers[-1].weight.grad)
                layers[-1].weight.grad.zero_()
            if layers[-1].bias is not None and layers[-1].bias.grad is not None:
                layers[-1].bias.add_(-eff_lr * layers[-1].bias.grad)
                layers[-1].bias.grad.zero_()

        ctx.metrics["loss"] = float(loss.detach().item())
        ctx.metrics["grad_norm"] = float(e_out.norm().item())
        ctx.metrics["lr"] = eff_lr
        ctx.metrics["skipped"] = 0.0

        if scheduler is not None and getattr(scheduler, "step_per_batch", True):
            scheduler.step()

        if bus is not None:
            bus.dispatch("on_step_end", step=ctx.step, metrics=ctx.metrics, batch=batch, model=model)

        return dict(ctx.metrics)


__all__ = ["DFAUpdateRule"]
