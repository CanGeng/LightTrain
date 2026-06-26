"""DiffusionObjective — DDPM eps / x0 / v prediction.

Implements three prediction targets for denoising diffusion probabilistic
models (Ho et al., 2020; Salimans & Ho 2022 for v-prediction):

    * "eps"  — predict the added noise (default, Ho 2020)
    * "x0"   — predict the clean data directly
    * "v"    — predict the velocity v = √ᾱ·ε − √(1−ᾱ)·x₀ (Salimans 2022)

The objective injects noise into ``batch["x"]`` during ``prepare_batch`` and
reads the model's ``outputs.outputs["pred"]`` tensor for loss computation.

Noise schedule: linear (default) or cosine (Nichol & Dhariwal 2021).

Usage in YAML::

    objective:
      name: diffusion
      target: eps          # eps | x0 | v
      noise_schedule: linear   # linear | cosine
      timesteps: 1000
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from lighttrain.protocols import LossContext, ModelOutput
from lighttrain.registry import register

# ---------------------------------------------------------------------------
# Noise schedule helpers
# ---------------------------------------------------------------------------

def _linear_schedule(T: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod) — shape (T,)."""
    betas = torch.linspace(1e-4, 0.02, T, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return alphas_cumprod.sqrt(), (1.0 - alphas_cumprod).sqrt()


def _cosine_schedule(T: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosine schedule (Nichol & Dhariwal 2021)."""
    s = 0.008
    steps = torch.arange(T + 1, device=device, dtype=torch.float32)
    f = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = f / f[0]
    alphas_cumprod = alphas_cumprod[1:]   # shape (T,)
    return alphas_cumprod.sqrt(), (1.0 - alphas_cumprod).sqrt()


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

@register("objective", "diffusion")
class DiffusionObjective:
    """DDPM diffusion objective (eps / x0 / v prediction).

    Args:
        target: Prediction target — "eps" | "x0" | "v".
        noise_schedule: "linear" | "cosine".
        timesteps: Total diffusion steps T.
        loss_weight: Optional per-timestep weighting ("uniform" or "snr").
    """

    loss_family: str = "diffusion"

    def __init__(
        self,
        target: str = "eps",
        noise_schedule: str = "linear",
        timesteps: int = 1000,
        loss_weight: str = "uniform",
    ) -> None:
        if target not in ("eps", "x0", "v"):
            raise ValueError(f"Unknown diffusion target '{target}'. Use eps/x0/v.")
        if noise_schedule not in ("linear", "cosine"):
            raise ValueError(f"Unknown noise schedule '{noise_schedule}'.")
        self.target = target
        self.noise_schedule = noise_schedule
        self.timesteps = timesteps
        self.loss_weight = loss_weight
        # schedule tensors are built lazily (device not known at init)
        self._sqrt_acp: torch.Tensor | None = None
        self._sqrt_one_minus_acp: torch.Tensor | None = None
        self._schedule_device: torch.device | None = None

    def _ensure_schedule(self, device: Any) -> None:
        dev = torch.device(device) if device is not None else torch.device("cpu")
        if self._schedule_device != dev or self._sqrt_acp is None:
            if self.noise_schedule == "linear":
                self._sqrt_acp, self._sqrt_one_minus_acp = _linear_schedule(
                    self.timesteps, dev
                )
            else:
                self._sqrt_acp, self._sqrt_one_minus_acp = _cosine_schedule(
                    self.timesteps, dev
                )
            self._schedule_device = dev

    # ------------------------------------------------------------------
    # ObjectiveProfile protocol
    # ------------------------------------------------------------------

    def prepare_batch(self, batch: dict, *, step: int, device: Any) -> dict:
        """Inject Gaussian noise into ``batch["x"]``.

        Expects ``batch["x"]`` — clean data tensor of any shape (B, …).
        Adds:
            * ``batch["noisy_x"]``  — x_t
            * ``batch["noise"]``    — sampled ε
            * ``batch["t"]``        — integer timesteps (B,)
        """
        self._ensure_schedule(device)
        assert self._sqrt_acp is not None and self._sqrt_one_minus_acp is not None
        x0 = batch["x"]
        B = x0.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=x0.device)
        noise = torch.randn_like(x0)

        sqrt_acp = self._sqrt_acp[t].view(B, *([1] * (x0.dim() - 1)))           # (B, 1, …)
        sqrt_omacp = self._sqrt_one_minus_acp[t].view(B, *([1] * (x0.dim() - 1)))

        noisy_x = sqrt_acp * x0 + sqrt_omacp * noise
        batch = {**batch, "noisy_x": noisy_x, "noise": noise, "t": t}
        return batch

    def __call__(
        self,
        outputs: ModelOutput,
        batch: dict,
        ctx: LossContext,
    ) -> dict:
        """Compute diffusion MSE loss between model prediction and target."""
        ctx.loss_family = self.loss_family

        pred = outputs.outputs.get("pred")
        if pred is None:
            raise KeyError(
                "DiffusionObjective expects ModelOutput.outputs['pred']. "
                "Ensure the model returns {'pred': tensor}."
            )

        x0 = batch["x"]
        noise = batch["noise"]
        t = batch["t"]
        self._ensure_schedule(x0.device)
        assert self._sqrt_acp is not None and self._sqrt_one_minus_acp is not None

        if self.target == "eps":
            gt = noise
        elif self.target == "x0":
            gt = x0
        else:  # v
            sqrt_acp = self._sqrt_acp[t].view(x0.shape[0], *([1] * (x0.dim() - 1)))
            sqrt_omacp = self._sqrt_one_minus_acp[t].view(x0.shape[0], *([1] * (x0.dim() - 1)))
            gt = sqrt_acp * noise - sqrt_omacp * x0

        mse = F.mse_loss(pred, gt, reduction="mean")
        return {"loss": mse, "diffusion_mse": mse.detach()}


__all__ = ["DiffusionObjective"]
