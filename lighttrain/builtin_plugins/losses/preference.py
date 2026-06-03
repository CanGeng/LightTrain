"""Preference losses — BT / DPO / IPO / SimPO / ORPO / KTO.

Batch contract
--------------
All losses read the following keys from ``batch``, computed and injected by
:class:`~lighttrain.builtin_plugins.trainers._preference_base.PreferenceTrainerMixin` before
the loss call:

    chosen_logps          (B,)  — mean per-token log-prob under current policy
    rejected_logps        (B,)  — mean per-token log-prob under current policy
    chosen_nll_loss       (B,)  — mean per-token NLL on chosen (ORPO SFT term)

Reference-model losses (BT / DPO / IPO / KTO) additionally require:

    ref_chosen_logps      (B,)  — pre-computed from artifact / freeze
    ref_rejected_logps    (B,)  — pre-computed from artifact / freeze

SimPO and ORPO are reference-free and ignore ``ref_*`` keys.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F

from lighttrain.protocols import LossContext, ModelOutput
from lighttrain.registry import register


def _require(batch: Mapping[str, Any], key: str, caller: str) -> torch.Tensor:
    if key not in batch:
        raise KeyError(
            f"{caller} needs '{key}' in the batch. "
            "Is PreferenceTrainerMixin computing and injecting log-probs?"
        )
    t = batch[key]
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t)
    return t


def _log1mexp(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable log(1 - exp(x)) for x < 0."""
    out = torch.empty_like(x)
    # For x close to 0 (> log(0.5) ≈ -0.693), use expm1 path.
    mask = x > -0.6931472
    if mask.any():
        out[mask] = torch.log(-torch.expm1(x[mask]))
    if (~mask).any():
        out[~mask] = torch.log1p(-torch.exp(x[~mask]))
    return out


# ---------------------------------------------------------------------------
# Bradley-Terry
# ---------------------------------------------------------------------------


@register("loss", "bradley_terry")
@register("loss", "bt")
class BradleyTerryLoss:
    """Pairwise Bradley-Terry preference loss.

    Treats the policy log-probability as the implicit reward, so no reference
    model is required. Optionally a reward ``margin`` can be applied.

    Loss = -logsigmoid(chosen_logps - rejected_logps - margin).mean()
    """

    def __init__(self, *, margin: float = 0.0) -> None:
        self.margin = float(margin)

    def __call__(
        self,
        model_output: ModelOutput | Mapping[str, Any],  # noqa: ARG002
        batch: Mapping[str, Any],
        ctx: LossContext,  # noqa: ARG002
    ) -> dict[str, Any]:
        chosen = _require(batch, "chosen_logps", "BradleyTerryLoss")
        rejected = _require(batch, "rejected_logps", "BradleyTerryLoss")
        rewards = chosen - rejected - self.margin
        loss = -F.logsigmoid(rewards).mean()
        return {
            "loss": loss,
            "reward_chosen": float(chosen.mean().detach()),
            "reward_rejected": float(rejected.mean().detach()),
            "reward_margin": float((chosen - rejected).mean().detach()),
        }


# ---------------------------------------------------------------------------
# DPO
# ---------------------------------------------------------------------------


@register("loss", "dpo")
class DPOLoss:
    """Direct Preference Optimization (Rafailov et al., 2023).

    Loss = -logsigmoid(β * ((π_chosen - ref_chosen) - (π_rejected - ref_rejected))).mean()
    """

    def __init__(self, *, beta: float = 0.1) -> None:
        self.beta = float(beta)

    def __call__(
        self,
        model_output: ModelOutput | Mapping[str, Any],  # noqa: ARG002
        batch: Mapping[str, Any],
        ctx: LossContext,  # noqa: ARG002
    ) -> dict[str, Any]:
        chosen = _require(batch, "chosen_logps", "DPOLoss")
        rejected = _require(batch, "rejected_logps", "DPOLoss")
        ref_chosen = _require(batch, "ref_chosen_logps", "DPOLoss")
        ref_rejected = _require(batch, "ref_rejected_logps", "DPOLoss")

        pi_logratios = chosen - rejected
        ref_logratios = ref_chosen - ref_rejected
        logits = self.beta * (pi_logratios - ref_logratios)
        loss = -F.logsigmoid(logits).mean()

        chosen_reward = self.beta * (chosen - ref_chosen).detach()
        rejected_reward = self.beta * (rejected - ref_rejected).detach()
        return {
            "loss": loss,
            "reward_chosen": float(chosen_reward.mean()),
            "reward_rejected": float(rejected_reward.mean()),
            "reward_margin": float((chosen_reward - rejected_reward).mean()),
            "dpo_accuracy": float((logits > 0).float().mean().detach()),
        }


# ---------------------------------------------------------------------------
# IPO
# ---------------------------------------------------------------------------


@register("loss", "ipo")
class IPOLoss:
    """Identity Preference Optimization (Azar et al., 2023).

    Replaces the log-sigmoid with a squared loss, making it less brittle near
    the boundary.

    Loss = ((π_logratios - ref_logratios) - 1/(2β))².mean()
    """

    def __init__(self, *, beta: float = 0.1) -> None:
        self.beta = float(beta)

    def __call__(
        self,
        model_output: ModelOutput | Mapping[str, Any],  # noqa: ARG002
        batch: Mapping[str, Any],
        ctx: LossContext,  # noqa: ARG002
    ) -> dict[str, Any]:
        chosen = _require(batch, "chosen_logps", "IPOLoss")
        rejected = _require(batch, "rejected_logps", "IPOLoss")
        ref_chosen = _require(batch, "ref_chosen_logps", "IPOLoss")
        ref_rejected = _require(batch, "ref_rejected_logps", "IPOLoss")

        h = (chosen - rejected) - (ref_chosen - ref_rejected) - 1.0 / (2.0 * self.beta)
        loss = h.pow(2).mean()
        return {
            "loss": loss,
            "ipo_h_mean": float(h.mean().detach()),
        }


# ---------------------------------------------------------------------------
# SimPO
# ---------------------------------------------------------------------------


@register("loss", "simpo")
class SimPOLoss:
    """Simple Preference Optimization (Yuan et al., 2024).

    Reference-free. Uses length-normalized (mean per-token) log-probs shifted
    by a target-reward margin γ.

    Loss = -logsigmoid(β * ((chosen_logps - γ) - (rejected_logps - γ))).mean()
         = -logsigmoid(β * (chosen_logps - rejected_logps)).mean()
    (γ cancels in the difference; it acts as a win-rate anchor when combined
    with the margin floor: only pairs where chosen > rejected + 2γ/β are
    considered clear wins.)
    """

    def __init__(self, *, beta: float = 2.5, gamma: float = 1.0) -> None:
        self.beta = float(beta)
        self.gamma = float(gamma)

    def __call__(
        self,
        model_output: ModelOutput | Mapping[str, Any],  # noqa: ARG002
        batch: Mapping[str, Any],
        ctx: LossContext,  # noqa: ARG002
    ) -> dict[str, Any]:
        chosen = _require(batch, "chosen_logps", "SimPOLoss")
        rejected = _require(batch, "rejected_logps", "SimPOLoss")

        logits = self.beta * (chosen - rejected) - self.gamma
        loss = -F.logsigmoid(logits).mean()
        return {
            "loss": loss,
            "reward_chosen": float(chosen.mean().detach()),
            "reward_rejected": float(rejected.mean().detach()),
            "simpo_accuracy": float((logits > 0).float().mean().detach()),
        }


# ---------------------------------------------------------------------------
# ORPO
# ---------------------------------------------------------------------------


@register("loss", "orpo")
class ORPOLoss:
    """Odds Ratio Preference Optimization (Hong et al., 2024).

    Reference-free. Combines SFT loss on chosen with an odds-ratio penalty.

    Loss = NLL_chosen + λ * (-logsigmoid(log_odds_chosen - log_odds_rejected))

    Batch requires ``chosen_nll_loss`` (B,) in addition to the log-prob keys.
    """

    def __init__(self, *, lam: float = 1.0) -> None:
        self.lam = float(lam)

    def __call__(
        self,
        model_output: ModelOutput | Mapping[str, Any],  # noqa: ARG002
        batch: Mapping[str, Any],
        ctx: LossContext,  # noqa: ARG002
    ) -> dict[str, Any]:
        chosen = _require(batch, "chosen_logps", "ORPOLoss")
        rejected = _require(batch, "rejected_logps", "ORPOLoss")
        nll = _require(batch, "chosen_nll_loss", "ORPOLoss")

        # log_odds = log_p - log(1-p) = logps - log1mexp(logps)
        log_odds_chosen = chosen - _log1mexp(chosen.clamp(max=-1e-7))
        log_odds_rejected = rejected - _log1mexp(rejected.clamp(max=-1e-7))
        log_odds_ratio = log_odds_chosen - log_odds_rejected

        sft_loss = nll.mean()
        ratio_loss = -F.logsigmoid(log_odds_ratio).mean()
        loss = sft_loss + self.lam * ratio_loss
        return {
            "loss": loss,
            "sft_loss": float(sft_loss.detach()),
            "ratio_loss": float(ratio_loss.detach()),
            "log_odds_ratio": float(log_odds_ratio.mean().detach()),
        }


# ---------------------------------------------------------------------------
# KTO
# ---------------------------------------------------------------------------


@register("loss", "kto")
class KTOLoss:
    """Kahneman-Tversky Optimization (Ethayarajh et al., 2023).

    Estimates the KL term from the within-batch chosen examples.

    loss_chosen   = λ_D * (1 - σ(β * (r_chosen - KL)))
    loss_rejected = λ_U * (1 - σ(β * (KL - r_rejected)))
    loss = mean(loss_chosen + loss_rejected) / 2
    """

    def __init__(
        self,
        *,
        beta: float = 0.1,
        lambda_desirable: float = 1.0,
        lambda_undesirable: float = 1.0,
    ) -> None:
        self.beta = float(beta)
        self.lambda_d = float(lambda_desirable)
        self.lambda_u = float(lambda_undesirable)

    def __call__(
        self,
        model_output: ModelOutput | Mapping[str, Any],  # noqa: ARG002
        batch: Mapping[str, Any],
        ctx: LossContext,  # noqa: ARG002
    ) -> dict[str, Any]:
        chosen = _require(batch, "chosen_logps", "KTOLoss")
        rejected = _require(batch, "rejected_logps", "KTOLoss")
        ref_chosen = _require(batch, "ref_chosen_logps", "KTOLoss")
        ref_rejected = _require(batch, "ref_rejected_logps", "KTOLoss")

        r_chosen = chosen - ref_chosen
        r_rejected = rejected - ref_rejected
        # KL estimate: mean implicit reward of chosen (desirable) examples.
        kl = r_chosen.detach().mean()

        loss_chosen = self.lambda_d * (1.0 - torch.sigmoid(self.beta * (r_chosen - kl)))
        loss_rejected = self.lambda_u * (1.0 - torch.sigmoid(self.beta * (kl - r_rejected)))
        loss = (loss_chosen + loss_rejected).mean() / 2.0
        return {
            "loss": loss,
            "kto_kl": float(kl.detach()),
            "reward_chosen": float(r_chosen.mean().detach()),
            "reward_rejected": float(r_rejected.mean().detach()),
        }


__all__ = [
    "BradleyTerryLoss",
    "DPOLoss",
    "IPOLoss",
    "KTOLoss",
    "ORPOLoss",
    "SimPOLoss",
]
