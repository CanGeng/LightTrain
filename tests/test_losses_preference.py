"""Preference loss tests (M6) — BT / DPO / IPO / SimPO / ORPO / KTO."""

from __future__ import annotations

import pytest
import torch

from lighttrain.builtin_plugins.losses.preference import (
    BradleyTerryLoss,
    DPOLoss,
    IPOLoss,
    KTOLoss,
    ORPOLoss,
    SimPOLoss,
)
from lighttrain.protocols import LossContext, ModelOutput

_DUMMY_OUT = ModelOutput(outputs={})
_CTX = LossContext()


def _batch(B: int = 4) -> dict:
    chosen = torch.randn(B) - 1.0      # negative log-probs
    rejected = torch.randn(B) - 2.0    # more negative (worse)
    return {
        "chosen_logps": chosen,
        "rejected_logps": rejected,
        "chosen_nll_loss": -chosen,
        "ref_chosen_logps": chosen.clone() + 0.1,
        "ref_rejected_logps": rejected.clone() + 0.1,
    }


# ---- BradleyTerry --------------------------------------------------------

def test_bt_loss_positive():
    b = _batch()
    out = BradleyTerryLoss()(_DUMMY_OUT, b, _CTX)
    assert out["loss"] > 0


def test_bt_loss_chosen_wins_lower():
    """When chosen >> rejected, BT loss should be near 0."""
    b = {
        "chosen_logps": torch.tensor([0.0, 0.0, 0.0]),
        "rejected_logps": torch.tensor([-10.0, -10.0, -10.0]),
    }
    out = BradleyTerryLoss()(_DUMMY_OUT, b, _CTX)
    assert float(out["loss"]) < 0.1


def test_bt_margin_shifts_loss():
    b = _batch()
    out_no = BradleyTerryLoss(margin=0.0)(_DUMMY_OUT, b, _CTX)
    out_mg = BradleyTerryLoss(margin=5.0)(_DUMMY_OUT, b, _CTX)
    assert float(out_mg["loss"]) > float(out_no["loss"])


# ---- DPO -----------------------------------------------------------------

def test_dpo_loss_positive():
    out = DPOLoss()(_DUMMY_OUT, _batch(), _CTX)
    assert out["loss"] > 0


def test_dpo_returns_accuracy():
    out = DPOLoss()(_DUMMY_OUT, _batch(), _CTX)
    assert "dpo_accuracy" in out
    assert 0.0 <= out["dpo_accuracy"] <= 1.0


def test_dpo_missing_ref_raises():
    b = {k: v for k, v in _batch().items() if not k.startswith("ref_")}
    with pytest.raises(KeyError):
        DPOLoss()(_DUMMY_OUT, b, _CTX)


# ---- IPO -----------------------------------------------------------------

def test_ipo_loss_nonneg():
    out = IPOLoss()(_DUMMY_OUT, _batch(), _CTX)
    assert float(out["loss"]) >= 0.0


def test_ipo_identity_zero_when_logratios_match():
    """If π and ref have identical logratios and β=0.5, h = 0 → loss = 1/4β²=1."""
    B = 3
    chosen = torch.zeros(B)
    rejected = torch.zeros(B)
    b = {
        "chosen_logps": chosen,
        "rejected_logps": rejected,
        "ref_chosen_logps": chosen.clone(),
        "ref_rejected_logps": rejected.clone(),
    }
    out = IPOLoss(beta=0.5)(_DUMMY_OUT, b, _CTX)
    expected = (1.0 / (2.0 * 0.5)) ** 2
    assert abs(float(out["loss"]) - expected) < 1e-5


# ---- SimPO ---------------------------------------------------------------

def test_simpo_no_ref_keys_needed():
    b = {"chosen_logps": torch.tensor([-0.5]), "rejected_logps": torch.tensor([-1.5])}
    out = SimPOLoss()(_DUMMY_OUT, b, _CTX)
    assert "loss" in out


def test_simpo_accuracy_binary():
    b = _batch()
    out = SimPOLoss()(_DUMMY_OUT, b, _CTX)
    assert "simpo_accuracy" in out


# ---- ORPO ----------------------------------------------------------------

def test_orpo_loss_has_sft_component():
    out = ORPOLoss()(_DUMMY_OUT, _batch(), _CTX)
    assert "sft_loss" in out and "ratio_loss" in out
    assert float(out["sft_loss"]) >= 0.0


# ---- KTO -----------------------------------------------------------------

def test_kto_loss_positive():
    out = KTOLoss()(_DUMMY_OUT, _batch(), _CTX)
    assert float(out["loss"]) > 0.0


# ---- SimPO gamma fix (bug fix verification) ---------------------------------

def test_simpo_gamma_affects_loss():
    """gamma must not cancel in the difference — gamma=0 vs gamma=2 must yield different losses."""
    torch.manual_seed(0)
    c = torch.tensor([1.0, 0.5])
    r = torch.tensor([0.0, -0.5])
    b = {"chosen_logps": c, "rejected_logps": r}
    loss_g0 = float(SimPOLoss(beta=1.0, gamma=0.0)(_DUMMY_OUT, b, _CTX)["loss"])
    loss_g2 = float(SimPOLoss(beta=1.0, gamma=2.0)(_DUMMY_OUT, b, _CTX)["loss"])
    assert loss_g0 != loss_g2, "gamma cancels in SimPO — bug not fixed"


def test_simpo_gamma_zero_equals_no_margin():
    """With gamma=0, logits = beta*(chosen - rejected) exactly (reference formula)."""
    c = torch.tensor([0.5])
    r = torch.tensor([-0.5])
    b = {"chosen_logps": c, "rejected_logps": r}
    import torch.nn.functional as F
    out = SimPOLoss(beta=2.0, gamma=0.0)(_DUMMY_OUT, b, _CTX)
    expected_logit = 2.0 * (0.5 - (-0.5))   # = 2.0
    expected_loss = float(-F.logsigmoid(torch.tensor(expected_logit)))
    assert abs(float(out["loss"]) - expected_loss) < 1e-5
