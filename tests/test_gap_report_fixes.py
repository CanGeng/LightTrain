"""Tests for the IMPLEMENTATION_GAPS_REPORT fixes (v0.2.3).

Covers: the objective canonical seam (Finding 1), the update_rule + loss/objective
config validators (Findings 1/2), the arch_profile resolver and _doc_boundary data
signal (Finding 5), and the target_ema callback (Finding 1.3). Demo recipes are
smoke-tested end-to-end (Finding 3).
"""

from __future__ import annotations

import math
import os
import tempfile
import textwrap
from types import SimpleNamespace

import pytest
import torch

from lighttrain.architectures.profile import ArchitectureProfile, LossOnlyObjective
from lighttrain.cli._runtime import (
    _build_arch_profile,
    _build_objective,
    _wire_objective,
)
from lighttrain.config import ConfigError, load_config
from lighttrain.builtin_plugins.losses.core import CrossEntropyLoss
from lighttrain.protocols import ModelOutput


def _write(txt: str) -> str:
    d = tempfile.mkdtemp()
    p = os.path.join(d, "r.yaml")
    with open(p, "w") as fh:
        fh.write(textwrap.dedent(txt))
    return p


# ---------------------------------------------------------------------------
# Finding 1/2 — config-level validators (must surface as ConfigError)
# ---------------------------------------------------------------------------

def test_loss_and_objective_mutually_exclusive():
    with pytest.raises(ConfigError):
        load_config(_write("mode: lab\nloss: {name: cross_entropy}\nobjective: {name: diffusion}\n"))


def test_nested_objective_loss_is_accepted():
    # The recommended nested form must NOT be flagged as a conflict (TC-3).
    c = load_config(_write("mode: lab\nobjective: {name: supervised, loss: {name: my_loss}}\n"))
    assert c.objective["loss"]["name"] == "my_loss"


def test_top_level_update_rule_rejected():
    with pytest.raises(ConfigError):
        load_config(_write("mode: lab\nupdate_rule: {name: mezo}\n"))


# ---------------------------------------------------------------------------
# Finding 1 — _build_objective + LossOnlyObjective
# ---------------------------------------------------------------------------

def test_build_objective_sources():
    assert _build_objective(SimpleNamespace(objective=None, loss=None)) == (None, "none")
    obj, src = _build_objective(SimpleNamespace(objective=None, loss={"name": "cross_entropy"}))
    assert src == "loss" and isinstance(obj, LossOnlyObjective)
    # family inherited from the wrapped loss (CrossEntropyLoss.loss_family) — A1
    assert obj.loss_family == "next_token"


def test_loss_only_objective_stamps_family_and_delegates():
    ctx = SimpleNamespace(loss_family=None)
    o = LossOnlyObjective(CrossEntropyLoss(), loss_family="next_token")
    out = o(
        ModelOutput(outputs={"logits": torch.randn(1, 3, 5)}),
        {"labels": torch.zeros(1, 3, dtype=torch.long)},
        ctx,
    )
    assert ctx.loss_family == "next_token"
    assert "loss" in out


# ---------------------------------------------------------------------------
# Finding 1 — _wire_objective contract (both directions + author bug)
# ---------------------------------------------------------------------------

def _fake_trainer(*, consume=True, prepare=True, require=False):
    class _T:
        consumes_objective = consume
        consumes_objective_prepare = prepare
        requires_objective = require

        def __init__(self):
            self.ctx = SimpleNamespace(loss_fn=None)
            self.objective = None

        def default_objective(self):
            return LossOnlyObjective(CrossEntropyLoss(), loss_family="next_token")

    return _T()


def _fake_engine():
    return SimpleNamespace(loss_fn=None)


def test_wire_consume_none_uses_default():
    t, e = _fake_trainer(), _fake_engine()
    _wire_objective(t, e, None, "none", "pretrain")
    assert isinstance(t.objective, LossOnlyObjective)
    assert t.ctx.loss_fn is t.objective and e.loss_fn is t.objective


def test_wire_requires_objective_raises():
    t = _fake_trainer(require=True)
    with pytest.raises(ConfigError):
        _wire_objective(t, _fake_engine(), None, "none", "preference")


def test_wire_inline_with_loss_raises():
    t = _fake_trainer(consume=False)
    with pytest.raises(ConfigError):
        _wire_objective(t, _fake_engine(), LossOnlyObjective(CrossEntropyLoss()), "loss", "reward_model")


def test_wire_real_objective_to_no_prepare_trainer_raises():
    t = _fake_trainer(prepare=False)
    real_obj = SimpleNamespace(
        loss_family="diffusion",
        prepare_batch=lambda b, *, step, device: b,
        __call__=lambda *a: {"loss": torch.tensor(0.0)},
    )
    with pytest.raises(ConfigError):
        _wire_objective(t, _fake_engine(), real_obj, "objective", "grpo")


def test_wire_inline_none_leaves_objective_none():
    t, e = _fake_trainer(consume=False), _fake_engine()
    out = _wire_objective(t, e, None, "none", "online_distill")
    assert out is None and t.objective is None and e.loss_fn is None


def test_wire_author_bug_consume_false_require_true_typeerror():
    t = _fake_trainer(consume=False, require=True)
    with pytest.raises(TypeError):
        _wire_objective(t, _fake_engine(), None, "none", "BadTrainer")


# ---------------------------------------------------------------------------
# Finding 5 — arch_profile resolver
# ---------------------------------------------------------------------------

def test_arch_profile_resolves_rwkv_string():
    import lighttrain.config._components as C

    C.import_all_components()
    cfg = SimpleNamespace(trainer=SimpleNamespace(arch_profile="rwkv"))
    prof = _build_arch_profile(cfg)
    assert isinstance(prof, ArchitectureProfile)
    assert prof.state_mode == "stateful" and prof.name == "rwkv"


def test_arch_profile_unknown_raises():
    import lighttrain.config._components as C

    C.import_all_components()
    cfg = SimpleNamespace(trainer=SimpleNamespace(arch_profile="nope"))
    with pytest.raises(ConfigError):
        _build_arch_profile(cfg)


def test_arch_profile_object_passthrough_and_none():
    obj = ArchitectureProfile(name="x", loss_family="next_token")
    assert _build_arch_profile(SimpleNamespace(trainer=SimpleNamespace(arch_profile=obj))) is obj
    assert _build_arch_profile(SimpleNamespace(trainer=SimpleNamespace(arch_profile=None))) is None


# ---------------------------------------------------------------------------
# Finding 5 — _doc_boundary data signal
# ---------------------------------------------------------------------------

def test_chunk_size_emits_doc_boundary():
    from lighttrain.builtin_plugins.data.core.datasets import LineFileTextDataset
    from lighttrain.builtin_plugins.data.core.tokenizers import ByteTokenizer

    d = tempfile.mkdtemp()
    p = os.path.join(d, "corpus.txt")
    # one document longer than chunk_size → multiple chunks
    with open(p, "w") as fh:
        fh.write("x" * 40 + "\n")
        fh.write("y" * 5 + "\n")  # short doc → single chunk
    ds = LineFileTextDataset(p, tokenizer=ByteTokenizer(), max_len=64, chunk_size=16)
    flags = [s["_doc_boundary"] for s in ds.samples]
    # doc1: 40 bytes → 3 chunks (T,F,F); doc2: 5 bytes → 1 chunk (T)
    assert flags == [True, False, False, True]


def test_causal_lm_collator_doc_boundary_passthrough_and_batch_guard():
    from lighttrain.builtin_plugins.data.core.collators import CausalLMCollator

    coll = CausalLMCollator(pad_id=0, max_len=16)
    out = coll([{"input_ids": [1, 2, 3], "labels": [1, 2, 3], "_doc_boundary": True}])
    assert out["_doc_boundary"] is True
    # batch_size > 1 with a boundary flag → hard error (Should #4)
    with pytest.raises(ValueError):
        coll([
            {"input_ids": [1, 2], "labels": [1, 2], "_doc_boundary": True},
            {"input_ids": [3, 4], "labels": [3, 4], "_doc_boundary": False},
        ])


def test_rwkv_doc_boundary_resets_only_at_boundaries():
    """Integration: reset_state_fn fires once per document, not per step.

    Drives the full data→collator→produce_batch chain (CPU, no model forward) so
    the assertion isolates the boundary semantics from model numerics.
    """
    from unittest.mock import MagicMock

    import torch.nn as nn
    from torch.utils.data import DataLoader, SequentialSampler

    from lighttrain.builtin_plugins.data.core.collators import CausalLMCollator
    from lighttrain.builtin_plugins.data.core.datasets import LineFileTextDataset
    from lighttrain.builtin_plugins.data.core.tokenizers import ByteTokenizer
    from lighttrain.trainers.base import Trainer

    tok = ByteTokenizer()
    d = tempfile.mkdtemp()
    corpus = os.path.join(d, "corpus.txt")
    with open(corpus, "w") as fh:
        fh.write("a" * 48 + "\n")  # doc1 → 3 chunks (16 each)
        fh.write("bc" * 16 + "\n")  # doc2 → 2 chunks
    ds = LineFileTextDataset(corpus, tokenizer=tok, max_len=64, chunk_size=16)
    n_docs = 2
    n_chunks = len(ds.samples)
    assert n_chunks > n_docs  # boundaries strictly fewer than steps

    loader = DataLoader(
        ds, batch_size=1, sampler=SequentialSampler(ds),
        collate_fn=CausalLMCollator(pad_id=tok.pad_id, max_len=64),
    )
    calls = {"n": 0}
    profile = ArchitectureProfile(
        name="rwkv", loss_family="next_token", state_mode="stateful",
        reset_state_fn=lambda m: calls.__setitem__("n", calls["n"] + 1),
    )
    trainer = Trainer(
        engine=MagicMock(), data_module=MagicMock(),
        optimizer=MagicMock(), model=nn.Linear(2, 2), device="cpu",
        arch_profile=profile,
    )
    for raw in loader:
        batch = trainer.produce_batch(raw)
        assert "_doc_boundary" not in batch  # popped before model forward (Nice #7)
    assert calls["n"] == n_docs


# ---------------------------------------------------------------------------
# Finding 1.3 — target_ema callback
# ---------------------------------------------------------------------------

def test_eval_strips_doc_boundary_before_model_forward():
    """Regression (Must-Fix #2): eval must drop the trainer-only _doc_boundary
    flag so a strict-signature model isn't passed an unexpected kwarg."""
    from unittest.mock import MagicMock

    import torch.nn as nn
    from torch.utils.data import DataLoader

    from lighttrain.protocols import ModelOutput
    from lighttrain.trainers.base import Trainer

    class _StrictModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Linear(2, 2)

        # No **kwargs: an unexpected '_doc_boundary' kwarg would raise TypeError.
        def forward(self, *, input_ids, labels=None, attention_mask=None):
            return ModelOutput(outputs={"logits": torch.zeros(1, 2, 3)})

    dm = MagicMock()
    dm.val_loader.return_value = DataLoader(
        [{"input_ids": torch.zeros(2, dtype=torch.long),
          "labels": torch.zeros(2, dtype=torch.long),
          "_doc_boundary": True}],
        batch_size=1, collate_fn=lambda xs: {
            "input_ids": torch.stack([x["input_ids"] for x in xs]),
            "labels": torch.stack([x["labels"] for x in xs]),
            "_doc_boundary": xs[0]["_doc_boundary"],
        },
    )
    t = Trainer(engine=MagicMock(), data_module=dm, optimizer=MagicMock(),
                model=_StrictModel(), device="cpu")
    t.ctx.loss_fn = lambda out, batch, ctx: {"loss": torch.tensor(0.5)}
    metrics = t.eval()  # must not raise about unexpected '_doc_boundary'
    assert "val_loss" in metrics


def test_target_ema_calls_update_ema_only_when_present():
    from lighttrain.builtin_plugins.callbacks.builtins.target_ema import TargetEMACallback

    cb = TargetEMACallback()
    hits = {"n": 0}

    class _WithEMA:
        def update_ema(self):
            hits["n"] += 1

    cb.on_optimizer_step_post(model=_WithEMA())
    assert hits["n"] == 1
    # no update_ema → harmless no-op (does not raise)
    cb.on_optimizer_step_post(model=SimpleNamespace())
    # reads model from ctx when not passed directly
    cb.on_optimizer_step_post(ctx=SimpleNamespace(model=_WithEMA()))
    assert hits["n"] == 2


# ---------------------------------------------------------------------------
# Finding 3 — demo recipes run end-to-end via user_modules
# ---------------------------------------------------------------------------

_DEMO_OBJECTIVE = {
    "diffusion_eps": "DiffusionObjective",
    "jepa": "JEPAObjective",
    "ff_demo": "LossOnlyObjective",   # default CE (unused by Forward-Forward)
    "pcn_demo": "LossOnlyObjective",
    "mezo_sft": "LossOnlyObjective",
}


@pytest.mark.parametrize("recipe", list(_DEMO_OBJECTIVE))
def test_demo_recipe_runs(recipe, tmp_path):
    from lighttrain.cli._runtime import setup_run_from_config

    cfg = load_config(
        f"recipes/{recipe}.yaml",
        overrides=[
            f"run_root={tmp_path}",
            "trainer.max_steps=2",
            "trainer.ckpt_every=0",
            "trainer.log_every=1",
        ],
    )
    bundle = setup_run_from_config(cfg)
    trainer = bundle["trainer"]
    assert type(trainer.objective).__name__ == _DEMO_OBJECTIVE[recipe]
    metrics = trainer.fit()
    assert "loss" in metrics and math.isfinite(float(metrics["loss"]))


def test_jepa_recipe_advances_target_encoder(tmp_path):
    """Regression (Finding 1.3): the `target_ema` callback in jepa.yaml must
    advance JEPAModel.target_encoder — guards against it silently freezing at
    init (the original bug) if the callback is dropped."""
    from lighttrain.cli._runtime import setup_run_from_config

    cfg = load_config(
        "recipes/jepa.yaml",
        overrides=[f"run_root={tmp_path}", "trainer.max_steps=3",
                   "trainer.ckpt_every=0", "trainer.log_every=1"],
    )
    bundle = setup_run_from_config(cfg)
    trainer = bundle["trainer"]
    p = next(iter(trainer.model.target_encoder.parameters()))
    before = p.detach().clone()
    trainer.fit()
    assert not torch.equal(before, p.detach())  # EMA target drifted


def test_rwkv_recipe_wires_arch_profile_object(tmp_path):
    """Regression (Finding 5): setup_run_from_config must resolve the
    `arch_profile: rwkv` string to an ArchitectureProfile object and trigger the
    stateful reset path during fit — guards against the bare-string regression."""
    from lighttrain.cli._runtime import setup_run_from_config

    cfg = load_config(
        "recipes/pretrain_rwkv.yaml",
        overrides=[f"run_root={tmp_path}", "trainer.max_steps=3",
                   "trainer.ckpt_every=0", "trainer.log_every=1"],
    )
    bundle = setup_run_from_config(cfg)
    trainer = bundle["trainer"]
    assert isinstance(trainer.arch_profile, ArchitectureProfile)
    assert trainer.arch_profile.state_mode == "stateful"
    calls = {"n": 0}
    orig = trainer.arch_profile.reset_state_fn
    trainer.arch_profile.reset_state_fn = lambda m: (calls.__setitem__("n", calls["n"] + 1), orig(m))[1]
    trainer.fit()
    assert calls["n"] >= 1  # the document-boundary reset path fired


def test_x_labels_one_hot_matches_mlp_toy_top_for_pcn_clamp():
    """M4: x_labels emits (B, output_dim) one-hot so the PCN supervised clamp
    (labels.shape == top_activation.shape) actually fires."""
    from examples.lab_components import MLPToy, XLabelsCollator

    coll = XLabelsCollator(num_classes=2)
    batch = coll([{"x": torch.randn(16), "label": 1}, {"x": torch.randn(16), "label": 0}])
    model = MLPToy(input_dim=16, hidden_dim=32, output_dim=2, num_layers=3)
    h = batch["x"]
    for layer in model.layers:
        h = torch.relu(layer(h))
    assert batch["labels"].shape == h.shape  # PCN clamp condition holds
