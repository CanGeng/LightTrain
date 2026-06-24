"""Adversarial tests for ``lighttrain.builtin_plugins.architectures.jepa``.

Coverage beyond ``tests/test_architectures_jepa.py``:

* **EMA closed-form at momentum=0 and momentum=1 endpoints** — these are
  the most error-prone corners of any EMA update.
* **EMA initialization is a deep copy** — target_encoder's params start
  identical to encoder's, by VALUE not just shape.
* **Target encoder forward output has requires_grad=False** — the JEPA
  paper's stop-gradient on the target branch.
* **JEPA forward output shape contract** — pred_embeddings is
  ``(B, num_target, embed_dim)``; target_embeddings is same.
* **EMA update only mutates target encoder, not source** — important
  invariant for the JEPA training loop.
* **jepa_profile block iterator yields encoder blocks + predictor blocks**
  in that order.
"""

from __future__ import annotations

import torch

from lighttrain.builtin_plugins.architectures.jepa import (
    EMATargetEncoder,
    JEPAEncoder,
    JEPAModel,
    JEPAModelConfig,
    JEPAPredictor,
    jepa_profile,
)


def _tiny_cfg(**overrides) -> JEPAModelConfig:
    """Tiny but valid JEPA config used by all tests."""
    base = dict(
        patch_dim=8, embed_dim=16, num_heads=4,
        depth=2, mlp_ratio=2.0, dropout=0.0, predictor_depth=2,
    )
    base.update(overrides)
    return JEPAModelConfig(**base)


# ---------------------------------------------------------------------------
# Component forward-shape contracts
# ---------------------------------------------------------------------------

def test_jepa_encoder_forward_output_shape():
    """``JEPAEncoder(patches)`` maps ``(B, T, patch_dim)`` → ``(B, T, embed_dim)``."""
    cfg = _tiny_cfg(patch_dim=16, embed_dim=32)
    enc = JEPAEncoder(cfg)
    patches = torch.randn(2, 10, 16)
    out = enc(patches)
    assert out.shape == (2, 10, 32)


def test_jepa_predictor_forward_output_shape():
    """``JEPAPredictor(context, target_pos)`` returns one embedding per target
    position: ``(B, num_target, embed_dim)``.
    """
    cfg = _tiny_cfg(embed_dim=32, num_heads=2)
    predictor = JEPAPredictor(cfg)
    context = torch.randn(2, 6, 32)
    target_pos = torch.randn(2, 4, 32)
    out = predictor(context, target_pos)
    assert out.shape == (2, 4, 32)


# ---------------------------------------------------------------------------
# EMA target encoder
# ---------------------------------------------------------------------------

def test_invariant_ema_target_initialized_to_deep_copy_of_source():
    """Invariant: at construction, every target encoder param equals the
    source encoder param by VALUE (deep copy).

    Setup: build encoder + EMA target wrapper.
    Expected: ``assert_close`` element-wise on every parameter pair.
    """
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    enc = JEPAEncoder(cfg)
    tgt = EMATargetEncoder(enc)

    src_params = dict(enc.named_parameters())
    tgt_params = dict(tgt.encoder.named_parameters())
    assert set(src_params) == set(tgt_params), (
        f"param-key mismatch between source and target: "
        f"{set(src_params) ^ set(tgt_params)}"
    )
    for name, p in src_params.items():
        torch.testing.assert_close(
            tgt_params[name], p, atol=1e-5, rtol=1e-4
        )


def test_invariant_ema_target_params_have_no_grad():
    """Invariant: every target encoder parameter has ``requires_grad=False``.

    Goal: target branch must not be in the optimizer's param list, so any
    parameter that's still ``requires_grad=True`` would be silently trained
    and break the JEPA contract.
    """
    enc = JEPAEncoder(_tiny_cfg())
    tgt = EMATargetEncoder(enc)
    leaks = [name for name, p in tgt.named_parameters() if p.requires_grad]
    assert not leaks, (
        f"target encoder params must all be frozen; trainable leaks: {leaks[:5]}"
    )


def test_invariant_ema_momentum_one_keeps_target_unchanged():
    """Closed-form: at momentum=1.0, the EMA update has no effect.

    Setup: build EMA wrapper with momentum=1.0. Take a snapshot. Set source
    to all zeros (very different). Call update. Compare target to snapshot.
    Expected: target unchanged — momentum=1 means "never adopt source".
    """
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    src = JEPAEncoder(cfg)
    tgt = EMATargetEncoder(src, momentum=1.0)

    snapshot = {n: p.detach().clone() for n, p in tgt.named_parameters()}
    # Drastically change source
    with torch.no_grad():
        for p in src.parameters():
            p.zero_()

    tgt.update(src)

    for name, p in tgt.named_parameters():
        torch.testing.assert_close(p, snapshot[name], atol=1e-5, rtol=1e-4)


def test_invariant_ema_momentum_zero_copies_source_into_target():
    """Closed-form: at momentum=0.0, the EMA update fully adopts the source.

    Setup: build EMA wrapper with momentum=0.0. Drastically change source.
    Update.
    Expected: target params equal source params element-wise.
    """
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    src = JEPAEncoder(cfg)
    tgt = EMATargetEncoder(src, momentum=0.0)

    with torch.no_grad():
        for p in src.parameters():
            p.fill_(1.5)

    tgt.update(src)

    src_params = dict(src.named_parameters())
    for name, p in tgt.encoder.named_parameters():
        torch.testing.assert_close(p, src_params[name], atol=1e-5, rtol=1e-4)


def test_invariant_ema_update_formula_at_arbitrary_momentum():
    """Closed-form: ``pt' = m·pt + (1-m)·ps`` for momentum=0.7.

    Setup: build EMA wrapper. Snapshot target. Set source to known constants.
    Call update.
    Expected: every target param == m·(snapshot) + (1-m)·(source) within
    assert_close.
    """
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    src = JEPAEncoder(cfg)
    m = 0.7
    tgt = EMATargetEncoder(src, momentum=m)

    # Snapshot the target's BEFORE values.
    before = {n: p.detach().clone() for n, p in tgt.encoder.named_parameters()}
    # Set source to known constants so we can hand-derive the expected.
    with torch.no_grad():
        for p in src.parameters():
            p.fill_(2.0)
    src_after = {n: p.detach().clone() for n, p in src.named_parameters()}

    tgt.update(src)

    for name, p in tgt.encoder.named_parameters():
        expected = m * before[name] + (1.0 - m) * src_after[name]
        torch.testing.assert_close(p, expected, atol=1e-5, rtol=1e-4)


def test_invariant_ema_update_does_not_mutate_source():
    """Invariant: ``target.update(source)`` does NOT mutate source params.

    Setup: snapshot source; update target; compare source to snapshot.
    Expected: source unchanged element-wise.
    """
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    src = JEPAEncoder(cfg)
    tgt = EMATargetEncoder(src, momentum=0.5)

    snap = {n: p.detach().clone() for n, p in src.named_parameters()}
    tgt.update(src)
    for name, p in src.named_parameters():
        torch.testing.assert_close(p, snap[name], atol=1e-5, rtol=1e-4)


def test_invariant_ema_forward_output_has_no_grad():
    """Invariant: ``target_encoder(patches)`` returns a tensor with
    ``requires_grad=False`` (stop-gradient on the target branch — JEPA core).

    Setup: build target encoder; pass through grad-requiring patches.
    Expected: output's ``requires_grad`` is False AND ``grad_fn`` is None.
    """
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    enc = JEPAEncoder(cfg)
    tgt = EMATargetEncoder(enc)

    patches = torch.randn(2, 5, cfg.patch_dim, requires_grad=True)
    out = tgt(patches)
    assert out.requires_grad is False
    assert out.grad_fn is None


# ---------------------------------------------------------------------------
# JEPAModel forward
# ---------------------------------------------------------------------------

def test_invariant_jepa_model_forward_output_shape_contract():
    """Invariant: ``JEPAModel.forward(**batch)`` returns a ModelOutput where:
      * outputs["pred_embeddings"].shape == (B, num_target, embed_dim)
      * extras["target_embeddings"].shape == (B, num_target, embed_dim)

    Setup: tiny config, batch with B=2, num_context=4, num_target=3.
    Expected: both shapes match exactly.
    """
    torch.manual_seed(0)
    cfg = _tiny_cfg(embed_dim=16)
    model = JEPAModel(cfg)
    model.eval()

    B, nc, nt = 2, 4, 3
    batch = {
        "context_patches": torch.randn(B, nc, cfg.patch_dim),
        "target_patches": torch.randn(B, nt, cfg.patch_dim),
        "target_idx": torch.randint(0, 100, (B, nt)),
    }
    out = model(**batch)
    assert out.outputs["pred_embeddings"].shape == (B, nt, cfg.embed_dim)
    assert out.extras["target_embeddings"].shape == (B, nt, cfg.embed_dim)


def test_invariant_jepa_target_embeddings_have_no_grad_in_full_model_forward():
    """Invariant: target_embeddings in ModelOutput.extras come from the
    no-grad target encoder; the gradient graph should NOT flow through them.

    Setup: build JEPAModel; run forward; check that
    ``out.extras["target_embeddings"].grad_fn is None``.
    """
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = JEPAModel(cfg)
    model.train()

    B, nc, nt = 2, 4, 3
    batch = {
        "context_patches": torch.randn(B, nc, cfg.patch_dim, requires_grad=True),
        "target_patches": torch.randn(B, nt, cfg.patch_dim, requires_grad=True),
    }
    out = model(**batch)
    assert out.extras["target_embeddings"].grad_fn is None
    assert out.extras["target_embeddings"].requires_grad is False
    # The pred branch should still be in the graph
    assert out.outputs["pred_embeddings"].requires_grad is True


def test_jepa_forward_uses_arange_when_target_idx_missing():
    """``target_idx`` absent → model defaults to arange(num_target) per row.

    Setup: provide ``context_patches`` and ``target_patches`` but no
    ``target_idx``.
    Expected: forward succeeds AND pred shape is (B, num_target, embed_dim).
    """
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = JEPAModel(cfg)
    model.eval()

    B, nc, nt = 2, 4, 3
    batch = {
        "context_patches": torch.randn(B, nc, cfg.patch_dim),
        "target_patches": torch.randn(B, nt, cfg.patch_dim),
    }
    out = model(**batch)
    assert out.outputs["pred_embeddings"].shape == (B, nt, cfg.embed_dim)


def test_jepa_update_ema_calls_through_to_target_encoder():
    """``JEPAModel.update_ema()`` is a delegating shim → calls
    ``self.target_encoder.update(self.encoder)``.

    Setup: monkey-patch ``model.encoder``'s weights to differ from the
    deep-copy snapshot; call ``update_ema()`` with momentum=0; assert
    target encoder weights now match the (mutated) source.
    """
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = JEPAModel(cfg)
    # Replace target encoder with a momentum=0 variant so update fully copies.
    model.target_encoder = EMATargetEncoder(model.encoder, momentum=0.0)
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.fill_(3.14)

    model.update_ema()

    enc_params = dict(model.encoder.named_parameters())
    for name, p in model.target_encoder.encoder.named_parameters():
        torch.testing.assert_close(p, enc_params[name], atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# jepa_profile
# ---------------------------------------------------------------------------

def test_pin_jepa_profile_name_and_loss_family():
    """Pin: jepa_profile().name == 'jepa', loss_family == 'jepa',
    state_mode == 'stateless'.
    """
    p = jepa_profile()
    assert p.name == "jepa"
    assert p.loss_family == "jepa"
    assert p.state_mode == "stateless"


def test_jepa_profile_iter_blocks_yields_encoder_then_predictor_blocks():
    """Block iterator visits ``encoder.blocks`` first, then ``predictor.blocks``.

    Setup: tiny JEPAModel with depth=2, predictor_depth=2.
    Expected: total of 4 blocks; first 2 are encoder, last 2 are predictor;
    yielded in declaration order.
    """
    torch.manual_seed(0)
    cfg = _tiny_cfg(depth=2, predictor_depth=2)
    model = JEPAModel(cfg)
    profile = jepa_profile()

    blocks = list(profile.iter_blocks(model))
    assert len(blocks) == 4
    enc_blocks = list(model.encoder.blocks)
    pred_blocks = list(model.predictor.blocks)
    # First half are encoder blocks in order
    for i in range(2):
        assert blocks[i] is enc_blocks[i], f"encoder block {i} mismatch"
    # Second half are predictor blocks
    for i in range(2):
        assert blocks[2 + i] is pred_blocks[i], f"predictor block {i} mismatch"


def test_jepa_profile_embedding_returns_encoder_proj():
    """``get_embedding`` on jepa_profile returns ``model.encoder.proj``
    (the patch-to-embed projection layer).
    """
    cfg = _tiny_cfg()
    model = JEPAModel(cfg)
    profile = jepa_profile()
    emb = profile.get_embedding(model)
    assert emb is model.encoder.proj
