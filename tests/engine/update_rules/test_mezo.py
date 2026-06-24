"""Adversarial tests for MeZOUpdateRule — gradient-free zeroth-order update.

Hard control-flow invariants this file pins:
  - ``.backward()`` is NEVER called.
  - ``optimizer.step()`` is NEVER called (MeZO modifies params directly).
  - param.grad stays None throughout.
  - The 3-perturb sequence (+ε, -2ε, +ε) is symmetric, so params return to
    original state when ``grad_est == 0``.
  - With ``seed_per_step=False``, two MeZO instances are bit-exactly
    reproducible from identical initial weights.

The numerical correctness of ``grad_est = (L+ − L−) / (2ε)`` w.r.t. the
true loss gradient is delegated to losses/rl group. We test the
*control-flow contract* here.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import torch
import torch.nn as nn

from lighttrain.builtin_plugins.update_rules.mezo import MeZOUpdateRule
from lighttrain.callbacks.base import EventBus
from lighttrain.engine._context import StepContext
from lighttrain.protocols import ModelOutput

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    def __init__(self, init_value: float = 1.0) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 1, bias=False)
        with torch.no_grad():
            self.linear.weight.fill_(init_value)

    def forward(self, x, **_):
        return ModelOutput(outputs={"logits": self.linear(x)})


def _const_loss(_out, _batch, _ctx):
    """Loss that is constant w.r.t. θ — ``grad_est`` will be ~0."""
    return {"loss": torch.tensor(1.0)}


def _quadratic_loss(model_output, batch, ctx):
    """MSE-to-ones — depends on θ, so grad_est != 0."""
    pred = model_output.outputs["logits"]
    return {"loss": (pred - 1.0).pow(2).mean()}


def _build_ctx(*, callbacks=None, loss_fn=_quadratic_loss, init_value: float = 1.0):
    model = _TinyModel(init_value=init_value)
    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    ctx = StepContext(
        model=model,
        optimizer=optim,
        bus=EventBus(callbacks or []),
        loss_fn=loss_fn,
    )
    return ctx, model, optim


def _batch():
    return {"x": torch.randn(2, 4)}


class _OrderedRecorder:
    def __init__(self) -> None:
        self.events: list[str] = []

    def _h(name):
        def _f(self, **_kw):
            self.events.append(name)

        return _f

    on_step_begin = _h("on_step_begin")
    on_forward_pre = _h("on_forward_pre")
    on_forward_post = _h("on_forward_post")
    on_loss_computed = _h("on_loss_computed")
    on_backward_pre = _h("on_backward_pre")
    on_backward_post = _h("on_backward_post")
    on_clip_grad = _h("on_clip_grad")
    on_optimizer_step_pre = _h("on_optimizer_step_pre")
    on_optimizer_step_post = _h("on_optimizer_step_post")
    on_step_end = _h("on_step_end")


# ===========================================================================
# Hard invariant: no backward
# ===========================================================================


def test_mezo_never_calls_backward():
    """Goal: ``Tensor.backward`` is never invoked during a MeZO step.

    Construction: patch ``torch.Tensor.backward`` with a recording wrapper.

    Catches a refactor that "fixes" MeZO by adding backward — would break
    its memory-free design (full activations would be stored).
    """
    backward_calls = [0]
    original_backward = torch.Tensor.backward

    def _recording_backward(self, *args, **kwargs):
        backward_calls[0] += 1
        return original_backward(self, *args, **kwargs)

    torch.Tensor.backward = _recording_backward
    try:
        ctx, model, _ = _build_ctx()
        MeZOUpdateRule(eps=1e-3).step(model, _batch(), ctx)
    finally:
        torch.Tensor.backward = original_backward

    assert backward_calls[0] == 0


def test_mezo_grads_remain_none_after_step():
    """Goal: after a MeZO step, every parameter's ``.grad`` is still None.

    Pins the "memory free" property: MeZO must never populate gradients,
    even transiently.
    """
    ctx, model, _ = _build_ctx()

    MeZOUpdateRule(eps=1e-3).step(model, _batch(), ctx)

    for p in model.parameters():
        assert p.grad is None


def test_mezo_regression_does_not_call_optimizer_step():
    """Goal: MeZO applies its own gradient-free update via ``_apply_update``;
    it must NOT delegate to ``optimizer.step()``.

    Catches a refactor that "harmonizes" MeZO with other update rules by
    calling optimizer.step — would double-update params (once via MeZO,
    once via the optimizer reading None grads which most optimizers
    treat as 0 but a bad fix could change).
    """
    ctx, model, optim = _build_ctx()
    step_spy = MagicMock(wraps=optim.step)
    optim.step = step_spy

    MeZOUpdateRule(eps=1e-3).step(model, _batch(), ctx)

    step_spy.assert_not_called()


# ===========================================================================
# Perturbation symmetry
# ===========================================================================


def test_mezo_perturb_restore_sequence_returns_to_original_theta():
    """Goal: when ``grad_est == 0`` (constant loss), the 3-perturb sequence
    (+ε, -2ε, +ε) is symmetric and θ ends exactly where it started.

    Then ``_apply_update`` does ``θ -= lr * 0 * z`` which is a no-op,
    so θ_final == θ_initial.

    Construction: loss_fn returns a constant. After step, weight must
    exactly equal the pre-step snapshot.

    Catches a refactor that breaks perturb-restore symmetry (e.g.,
    drops the final +ε restore at line 164).
    """
    ctx, model, _ = _build_ctx(loss_fn=_const_loss, init_value=2.5)
    before = model.linear.weight.detach().clone()

    metrics = MeZOUpdateRule(eps=1e-3).step(model, _batch(), ctx)

    after = model.linear.weight.detach().clone()
    torch.testing.assert_close(after, before, atol=1e-6, rtol=1e-5)
    # grad_est should be 0 (loss was constant)
    torch.testing.assert_close(
        torch.tensor(metrics["grad_est"]),
        torch.tensor(0.0),
        atol=1e-6,
        rtol=1e-5,
    )


def test_mezo_perturbation_seed_reproducible():
    """Goal: with ``seed_per_step=False``, two MeZO instances starting from
    identical θ converge to identical θ after one step on the same batch
    and loss_fn.

    Catches a refactor that introduces nondeterminism (e.g., reads
    torch.randn from global RNG instead of the seeded generator).
    """
    torch.manual_seed(42)
    batch_a = _batch()
    torch.manual_seed(42)
    batch_b = _batch()

    ctx_a, model_a, _ = _build_ctx(init_value=1.0)
    ctx_b, model_b, _ = _build_ctx(init_value=1.0)

    rule_a = MeZOUpdateRule(eps=1e-3, seed_per_step=False)
    rule_b = MeZOUpdateRule(eps=1e-3, seed_per_step=False)

    rule_a.step(model_a, batch_a, ctx_a)
    rule_b.step(model_b, batch_b, ctx_b)

    torch.testing.assert_close(
        model_a.linear.weight, model_b.linear.weight, atol=1e-6, rtol=1e-5
    )


def test_mezo_seed_per_step_true_rotates_seed_across_steps():
    """Goal (adversarial — attack path 'seed_per_step is just for show'):
    a refactor that uses a constant seed even when ``seed_per_step=True``
    would silently break ZO optimization — every step would perturb
    in the same direction, defeating the gradient estimator.

    Construction: with seed_per_step=True, run TWO consecutive steps
    from identical θ on identical batches and loss_fn. Snapshot the
    weight deltas Δ_a = θ_after_step1 - θ_initial and
    Δ_b = θ_after_step2 - θ_after_step1.

    Expected: the two deltas point in DIFFERENT directions
    (cosine similarity < 1, allowing some slack).

    Catches a refactor that hardcodes the seed regardless of
    seed_per_step.
    """
    torch.manual_seed(99)

    ctx, model, _ = _build_ctx(init_value=2.0)
    rule = MeZOUpdateRule(eps=1e-3, seed_per_step=True)

    before_step1 = model.linear.weight.detach().clone()
    rule.step(model, _batch(), ctx)
    after_step1 = model.linear.weight.detach().clone()
    rule.step(model, _batch(), ctx)
    after_step2 = model.linear.weight.detach().clone()

    delta_1 = (after_step1 - before_step1).flatten()
    delta_2 = (after_step2 - after_step1).flatten()

    # Both updates must be non-zero
    assert delta_1.norm().item() > 1e-9
    assert delta_2.norm().item() > 1e-9

    # Cosine similarity strictly less than 1 — directions differ.
    cos = torch.nn.functional.cosine_similarity(
        delta_1.unsqueeze(0), delta_2.unsqueeze(0), dim=1
    ).item()
    assert cos < 0.999, (
        f"Two consecutive steps used same perturbation direction (cos={cos}); "
        "seed_per_step=True is being ignored."
    )


def test_mezo_apply_update_uses_current_lr():
    """Goal: the update step ``θ -= lr * grad_est * z`` reads lr from the
    optimizer at apply-time. Doubling lr must roughly double the update
    magnitude (for the same grad_est and z).

    Construction:
      - run two MeZO steps starting from identical θ on identical batches
      - one with lr=1e-3, one with lr=2e-3 (seed_per_step=False, same eps)
      - measure ||Δθ|| in each case; assert ratio ≈ 2.

    Catches a refactor that hardcodes lr or reads it from a stale field.
    """
    torch.manual_seed(0)
    batch_a = _batch()
    torch.manual_seed(0)
    batch_b = _batch()

    model_a = _TinyModel(init_value=2.0)
    optim_a = torch.optim.SGD(model_a.parameters(), lr=1e-3)
    ctx_a = StepContext(
        model=model_a,
        optimizer=optim_a,
        bus=EventBus([]),
        loss_fn=_quadratic_loss,
    )

    model_b = _TinyModel(init_value=2.0)
    optim_b = torch.optim.SGD(model_b.parameters(), lr=2e-3)
    ctx_b = StepContext(
        model=model_b,
        optimizer=optim_b,
        bus=EventBus([]),
        loss_fn=_quadratic_loss,
    )

    before_a = model_a.linear.weight.detach().clone()
    before_b = model_b.linear.weight.detach().clone()

    MeZOUpdateRule(eps=1e-3, seed_per_step=False).step(model_a, batch_a, ctx_a)
    MeZOUpdateRule(eps=1e-3, seed_per_step=False).step(model_b, batch_b, ctx_b)

    delta_a = (model_a.linear.weight - before_a).norm().item()
    delta_b = (model_b.linear.weight - before_b).norm().item()

    # Update magnitude should scale linearly with lr (ratio ≈ 2).
    # Allow slack because z, grad_est are sample-noisy.
    assert delta_a > 1e-8
    ratio = delta_b / delta_a
    torch.testing.assert_close(
        torch.tensor(ratio), torch.tensor(2.0), atol=1e-3, rtol=1e-3
    )


# ===========================================================================
# Lifecycle
# ===========================================================================


def test_mezo_emits_step_begin_forward_post_step_end_only():
    """Goal: MeZO fires exactly these three bus events, in this order:
      [on_step_begin, on_forward_post, on_step_end]

    No on_backward_pre / on_backward_post / on_optimizer_step_* / on_clip_grad
    — MeZO bypasses all of those.

    Catches a refactor that adds spurious events to mimic the other rules.
    """
    rec = _OrderedRecorder()
    ctx, model, _ = _build_ctx(callbacks=[rec])

    MeZOUpdateRule(eps=1e-3).step(model, _batch(), ctx)

    assert rec.events == ["on_step_begin", "on_forward_post", "on_step_end"]


def test_mezo_state_dict_roundtrip():
    """Goal: state_dict persists eps, seed_per_step, step_count."""
    rule = MeZOUpdateRule(eps=2e-4, seed_per_step=False)
    rule._step_count = 7
    sd = rule.state_dict()
    assert sd == {"eps": 2e-4, "seed_per_step": False, "step_count": 7}

    rule2 = MeZOUpdateRule()
    rule2.load_state_dict(sd)
    assert rule2.eps == 2e-4
    assert rule2.seed_per_step is False
    assert rule2._step_count == 7
