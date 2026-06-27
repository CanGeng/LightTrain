"""Tests for ``lighttrain.builtin_plugins.layer_offload._activation`` Block D.

* T7 ``test_cpu_fail_loud_*``: ``ActivationManager(mode='offload',
  device=cpu)`` raises ``RuntimeError`` at ``wrap()`` time — fail-loud
  contract (silent fallback would re-create the v0.2.3 "selectable-but-
  no-op" footgun).
* T6 ``test_offload_equals_recompute_gpu`` (``@pytest.mark.gpu``): on a real
  CUDA device, ``mode='offload'`` produces a loss within ``atol=1e-5`` of
  ``mode='recompute'`` after one forward+backward on the same model+seed.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from lighttrain.builtin_plugins.layer_offload._activation import (
    ActivationManager,
    _OffloadWrap,
)

# ---------------------------------------------------------------------------
# T7 — CPU fail-loud
# ---------------------------------------------------------------------------


def test_cpu_fail_loud_wrap_raises_runtime_error() -> None:
    """``wrap()`` with ``mode='offload'`` on a CPU ``device`` raises
    ``RuntimeError`` — instead of silently falling back to recompute."""
    mgr = ActivationManager(mode="offload", device=torch.device("cpu"))
    layer = nn.Linear(4, 4)
    raised: str | None = None
    try:
        mgr.wrap(layer)
    except RuntimeError as exc:
        raised = str(exc)
    assert raised is not None, (
        "ActivationManager(mode='offload', device='cpu').wrap(...) must raise"
    )
    assert "CUDA" in raised, f"expected CUDA-hinting RuntimeError, got: {raised!r}"


def test_cpu_fail_loud_unknown_device_raises() -> None:
    """``wrap()`` with ``device=None`` (default) and ``mode='offload'`` raises."""
    mgr = ActivationManager(mode="offload", device=None)
    raised = False
    try:
        mgr.wrap(nn.Linear(4, 4))
    except RuntimeError:
        raised = True
    assert raised


def test_offload_wrap_forward_rejects_cpu_grad_input() -> None:
    """``_OffloadWrap.forward`` raises when its input ``x`` is non-CUDA
    AND ``requires_grad=True`` — defensive in depth for the rare case of
    direct construction bypassing the manager.

    Without requires_grad, ``forward`` short-circuits (no offload, no raise)
    — covered by ``test_offload_wrap_forward_no_grad_input_skips_offload``.
    """
    wrap = _OffloadWrap(nn.Linear(4, 4))
    x = torch.randn(2, 4, requires_grad=True)  # CPU, requires_grad
    raised: str | None = None
    try:
        wrap(x)
    except RuntimeError as exc:
        raised = str(exc)
    assert raised is not None
    assert "CUDA" in raised


def test_offload_wrap_forward_no_grad_input_skips_offload() -> None:
    """When the input doesn't require grad, ``forward`` runs the layer
    and skips the offload bookkeeping — important for inference / no-grad."""
    layer = nn.Linear(4, 4).eval()
    wrap = _OffloadWrap(layer)
    x = torch.randn(2, 4, requires_grad=False)
    y = wrap(x)
    assert y.shape == (2, 4)


def test_recompute_modes_unchanged_on_cpu() -> None:
    """Regression: ``mode='recompute'`` and ``mode='recompute_or_offload'``
    must still work on CPU — Block D must not perturb the recompute path."""
    for mode in ("recompute", "recompute_or_offload"):
        mgr = ActivationManager(mode=mode, device=torch.device("cpu"))
        wrapped = mgr.wrap(nn.Linear(4, 4))
        assert wrapped is not None


# ---------------------------------------------------------------------------
# T6 — GPU numerical equivalence vs recompute
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_offload_equals_recompute_gpu() -> None:
    """On a real CUDA device, ``mode='offload'`` matches ``mode='recompute'``
    in loss (atol=1e-5) and produces a gradient of the same shape.

    Setup:
      * Two identical models with the same seed (Linear+ReLU+Linear).
      * Layer 0 is wrapped via the chosen activation policy.
      * One forward + backward on a fixed input/target.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(0)
    model_a = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8))
    torch.manual_seed(0)
    model_b = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8))

    device = torch.device("cuda")
    model_a = model_a.to(device).train()
    model_b = model_b.to(device).train()

    mgr_recompute = ActivationManager(mode="recompute", device=device)
    mgr_offload = ActivationManager(mode="offload", device=device)

    model_a[0] = mgr_recompute.wrap(model_a[0]).to(device).train()
    model_b[0] = mgr_offload.wrap(model_b[0]).to(device).train()

    torch.manual_seed(42)
    x = torch.randn(3, 8, device=device, requires_grad=True)
    target = torch.randn(3, 8, device=device)

    loss_a = ((model_a(x) - target) ** 2).mean()
    loss_a.backward()
    loss_b = ((model_b(x) - target) ** 2).mean()
    loss_b.backward()

    assert torch.allclose(loss_a, loss_b, atol=1e-5), (
        f"loss mismatch: recompute={loss_a.item():.6e} "
        f"offload={loss_b.item():.6e}"
    )

    grad_a = model_a[0].layer.weight.grad  # type: ignore[union-attr]
    grad_b = model_b[0].layer.weight.grad  # type: ignore[union-attr]
    assert grad_a is not None and grad_b is not None
    assert grad_a.shape == grad_b.shape
    assert torch.allclose(grad_a, grad_b, atol=1e-5), (  # type: ignore[arg-type]
        f"grad mismatch: |recompute|={grad_a.norm().item():.6e} "  # type: ignore[operator]
        f"|offload|={grad_b.norm().item():.6e}"  # type: ignore[operator]
    )
