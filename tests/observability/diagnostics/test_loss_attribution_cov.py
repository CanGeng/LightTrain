"""Edge-case coverage for ``loss_attribution`` beyond the happy-path file.

Companion to ``test_loss_attribution.py`` (which pins the sample/token/module
shapes against a real ``TinyCausalLM``). Here we drive the remaining branches:

* :func:`compute_loss_attribution` module level — the inner grad-norm loop
  (non-``None`` grads recorded, ``None`` grads skipped) and the forward-hook
  removal failure warning path, exercised with tiny stub modules whose hook
  captures the exact tensor ``loss`` was built from (the real model re-run
  produces fresh, disconnected tensors so this loop is otherwise unreachable).
* :func:`render_attribution_markdown` — all three section branches plus the
  empty-report base case.
* :class:`LossAttributionCallback` — constructor clamping/coercion,
  ``on_train_start`` ctx/trainer/none run-dir + model resolution,
  ``on_loss_computed`` latching, ``on_step_end`` / ``on_nan_detected`` gating,
  and ``_dump`` (guard early-return, force-module append, compute-failure
  swallow, and the JSON + Markdown artifact write).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from lighttrain.builtin_plugins.losses.core import CrossEntropyLoss
from lighttrain.builtin_plugins.models.text.tiny_lm import TinyCausalLM
from lighttrain.builtin_plugins.observability.diagnostics.loss_attribution import (
    LossAttributionCallback,
    compute_loss_attribution,
    render_attribution_markdown,
)
from lighttrain.protocols import LossContext

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _ConstChild(nn.Module):
    """Child whose forward returns a fixed pre-built graph tensor.

    The module-level forward hook in ``compute_loss_attribution`` only records a
    grad norm when the captured output is part of ``loss``'s graph. A real model
    re-run yields fresh tensors, so we return the *same* tensor object that the
    caller used to build ``loss``. The dummy parameter keeps the child distinct
    from the root in ``named_modules``.
    """

    def __init__(self, out: torch.Tensor) -> None:
        super().__init__()
        self._out = out
        self.p = nn.Parameter(torch.zeros(1))

    def forward(self, _x: Any = None) -> torch.Tensor:
        return self._out


class _StubModel(nn.Module):
    """Root model exposing one or two ``_ConstChild`` submodules."""

    def __init__(self, **children: torch.Tensor) -> None:
        super().__init__()
        for name, tensor in children.items():
            setattr(self, name, _ConstChild(tensor))

    def forward(self, **_batch: Any) -> torch.Tensor:
        last = None
        for child in self.children():
            last = child(None)
        return last  # type: ignore[return-value]


class _BadHandle:
    """Hook handle whose ``remove`` raises, to drive the warning branch."""

    def remove(self) -> None:
        raise RuntimeError("boom")


class _BadHookChild(_ConstChild):
    """Child that hands back a handle whose ``remove`` raises."""

    def register_forward_hook(self, _hook: Any) -> Any:  # type: ignore[override]  # noqa: D401
        return _BadHandle()


class _Ctx:
    """Minimal context object exposing ``run_dir`` / ``model`` attributes."""

    def __init__(self, *, run_dir: Any, model: Any) -> None:
        self.run_dir = run_dir
        self.model = model


class _Trainer:
    """Minimal trainer object exposing ``_run_dir`` / ``model``."""

    def __init__(self, *, run_dir: Any, model: Any) -> None:
        self._run_dir = run_dir
        self.model = model


class _BadOutputs:
    """Outputs whose ``.outputs['logits']`` is 1-D, breaking the (B,T,V) unpack."""

    outputs = {"logits": torch.randn(5)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _real_setup() -> tuple[TinyCausalLM, dict[str, torch.Tensor], Any, torch.Tensor]:
    torch.manual_seed(0)
    model = TinyCausalLM(vocab_size=32, d_model=16, n_layers=2, n_heads=2, max_seq_len=8)
    batch = {
        "input_ids": torch.randint(0, 32, (3, 6)),
        "attention_mask": torch.ones(3, 6, dtype=torch.long),
        "labels": torch.randint(0, 32, (3, 6)),
    }
    out = model(**batch)
    loss = CrossEntropyLoss()(out, batch, LossContext())["loss"]
    return model, batch, out, loss


# ---------------------------------------------------------------------------
# compute_loss_attribution — module level inner loop
# ---------------------------------------------------------------------------


def test_invariant_module_records_norm_for_connected_capture():
    """A captured output in ``loss``'s graph yields its 2-norm in ``top_k``."""
    torch.manual_seed(0)
    base = torch.randn(2, 4, requires_grad=True)
    captured = base * 2.0
    model = _StubModel(inner=captured)
    loss = captured.sum()

    report = compute_loss_attribution(
        model=model, batch={}, outputs=None, loss=loss, levels=("module",), top_k_modules=3
    )

    top = report["module"]["top_k"]
    assert top == [("inner", pytest.approx(float(torch.ones_like(captured).norm(2))))]


def test_invariant_module_skips_unused_grad_keeps_connected():
    """``allow_unused`` ``None`` grads are skipped; only the connected child survives."""
    torch.manual_seed(0)
    base = torch.randn(2, 4, requires_grad=True)
    connected = base * 3.0
    unconnected = torch.randn(2, 4, requires_grad=True)  # separate graph -> grad None
    model = _StubModel(c=connected, u=unconnected)
    loss = connected.sum()

    report = compute_loss_attribution(
        model=model, batch={}, outputs=None, loss=loss, levels=("module",), top_k_modules=5
    )

    top = dict(report["module"]["top_k"])
    assert "c" in top and "u" not in top
    assert top["c"] == pytest.approx(float(torch.ones_like(connected).norm(2)))


def test_invariant_module_top_k_truncates_and_sorts_desc():
    """``top_k`` is truncated to ``top_k_modules`` and sorted by descending norm."""
    torch.manual_seed(0)
    base = torch.randn(2, 4, requires_grad=True)
    # Distinct scalings -> distinct, ordered grad norms (grad of sum is ones,
    # so norm scales with element count which is equal; use distinct sizes).
    big = base.sum() * torch.ones(8, requires_grad=True)
    small = base.sum() * torch.ones(2, requires_grad=True)
    model = _StubModel(big=big, small=small)
    loss = big.sum() + small.sum()

    report = compute_loss_attribution(
        model=model, batch={}, outputs=None, loss=loss, levels=("module",), top_k_modules=1
    )

    top = report["module"]["top_k"]
    assert len(top) == 1
    assert top[0][0] == "big"  # larger tensor -> larger grad norm -> first


def test_invariant_module_hook_remove_failure_is_warned_not_raised(caplog):
    """A handle whose ``remove`` raises is swallowed with a warning, not re-raised."""
    torch.manual_seed(0)
    captured = torch.randn(2, 4, requires_grad=True) * 2.0
    model = _StubModel(inner=captured)
    model.inner = _BadHookChild(captured)  # swap in a child with a failing handle
    loss = captured.sum()

    with caplog.at_level("WARNING"):
        report = compute_loss_attribution(
            model=model, batch={}, outputs=None, loss=loss, levels=("module",)
        )

    assert "module" in report  # function still returned normally
    assert any("failed to remove a forward hook" in r.message for r in caplog.records)


def test_invariant_module_skipped_when_loss_not_tensor():
    """``module`` level is a no-op when ``loss`` is not a tensor."""
    model = _StubModel(inner=torch.randn(2, 2, requires_grad=True))
    report = compute_loss_attribution(
        model=model, batch={}, outputs=None, loss=0.5, levels=("module",)
    )
    assert "module" not in report
    assert report["levels"] == ["module"]


def test_invariant_module_skipped_when_model_none():
    """``module`` level is a no-op when ``model`` is ``None`` even with a tensor loss."""
    report = compute_loss_attribution(
        model=None,
        batch={},
        outputs=None,
        loss=torch.tensor(1.0, requires_grad=True),
        levels=("module",),
    )
    assert "module" not in report


def test_invariant_module_real_model_has_no_connected_capture():
    """Real model re-run yields disconnected tensors -> empty ``top_k`` (current design)."""
    model, batch, out, loss = _real_setup()
    report = compute_loss_attribution(
        model=model, batch=batch, outputs=out, loss=loss, levels=("module",), top_k_modules=5
    )
    assert report["module"]["top_k"] == []


# ---------------------------------------------------------------------------
# render_attribution_markdown
# ---------------------------------------------------------------------------


def test_invariant_render_all_three_sections():
    """All three sections render with formatted values and the step header."""
    report = {
        "sample": {"loss_per_sample": [1.5, 2.25]},
        "module": {"top_k": [("blocks.0", 0.5), ("lm_head", 0.25)]},
        "token": {"loss_per_token": [[1.0]]},
    }
    md = render_attribution_markdown(report, step=7)

    assert md.startswith("# Loss attribution — step 7")
    assert "## Per-sample loss" in md
    assert "- sample[0] = 1.5000" in md
    assert "- sample[1] = 2.2500" in md
    assert "## Top modules by ∂loss/∂out norm" in md
    assert "- `blocks.0` :: 0.5000" in md
    assert "- `lm_head` :: 0.2500" in md
    assert "## Per-token loss matrix" in md
    assert "omitted from markdown" in md


def test_invariant_render_empty_report_only_header():
    """An empty report renders just the header line."""
    assert render_attribution_markdown({}, step=0) == "# Loss attribution — step 0\n"


def test_invariant_render_sample_only_omits_other_sections():
    """A sample-only report omits the module and token sections."""
    md = render_attribution_markdown({"sample": {"loss_per_sample": []}}, step=3)
    assert "## Per-sample loss" in md
    assert "Top modules" not in md
    assert "Per-token loss matrix" not in md


# ---------------------------------------------------------------------------
# LossAttributionCallback — constructor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw, expected", [(0, 1), (-5, 1), (1, 1), (250, 250)])
def test_invariant_ctor_clamps_every_n_steps_to_at_least_one(raw, expected):
    """``every_n_steps`` is coerced to ``int`` and floored at 1."""
    cb = LossAttributionCallback(every_n_steps=raw)
    assert cb.every_n_steps == expected


def test_invariant_ctor_coerces_levels_and_on_nan():
    """``levels`` is tuple-ified and ``on_nan`` is coerced to ``bool``."""
    cb = LossAttributionCallback(levels=["sample"], on_nan=0)  # type: ignore[arg-type]
    assert cb.levels == ("sample",)
    assert cb.on_nan is False
    assert cb._run_dir is None and cb._model is None


# ---------------------------------------------------------------------------
# LossAttributionCallback — on_train_start
# ---------------------------------------------------------------------------


def test_invariant_on_train_start_prefers_ctx(tmp_path):
    """When ctx carries a run_dir/model, both are taken from ctx."""
    cb = LossAttributionCallback()
    cb.on_train_start(ctx=_Ctx(run_dir=tmp_path, model="ctx-model"), trainer=None)
    assert cb._run_dir == Path(tmp_path)
    assert cb._model == "ctx-model"


def test_invariant_on_train_start_falls_back_to_trainer(tmp_path):
    """ctx with a ``None`` run_dir falls back to ``trainer._run_dir``; model from ctx."""
    cb = LossAttributionCallback()
    cb.on_train_start(
        ctx=_Ctx(run_dir=None, model="ctx-model"),
        trainer=_Trainer(run_dir=tmp_path, model="trainer-model"),
    )
    assert cb._run_dir == Path(tmp_path)
    # model is read from ctx (ctx is not None branch), not the trainer.
    assert cb._model == "ctx-model"


def test_invariant_on_train_start_trainer_only(tmp_path):
    """With ctx ``None`` the trainer supplies both run_dir and model."""
    cb = LossAttributionCallback()
    cb.on_train_start(trainer=_Trainer(run_dir=tmp_path, model="trainer-model"))
    assert cb._run_dir == Path(tmp_path)
    assert cb._model == "trainer-model"


def test_invariant_on_train_start_none_leaves_run_dir_none():
    """With neither ctx nor trainer, run_dir and model stay ``None``."""
    cb = LossAttributionCallback()
    cb.on_train_start()
    assert cb._run_dir is None
    assert cb._model is None


# ---------------------------------------------------------------------------
# LossAttributionCallback — on_loss_computed latching
# ---------------------------------------------------------------------------


def test_invariant_on_loss_computed_latches_inputs():
    """The latest outputs/batch/loss are stored; a provided model overrides."""
    cb = LossAttributionCallback()
    cb.on_loss_computed(step=1, loss="L", outputs="O", batch="B", model="M")
    assert (cb._latest_outputs, cb._latest_batch, cb._latest_loss) == ("O", "B", "L")
    assert cb._model == "M"


def test_invariant_on_loss_computed_keeps_model_when_none_passed():
    """A ``None`` model does not clobber a previously-resolved model."""
    cb = LossAttributionCallback()
    cb._model = "existing"
    cb.on_loss_computed(loss="L", outputs="O", batch="B", model=None)
    assert cb._model == "existing"


# ---------------------------------------------------------------------------
# LossAttributionCallback — on_step_end / on_nan_detected gating
# ---------------------------------------------------------------------------


def test_invariant_on_step_end_noop_without_run_dir(tmp_path):
    """No run_dir -> ``on_step_end`` writes nothing."""
    cb = LossAttributionCallback(every_n_steps=1)
    model, batch, out, loss = _real_setup()
    cb.on_loss_computed(loss=loss, outputs=out, batch=batch, model=model)
    cb.on_step_end(step=1)  # run_dir is None
    assert not (tmp_path / "diagnostics").exists()


@pytest.mark.parametrize("step", [0, -1, 3])
def test_invariant_on_step_end_skips_non_multiple_or_nonpositive(tmp_path, step):
    """Steps <= 0 or not a multiple of ``every_n_steps`` produce no artifacts."""
    cb = LossAttributionCallback(every_n_steps=10)
    cb._run_dir = tmp_path
    model, batch, out, loss = _real_setup()
    cb.on_loss_computed(loss=loss, outputs=out, batch=batch, model=model)
    cb.on_step_end(step=step)  # 0, -1 gated; 3 not a multiple of 10
    assert not (tmp_path / "diagnostics").exists()


def test_invariant_on_step_end_writes_on_multiple(tmp_path):
    """A positive multiple of ``every_n_steps`` writes JSON + Markdown artifacts."""
    cb = LossAttributionCallback(every_n_steps=10)
    cb._run_dir = tmp_path
    model, batch, out, loss = _real_setup()
    cb.on_loss_computed(loss=loss, outputs=out, batch=batch, model=model)
    cb.on_step_end(step=10)

    diag = tmp_path / "diagnostics"
    assert (diag / "loss_attribution_10.json").exists()
    assert (diag / "loss_attribution_10.md").exists()
    payload = json.loads((diag / "loss_attribution_10.json").read_text(encoding="utf-8"))
    assert set(payload["levels"]) == {"sample", "token"}
    assert "module" not in payload  # force_module=False on the periodic path


def test_invariant_on_nan_detected_gated_by_flag(tmp_path):
    """``on_nan=False`` suppresses the NaN dump entirely."""
    cb = LossAttributionCallback(every_n_steps=10, on_nan=False)
    cb._run_dir = tmp_path
    model, batch, out, loss = _real_setup()
    cb.on_loss_computed(loss=loss, outputs=out, batch=batch, model=model)
    cb.on_nan_detected(step=5)
    assert not (tmp_path / "diagnostics").exists()


def test_invariant_on_nan_detected_noop_without_run_dir():
    """No run_dir -> ``on_nan_detected`` returns without writing."""
    cb = LossAttributionCallback()
    # Should not raise even though no inputs were latched.
    cb.on_nan_detected(step=5)
    assert cb._run_dir is None


def test_invariant_on_nan_detected_forces_module_level(tmp_path):
    """The NaN path appends ``module`` to the levels even off the periodic cadence."""
    cb = LossAttributionCallback(every_n_steps=1000, levels=("sample",))
    cb._run_dir = tmp_path
    model, batch, out, loss = _real_setup()
    cb.on_loss_computed(loss=loss, outputs=out, batch=batch, model=model)
    cb.on_nan_detected(step=5)

    payload = json.loads(
        (tmp_path / "diagnostics" / "loss_attribution_5.json").read_text(encoding="utf-8")
    )
    assert "module" in payload["levels"]
    assert "module" in payload  # module section present (top_k may be empty for real model)


# ---------------------------------------------------------------------------
# LossAttributionCallback — _dump guards / failure swallow
# ---------------------------------------------------------------------------


def test_invariant_dump_noop_when_no_latched_outputs(tmp_path):
    """``_dump`` returns early when no outputs have been latched yet."""
    cb = LossAttributionCallback(every_n_steps=1)
    cb._run_dir = tmp_path
    cb.on_step_end(step=1)  # _latest_outputs/_latest_batch still None
    assert not (tmp_path / "diagnostics").exists()


def test_invariant_dump_swallows_compute_failure(tmp_path, caplog):
    """A raising ``compute_loss_attribution`` is logged and produces no artifacts."""
    cb = LossAttributionCallback(every_n_steps=1)
    cb._run_dir = tmp_path
    # 1-D logits -> the (B,T,V) unpack raises inside compute_loss_attribution.
    cb._latest_outputs = _BadOutputs()
    cb._latest_batch = {"labels": torch.zeros(5, dtype=torch.long)}
    cb._latest_loss = None
    cb._model = None

    with caplog.at_level("WARNING"):
        cb.on_step_end(step=1)

    assert not (tmp_path / "diagnostics").exists()
    assert any("compute failed at step" in r.message for r in caplog.records)
